# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import shutil

import numpy as np
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from layoutlm import FunsdDataset, LayoutlmConfig, LayoutlmForTokenClassification

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, LayoutlmConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "layoutlm": (LayoutlmConfig, LayoutlmForTokenClassification, BertTokenizer),
}


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def collate_fn(data):
    batch = [i for i in zip(*data)]
    for i in range(len(batch)):
        if i < len(batch) - 2:
            batch[i] = torch.stack(batch[i], 0)
    return tuple(batch)


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def train(train_dataset, model, tokenizer, labels, pad_token_label_id):  # noqa C901
    """Train the model"""
    if local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir="runs/" + os.path.basename(output_dir))

    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=None,
    )

    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = (
            max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
        )
    else:
        t_total = (
            len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size
        * gradient_accumulation_steps
        * (torch.distributed.get_world_size() if local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0]
    )
    set_seed(
        seed, n_gpu
    )  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "labels": batch[3].to(device),
            }
            if model_type in ["layoutlm"]:
                inputs["bbox"] = batch[4].to(device)
            inputs["token_type_ids"] = (
                batch[2].to(device) if model_type in ["bert", "layoutlm"] else None
            )  # RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    local_rank in [-1, 0]
                    and logging_steps > 0
                    and global_step % logging_steps == 0
                ):
                    # Log metrics
                    if (
                        local_rank in [-1, 0] and evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(
                            args,
                            model,
                            tokenizer,
                            labels,
                            pad_token_label_id,
                            mode="dev",
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                    local_rank in [-1, 0]
                    and save_steps > 0
                    and global_step % save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(os.path.join(output_dir, "training_bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    if local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = FunsdDataset(tokenizer, labels, pad_token_label_id, mode=mode)

    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
        collate_fn=None,
    )

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "labels": batch[3].to(device),
            }
            if model_type in ["layoutlm"]:
                inputs["bbox"] = batch[4].to(device)
            inputs["token_type_ids"] = (
                batch[2].to(device) if model_type in ["bert", "layoutlm"] else None
            )  # RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if n_gpu > 1:
                tmp_eval_loss = (
                    tmp_eval_loss.mean()
                )  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    report = classification_report(out_label_list, preds_list)
    logger.info("\n" + report)

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def main(
    data_dir,
    model_type,
    model_name_or_path,
    do_lower_case,
    max_seq_length,
    do_train,
    do_eval,
    evaluate_during_training,
    num_train_epochs,
    logging_steps,
    save_steps,
    output_dir,
    labels,
    per_gpu_train_batch_size,
    per_gpu_eval_batch_size,
    no_cuda,
):
    if os.path.exists(output_dir) and os.listdir(output_dir) and do_train:
        if not overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    output_dir
                )
            )
        else:
            if local_rank in [-1, 0]:
                shutil.rmtree(output_dir)

    if not os.path.exists(output_dir) and (do_eval or do_predict):
        raise ValueError(
            "Output directory ({}) does not exist. Please train and save the model before inference stage.".format(
                output_dir
            )
        )

    if not os.path.exists(output_dir) and do_train and local_rank in [-1, 0]:
        os.makedirs(output_dir)

    # Setup distant debugging if needed
    if server_ip and server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(server_ip, server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if local_rank == -1 or no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        )
        n_gpu = 0 if no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1
    device = device

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(output_dir, "train.log")
        if local_rank in [-1, 0]
        else None,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        local_rank,
        device,
        n_gpu,
        bool(local_rank != -1),
        fp16,
    )

    # Set seed
    set_seed(seed, n_gpu)

    labels = get_labels(labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model_type = model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(
        config_name if config_name else model_name_or_path,
        num_labels=num_labels,
        cache_dir=cache_dir if cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        do_lower_case=do_lower_case,
        cache_dir=cache_dir if cache_dir else None,
    )
    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir if cache_dir else None,
    )

    if local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)
    # Training
    if do_train:
        train_dataset = FunsdDataset(
            tokenizer, labels, pad_token_label_id, mode="train"
        )
        global_step, tr_loss = train(
            train_dataset, model, tokenizer, labels, pad_token_label_id
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if do_train and (local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(output_dir) and local_rank in [-1, 0]:
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(os.path.join(output_dir, "training_bin"))

    # Evaluation
    results = {}
    if do_eval and local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            output_dir, do_lower_case=do_lower_case
        )
        checkpoints = [output_dir]
        if eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(device)
            result, _ = evaluate(
                args,
                model,
                tokenizer,
                labels,
                pad_token_label_id,
                mode="test",
                prefix=global_step,
            )
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if do_predict and local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            model_name_or_path, do_lower_case=do_lower_case
        )
        model = model_class.from_pretrained(output_dir)
        model.to(device)
        result, predictions = evaluate(
            model, tokenizer, labels, pad_token_label_id, mode="test"
        )
        # Save results
        output_test_results_file = os.path.join(output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w", encoding="utf8") as writer:
            with open(os.path.join(data_dir, "test.txt"), "r", encoding="utf8") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        output_line = (
                            line.split()[0]
                            + " "
                            + predictions[example_id].pop(0)
                            + "\n"
                        )
                        writer.write(output_line)
                    else:
                        logger.warning(
                            "Maximum sequence length exceeded: No prediction for '%s'.",
                            line.split()[0],
                        )

    return results


if __name__ == "__main__":
    main(
        data_dir="data/data_resumes_2",
        model_type="layoutlm",
        model_name_or_path="layoutlm-large-uncased",
        do_lower_case=True,
        max_seq_length=512,
        do_train=True,
        do_eval=True,
        evaluate_during_training=True,
        num_train_epochs=100.0,
        logging_steps=1130,
        save_steps=22600,
        output_dir="resumes_512_2",
        labels="data/data_resumes_2/labels.txt",
        per_gpu_train_batch_size=2,
        per_gpu_eval_batch_size=2,
        no_cuda=True,
    )
