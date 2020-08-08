import argparse
import glob
import json
import logging
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import (DataLoader, 
                              RandomSampler, 
                              SequentialSampler, 
                              TensorDataset)
from tqdm import tqdm, trange

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from utils import processors, output_modes, MODEL_CLASSES
from config import (NEGATION_MODEL, FMROLE_MODEL, FMSIDE_MODEL, LS_MODEL, RELATION_MODEL,
                    CLS_OUTPUT_ROOT, OBN_TEST, FMR_TEST, FMS_TEST, LSS_TEST, REL_TEST, REL_OUTPUT_ROOT)


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, task, tokenizer, evaluate=0):
    mode_dict = {0: 'train', 1: 'eval', 2: 'pred'}

    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.test_data_dir,
        "cached_{}_{}_{}_{}".format(
            # "dev" if evaluate else "train",
            mode_dict[evaluate],
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.test_data_dir)
        label_list = processor.get_labels()

        if evaluate == 0:
            examples = processor.get_train_examples(args.test_data_dir)
        elif evaluate == 1:
            examples = processor.get_dev_examples(args.test_data_dir)
        elif evaluate == 2:
            examples = processor.get_test_examples(args.test_data_dir)
        else:
            raise Exception("evaluate id must be 0, 1, 2 but get {}".format(evaluate))
        
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    logger.info("features: \n{}".format(features[:3]))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    else:
        raise RuntimeError("the {} is not classification or regression.".format(output_mode))

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def evaluate(args, model, tokenizer, prefix=1):

    eval_task = args.task_name 
    
    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=prefix)

    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type[:4] in ["bert", "xlne", "albe"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    return preds, out_label_ids, eval_loss


class Args:
    def __init__(self):
        self.test_data_dir = None
        self.model_type = None
        self.model_name_or_path = None
        self.task_name = None
        self.test_output_dir = None
        self.config_name = None
        self.tokenizer_name = None
        self.cache_dir = "./"
        self.max_seq_length = 256
        self.per_gpu_eval_batch_size = 8
        self.do_lower_case = True
        self.use_spec_tag = True
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.seed = 13
        self.local_rank = -1
        self.n_gpu = 0
        self.device = None
        self.fp16 = False
        self.output_mode = "classification"


def cls_rel_prediction(args):

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    
    processor = processors[args.task_name]()
    output_mode = output_modes[args.task_name]
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=False,
                config=config,
    )

    model.to(args.device)
    
    preds, out_label_ids, eval_loss = evaluate(args, model, tokenizer, prefix=2)
    Path(args.test_output_dir).mkdir(parents=True, exist_ok=True)
    output_test_file = os.path.join(args.test_output_dir, "test_results.txt")

    with open(output_test_file, "w") as writer:
        if output_mode == "classification":
            writer.write("\n".join([label_list[e] for e in preds]))
        elif output_mode == "regression":
            writer.write("\n".join(map(lambda x: f"{x:.4f}", preds)))


def run_cls():
    args = Args()

    configs = {
        # task_name: (model_path, data_path, model_type)
        "obn": (NEGATION_MODEL, OBN_TEST, "bert"),
        "fmr": (FMROLE_MODEL, FMR_TEST, "berta"),
        "fms": (FMSIDE_MODEL, FMS_TEST, "berta"),
        "lss": (LS_MODEL, LSS_TEST, "berta")
    }

    for i in range(5):
        for k, v in configs.items():
            args.model_name_or_path = v[0]
            args.test_data_dir = v[1].format(i)
            args.model_type = v[2]
            args.task_name = k
            args.test_output_dir = CLS_OUTPUT_ROOT.format(k, i)
            args.config_name = v[0]
            args.tokenizer_name = v[0]
            cls_rel_prediction(args)


def run_rel():
    args = Args()
    conf = ("rel", RELATION_MODEL, REL_TEST, "bertr")
    args.model_name_or_path = conf[1]
    args.test_data_dir = conf[2]
    args.model_type = conf[3]
    args.task_name = conf[0]
    args.test_output_dir = REL_OUTPUT_ROOT
    args.config_name = conf[1]
    args.tokenizer_name = conf[1]
    cls_rel_prediction(args)
