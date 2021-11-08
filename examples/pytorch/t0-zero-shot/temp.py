import torch

from transformers import AutoModelForSeq2SeqLM
import numpy as np
import sys
sys.path.insert(2, "./")

import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from ttt.options import *
from ttt.dataloader import Task, extract_original_task_prompts

import logging
import deepspeed

# reload the t0 model after each test point or reset the biases when using bitfit;
# prompt tuning looks easier
#
logger = logging.getLogger(__name__)


def chunks(tot, bsz):
    batches = [(i, i+bsz if i+bsz < tot else tot) for i in range(0, tot, bsz)]
    return batches


def batched_evalute_t0(model, tokenizer, test_data, data_args, batch_size, fp16, use_deepspeed=False):
    golds = []
    input_dataset = []
    output_dataset = []
    choice_nums = []

    if use_deepspeed:
        print("world size = {}".format(torch.distributed.get_world_size()))
        ds_engine = deepspeed.init_inference(model, mp_size=torch.distributed.get_world_size(),
                                         dtype=torch.half if fp16 else torch.float,
                                         checkpoint=None,
                                         replace_method='auto')
        model = ds_engine.module
    else:
        model.eval()
        if fp16:
            model = model.half()
        model = model.to(torch.cuda.current_device())

    for sidx in range(test_data.size):
        test_inputs, test_outputs, label = test_data[sidx]
        if isinstance(test_inputs[0], list):
            assert data_args.task_type == "classification"
        # single prompt
        for pidx, (prompted_test_input, prompted_test_output) in enumerate(zip(test_inputs, test_outputs)):
            for ii, (pin, pout) in enumerate(zip(prompted_test_input, prompted_test_output)):
                input_dataset.append(pin)
                output_dataset.append(pout)
            choice_nums.append(len(prompted_test_input))
        golds.append(label)

    all_loglikelihoods = []
    processed_batch = 0
    vocab = tokenizer.get_vocab()
    vocab = {v:k for k, v in vocab.items()}
    print(vocab[0], vocab[1], vocab[2])
    for bid1, bid2 in chunks(len(input_dataset), batch_size):
        tokenized_input = tokenizer(input_dataset[bid1:bid2], return_tensors="pt", padding='longest', truncation=True)
        input_ids, attention_mask = tokenized_input.input_ids, tokenized_input.attention_mask
        output_ids = tokenizer(output_dataset[bid1:bid2], return_tensors="pt", padding='longest', truncation=True).input_ids
        target_mask = torch.tensor([[1. if l != tokenizer.pad_token_id and l != tokenizer.eos_token_id else 0. for l in x] for x in output_ids]).float()
        output_ids = torch.tensor([[(l if l != tokenizer.pad_token_id else -100) for l in x] for x in output_ids])

        # fixme: deepspeed offload to cpu, which device should the inputs be put on?
        attention_mask = attention_mask.to(model.device)
        input_ids = input_ids.to(model.device)
        output_ids = output_ids.to(model.device)

        with torch.no_grad():
            if data_args.task_type == "classification":
                # log-likelihood per sequence
                ll = -model.forward(input_ids=input_ids, labels=output_ids, attention_mask=attention_mask).loss
                ll = ll.view(output_ids.size())
                ll = (ll * target_mask.to(ll.device)).sum(1).cpu().numpy()
                all_loglikelihoods.extend(ll)
            else:
                # it seems that there is no actual generation tasks in T0 evaluation
                decoded = model.generate(input)
        processed_batch += 1
        if processed_batch % 10 == 0:
            logger.info("evaluating {} batches of test examples".format(processed_batch))

    predictions = [[] for _ in range(test_data.num_prompts)]
    idx = 0
    for eidx in range(test_data.size):
        for pidx in range(test_data.num_prompts):
            max_ll, pred_label = -np.inf, -1
            # actually, the number of labels of each prompt should be the same
            for ii in range(choice_nums[eidx * test_data.num_prompts + pidx]):
                if all_loglikelihoods[idx] > max_ll:
                    max_ll, pred_label = all_loglikelihoods[idx], ii
                idx += 1
            predictions[pidx].append(pred_label)

    accuracies = []
    for ppred in predictions:
        accuracies.append(sum(np.array(ppred) == np.array(golds)) * 1.0 / len(golds))
    logger.info("median accuracy = {}, max acc = {}, min acc ={}, mean = {}, var = {}".format(np.median(accuracies),
                                                                             np.max(accuracies),
                                                                             np.min(accuracies),
                                                                             np.mean(accuracies),
                                                                             np.var(accuracies)))


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, TestArguments))
    model_args, data_args, training_args, test_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # set additional args
    for k, v in vars(test_args).items():
        if not hasattr(config, k):
            setattr(config, k, v)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    test_data = Task(data_args, model_args.cache_dir)

    # without batching, to batch, collect all the processed examples first
    # todo: make this a function
    if test_args.test_mode == "t0":
        batched_evalute_t0(model, tokenizer, test_data, data_args, training_args.per_gpu_eval_batch_size,
                           training_args.fp16, test_args.use_deepspeed)
    elif test_args.test_mode == "ttt_t0":
        test_dataset = datasets.load_dataset(data_args.dataset_name, data_args.subset_name, cache_dir=model_args.cache_dir)
        prompts, original_task_pnames = extract_original_task_prompts(data_args.dataset_name, data_args.subset_name)
        num_prompts = len(original_task_pnames)
        def preprocess_func(item):
            inputs = []
            outputs = []
            for pname in original_task_pnames:
                input_template, output_template = prompts[pname].apply(item)
            label = item['label']
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            test_dataset = test_dataset.map(
                preprocess_func,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=None,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()