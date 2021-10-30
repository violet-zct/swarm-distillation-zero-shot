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
from ttt.dataloader import Task

import logging


# reload the t0 model after each test point or reset the biases when using bitfit;
# prompt tuning looks easier
#
logger = logging.getLogger(__name__)

def evalute_t0(model, tokenizer, test_data, data_args):
    predictions = [[] for _ in range(test_data.num_prompts)]
    golds = []
    for sidx in range(test_data.size):
        test_inputs, test_outputs, label = test_data[sidx]
        if isinstance(test_inputs[0], list):
            assert data_args.task_type == "classification"
        # single prompt
        for pidx, (prompted_test_input, prompted_test_output) in enumerate(zip(test_inputs, test_outputs)):
            max_ll, pred = 0, -1
            for ii, (pin, pout) in enumerate(zip(prompted_test_output, prompted_test_output)):
                input_ids = tokenizer.encode(pin, return_tensors="pt")  # .input_ids
                output_ids = tokenizer.encode(pout, return_tensors="pt")
                input_ids.to('cuda')
                output_ids.to('cuda')
                with torch.no_grad():
                    if data_args.task_type == "classification":
                        # log-likelihood per sequence
                        ll = -model.forward(input_ids=input_ids, labels=output_ids).loss
                        if ll > max_ll:
                            pred = ii
                            max_ll = ll
                    else:
                        # it seems that there is no actual generation tasks in T0 evaluation
                        decoded = model.generate(input)
            predictions[pidx].append(pred)
        if sidx % 100 == 0:
            logger.info("evaluating {}-th test examples".format(sidx))
        golds.append(label)
    accuracies = []
    for ppred in predictions:
        accuracies.append(sum(np.array(ppred) == np.array(golds)) * 1.0 / len(golds))
    logger.info("median accuracy = {}, max acc = {}, min acc ={}, var = {}".format(np.median(accuracies),
                                                                             np.max(accuracies),
                                                                             np.min(accuracies),
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

    test_data = Task(data_args, model_args.cache_dir)

    # without batching, to batch, collect all the processed examples first
    # todo: make this a function
    if test_args.test_mode == "t0":
        evalute_t0(model, tokenizer, test_data, data_args)
    elif test_args.test_mode == "ttt_t0":
        pass


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()