import math
import os.path

import torch

from transformers import AutoModelForSeq2SeqLM
import numpy as np
import sys
sys.path.insert(2, "./")

import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from ttt.options import *
from ttt.utils import compute_metrics, summarize_metrics
from ttt.dataloader import DatasetByPrompt, TTTOnlineDataset, TTTOfflineDataset, TTTEvalDataset, \
    TTTOnlineTokenLossDataset, TTTOfflineTokenLossDataset, TTTOfflineLoopDataset
import logging

# reload the t0 model after each test point or reset the biases when using bitfit;
# prompt tuning looks easier
#
logger = logging.getLogger(__name__)


def chunks(tot, bsz):
    batches = [(i, i+bsz if i+bsz < tot else tot) for i in range(0, tot, bsz)]
    return batches


def batched_evalute_t0(model, tokenizer, test_data, data_args, batch_size, fp16, data_collator, metrics, model_name):
    # print("world size = {}".format(torch.distributed.get_world_size()))
    # ds_engine = deepspeed.init_inference(model, mp_size=torch.distributed.get_world_size(),
    #                                  dtype=torch.half if fp16 else torch.float,
    #                                  checkpoint=None,
    #                                  replace_method='auto')
    # model = ds_engine.module
    fout_name = "results/" + "_".join([data_args.dataset_name, data_args.subset_name, data_args.testset_name, model_name.replace("/", ".")])
    model.eval()
    if fp16:
        model = model.half()
    model = model.to(torch.cuda.current_device())

    all_data = []
    golds = []
    for sidx in range(len(test_data)):
        prompted_example, glabel = test_data[sidx]
        all_data.extend(prompted_example)
        golds.append(glabel)

    all_loglikelihoods = []
    processed_batch = 0
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    print(vocab[0], vocab[1], vocab[2])
    for bid1, bid2 in chunks(len(all_data), batch_size):
        model_inputs = all_data[bid1: bid2]
        model_inputs = data_collator(model_inputs)

        # fixme: deepspeed offload to cpu, put onto cuda:0?
        for k, v in model_inputs.items():
            model_inputs[k] = model_inputs[k].to(model.device)

        with torch.no_grad():
            if data_args.task_type == "classification":
                # log-likelihood per sequence
                ll = model(**model_inputs).loss
                all_loglikelihoods.extend(ll.cpu().numpy())
            else:
                # it seems that there is no actual generation tasks in T0 evaluation
                decoded = model.generate(input)
        processed_batch += 1
        if processed_batch % 10 == 0:
            logger.info("evaluating {} batches of test examples".format(processed_batch))

    results = compute_metrics(all_loglikelihoods, len(test_data), test_data.num_choices, test_data.num_prompts,
                              golds, metrics, fout_name=fout_name)
    for k, v in results.items():
        logger.info("{} = {}".format(k, v))


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, TestArguments))
    model_args, data_args, training_args, test_args = parser.parse_args_into_dataclasses()
    model_args.model_name_or_path = "bigscience/" + model_args.model_name_or_path

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

    if test_args.train_data_source == 'stream':
        training_args.per_gpu_train_batch_size = test_args.train_random_n_prompts

    # set additional args
    for k, v in vars(test_args).items():
        if not hasattr(config, k):
            setattr(config, k, v)
            setattr(training_args, k, v)

    logger.info(f"Training/evaluation parameters {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        expand_list=(test_args.loss_option in ["consistency", "pseudo_train", "consistency_pseudo_train"]),
    )
    if test_args.loss_option in ["consistency", "pseudo_train", "consistency_pseudo_train"]:
        test_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            expand_list=False,
        )
    else:
        test_data_collator = None

    test_data = DatasetByPrompt(data_args, model_args.cache_dir, tokenizer)
    if test_args.train_random_n_prompts <= 0:
        test_args.train_random_n_prompts = test_data.num_prompts

    config.num_choices = test_data.num_choices
    if test_args.metric_name == "none":
        metrics = datasets.load_metric(data_args.dataset_name, data_args.subset_name, cache_dir=model_args.cache_dir)
    else:
        metrics = datasets.load_metric(test_args.metric_name, cache_dir=model_args.cache_dir)

    logger.info(f"Model parameters {config}")

    def _model_init():
        # very slow
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        for n, p in model.named_parameters():
            if test_args.peft_option == 'bitfit' and "bias" in n:
                print("tune " + n)
                p.requires_grad = True
            elif test_args.peft_option in ['lora', 'prompt_tuning'] and "ef_" in n:
                print("tune " + n)
                p.requires_grad = True
            else:
                p.requires_grad = False
        return model

    if test_args.test_mode == "t0":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.resize_token_embeddings(len(tokenizer))
        batched_evalute_t0(model, tokenizer, test_data, data_args, training_args.per_device_eval_batch_size,
                           training_args.fp16, data_collator, metrics, model_args.model_name_or_path)
    elif test_args.test_mode == "ttt_t0" and test_args.train_data_source == 'stream':
        predictions = [[] for _ in range(test_data.num_prompts)]
        avg_ensemble_predictions = []
        vote_ensemble_predictions = []
        golds = []
        model = _model_init()
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            test_data_collator=test_data_collator,
        )

        # if test_args.train_data_source == "stream": todo: support unlimited
        test_size = len(test_data) if test_args.debug_size < 0 else test_args.debug_size
        for i in range(test_size):
            # create dataset for one example
            if test_args.loss_option == "entropy":
                # loss computed over answer space and prompts
                train_data = TTTOnlineDataset(test_data, test_args, idx=i)
            elif test_args.loss_option in ["token_level_divergence", "token_level_entropy"]:
                # loss computed at token level over prompts
                train_data = TTTOnlineTokenLossDataset(test_data, test_args, idx=i)
            elif test_args.loss_option in ["consistency", "pseudo_train", "consistency_pseudo_train"]:
                train_data = TTTOfflineLoopDataset(test_data, test_args, test_args.train_random_n_prompts,
                                                    training_args.per_device_eval_batch_size, idx=i)
            else:
                raise NotImplementedError
            trainer.train_dataset = train_data
            trainer.eval_dataset = TTTOnlineDataset(test_data, test_args, idx=i)

            # run train
            if test_args.loss_option in ["consistency", "pseudo_train", "consistency_pseudo_train"]:
                trainer.train_ttt(resume_from_checkpoint=None, reinit_model=(i==0))
            else:
                trainer.train(resume_from_checkpoint=None, reinit_model=(i == 0))
            prompt_preds, avg_ensemble_pred, vote_ensemble_pred = trainer.evaluate()
            avg_ensemble_predictions.append(avg_ensemble_pred)
            vote_ensemble_predictions.append(vote_ensemble_pred)
            golds.append(train_data.gold_label if hasattr(train_data, 'gold_label') else train_data.gold_labels[0])
            for ii, pred in enumerate(prompt_preds):
                predictions[ii].append(pred)
            logger.info("Finish TTT of example {}, avg ensemble pred = {}, "
                        "gold label = {}".format(i, avg_ensemble_pred, golds[-1]))

        fout_name = training_args.output_dir
        results = summarize_metrics(predictions, avg_ensemble_predictions, vote_ensemble_predictions, golds, metrics, fout_name=fout_name)
        for k, v in results.items():
            logger.info("{} = {}".format(k, v))
    else:
        model = _model_init()
        print(f'there are {test_data.num_prompts} prompts in total')
        print(f'using {test_args.train_random_n_prompts} prompts  during training')

        data = DatasetByPrompt(data_args, model_args.cache_dir, tokenizer, split='train') \
            if test_args.train_data_source == 'train' else test_data
        if test_args.loss_option == "entropy":
            train_data = TTTOfflineDataset(data, test_args, test_args.train_random_n_prompts)
        elif test_args.loss_option in ["token_level_divergence", "token_level_entropy"]:
            train_data = TTTOfflineTokenLossDataset(data, test_args, test_args.train_random_n_prompts)
        elif test_args.loss_option in ["consistency", "pseudo_train", "consistency_pseudo_train"]:
            train_data = TTTOfflineLoopDataset(data, test_args, test_args.train_random_n_prompts,
                                               training_args.per_device_eval_batch_size)
        else:
            raise NotImplementedError
        test_set = TTTEvalDataset(test_data)

        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=test_set,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            test_data_collator=test_data_collator,
            compute_metrics=compute_metrics,  # todo: add metrics
            additional_metrics=metrics,
        )
        if test_args.loss_option in ["consistency", "pseudo_train", "consistency_pseudo_train"]:
            trainer.train_ttt(resume_from_checkpoint=None)
        else:
            trainer.train(resume_from_checkpoint=None)
        eval_results = trainer.evaluate()
        for k, v in eval_results.items():
            logger.info("{} = {}".format(k, v))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
