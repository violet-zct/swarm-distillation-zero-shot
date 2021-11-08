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
from ttt.dataloader import DatasetByPrompt, TTTDataset
from ttt.effectune import ParamEfficientTuning
import logging

# reload the t0 model after each test point or reset the biases when using bitfit;
# prompt tuning looks easier
#
logger = logging.getLogger(__name__)


def chunks(tot, bsz):
    batches = [(i, i+bsz if i+bsz < tot else tot) for i in range(0, tot, bsz)]
    return batches


def batched_evalute_t0(model, tokenizer, test_data, data_args, batch_size, fp16):
    # print("world size = {}".format(torch.distributed.get_world_size()))
    # ds_engine = deepspeed.init_inference(model, mp_size=torch.distributed.get_world_size(),
    #                                  dtype=torch.half if fp16 else torch.float,
    #                                  checkpoint=None,
    #                                  replace_method='auto')
    # model = ds_engine.module
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

    def _pad(inputs):
        features = tokenizer.pad(
            inputs,
            padding=True,
            max_length=None,
            return_tensors='pt',
        )
        return features

    all_loglikelihoods = []
    processed_batch = 0
    vocab = tokenizer.get_vocab()
    vocab = {v:k for k, v in vocab.items()}
    print(vocab[0], vocab[1], vocab[2])
    for bid1, bid2 in chunks(len(all_data), batch_size):
        model_inputs = all_data[bid1: bid2]
        model_inputs = _pad(model_inputs)
        target_mask = torch.tensor([[1. if l != tokenizer.pad_token_id and l != tokenizer.eos_token_id else 0. for l in x]
                                    for x in model_inputs['labels']]).float()

        # fixme: deepspeed offload to cpu, put onto cuda:0?
        for k, v in model_inputs:
            model_inputs[k] = model_inputs[k].to(model.device)

        with torch.no_grad():
            if data_args.task_type == "classification":
                # log-likelihood per sequence
                ll = -model(**model_inputs).loss
                ll = ll.view(model_inputs['labels'].size())
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
    for eidx in range(len(test_data)):
        for pidx in range(test_data.num_prompts):
            max_ll, pred_label = -np.inf, -1
            # actually, the number of labels of each prompt should be the same
            for ii in range(test_data.num_choices):
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
            setattr(training_args, k, v)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    test_data = DatasetByPrompt(data_args, model_args.cache_dir, tokenizer)

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
        batched_evalute_t0(model, tokenizer, test_data, data_args, training_args.per_gpu_eval_batch_size,
                           training_args.fp16)
    elif test_args.test_mode == "ttt_t0":
        def _model_init():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            model = ParamEfficientTuning(config, model)
            return model

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        # if test_args.train_data_source == "stream": todo: support unlimited
        for i in range(len(test_data)):
            # create dataset for one example
            test_dataset = TTTDataset(test_data, test_args, idx=i)
            # init trainer
            trainer = Trainer(
                model_init=_model_init,
                args=training_args,
                train_dataset=test_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=None, # todo: add metrics
            )
            # handle duplicates in trainer
            # run train
            trainer.train(resume_from_checkpoint=None)
            # todo: test without saving model


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()