from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"choices": ["T0_3B", "T0pp", "T0"],
                "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        metadata={"help": "name of dataset, e.g. super_glue"}
    )

    prompt_set_name: str = field(metadata={"help": ""})  # same as dataset name?

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    subset_name: Optional[str] = field(
        default="none",
        metadata={"help": "name of dataset, e.g. "}
    )

    task_type: Optional[str] = field(
        default="classification",
        metadata={"choices": ["generation", "classification"],
                  "help": ""}
    )

    testset_name: Optional[str] = field(
        default="test",
        metadata={"help": ""}
    )

    cb_surgery: Optional[int] = field(
        default=0,
        metadata={"help": ""}
    )

@dataclass
class TestArguments:
    test_mode: str = field(
        default="t0",
        metadata={"choices": ["t0", "ttt_t0"],
                  "help": ""}
    )

    train_data_source: Optional[str] = field(
        default="stream",
        metadata={"choices": ["stream", "train", "validation"],
                  "help": "stream trains on one single test data"}
    )

    train_duplicates: Optional[int] = field(
        default=1,
        metadata={"help": "> 1 to create larger batch size"}
    )

    peft_option: str = field(
        default="none",
        metadata={"choices": ["prompt_tuning", "lora", "bitfit", "none"],
                "help": ""}
    )

    use_deepspeed: Optional[bool] = field(
        default=False,
    )

    debug_size: Optional[int] = field(
        default=-1,
        metadata={"help": ""},
    )

    max_dev_size: Optional[int] = field(
        default=1000,
        metadata={"help": "maximum number of examples for unsupervised dev metric"},
    )

    metric_name: Optional[str] = field(
        default="none",
    )

    train_random_n_prompts: Optional[int] = field(
        default=-1,
        metadata={"help": "number of prompts for one single example when minimizing the entropy"}
    )

    prob_temperature: Optional[float] = field(
        default=1.,
        metadata={"help": "peakify the probability distribution"}
    )

    loss_option: Optional[str] = field(
        default="entropy",
        metadata={"help": "loss type for test mode",
                  "choices": ["token_level_divergence", "entropy", "token_level_entropy",
                              "consistency", "pseudo_train", "consistency_pseudo_train"]}
    )

    pseudo_train_loss_weight: Optional[float] = field(
        default=1.,
        metadata={"help": "used to"}
    )

    pseudo_dist: Optional[str] = field(
        default="smooth",
        metadata={"help": "type of pseudo distribution",
                  "choices": ["smooth", "argmax"]}
    )

    # options for consistency loss
    jsd: Optional[int] = field(
        default=1,
        metadata={
            "help": "jsd"
        },
    )

    detach_kl_left: Optional[int] = field(
        default=0,
        metadata={
            "help": "detach the left side of KL"
        },
    )

    detach_kl_right: Optional[int] = field(
        default=0,
        metadata={
            "help": "detach the right side of KL"
        },
    )

    combine_option: Optional[str] = field(
        default="uniform",
        metadata={"help": "how to compute marginal distribution",
                  "choices": ["uniform", "entropy"]}
    )

    # parameter-efficient tuning specific options:
    bottleneck_dim: Optional[int] = field(
        default=3,
        metadata={
            "help": "length of prompt vectors"
        },
    )

    # lora
    lora_alpha: Optional[float] = field(
        default=4,
        metadata={
            "help": ""
        },
    )

    lora_pos: Optional[str] = field(
        default="encdec",
        metadata={"choices": ["dec", "encdec"]}
    )

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": ""
        },
    )

    prune_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "format is xx:yy, xx is the option, yy is the hyperparameter"}
    )

    ensemble_option: Optional[str] = field(
        default="avg_prob",
        metadata={"choices": ["avg_prob", "majority_vote"]}
    )

    split_answer_groups: Optional[int] = field(
        default=1,
        metadata={"help": "if 0, use the buggy version of L1"}
    )

    disable_eval_mode: Optional[int] = field(
        default=0,
        metadata={"help": "if 1, disable eval mode at inference time per train step"}
    )

    # random ensemble not implemented yet
    pseudo_target_mode: Optional[str] = field(
        default="pairwise",
        metadata={"help": "how to produce the pseudo target",
                  "choices": ["pairwise", "full_ensemble", "random_ensemble"]}
    )

    ensemble_subset_size: Optional[float] = field(
        default=-1.0,
        metadata={"help": "<1, > 0, set when pseudo_target_mode=random_ensemble, "
                          "use this ratio of prompts to compute ensemble"}
    )

    min_train_steps: Optional[int] = field(
        default=300,
        metadata={"help": "get best ckpt after this many steps with the unsupervised metric"}
    )