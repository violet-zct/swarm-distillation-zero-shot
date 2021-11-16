from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"choices": ["bigscience/T0_3B", "bigscience/T0pp", "bigscience/T0"],
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

@dataclass
class TestArguments:
    test_mode: str = field(
        default="t0",
        metadata={"choices": ["t0", "ttt_t0"],
                  "help": ""}
    )

    train_data_source: Optional[str] = field(
        default="stream",
        metadata={"choices": ["stream", "unlimited"],
                  "help": "stream trains on one single test data"}
    )

    train_duplicates: Optional[int] = field(
        default=1,
        metadata={"help": "> 1 to create larger batch size"}
    )

    peft_option: str = field(
        default="bitfit",
        metadata={"choices": ["prompt_tuning", "bitfit"],
                "help": ""}
    )

    use_deepspeed: Optional[bool] = field(
        default=False,
    )

    prompt_tuning_L: Optional[int] = field(
        default=3,
        metadata={
            "help": "length of prompt vectors"
        },
    )

    debug_size: Optional[int] = field(
        default=-1,
        metadata={"help": ""},
    )

    metric_name: Optional[str] = field(
        default="none",
    )