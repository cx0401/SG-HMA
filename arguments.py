import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass, field
from transformers import TrainingArguments, MODEL_WITH_LM_HEAD_MAPPING, GPT2LMHeadModel, GPT2Tokenizer, BartTokenizer, AutoTokenizer
from score_models.gpt2score import GPT2CausalLM 
from score_models.bartscore import BartForCausalLM
from score_models.t5score import T5Scorer


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
MODEL_CLASSES = {
    "gpt2": (GPT2CausalLM, GPT2Tokenizer),
    "bart": (BartForCausalLM, BartTokenizer),
    "t5": (T5Scorer, AutoTokenizer)
}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    prefixModel_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The prefix model checkpoint for weights initialization. "
                    "Leave None if you want to train a model from scratch."
        },
    )

    prefix_mode: Optional[str] = field(
        default='activation',
        metadata={
            "help": "activation or embedding"
        },
    )

    preseqlen: Optional[int] = field(
        default=0,
        metadata={
            "help": "preseqlen for how many tokens of prefix should we include."
        },
    )

    optim_prefix: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether we are optimizing the prefix directly, or optimize another amortized function that "
                    "genrate the prefix."
        },
    )

    tuning_mode: Optional[str] = field(
        default='finetune',
        metadata={
            "help": "whether it's doing prefixtune or finetune."
        },
    )

    objective_mode: Optional[int] = field(
        default=2,
        metadata={
            "help": "In prefixtuning setting, the objective function... "
        },
    )

    top_layers: Optional[int] = field(
        default=2,
        metadata={
            "help": "In finetuning setting, if we only tune the top k layers. "
        },
    )

    adapter_design: Optional[int] = field(
        default=2,
        metadata={
            "help": "For Baseline of the adapter module... (1) means using the NLG adapter reference. "
                    "(2) means using a design similar to adapter module"
        },
    )

    adapter_bottleneck: Optional[int] = field(
        default=100,
        metadata={
            "help": "For baseline adapter module: the mid dim of the adapter. "
        },
    )

    parametrize_emb: Optional[str] = field(
        default='MLP',
        metadata={
            "help": "MLP or Emb to parametrize when we optimize for the embeddings."
        },
    )

    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "dropout rate for the prefix tuning model. "
        },
    )

    teacher_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "dropout rate for the teacher model. "
        },
    )


    init_random: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to init a random embedding, or use GPT2 embedding for the prefix tuning model. "
        },
    )

    init_shallow : Optional[str] = field(
        default='no',
        metadata={
            "help": "shallow is default to be no, because we add reparametrization trick. If shallow=yes, "
                    "then no reparametrization "
        },
    )

    init_shallow_word: Optional[str] = field(
        default='no',
        metadata={
            "help": "when init_shallow is yes, what word to use... "
        },
    )


    use_dropout: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to use dropout of GPT2 on trainer. "
        },
    )

    use_custom_teacher_dropout: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to use dropout of GPT2 on trainer. "
        },
    )

    mid_dim: Optional[int] = field(
        default=512,
        metadata={
            "help": "the mid dim."
        },
    )


    gumbel: Optional[str] = field(
        default='no',
        metadata={
            "help": "use the gumbel softmax trick in training."
        },
    )

    replay_buffer: Optional[str] = field(
        default='no',
        metadata={
            "help": "use the replay buffer in training."
        },
    )

    training_obj: Optional[int] = field(
        default=0,
        metadata={
            "help": "use a specified training objective"
        },
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    data2text_dataset: Optional[str] = field(
        default="e2e", metadata={"help": ""}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    task_mode: Optional[str] = field(
        default=None, metadata={"help": "The task mode"}
    )

    matching_objective: Optional[str] = field(
        default='kl', metadata={"help": "The distillation objective"}
    )

    distill: Optional[str] = field(
        default='no', metadata={"help": "yes/no"}
    )

    finetuned_model_path: Optional[str] = field(
        default="/u/scr/xlisali/contrast_LM/transformers/examples/full/full/webnlgfinetune_n_20_act_cat_b=6-e"
                "=10_d=0.0_u=no_lr=1e-05_w=0.0_s=101_r=n_m=512_earlystop", metadata={"help": "finetuned model path (teacher model)"}
    )

    format_mode: Optional[str] = field(
        default='cat', metadata={"help": "The mode of data2text format (cat, peek, nopeek)"}
    )

    lowdata_token: Optional[str] = field(
        default='summarize', metadata={"help": "The token to be prepended at initialization time. "}
    )

    use_lowdata_token: Optional[str] = field(
        default='yes', metadata={"help": "Whether we should use the lowdata token and pass it to the prefixTuning Model "
                                         "for the initialization trick.  "}
    )

    train_embs: Optional[str] = field(
        default='no', metadata={"help": "whether the train word embeddings"}
    )

    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "the max source length of summarization data. "}
    )

    train_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for training data. "}
    )

    val_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for dev data. "}
    )

    # controlprefix: Optional[str] = field(
    #     default="yes", metadata={"help": "The control mode"}
    # )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    rank_weight: float = field(
        default=1,
        metadata={
        },
    )
    mle_weight: float = field(
        default=1,
        metadata={
        },
    )

@dataclass
class GenerationArguments:
    max_length: int = field(default = 100)
    min_length: int = field(default = 1)
    temperature: float = field(default = 1.0)
    repetition_penalty: float = field(default = 1.0)
    k: int = field(default = 1)
    p: float = field(default = 1)
    objective_mode: int = field(default = 2)
    prefix: str = field(default = "")
    seed: int = field(default=42)
    num_beams: int = field(default=4)
    num_return_sequences: int = field(default=1)
    stop_token: str = field(default="[EOS]")
    batch_size: int = field(default=32)
    diversity_penalty:float = field(default=1.0)
    no_repeat_ngram_size: int = field(default = 3)
    do_sample: bool = field(default=False)

def e2eTrainingArguments(rewrite="no", output_dir=None, tuning_mode="finetune"):
    model_args = ModelArguments()
    model_args.model_name_or_path = "gpt2-medium"
    model_args.tokenizer_name = "gpt2-medium"
    model_args.mid_dim = 512
    model_args.init_random = "no"
    model_args.use_dropout = "no"
    model_args.objective_mode = 1
    model_args.tuning_mode = tuning_mode

    data_args = DataTrainingArguments()
    data_args.train_data_file = "data/e2e/src1_train.txt"
    data_args.eval_data_file = "data/e2e/src1_valid.txt"
    data_args.format_mode = "cat"

    assert output_dir is not None
    training_args = TrainingArguments(output_dir=output_dir)

    if rewrite == "yes":
        training_args.per_device_train_batch_size = 4
        training_args.per_device_eval_batch_size = 4
        training_args.rank_weight = 1
        training_args.save_steps = 50000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 2
        training_args.learning_rate = 1e-6
    else:
        training_args.per_device_train_batch_size = 32
        training_args.per_device_eval_batch_size = 32
        training_args.rank_weight = 0
        training_args.save_steps = 500000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 5
        training_args.learning_rate = 5e-5

    training_args.mle_weight = 1
    training_args.gradient_accumulation_steps = 2
    
    training_args.weight_decay = 0.0
    training_args.seed = 10
    training_args.logging_steps = 50
    training_args.save_total_limit = 4
    training_args.disable_tqdm = False

    ## prefix parameters 
    if tuning_mode == "prefixtune":
        model_args.preseqlen = 5
        model_args.optim_prefix = "yes"

    return model_args, data_args, training_args

def e2eGenerationArguments():
    model_args = GenerationArguments()
    return model_args   

@dataclass
class PrefixArguments:
    length: int = field(default = 50)
    stop_token: str = field(default = None)
    temperature: float = field(default = 1.0)
    repetition_penalty: float = field(default = 1.0)
    k: int = field(default = 0)
    p: float = field(default = 0.9)
    objective_mode: int = field(default = 2)
    prefix: str = field(default = "")
    num_return_sequences: int = field(default=1)
    seed: int = field(default=42)
    num_beams: int = field(default=4)
    stop_token: str = field(default="[EOS]")
    batch_size: int = field(default=16)

def webnlgTrainingArguments(rewrite="no", output_dir=None, tuning_mode="finetune"):
    model_args = ModelArguments()
    model_args.model_name_or_path = "gpt2-medium"
    model_args.tokenizer_name = "gpt2-medium"
    model_args.mid_dim = 512
    model_args.init_random = "no"
    model_args.use_dropout = "no"
    model_args.objective_mode = 1
    model_args.tuning_mode = tuning_mode

    data_args = DataTrainingArguments()
    data_args.train_data_file = "data/webnlg/train.json"
    data_args.eval_data_file = "data/webnlg/eval.json"
    data_args.format_mode = "cat"

    assert output_dir is not None
    training_args = TrainingArguments(output_dir=output_dir)

    if rewrite == "yes":
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        training_args.rank_weight = 0.01
        training_args.save_steps = 5000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 5
        training_args.learning_rate = 1e-4
    else:
        training_args.per_device_train_batch_size = 16
        training_args.per_device_eval_batch_size = 16
        training_args.rank_weight = 0
        training_args.save_steps = 1000
        training_args.eval_steps = 100
        training_args.num_train_epochs = 5
        training_args.learning_rate = 1e-4

    training_args.mle_weight = 1
    training_args.gradient_accumulation_steps = 2
    
    training_args.weight_decay = 0.0
    training_args.seed = 10
    training_args.logging_steps = 50
    training_args.save_total_limit = 1
    training_args.disable_tqdm = False

    ## prefix parameters 
    if tuning_mode == "prefixtune":
        model_args.preseqlen = 5
        model_args.optim_prefix = "yes"

    return model_args, data_args, training_args

def webnlgGenerationArguments():
    model_args = GenerationArguments()
    return model_args   

def wikibioTrainingArguments(rewrite="no", output_dir=None, tuning_mode="finetune"):
    model_args = ModelArguments()
    model_args.model_name_or_path = "gpt2-medium"
    model_args.tokenizer_name = "gpt2-medium"
    model_args.mid_dim = 512
    model_args.init_random = "no"
    model_args.use_dropout = "no"
    model_args.objective_mode = 1
    model_args.tuning_mode = tuning_mode

    data_args = DataTrainingArguments()
    data_args.train_data_file = "data/e2e/src1_train.txt"
    data_args.eval_data_file = "data/e2e/src1_valid.txt"
    data_args.format_mode = "cat"

    assert output_dir is not None
    training_args = TrainingArguments(output_dir=output_dir)

    if rewrite == "yes":
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        training_args.rank_weight = 10
        training_args.save_steps = 50000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 1
        training_args.learning_rate = 1e-5
    else:
        training_args.per_device_train_batch_size = 2
        training_args.per_device_eval_batch_size = 2
        training_args.rank_weight = 0
        training_args.save_steps = 50000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 2
        training_args.learning_rate = 1e-4

    training_args.mle_weight = 1
    training_args.gradient_accumulation_steps = 2
    
    training_args.weight_decay = 0.0
    training_args.seed = 10
    training_args.logging_steps = 50
    training_args.save_total_limit = 1
    training_args.disable_tqdm = False

    ## prefix parameters 
    if tuning_mode == "prefixtune":
        model_args.preseqlen = 5
        model_args.optim_prefix = "yes"

    return model_args, data_args, training_args

def wikibioGenerationArguments():
    model_args = GenerationArguments()
    model_args.batch_size = 16
    return model_args   

def numericNLGTrainingArguments(rewrite="no", output_dir=None, tuning_mode="finetune"):
    model_args = ModelArguments()
    model_args.model_name_or_path = "gpt2-medium"
    model_args.tokenizer_name = "gpt2-medium"
    model_args.mid_dim = 512
    model_args.init_random = "no"
    model_args.use_dropout = "no"
    model_args.objective_mode = 1
    model_args.tuning_mode = tuning_mode

    data_args = DataTrainingArguments()
    data_args.train_data_file = "data/e2e/src1_train.txt"
    data_args.eval_data_file = "data/e2e/src1_valid.txt"
    data_args.format_mode = "cat"

    assert output_dir is not None
    training_args = TrainingArguments(output_dir=output_dir)

    if rewrite == "yes":
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        training_args.rank_weight = 10
        training_args.save_steps = 5000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 10
        training_args.learning_rate = 1e-4
    else:
        training_args.per_device_train_batch_size = 4
        training_args.per_device_eval_batch_size = 4
        training_args.rank_weight = 0
        training_args.save_steps = 5000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 30
        training_args.learning_rate = 1e-4

    training_args.mle_weight = 1
    training_args.gradient_accumulation_steps = 2
    
    training_args.weight_decay = 0.0
    training_args.seed = 10
    training_args.logging_steps = 50
    training_args.save_total_limit = 1
    training_args.disable_tqdm = False

    ## prefix parameters 
    if tuning_mode == "prefixtune":
        model_args.preseqlen = 5
        model_args.optim_prefix = "yes"

    return model_args, data_args, training_args

def numericNLGGenerationArguments():
    model_args = GenerationArguments()
    model_args.batch_size = 8
    return model_args   

def TottoTrainingArguments(rewrite="no", output_dir=None, tuning_mode="finetune"):
    model_args = ModelArguments()
    model_args.model_name_or_path = "gpt2-medium"
    model_args.tokenizer_name = "gpt2-medium"
    model_args.mid_dim = 512
    model_args.init_random = "no"
    model_args.use_dropout = "no"
    model_args.objective_mode = 1
    model_args.tuning_mode = tuning_mode

    data_args = DataTrainingArguments()

    assert output_dir is not None
    training_args = TrainingArguments(output_dir=output_dir)

    if rewrite == "yes":
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        training_args.rank_weight = 10
        training_args.save_steps = 5000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 10
        training_args.learning_rate = 1e-4
    else:
        training_args.per_device_train_batch_size = 4
        training_args.per_device_eval_batch_size = 4
        training_args.rank_weight = 0
        training_args.save_steps = 5000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 30
        training_args.learning_rate = 1e-4

    training_args.mle_weight = 1
    training_args.gradient_accumulation_steps = 2
    
    training_args.weight_decay = 0.0
    training_args.seed = 10
    training_args.logging_steps = 50
    training_args.save_total_limit = 1
    training_args.disable_tqdm = False

    ## prefix parameters 
    if tuning_mode == "prefixtune":
        model_args.preseqlen = 5
        model_args.optim_prefix = "yes"

    return model_args, data_args, training_args

def TottoGenerationArguments():
    model_args = GenerationArguments()
    model_args.batch_size = 8
    return model_args   

def webnlgTrainingArguments(rewrite="no", output_dir=None, tuning_mode="finetune"):
    model_args = ModelArguments()
    model_args.model_name_or_path = "gpt2-medium"
    model_args.tokenizer_name = "gpt2-medium"
    model_args.mid_dim = 512
    model_args.init_random = "no"
    model_args.use_dropout = "no"
    model_args.objective_mode = 1
    model_args.tuning_mode = tuning_mode

    data_args = DataTrainingArguments()

    assert output_dir is not None
    training_args = TrainingArguments(output_dir=output_dir)

    if rewrite == "yes":
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        training_args.rank_weight = 10
        training_args.save_steps = 5000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 10
        training_args.learning_rate = 1e-4
    else:
        training_args.per_device_train_batch_size = 4
        training_args.per_device_eval_batch_size = 4
        training_args.rank_weight = 0
        training_args.save_steps = 5000
        training_args.eval_steps = 1000
        training_args.num_train_epochs = 30
        training_args.learning_rate = 1e-4

    training_args.mle_weight = 1
    training_args.gradient_accumulation_steps = 2
    
    training_args.weight_decay = 0.0
    training_args.seed = 10
    training_args.logging_steps = 50
    training_args.save_total_limit = 1
    training_args.disable_tqdm = False

    ## prefix parameters 
    if tuning_mode == "prefixtune":
        model_args.preseqlen = 5
        model_args.optim_prefix = "yes"

    return model_args, data_args, training_args

def webnlgGenerationArguments():
    model_args = GenerationArguments()
    model_args.batch_size = 48
    return model_args   
