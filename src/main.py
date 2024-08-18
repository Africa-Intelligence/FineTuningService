import os

from transformers import TrainingArguments
from transformers.integrations import WandbCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from dataset_processor import DatasetProcesser
from dotenv import load_dotenv

load_dotenv()
assert "HF_API_KEY" in os.environ, "Please add your Hugging Face API key to the environment variables"

dataset_names = [
    'africa-intelligence/yahma-alpaca-cleaned-af',
    'africa-intelligence/yahma-alpaca-cleaned-zu',
    'africa-intelligence/yahma-alpaca-cleaned-xh',
    'africa-intelligence/yahma-alpaca-cleaned-tn'
]
dataset_processor = DatasetProcesser(dataset_names)
train_dataset, eval_dataset = dataset_processor.get_processed_train_eval_split(0.9)

max_seq_length = 2048  # Supports automatic RoPE Scaling, so choose any number

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
    token="hf_...",
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Dropout = 0 is currently optimized
    bias="none",  # Bias = "none" is currently optimized
    use_gradient_checkpointing=True,
    random_state=3407,
)

os.environ["WANDB_PROJECT"] = "alpaca_ft"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    )

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    packing=True,  # pack samples together for efficient training
    max_seq_length=1024,  # maximum packed length
    args=args,
    callbacks=[WandbCallback()]
)
