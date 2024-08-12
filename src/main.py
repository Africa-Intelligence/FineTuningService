import os

from transformers.integrations import WandbCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from .dataset_processor import DatasetProcesser
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

args = SFTConfig(
    report_to="wandb",  # enables logging to W&B 😎
    per_device_train_batch_size=16,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    gradient_accumulation_steps=2,  # simulate larger batch sizes
    output_dir="./output",
    max_seq_length=max_seq_length,
    dataset_text_field="text",
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
