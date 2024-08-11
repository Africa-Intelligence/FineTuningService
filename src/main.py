import os

from transformers.integrations import WandbCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

max_seq_length = 2048  # Supports automatic RoPE Scaling, so choose any number


def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}").format_map(row)


def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}").format_map(row)


# TODO: update function to rather pre-process data first
def create_alpaca_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)


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
    callbacks=[WandbCallback()],
    formatting_func=create_alpaca_prompt,
)
