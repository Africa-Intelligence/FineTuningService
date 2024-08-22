import os

from transformers import TrainingArguments
from transformers.integrations import WandbCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from dataset_processor import DatasetProcesser
from dotenv import load_dotenv
import yaml

load_dotenv()
assert "HF_API_KEY" in os.environ, "Please add your Hugging Face API key to the environment variables"
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yaml')
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

dataset_names = config['dataset_args']['dataset_names']
dataset_processor = DatasetProcesser(dataset_names)
train_dataset, eval_dataset = dataset_processor.get_processed_train_eval_split(
    config['dataset_args']['train_split']
)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config['training_args']['model_name'],
    dtype=None,
    load_in_4bit=True,
    token=os.environ["HF_API_KEY"],
    token=os.environ["HF_API_KEY"],
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=config['model_args']['lora_rank'],
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=config['model_args']['lora_alpha'],
    lora_dropout=config['model_args']['lora_dropout'], 
    bias=config['model_args']['bias'],
    use_gradient_checkpointing=True,
    random_state=3407,
)

os.environ['WANDB_PROJECT'] = 'llama3.1-8b-alpaca-fine-tuning'
os.environ['WANDB_LOG_MODEL'] = 'checkpoint'

args = TrainingArguments(
        per_device_train_batch_size = config['training_args']['batch_size'],
        gradient_accumulation_steps = config['training_args']['gradient_accumulation_steps'],
        warmup_steps = config['training_args']['warmup_steps'],
        num_train_epochs = config['training_args']['epochs'],
        max_steps = config['training_args']['max_steps'],
        learning_rate = config['training_args']['learning_rate'],
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = config['training_args']['logging_steps'],
        optim = config['training_args']['optimizer'],
        weight_decay = config['training_args']['weight_decay'],
        lr_scheduler_type = config['training_args']['lr_scheduler_type'],
        seed = 3407,
        output_dir = "outputs",
    )

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    packing=True,  # pack samples together for efficient training
    max_seq_length=config['dataset_args']['max_seq_length'],
    args=args,
    dataset_text_field="text",
    callbacks=[WandbCallback()]
)

trainer.train()
