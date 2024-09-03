import os

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModelForCausalLM, LoraConfig
from trl import SFTTrainer, SFTConfig
from dataset_processor import DatasetProcesser
from llm_sample_cb import LLMSampleCB
from dotenv import load_dotenv
import yaml
import wandb


load_dotenv()
assert "HF_API_KEY" in os.environ, "Please add your Hugging Face API key to the environment variables"
login(token=os.environ["HF_API_KEY"])
assert "WANDB_API_KEY" in os.environ, "Please add your Weights and Biases API key to the environment variables"
wandb.login(token=os.environ["WANDB_API_KEY"])
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yaml')
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

dataset_names = config['dataset_args']['dataset_names']
dataset_processor = DatasetProcesser(dataset_names)
train_dataset, eval_dataset, test_dataset = dataset_processor.get_processed_train_eval_split(
    config['dataset_args']['train_split']
)

# Define the quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # or torch.bfloat16 if our GPU supports it
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
# Load model
model = AutoModelForCausalLM.from_pretrained(
    config['model_args']['model_name'],
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    token=os.environ["HF_API_KEY"],
)
tokenizer = AutoTokenizer.from_pretrained(
    config['model_args']['model_name'],
    use_fast=True,
    trust_remote_code=True,
    token=os.environ["HF_API_KEY"],
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# Do model patching and add fast LoRA weights
lora_config = LoraConfig(
    r=config['model_args']['lora_rank'],
    modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
    lora_alpha=config['model_args']['lora_alpha'],
    lora_dropout=config['model_args']['lora_dropout'], 
    bias=config['model_args']['bias'],
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)
model = PeftModelForCausalLM(
    model,
    peft_config=lora_config
)

project_name = config['model_args']['model_name'] + 'alpaca-fine-tuning'
os.environ['WANDB_PROJECT'] = project_name
os.environ['WANDB_LOG_MODEL'] = 'checkpoint'
# wandb stuff
wandb.init(project=project_name, 
            entity='africa-intelligence', 
            job_type="train",
            tags=['8b', 'hf_sft'],
            config=config)


args = SFTConfig(
        dataset_text_field=config['dataset_args']['text_field'],
        dataset_batch_size = config['training_args']['per_device_train_batch_size'],
        gradient_accumulation_steps = config['training_args']['gradient_accumulation_steps'],
        warmup_steps = config['training_args']['warmup_steps'],
        num_train_epochs = config['training_args']['epochs'],
        learning_rate = config['training_args']['learning_rate'],
        # fp16 = not is_bfloat16_supported(),
        # bf16 = is_bfloat16_supported(),
        optim = config['training_args']['optimizer'],
        weight_decay = config['training_args']['weight_decay'],
        lr_scheduler_type = config['training_args']['lr_scheduler_type'],
        # gradient_checkpointing=True,  # Enable gradient checkpointing
        seed = 3407,
        packing=config['dataset_args']['packing'],  # pack samples together for efficient training
        max_seq_length=config['dataset_args']['max_seq_length'],
        
        output_dir = "./output/",
        per_device_eval_batch_size=config['eval_args']['per_device_eval_batch_size'],
        logging_steps = config['eval_args']['logging_steps'],
        eval_strategy=config['eval_args']['eval_strategy'],
        eval_steps=config['eval_args']['eval_steps'],
        save_strategy=config['eval_args']['save_strategy'],
        save_steps=config['eval_args']['save_steps'],
        save_total_limit=config['eval_args']['save_total_limit'],
        load_best_model_at_end=config['eval_args']['load_best_model_at_end'],
        metric_for_best_model=config['eval_args']['metric_for_best_model'],
    )

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=args,
)

wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=1, max_new_tokens=256)
trainer.add_callback(wandb_callback)
trainer.train()
wandb.finish()