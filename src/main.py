import os

from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from dataset_processor import DatasetProcesser
from llm_sample_cb import LLMSampleCB
from dotenv import load_dotenv
import yaml
import wandb

load_dotenv()
assert "HF_API_KEY" in os.environ, "Please add your Hugging Face API key to the environment variables"
assert "WANDB_API_KEY" in os.environ, "Please add your Weights and Biases API key to the environment variables"
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yaml')
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

dataset_names = config['dataset_args']['dataset_names']
dataset_processor = DatasetProcesser(dataset_names)
train_dataset, eval_dataset, test_dataset = dataset_processor.get_processed_train_eval_split(
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
    modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
    lora_alpha=config['model_args']['lora_alpha'],
    lora_dropout=config['model_args']['lora_dropout'], 
    bias=config['model_args']['bias'],
    use_gradient_checkpointing=True,
    random_state=3407,
)

os.environ['WANDB_PROJECT'] = 'llama3.1-8b-alpaca-fine-tuning'
os.environ['WANDB_LOG_MODEL'] = 'checkpoint'
# wandb stuff
wandb.init(project='llama3.1-alpaca-fine-tuning', 
            entity='africa-intelligence', 
            job_type="train",
            tags=['8b', 'hf_sft'],
            config=config)


args = TrainingArguments(
        per_device_train_batch_size = config['training_args']['batch_size'],
        gradient_accumulation_steps = config['training_args']['gradient_accumulation_steps'],
        warmup_steps = config['training_args']['warmup_steps'],
        num_train_epochs = config['training_args']['epochs'],
        max_steps = config['training_args']['max_steps'],
        learning_rate = config['training_args']['learning_rate'],
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = config['training_args']['optimizer'],
        weight_decay = config['training_args']['weight_decay'],
        lr_scheduler_type = config['training_args']['lr_scheduler_type'],
        seed = 3407,
        
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
    packing=True,  # pack samples together for efficient training
    max_seq_length=config['dataset_args']['max_seq_length'],
    args=args,
    dataset_text_field=config['dataset_args']['text_field']
)

wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=1, max_new_tokens=256)
trainer.add_callback(wandb_callback)
trainer.train()
wandb.finish()
