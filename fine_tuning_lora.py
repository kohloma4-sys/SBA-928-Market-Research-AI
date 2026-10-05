Python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Load Base Model & Tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, load_in_4bit=True, torch_dtype=torch.float16, device_map="auto"
)

# 2. Configure LoRA
peft_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# 3. Training Parameters
training_args = TrainingArguments(
    output_dir="./market_research_lora", per_device_train_batch_size=4,
    gradient_accumulation_steps=2, learning_rate=2e-4, max_steps=100, fp16=True
)

# 4. Execute Supervised Fine-Tuning
dataset = load_dataset("json", data_files="market_research_train.jsonl", split="train")
trainer = SFTTrainer(
    model=model, train_dataset=dataset, peft_config=peft_config,
    dataset_text_field="output", max_seq_length=512, args=training_args
)
trainer.train()
