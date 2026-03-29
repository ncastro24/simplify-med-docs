import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

# -------------------------
# Base model
# -------------------------
base_model_name = "microsoft/phi-2"

# -------------------------
# Paths relative to this script
# -------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "biomed_dataset.json")
checkpoint_path = os.path.join(base_dir, "fine_tuned_model")  # folder to load/save checkpoints

# -------------------------
# Load dataset
# -------------------------
dataset = load_dataset("json", data_files=data_path)["train"]

# -------------------------
# Load tokenizer
# -------------------------
if os.path.exists(checkpoint_path):
    print(f"Loading tokenizer from checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
else:
    print(f"No checkpoint found. Loading tokenizer from base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # ensure padding token

# -------------------------
# Load model (checkpoint if exists, otherwise base)
# -------------------------
if os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
else:
    print(f"No checkpoint found. Loading model from base: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

# -------------------------
# Format & tokenize dataset (multi-sentence support)
# -------------------------
def format_example(example):
    input_text = "\n".join(example["input"]) if isinstance(example["input"], list) else example["input"]
    output_text = "\n".join(example["output"]) if isinstance(example["output"], list) else example["output"]
    return f"{example['instruction']}\nInput: {input_text}\nOutput: {output_text}"

def tokenize(example):
    text = format_example(example)
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256  # adjust for long multi-sentence examples
    )

dataset = dataset.map(tokenize, batched=True, batch_size=16)
dataset = dataset.remove_columns(list(dataset.features.keys()))

# -------------------------
# LoRA / QLoRA config
# -------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adjust for Phi-2
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# -------------------------
# Data collator for dynamic padding
# -------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# -------------------------
# Training arguments
# -------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(base_dir, "results"),
    per_device_train_batch_size=2,  # adjust for CPU/GPU
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="no",
    fp16=False  # set True if GPU is available
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# -------------------------
# Train
# -------------------------
trainer.train()

# -------------------------
# Save fine-tuned model
# -------------------------
model.save_pretrained(checkpoint_path)
tokenizer.save_pretrained(checkpoint_path)

print("Training complete and model saved.")