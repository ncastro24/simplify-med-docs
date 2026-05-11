import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from peft import LoraConfig, get_peft_model

from evaluate import load as load_metric

# -------------------------
# Base model
# -------------------------
base_model_name = "microsoft/phi-2"

# -------------------------
# Experiment Configuration
# -------------------------

TRAIN_DATASET = "biomed"   # only option for now
TEST_SCENARIO = "short"    # "short" or "clinical"

# -------------------------
# Base directories
# -------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

# -------------------------
# Dataset selection
# -------------------------
if TRAIN_DATASET == "biomed":
    data_path = os.path.join(data_dir, "biomed_dataset.json")
else:
    raise ValueError("Invalid TRAIN_DATASET")

if TEST_SCENARIO == "short":
    test_scenario_path = os.path.join(data_dir, "test_short_sentences.json")
elif TEST_SCENARIO == "clinical":
    test_scenario_path = os.path.join(data_dir, "test_clinical_summaries.json")
else:
    raise ValueError("Invalid TEST_SCENARIO")

# -------------------------
# Output folders
# -------------------------
checkpoint_path = os.path.join(base_dir, "fine_tuned_model")
results_path = os.path.join(base_dir, "results")
logs_path = os.path.join(base_dir, "logs")
plots_path = os.path.join(base_dir, "plots")
csv_path = os.path.join(base_dir, "csv_results")

os.makedirs(results_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)
os.makedirs(csv_path, exist_ok=True)

print("\n=== EXPERIMENT SETUP ===")
print(f"Training dataset: {data_path}")
print(f"Test scenario: {test_scenario_path}")
print("=========================\n")

# -------------------------
# Load dataset
# -------------------------
dataset = load_dataset("json", data_files=data_path)["train"]

print(f"Dataset size: {len(dataset)}")

# -------------------------
# Train / Validation Split
# -------------------------
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_valid = split_dataset["train"].train_test_split(test_size=0.125, seed=42)

train_dataset = train_valid["train"]
validation_dataset = train_valid["test"]
test_dataset = split_dataset["test"]

print("\n=== DATA SPLIT ===")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(validation_dataset)}")
print(f"Test: {len(test_dataset)}")

# -------------------------
# Tokenizer
# -------------------------
if os.path.exists(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Model
# -------------------------
if os.path.exists(checkpoint_path):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
else:
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

# -------------------------
# SAFE FORMATTER (handles nested / inconsistent inputs)
# -------------------------
def safe_join(text_data):

    if isinstance(text_data, str):
        return text_data

    if not isinstance(text_data, list):
        return str(text_data)

    cleaned = []

    for item in text_data:

        if isinstance(item, str):
            cleaned.append(item)

        elif isinstance(item, list):
            cleaned.append(" ".join(map(str, item)))

        else:
            cleaned.append(str(item))

    return "\n".join(cleaned)


# -------------------------
# Format examples
# -------------------------
def format_example(example):

    input_text = safe_join(example["input"])
    output_text = safe_join(example["output"])

    instruction = example.get(
        "instruction",
        "Simplify the following medical text into plain English."
    )

    return f"{instruction}\nInput: {input_text}\nOutput: {output_text}"

# -------------------------
# Tokenization
# -------------------------
def tokenize(example):

    text = format_example(example)

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length = 128 #reduced for memory
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": tokens["input_ids"].copy()
    }

train_dataset = train_dataset.map(tokenize, batched=False)
validation_dataset = validation_dataset.map(tokenize, batched=False)
test_dataset = test_dataset.map(tokenize, batched=False)

train_dataset = train_dataset.map(tokenize, batched=False)
validation_dataset = validation_dataset.map(tokenize, batched=False)
test_dataset = test_dataset.map(tokenize, batched=False)

# -------------------------
# QLoRA setup
# -------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# -------------------------
# Metrics
# -------------------------
bleu_metric = load_metric("bleu")
rouge_metric = load_metric("rouge")

def compute_metrics(eval_pred):

    preds, labels = eval_pred

    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu = bleu_metric.compute(
        predictions=[p.split() for p in preds],
        references=[[l.split()] for l in labels]
    )

    rouge = rouge_metric.compute(
        predictions=preds,
        references=labels
    )

    return {
        "bleu": bleu["bleu"],
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"]
    }

# -------------------------
# Training args
# -------------------------
training_args = TrainingArguments(
    output_dir=results_path,

    # -------------------------
    # Memory-safe settings
    # -------------------------
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,

    weight_decay=0.01,
    warmup_ratio=0.03,

    evaluation_strategy="no",   # keep OFF (good)
    save_strategy="no",         # prevents memory spikes on Mac

    # -------------------------
    # Logging (reduce overhead slightly)
    # -------------------------
    logging_dir=logs_path,
    logging_steps=10,           # increase slightly

    fp16=False,
    report_to="none"
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    compute_metrics=compute_metrics
)

# -------------------------
# Train
# -------------------------

print("\n=== DATASET CHECK ===")
print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(validation_dataset)}")
print(f"Test size: {len(test_dataset)}")

print("\nTraining started...\n")
trainer.train()
print("\nTraining complete.\n")

# -------------------------
# Save model
# -------------------------
model.save_pretrained(checkpoint_path)
tokenizer.save_pretrained(checkpoint_path)

# -------------------------
# Final evaluation
# -------------------------
results = trainer.evaluate(test_dataset)

print("\n=== FINAL RESULTS ===")
for k, v in results.items():
    print(f"{k}: {v}")

# -------------------------
# Save CSV results
# -------------------------
results_df = pd.DataFrame([results])
results_csv = os.path.join(csv_path, f"eval_{TEST_SCENARIO}.csv")
results_df.to_csv(results_csv, index=False)

log_df = pd.DataFrame(trainer.state.log_history)
log_csv = os.path.join(csv_path, f"logs_{TEST_SCENARIO}.csv")
log_df.to_csv(log_csv, index=False)

# -------------------------
# Loss graph
# -------------------------
train_loss = []
train_steps = []
eval_loss = []
eval_steps = []

for entry in trainer.state.log_history:
    if "loss" in entry:
        train_loss.append(entry["loss"])
        train_steps.append(entry["step"])
    if "eval_loss" in entry:
        eval_loss.append(entry["eval_loss"])
        eval_steps.append(entry["step"])

plt.figure()
plt.plot(train_steps, train_loss, label="Train Loss")
plt.plot(eval_steps, eval_loss, label="Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()

plot_path = os.path.join(plots_path, f"loss_{TEST_SCENARIO}.png")
plt.savefig(plot_path)

print(f"\nSaved loss plot: {plot_path}")

# -------------------------
# Experiment summary
# -------------------------
summary_path = os.path.join(csv_path, f"summary_{TEST_SCENARIO}.txt")

with open(summary_path, "w") as f:
    f.write("EXPERIMENT SUMMARY\n\n")
    f.write(f"Model: {base_model_name}\n")
    f.write(f"Training: {data_path}\n")
    f.write(f"Test: {test_scenario_path}\n")
    f.write(f"Epochs: {training_args.num_train_epochs}\n")
    f.write(f"LR: {training_args.learning_rate}\n\n")
    f.write("FINAL RESULTS:\n")

    for k, v in results.items():
        f.write(f"{k}: {v}\n")

print(f"\nSaved summary: {summary_path}")
print("\nDONE")