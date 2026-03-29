import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Paths relative to this script
# -------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
fine_tuned_path = os.path.join(base_dir, "fine_tuned_model")
test_data_path = os.path.join(base_dir, "data", "test_examples.json")
base_model_name = "microsoft/phi-2"

# -------------------------
# Load tokenizer + model
# -------------------------
if os.path.exists(fine_tuned_path):
    print(f"Loading fine-tuned model from: {fine_tuned_path}")
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
    model = AutoModelForCausalLM.from_pretrained(fine_tuned_path)
else:
    print(f"No fine-tuned model found. Using base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

# -------------------------
# Load test examples from JSON
# JSON format:
# [
#   {
#       "instruction": "...",
#       "input": ["sentence1", "sentence2"]
#   },
#   ...
# ]
# -------------------------
with open(test_data_path, "r") as f:
    examples = json.load(f)

# -------------------------
# Function to format input for model
# -------------------------
def format_input(example):
    input_text = "\n".join(example["input"]) if isinstance(example["input"], list) else example["input"]
    return f"{example['instruction']}\nInput: {input_text}\nOutput:"

# -------------------------
# Loop through examples
# -------------------------
for i, ex in enumerate(examples):
    input_text = format_input(ex)
    inputs = tokenizer(input_text, return_tensors="pt")
    
    outputs = model.generate(**inputs, max_new_tokens=150)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n=== EXAMPLE {i+1} OUTPUT ===")
    print(result)