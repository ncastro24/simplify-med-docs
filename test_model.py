import os
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel

# -------------------------
# Base paths
# -------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

# Folder containing LoRA adapters
adapter_path = os.path.join(
    base_dir,
    "models",
    "phi2-biomed-qlora"
)

# Test examples
test_data_path = os.path.join(
    base_dir,
    "data",
    "test_examples.json"
)

# Base model
base_model_name = "microsoft/phi-2"

# -------------------------
# Device
# -------------------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing device: {device}")

# -------------------------
# Load tokenizer
# -------------------------
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name
)

tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Load base Phi-2 model
# -------------------------
print("\nLoading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# -------------------------
# Load LoRA adapters
# -------------------------
if os.path.exists(adapter_path):

    print("\nLoading LoRA adapters...")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )

else:

    print("\nNo LoRA adapters found.")
    print("Using base model only.")

    model = base_model

model.to(device)

# -------------------------
# Evaluation mode
# -------------------------
model.eval()

# -------------------------
# Load test examples
# -------------------------
if os.path.exists(test_data_path):

    with open(test_data_path, "r") as f:
        examples = json.load(f)

else:

    print("\nNo test_examples.json found.")
    examples = []

# -------------------------
# Safe join helper
# -------------------------
def safe_join(value):

    if isinstance(value, list):

        cleaned = []

        for item in value:

            if isinstance(item, list):
                cleaned.append(" ".join(map(str, item)))

            else:
                cleaned.append(str(item))

        return "\n".join(cleaned)

    return str(value)

# -------------------------
# Format input
# -------------------------
def format_input(example):

    input_text = safe_join(example["input"])

    return (
        f"{example['instruction']}\n"
        f"Input: {input_text}\n"
        f"Output:"
    )

# -------------------------
# Generate function
# -------------------------
def generate_response(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

# -------------------------
# Run test examples
# -------------------------
print("\n==============================")
print("RUNNING TEST EXAMPLES")
print("==============================")

for i, ex in enumerate(examples):

    formatted_input = format_input(ex)

    result = generate_response(formatted_input)

    print(f"\n=== EXAMPLE {i+1} ===")
    print(result)

# -------------------------
# Interactive testing
# -------------------------
print("\n==============================")
print("INTERACTIVE TESTING")
print("==============================")

while True:

    user_input = input(
        "\nEnter a medical sentence "
        "(or type 'quit'): "
    )

    if user_input.lower() == "quit":
        break

    prompt = (
        "Simplify the following medical "
        "sentence into plain English.\n"
        f"Input: {user_input}\n"
        "Output:"
    )

    result = generate_response(prompt)

    print("\n=== MODEL OUTPUT ===")
    print(result)