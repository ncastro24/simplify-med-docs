CPSC543 Final Project 2026

## Environment Setup

This project uses **Python 3.10.x**. To reproduce the training and testing of the biomedical text simplifier, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/simplify-med-docs.git
cd simplify-med-docs
```

### 2. Create a virtual environment

```bash
# macOS/Linux
python3 -m venv nlp_env
source nlp_env/bin/activate

# Windows
python -m venv nlp_env
nlp_env\Scripts\activate
```

### 3. Install required packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This will install all Python packages required for training and testing.

#### Recommended Python packages versions:

torch==2.4.0
transformers==4.37.2
peft==0.8.2
datasets==4.8.4
numpy==1.26.4
safetensors==0.7.0
tqdm==4.67.3
requests==2.33.0

### 4. GPU Usage (optional)
If you have a CUDA-enabled GPU, PyTorch will automatically use it.
If not, training will run on CPU, but may be slower.
### 5. Run training or testing
```bash
# Train the model
python train.py

# Test the model on examples
python test_model.py
```

#### Notes:

If fine_tuned_model/ exists, train.py will continue training from the last checkpoint.
If no checkpoint exists, it will start fine-tuning from the base Phi-2 model.
