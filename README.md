# Medical Misinformation Detection
### Applied AI & Deep Learning (ITMD-524) · Prof. Yibo Hu · Illinois Institute of Technology · Spring 2026
### Student: Ashokkumar Chandrasekaran

---

## Project Overview

This project builds an automated health misinformation detection system using NLP and domain-adapted deep learning. Three models are implemented and compared:

- **TF-IDF + Logistic Regression** — interpretable baseline (Macro F1 = 0.933)
- **BioBERT Full Fine-Tune** — medical domain deep learning (Macro F1 = 0.866, AUC = 1.000)
- **BioBERT + LoRA** — parameter-efficient fine-tuning (Macro F1 = 0.700, 99.7% fewer parameters)

**Live Demo:** https://huggingface.co/spaces/ashokakjr/medical-misinfo-shield

---

## Repository Structure

```
medical_misinfo_shield/
├── MASTER_notebook.ipynb          ← Main notebook (all 5 experiments)
├── README.md                      ← This file
├── requirements.txt               ← Python dependencies
├── app.py                         ← HuggingFace Gradio web app
│
├── outputs/
│   ├── models/
│   │   ├── tfidf_lr_pipeline.pkl      ← Saved TF-IDF + LR model
│   │   ├── biobert_finetuned/         ← Saved BioBERT full fine-tune
│   │   └── biobert_lora/              ← Saved BioBERT + LoRA adapters
│   ├── figures/                       ← All generated charts (PNG)
│   ├── baseline_results.json          ← Experiment 1 results
│   ├── biobert_results.json           ← Experiment 2 results
│   ├── lora_results.json              ← Experiment 3 results
│   ├── ablation_results.csv           ← Experiment 4 results (18 configs)
│   └── robustness_results.json        ← Experiment 5 results
```

---

## Dependencies

### Python Version
Python 3.9 or higher (tested on Google Colab with Python 3.10)

### Install all dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt contents

```
torch>=2.0.0
transformers==4.38.0
peft==0.9.0
datasets
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
gradio>=4.0.0
accelerate>=0.27.0
```

### Hardware Requirements

| Experiment | Minimum Hardware | Recommended |
|---|---|---|
| TF-IDF Baseline | CPU only | Any laptop |
| BioBERT Full Fine-Tune | GPU required | A100 (Colab Pro) |
| BioBERT + LoRA | GPU required | A100 (Colab Pro) |
| Ablation Study | GPU required | A100 (Colab Pro) |
| Robustness Test | CPU or GPU | T4 or higher |

---

## Dataset Instructions

### Option A — Use the synthetic dataset (built automatically in notebook)

The notebook automatically builds the 100-sentence synthetic dataset at runtime. **No download required.** Simply run Cell 4 in the notebook and the dataset is created in memory and saved to Google Drive.

### Option B — Use the original research datasets (recommended for real results)

**CoAID Dataset** (Cui & Lee, 2020)
```
Source: https://github.com/cuilimeng/CoAID
Download: git clone https://github.com/cuilimeng/CoAID
Files needed: NewsFakeCOVID-19.csv, NewsRealCOVID-19.csv
```

**FakeHealth Dataset** (Dai et al., 2020)
```
Source: https://zenodo.org/record/3765970
Download: wget https://zenodo.org/record/3765970/files/FakeHealth.zip
Files needed: HealthStory/reviews/, HealthStory/content/
```

After downloading, place files in:
```
/content/drive/MyDrive/medical_misinfo_shield/raw_data/
```

Then update Cell 4 in the notebook to load from raw files instead of synthetic data.

---

## How to Run

### Step 1 — Open the notebook in Google Colab

Click the link below to open the master notebook directly in Colab:

```
https://colab.research.google.com/drive/[YOUR_COLAB_LINK_HERE]
```

Or upload `MASTER_notebook.ipynb` manually to Google Colab.

### Step 2 — Connect to A100 GPU

In Colab:
1. Click **Runtime** → **Change runtime type**
2. Set Hardware accelerator to **A100 GPU**
3. Click **Save**

> ⚠️ A100 requires Colab Pro. T4 GPU (free tier) works but BioBERT training will take ~25 minutes instead of 1.4 minutes.

### Step 3 — Mount Google Drive

Run **Cell 1** — this mounts your Drive and creates the folder structure:
```
My Drive/
└── medical_misinfo_shield/
    ├── data/
    ├── outputs/
    │   ├── models/
    │   └── figures/
```

### Step 4 — Install dependencies

Run **Cell 2** — installs all required packages:
```python
!pip install transformers==4.38.0 peft==0.9.0 accelerate datasets --quiet
```

### Step 5 — Run all experiments

Run cells in order:

| Cell | Content | Time |
|---|---|---|
| Cell 1 | Mount Google Drive | 30 sec |
| Cell 2 | Install dependencies | 2 min |
| Cell 3 | Import libraries | 10 sec |
| Cell 4 | Build dataset (100 synthetic sentences) | 10 sec |
| Cell 5 | Experiment 1 — TF-IDF Baseline | < 5 sec |
| Cell 6 | Experiment 2 — BioBERT Full Fine-Tune | ~1.4 min |
| Cell 7 | Experiment 3 — BioBERT + LoRA | ~3.4 min |
| Cell 8 | Experiment 4 — Ablation Study (18 configs) | ~15 min |
| Cell 9 | Experiment 5 — Robustness Testing | ~2 min |
| Cell 10 | Generate figures and charts | 1 min |
| Cell 11 | Print final summary table | 10 sec |

**Total runtime: approximately 25 minutes on A100 GPU**

### Step 6 — View results

All results are saved automatically to Google Drive:
```
outputs/baseline_results.json
outputs/biobert_results.json
outputs/lora_results.json
outputs/ablation_results.csv
outputs/robustness_results.json
outputs/figures/*.png
```

---

## Reproduction Steps

To exactly reproduce the reported results:

### 1. Set random seeds (already done in notebook)
```python
import torch, numpy as np, random
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
```

### 2. Use exact model checkpoint
```python
BASE_MODEL = "dmis-lab/biobert-v1.1"
```

### 3. Use exact hyperparameters

**BioBERT Full Fine-Tune:**
```python
learning_rate = 2e-5
dropout = 0.2
batch_size = 8
max_epochs = 10
patience = 3   # early stopping
```

**BioBERT + LoRA:**
```python
lora_r = 8
lora_alpha = 16
lora_dropout = 0.1
target_modules = ["query", "value"]
```

**Best Ablation Config:**
```python
learning_rate = 3e-5
dropout = 0.3
batch_size = 8
```

### 4. Expected results

| Model | Macro F1 | AUC-ROC | Train Time |
|---|---|---|---|
| TF-IDF + LR | 0.933 | 0.964 | < 5 sec |
| BioBERT Full | 0.866 | 1.000 | ~1.4 min |
| BioBERT + LoRA | 0.700 | 0.732 | ~3.4 min |
| Ablation Best | 1.000 (val) | — | ~15 min total |
| Robustness F1 Drop | 0.000 | — | ~2 min |

> Note: Minor variation in results (±0.01) may occur due to GPU non-determinism even with fixed seeds. This is expected and normal for transformer models.

---

## Running the Web App Locally

```bash
# Install dependencies
pip install gradio transformers peft joblib scikit-learn torch

# Run the app
python app.py
```

The app will start at `http://localhost:7860`

Or visit the live deployed version:
**https://huggingface.co/spaces/ashokakjr/medical-misinfo-shield**

---

## Common Issues and Fixes

| Issue | Fix |
|---|---|
| `CUDA out of memory` | Reduce batch size to 4 or restart runtime |
| `ModuleNotFoundError: peft` | Run `pip install peft==0.9.0` |
| `ImportError: torchao` | Run `pip install torchao==0.16.0` |
| `NameError: train_df not defined` | Run Cell 4 first to rebuild dataset |
| `Drive not mounted` | Run Cell 1 first and authorize Drive access |
| BioBERT slow on free Colab | Upgrade to Colab Pro or use T4 GPU |

---

## Contact

**Student:** Ashokkumar Chandrasekaran
**Course:** Applied AI & Deep Learning (ITMD-524-01)
**Institution:** Illinois Institute of Technology
**Semester:** Spring 2026
**Instructor:** Prof. Yibo Hu
