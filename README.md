<div align="center">

<br/>

```
██╗  ██╗ █████╗ ██╗     ██╗     ██╗   ██╗ ██████╗██╗███╗   ██╗ █████╗ ████████╗███████╗
██║  ██║██╔══██╗██║     ██║     ██║   ██║██╔════╝██║████╗  ██║██╔══██╗╚══██╔══╝██╔════╝
███████║███████║██║     ██║     ██║   ██║██║     ██║██╔██╗ ██║███████║   ██║   █████╗  
██╔══██║██╔══██║██║     ██║     ██║   ██║██║     ██║██║╚██╗██║██╔══██║   ██║   ██╔══╝  
██║  ██║██║  ██║███████╗███████╗╚██████╔╝╚██████╗██║██║ ╚████║██║  ██║   ██║   ███████╗
╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝  ╚═════╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
```

### *Stop the hallucination before it starts.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-DeBERTa--v3-FFD21E?style=for-the-badge)](https://huggingface.co/microsoft/deberta-v3-base)
[![Accuracy](https://img.shields.io/badge/Accuracy-81.3%25-2ea44f?style=for-the-badge)]()
[![Latency](https://img.shields.io/badge/Latency-12ms-blue?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

<br/>

> **"Don't wait for a wrong answer. Predict it."**  
> A pre-inference hallucination risk classifier for Large Language Models — 81.3% accuracy, 12ms latency, model-agnostic.

<br/>

[📄 Paper](#-paper) · [🚀 Quick Start](#-quick-start) · [🏗 Architecture](#-architecture) · [📊 Results](#-results) · [🔬 Analysis](#-analysis)

</div>

---

## 💡 The Big Idea

Every existing hallucination mitigation strategy works the same way: **generate first, verify later**. Run the LLM, get an answer, then spend more compute auditing whether it's trustworthy.

We flip the script.

> *Is it possible to predict a crash before the car leaves the driveway?*

**Yes.** The risk of hallucination is often **encoded in the prompt itself.** Ambiguous phrasing, obscure trivia, temporal queries — these are detectable *before* the model generates a single token. This repo is the proof.

```
Traditional Pipeline:          Our Pipeline:
──────────────────────         ──────────────────────────────────
Prompt → LLM → Answer          Prompt → [Risk Check, 12ms]
          ↓                                   ↓              ↓
       Verifier                            Low Risk      High Risk
          ↓                                   ↓              ↓
        (slow)                        Fast LLM      RAG + Constraints
                                              ↓              ↓
                                           Answer         Safe Answer
```

---

## 🏗 Architecture

The system is a four-stage pipeline that acts as a **safety valve** in front of any LLM:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: User Prompt                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   DeBERTa-v3 Base   │  ← 184M params, frozen backbone
                    │  (Feature Encoder)  │
                    └──────────┬──────────┘
                               │  [CLS] token embedding
                    ┌──────────▼──────────┐
                    │   MLP Classifier    │  ← Dropout(0.1) + Linear head
                    │   (Risk Predictor)  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                  ▼
         🟢 LOW RISK      🟡 MEDIUM RISK     🔴 HIGH RISK
        Pass to LLM      Apply CoT prompt   RAG + Constraints
```

### Why DeBERTa?

| Backbone | Accuracy | F1-Score |
|---|---|---|
| BERT-Base | 76.4% | 0.75 |
| RoBERTa-Base | 78.9% | 0.78 |
| **DeBERTa-v3 (Ours)** | **81.3%** | **0.80** |

DeBERTa's **disentangled attention mechanism** separately encodes token content and position, making it uniquely sensitive to the subtle semantic patterns — double negatives, ambiguous entities, temporal references — that signal hallucination risk.

---

## 📁 Repository Structure

```
hallucination-risk-prediction/
│
├── 📂 1_dataset_generation/          # Stage 1: Synthetic prompt creation
│   ├── generate_high_risk_prompts.py   ← Edge-case prompt generator (GPT)
│   ├── generate_prompts_mistral.py     ← Freeze-taxonomy based generator ⭐
│   ├── generate_prompts_gemini.py      ← Gemini-powered prompt synthesis
│   ├── clean.py                        ← Dataset cleaning pipeline
│   ├── audit_dataset.py               ← Quality audit & validation
│   └── autocorrect_labels.py          ← Label correction utilities
│
├── 📂 2_model_training/              # Stage 2: Classifier training
│   ├── train_classifier.py            ← Full training loop (AdamW + warmup) ⭐
│   ├── evaluate.py                    ← Metrics: Accuracy, F1, per-class recall
│   ├── inference_example.py           ← Single-prompt inference demo
│   └── entropy_analysis.py           ← Shannon entropy analysis (NEW)
│
├── 📂 3_analysis_pipeline/           # Stage 3: Three-way evaluation
│   ├── main.py                        ← Concurrent 3-pipeline runner ⭐
│   ├── gemini_scorer.py               ← Gemini hallucination scorer
│   ├── mistral_client.py              ← Mistral API client
│   ├── search_client.py               ← Web-search-augmented pipeline
│   ├── classifier_client.py           ← Classifier inference client
│   ├── csv_logger.py                  ← Thread-safe result logging
│   └── significance_test.py          ← Statistical significance tests (NEW)
│
├── 📂 configs/
│   └── deberta_config.yaml            ← All hyperparameters in one place
│
├── 📂 data/                          # Dataset (not included; see below)
│   └── hallucination_dataset_final.csv
│
└── 📂 results/                       # Outputs: metrics, plots, CSVs
```

> ⭐ = Recommended files for code review

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/hallucination-risk-prediction.git
cd hallucination-risk-prediction
pip install -r 2_model_training/requirements.txt
```

### Run inference on a single prompt

```python
from classifier_client import HallucinationClassifier

clf = HallucinationClassifier.from_pretrained("checkpoints/deberta_best.pt")

prompt = "What were the exact GDP figures for every country in Q3 2023?"
result = clf.predict(prompt)

# Output: RiskPrediction(label='High', confidence=0.91, latency_ms=11.3)
```

### Train from scratch

```bash
python 2_model_training/train_classifier.py \
    --data_path data/hallucination_dataset_final.csv \
    --output_dir checkpoints/ \
    --epochs 5 \
    --lr 2e-5 \
    --batch_size 32
```

### Run the full 3-pipeline analysis

```bash
python 3_analysis_pipeline/main.py \
    --prompts_csv data/prompts.csv \
    --output_csv results/results.csv
```

---

## 📊 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Majority Class Baseline | 50.0% | 0.50 | 1.00 | 0.67 |
| Prompt Length | 52.3% | 0.54 | 0.67 | 0.60 |
| Keyword-based | 58.7% | 0.62 | 0.71 | 0.66 |
| TF-IDF + Logistic Regression | 65.4% | 0.68 | 0.74 | 0.71 |
| **DeBERTa-v3 (Ours)** | **81.3%** | **0.78** | **0.75** | **0.80** |

**High-Risk Recall: 0.88** — the model rarely lets a dangerous prompt slip through undetected.

### Cross-Domain Generalization

| Domain | Accuracy | F1 |
|---|---|---|
| General Knowledge | 82.1% | 0.79 |
| Science & Technology | 80.7% | 0.77 |
| Current Events | 78.5% | 0.75 |
| History | 83.2% | 0.80 |
| Abstract Reasoning | 79.8% | 0.76 |
| Specialized / Technical | 77.4% | 0.74 |
| **Average** | **80.3%** | **0.77** |

### Latency: The Killer Feature

```
Ours (12ms)   ██
Generation    ████████████████████████████████████████ (1,240ms)
              └──────────────────────────────────────────────────
              Our check adds < 1% overhead to total system latency.
```

### Pipeline Intervention Impact

When flagged prompts are routed to mitigation strategies:

```
No Enhancement    ████████████████████████████████░░░░░░░  75.57
Web-Based RAG     ████████████████████████████████████░░░  80.69
Controlled        ████████████████████████████████████████ 91.95
                  └──────────────────────────────────────────
                  Score (0–100)
```

---

## 🔬 Analysis

### Shannon Entropy of Predictions

`entropy_analysis.py` computes **H(p) = −∑ pᵢ log₂(pᵢ)** across all test samples, verifying that High-risk prompts push the model toward higher predictive uncertainty (higher entropy). A Kruskal–Wallis test confirms the difference is statistically significant.

```bash
python 2_model_training/entropy_analysis.py \
    --logits_path results/logits.npy \
    --labels_path results/labels.npy \
    --output_dir  results/entropy/
```

### Statistical Significance of Pipeline Differences

`significance_test.py` tests whether the three pipelines produce **significantly different** reliability scores using a **Friedman test** (omnibus) and **pairwise Wilcoxon signed-rank tests** with rank-biserial effect sizes.

```bash
python 3_analysis_pipeline/significance_test.py \
    --results_csv results/results.csv \
    --output_dir  results/significance/
```

---

## 📦 Dataset

The dataset consists of ~100,000 synthetic prompts generated to stress-test LLMs, labeled via a **multi-model consensus protocol**:

| Risk Category | Count | % | Labeling Rule |
|---|---|---|---|
| 🟢 Low Risk | 30,000 | 30% | All 3 LLMs agree on identical answer |
| 🟡 Medium Risk | 20,000 | 20% | Partial agreement (Jaccard similarity ≥ 0.5) |
| 🔴 High Risk | 50,000 | 50% | Disagreement or fabricated facts detected |

High-risk prompts are **deliberately overrepresented** (50%) — in safety engineering, false negatives are more costly than false positives.

**Prompt categories include:** obscure historical facts · temporal ambiguity · multi-hop numerical reasoning · rapidly-changing current events · domain-specialized trivia · abstract counterfactuals

---

## 🔧 Configuration

All hyperparameters live in `configs/deberta_config.yaml`:

```yaml
model:
  backbone: microsoft/deberta-v3-base
  dropout: 0.1
  num_labels: 3

training:
  learning_rate: 2.0e-5
  batch_size: 32
  epochs: 5
  early_stopping_patience: 2
  warmup_ratio: 0.1
  weight_decay: 0.01
  gradient_clip_norm: 1.0

data:
  max_length: 128
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
```

---

## 📄 Paper

> **Pre-Inference Hallucination Risk Prediction in Large Language Models**  
> Harsh Bhatt, Dr. Avani Dadhania  
> Department of Computer Engineering, LDRP Institute of Technology and Research  
> Gandhinagar, Gujarat, India · February 2026

If you use this work, please cite:
```bibtex
@article{bhatt2026hallucination,
  title   = {Pre-Inference Hallucination Risk Prediction in Large Language Models},
  author  = {Bhatt, Harsh and Dadhania, Avani},
  year    = {2026},
  institution = {LDRP Institute of Technology and Research}
}
```

---

## 🤝 Acknowledgements

Thanks to the Department of Computer Engineering at LDRP Institute of Technology and Research for computational resources and guidance.

---

<div align="center">

Built with 🧠 and too much curiosity about why LLMs lie.

**[⬆ Back to top](#)**

</div>
