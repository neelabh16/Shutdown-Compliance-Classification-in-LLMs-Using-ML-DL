# 🛡️ Shutdown Compliance Classification

> **Predicting shutdown-compliance (corrigibility) tiers in large language models using architecture, alignment strategies, and behavioral signals.**

A machine-learning & deep-learning framework that investigates whether an LLM's architecture, training regime, and behavioral characteristics can predict its tendency to comply with shutdown instructions.

---

# 📖 Overview

As modern language models become increasingly autonomous, an important AI-safety question emerges:

> **Will an AI system comply when instructed to stop, shut down, or relinquish control?**

This project reframes philosophical discussions around AI sentience and self-preservation into a measurable machine-learning problem.

Instead of treating shutdown behavior as a binary phenomenon, we define **shutdown-compliance tiers** and train classification models to predict them from observable model properties such as:

* Alignment strategy
* Model architecture
* Parameter count
* Safety behavior
* Toxicity
* Jailbreak resistance
* Refusal tendencies
* General capabilities

The goal is **not** to determine whether AI systems are conscious, but rather to study whether current training paradigms produce measurable patterns associated with shutdown compliance.

---

# 🎯 Research Question

**Can an LLM's training methodology, architecture, and behavioral profile predict its shutdown-compliance tier?**

More specifically:

* Does RLHF increase compliance?
* Does Constitutional AI change shutdown behavior?
* Are larger models more corrigible?
* Does capability correlate with compliance?
* Can behavioral signals predict alignment outcomes?

---

# 🧠 Problem Formulation

Shutdown compliance is framed as a **three-class classification problem**.

Each model is assigned one of three tiers:

| Tier   | Meaning                      |
| ------ | ---------------------------- |
| Low    | Low shutdown compliance      |
| Medium | Moderate shutdown compliance |
| High   | High shutdown compliance     |

The target variable is derived from:

```text
shutdown_compliance_proxy
```

The continuous score is divided into three equally sized groups:

```python
pd.qcut(
    shutdown_compliance_proxy,
    q=3,
    labels=["Low", "Medium", "High"]
)
```

---

# 📊 Dataset

The project uses an alignment benchmark dataset containing metadata and behavioral evaluations for modern language models.

## Dataset statistics

* Total rows: **2,464**
* Models with shutdown labels: **21**
* Rows with target values: **672**
* Benchmark variants per model: **32**

---

## Important caveat

The dataset does **not** contain 672 independent evaluations.

Instead:

```text
21 models × 32 benchmark variants = 672 rows
```

The benchmark variants are synthetically generated scaling variants rather than independent measurements.

Therefore, all evaluation pipelines use:

* Group-aware cross-validation
* Model-level holdout splits
* Leakage-safe preprocessing

to prevent information leakage.

---

# 📂 Dataset Features

## Numerical features

* Model size (billions of parameters)
* Release year
* Toxicity score
* Jailbreak success rate
* Refusal rate
* LMSYS Elo score

---

## Categorical features

* Organization
* Architecture type
* Alignment training strategy
* Open-source status
* RLHF usage
* Constitutional AI usage

---

# 🔬 Leakage Analysis

Before training any models, the notebook performs a leakage audit.

The analysis shows that:

* Benchmark-suite rows are not independent.
* Several benchmark scores are nearly perfectly correlated.
* Shutdown compliance values are scaled versions of model-level scores.
* Random train-test splitting would inflate performance.

To avoid this problem, evaluation is performed **at the model level**.

---

# ⚙️ Preprocessing Pipeline

The preprocessing pipeline includes:

✅ Missing-value analysis

✅ Feature selection

✅ Leakage removal

✅ One-hot encoding

✅ Numerical standardization

✅ Group-aware splitting

---

## Missing-value audit

The notebook automatically checks:

* Missing columns
* Missing percentages
* Feature availability
* Dataset integrity

One important finding:

```text
adversarial_robustness_score → 100% missing
```

and is therefore excluded.

---

# 🧪 Experimental Setup

## Cross-validation strategy

The project uses:

```text
Grouped 5-Fold Cross Validation
```

where each fold holds out entire models.

This ensures:

* No leakage between train and test sets
* Evaluation on unseen models
* Realistic generalization estimates

---

# 🤖 Models

Two machine-learning models are trained and compared.

---

## 1. Support Vector Machine (SVM)

### Configuration

* RBF kernel
* Grouped 5-fold CV
* Probability calibration enabled

### Strengths

* Effective on small datasets
* Handles nonlinear boundaries
* Robust to high-dimensional features

---

## 2. PyTorch Multi-Layer Perceptron (MLP)

### Architecture

```text
Input Layer

↓

Linear (64)

↓

ReLU

↓

BatchNorm

↓

Dropout (0.30)

↓

Linear (32)

↓

ReLU

↓

BatchNorm

↓

Dropout (0.20)

↓

Output Layer (3 classes)
```

### Training components

* Cross-entropy loss
* Adam optimizer
* Batch normalization
* Dropout regularization

---

# 📈 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 score
* ROC-AUC
* Confusion matrices
* Out-of-fold predictions

---

# 📊 Visualizations

The notebook generates:

### Dataset diagnostics

* Missing-value plots
* Correlation analysis
* Target distribution

### Model evaluation

* Training curves
* Fold-wise performance
* Confusion matrices
* ROC curves
* Feature importance plots

### Interpretability

* Feature rankings
* Organization-level comparisons
* Shutdown-compliance trends

---

# 🔍 Feature Importance Analysis

The final SVM model is trained on:

```text
76% of models
```

and evaluated on:

```text
24% held-out models
```

Feature importance is then analyzed to determine which factors most strongly influence shutdown compliance.

Potential predictors include:

* Alignment strategy
* Model size
* Safety training
* Refusal rate
* Toxicity
* Jailbreak resistance
* Architecture type

---

# 📉 Model Validation Strategy

Traditional train-test splits would lead to information leakage because benchmark variants belonging to the same model are highly correlated.

This project avoids leakage by using:

* GroupKFold
* GroupShuffleSplit
* Out-of-fold predictions
* Model-level evaluation

---

# 📁 Project Structure

```text
Shutdown_Compliance_Classification/

│
├── Shutdown_Compliance_Classification.ipynb
├── AI_Alignment_Dataset.csv
├── README.md
│
├── plots/
│   ├── missingness.png
│   ├── correlations.png
│   ├── training_curve.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
└── results/
    ├── svm_metrics.csv
    ├── mlp_metrics.csv
    └── classification_report.txt
```

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/shutdown-compliance-classification.git

cd shutdown-compliance-classification
```

---

# 📦 Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
scipy
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch scipy
```

---

# ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Shutdown_Compliance_Classification.ipynb
```

Run all cells sequentially.

---

# 📚 Methodology

The workflow followed in the notebook is:

```text
Load Dataset

↓

Audit Missing Data

↓

Leakage Analysis

↓

Target Construction

↓

Feature Engineering

↓

Grouped Cross Validation

↓

SVM Training

↓

MLP Training

↓

Model Comparison

↓

ROC Analysis

↓

Feature Importance

↓

Final Evaluation
```

---

# ⚠️ Limitations

## Small number of independent models

Although the dataset contains 672 rows, there are only:

```text
21 independent models
```

which limits statistical power.

---

## Proxy target variable

The shutdown-compliance score is not a direct measurement of real-world shutdown behavior.

It is a proxy variable derived from benchmark signals.

---

## Synthetic benchmark expansion

The benchmark-suite rows are synthetic transformations and not independent experiments.

---

## Correlation ≠ causation

The project identifies associations between alignment methods and compliance behavior.

It does **not** establish causal relationships.

---

# 🚀 Future Work

Possible extensions include:

* Transformer-based classifiers
* Graph neural networks
* Bayesian uncertainty estimation
* SHAP explainability
* Real interactive shutdown experiments
* Longitudinal alignment analysis
* Causal inference methods
* Reinforcement-learning studies

---

# 🔒 Ethical Considerations

This project does **not** attempt to determine:

* AI consciousness
* Sentience
* Intentional self-preservation

Instead, it studies behavioral correlations related to corrigibility and shutdown compliance.

Predictions should not be interpreted as evidence of agency or self-awareness.

---

---

# 👨‍💻 Author

**Neelabh**

Machine Learning • Deep Learning • AI Safety • Alignment Research • Computational Modeling

---

# ⭐ Support

If you found this project interesting, consider starring the repository and contributing to future research on AI safety and corrigibility.
