# 1st Place Kaggle Solution: Architectural Deep-Dive

Welcome to the repository for my 1st place solution. This project was an exercise in systematic machine learning design. Rather than relying solely on trial and error, the winning pipeline was built on four deliberate architectural decisions designed to overcome specific challenges in the dataset.

## 🧠 Key Design Decisions

### 1. Model Selection: Transformers over Trees
Exploratory Data Analysis revealed highly complex, entangled relationships between the numerical features. Standard Gradient Boosted Decision Trees (GBDTs) like XGBoost often require massive depth to capture these non-linear interactions, which quickly leads to overfitting. 

**The Decision**: I opted for **TabPFN**, a transformer architecture designed for tabular data. By utilizing self-attention mechanisms, TabPFN evaluates all feature interactions simultaneously in a single forward pass, naturally modeling the dense correlations that tree-based models struggled with.

### 2. Loss Function Design: Addressing Extreme Imbalance
The dataset exhibited a severe class imbalance. When trained with standard Cross-Entropy loss, models converged to a naive strategy: predicting the majority class to minimize average loss.

**The Decision**: I implemented a custom **Focal Loss** (`alpha=0.75`, `gamma=2.0`) in the TabPFN finetuning loop. Focal Loss introduces a dynamically scaled cross-entropy that mathematically down-weights the loss assigned to well-classified (easy) examples. This forced the gradient updates to focus almost entirely on the hard-to-classify minority cases, stabilizing the learning process.

### 3. Threshold Optimization Strategy
Classification models output probabilities, but the competition evaluated on the **F1 Score**—a hard metric that requires a specific decision boundary. On highly imbalanced data, the default `0.5` threshold is rarely optimal.

**The Decision**: 
- **Monte Carlo Simulations**: To find a threshold robust to distribution shifts, I built a GPU-accelerated Monte Carlo simulation. This generated millions of synthetic probability distributions to evaluate which thresholds maximized the *expected* F1 score across varying conditions.
- **Deterministic PR-Curve Maximization**: For the final finetuned model, I extracted the exact peak of the Precision-Recall curve to mathematically pinpoint the threshold that yielded the absolute maximum F1 score on the validation set.

### 4. Variance Reduction via Ensembling
Deep learning models and complex transformers can suffer from high variance, making their predictions slightly unstable across different data folds.

**The Decision**: I integrated an AutoGluon stacked ensemble as a stabilizing layer. By employing **20x Bagging** and **3-layer Stacking**, the ensemble aggregated predictions across diverse model types, effectively crushing model variance and ensuring a highly stable final prediction.

## 📊 Performance Summary

| Architecture / Setup | Key Design Choice | F1 Score |
| :--- | :--- | :--- |
| **TabPFN (Finetuned)** | Focal Loss + Deterministic PR Curve Optimization | **0.82** |
| **AutoGluon Ensemble** | 20x Bagging + 3-Layer Stacking | **0.80** |
| **TabPFN Baseline** | GPU-Accelerated Expected F1 Maximization | **0.79** |

## 📂 Repository Structure

```text
Project_Repo/
├── data/                   # Dataset directory (train.csv, test.csv)
├── notebooks/
│   └── Master_Guide.ipynb  # Comprehensive deep-dive into the design decisions
├── resources/              # EDA plots and visual insights
├── src/
│   ├── generate_eda.py     # Script to generate dataset insights
│   ├── monte_carlo.py      # GPU-accelerated MC Threshold Simulation
│   ├── train_autogluon.py  # AutoGluon ensemble logic
│   ├── train_tabpfn.py     # TabPFN baseline training
│   └── train_finetune.py   # Finetuned TabPFN Pipeline
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## 🛠️ Getting Started

1. Install the dependencies:
```bash
pip install -r requirements.txt
```

2. Follow the architectural deep-dive in the Jupyter Notebook:
```bash
jupyter notebook notebooks/Master_Guide.ipynb
```

3. Or execute the finetuning script directly:
```bash
python src/train_finetune.py
```
