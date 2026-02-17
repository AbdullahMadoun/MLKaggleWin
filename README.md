# 🏆 Advanced Ensemble Learning for Classification - 1st Place Solution

This repository contains the code and methodology for the course project where we achieved **1st Place**. The solution leverages a powerful ensemble of modern tabular deep learning methods, including **TabPFN** and **AutoGluon**, optimized via **Monte Carlo Threshold Finetuning**.

## 📊 Performance Overview

| Model | Technique | F1 Score |
| :--- | :--- | :--- |
| **Finetune Model** | Custom Deep Learning + Finetuning | **0.82** 🥇 |
| **AutoGluon** | Stacked Ensemble (Best Quality Preset) | **0.80** 🥈 |
| **TabPFN** | Prior-Data Fitting w/ Monte Carlo Optimization | **0.79** 🥉 |

## 🚀 Key Features

*   **Modular Architecture**: Clean separation of data loading, model training, and optimization logic.
*   **Monte Carlo Threshold Optimization**: A novel technique using GPU-accelerated simulations to find the optimal decision threshold that maximizes F1 score.
*   **Advanced Ensembling**: Utilizing `AutoTabPFNClassifier` for posterior ensemble approximation and `AutoGluon`'s multi-layer stacking.

## 📂 Repository Structure

```
Project_Repo/
├── data/                   # Dataset directory
├── notebooks/
│   └── Master_Guide.ipynb  # 📖 The detailed story of our solution
├── src/
│   ├── monte_carlo.py      # 🎲 GPU-accelerated MC Threshold Finetuning
│   ├── train_autogluon.py  # 🤖 AutoGluon training logic
│   ├── train_tabpfn.py     # 🧠 TabPFN training logic
│   └── train_finetune.py   # 🔧 Custom Finetuning logic
├── resources/              # Images and assets
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 💡 Usage

Run the **Master Guide** notebook for a step-by-step walkthrough of the solution:

```bash
jupyter notebook notebooks/Master_Guide.ipynb
```

Or run individual training scripts:

```python
from src.train_autogluon import train_autogluon
train_autogluon('data/train.csv', 'data/test.csv')
```

## 🧠 Methodology Highlight: Monte Carlo Optimization

To squeeze every bit of performance out of our models, we implemented a custom Monte Carlo simulation. Instead of assuming a standard decision threshold of 0.5, we:
1.  Simulated 100,000,000+ probable realities based on the model's predicted probabilities.
2.  Calculated the expected F1 score for thousands of candidate thresholds.
3.  Selected the threshold that maximizes the *expected* return, rather than just the point estimate.

This robust approach helped us secure the top spot on the leaderboard.
