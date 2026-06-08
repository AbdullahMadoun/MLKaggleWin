import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create resources directory
os.makedirs('resources', exist_ok=True)

# Load data
df = pd.read_csv('data/dataset-train-vf.csv')

# Drop ID
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# Target mapping
if 'y' in df.columns and df['y'].dtype == 'object':
    df['y'] = df['y'].map({'square': 0, 'circle': 1})

# 1. Class Imbalance Plot
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='y', data=df, palette='viridis')
plt.title('Extreme Class Imbalance (Why Focal Loss is Needed)', fontsize=14)
plt.xlabel('Target Class (0: square, 1: circle)', fontsize=12)
plt.ylabel('Count', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.tight_layout()
plt.savefig('resources/class_imbalance.png', dpi=300)
plt.close()

# 2. Correlation Matrix
numerical_cols = [c for c in df.columns if c.startswith('f') and c != 'f11']
if len(numerical_cols) > 0:
    plt.figure(figsize=(10, 8))
    corr = df[numerical_cols + ['y']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix (The GBDT Plateau)', fontsize=14)
    plt.tight_layout()
    plt.savefig('resources/correlation_matrix.png', dpi=300)
    plt.close()

print("EDA plots generated in resources/")
