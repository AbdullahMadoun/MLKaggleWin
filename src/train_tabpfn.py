import pandas as pd
import numpy as np
from tabpfn_extensions.post_hoc_ensembles import AutoTabPFNClassifier
from sklearn.metrics import f1_score

def train_tabpfn(train_path, test_path, n_ensemble=40, n_estimators=32, max_time=3600):
    """
    Trains a TabPFN ensemble model optimized for F1 score.
    """
    print(f"🚀 Loading Data from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Handle ID columns
    if 'ID' in train_df.columns:
        train_df = train_df.drop('ID', axis=1)
    if 'ID' in test_df.columns:
        test_ids = test_df['ID']
        test_df = test_df.drop('ID', axis=1)
    else:
        test_ids = range(len(test_df))

    # 2. Map targets
    target_map = {'square': 0, 'circle': 1}
    y = train_df['y'].map(target_map).values
    X = train_df.drop('y', axis=1)
    X_submission = test_df

    print("🚀 Training Optimized AutoTabPFN Ensemble...")

    model = AutoTabPFNClassifier(
        n_ensemble_models=n_ensemble,
        n_estimators=n_estimators,
        balance_probabilities=False, # Critical for imbalanced datasets
        eval_metric='f1',
        max_time=max_time,
        presets='best_quality',
        device='cuda',
        random_state=42
    )
    
    model.fit(X, y)

    print("\n🔮 Predicting Probabilities for Test Set...")
    test_probs = model.predict_proba(X_submission)[:, 1]

    return model, test_probs, test_ids

if __name__ == "__main__":
    # Example usage
    model, probs, ids = train_tabpfn('dataset-train-vf (2).csv', 'dataset-test-vf.csv', max_time=60)
