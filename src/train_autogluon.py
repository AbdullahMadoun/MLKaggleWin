import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np

def train_autogluon(train_path, test_path, time_limit=300, presets='best_quality'):
    """
    Trains an AutoGluon model optimized for F1 score.
    """
    print(f"🚀 Loading Data from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Handle ID columns
    if 'ID' in train_df.columns:
        train_ids = train_df['ID']
        train_data = train_df.drop('ID', axis=1)
    else:
        train_data = train_df

    if 'ID' in test_df.columns:
        test_ids = test_df['ID']
        test_data = test_df.drop('ID', axis=1)
    else:
        test_data = test_df

    # 2. Map targets
    target_map = {'square': 0, 'circle': 1}
    if train_data['y'].dtype == 'object':
        print("Mapping target labels to integers...")
        train_data['y'] = train_data['y'].map(target_map)
    
    label = 'y'

    print(f"🚀 Training AutoGluon with metric='f1' and preset='{presets}'...")

    predictor = TabularPredictor(
        label=label,
        eval_metric='f1',
        path='ag_models_f1_opt'
    ).fit(
        train_data,
        presets=presets,
        time_limit=time_limit,
        ag_args_fit={'num_gpus': 1},
        num_bag_folds=10,
        num_bag_sets=20,
        num_stack_levels=3
    )

    results = predictor.fit_summary()
    
    # Inference
    print("\n🔮 Predicting Probabilities for Test Set...")
    # AutoGluon returns DataFrame where columns are class labels.
    test_pred_probs = predictor.predict_proba(test_data)
    minority_class_label = 1
    test_probs = test_pred_probs[minority_class_label]
    
    return predictor, test_probs, test_ids

if __name__ == "__main__":
    # Example usage
    predictor, probs, ids = train_autogluon('dataset-train-vf (2).csv', 'dataset-test-vf.csv')
