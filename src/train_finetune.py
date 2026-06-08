# ==================== TABPFN PEAK HUNTER (THRESHOLD OPTIMIZED) ====================
# Fixes "Identical Results" by:
# 1. Verifying weight updates occur.
# 2. Optimizing the Decision Threshold (Crucial for Imbalanced Data).
# 3. Visualizing probability shifts.

from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabpfn import TabPFNClassifier
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
FINETUNE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 42,
    'val_ratio': 0.1,
    'n_inference_context_samples': 4000, 
}

FINETUNE_CONFIG['finetuning'] = {
    'epochs': 30, 
    'learning_rate': 3e-5, 
    'weight_decay': 1e-4, 
    'meta_batch_size': 1,
    'batch_size': 2048, 
}

print("=" * 80)
print("TABPFN PEAK HUNTER (DIAGNOSTIC & THRESHOLD MODE)")
print("=" * 80)

# ==================== FOCAL LOSS IMPL ====================
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# ==================== DATA PREPARATION ====================
class FineTuneDataLoader:
    @staticmethod
    def prepare(train_path, config):
        print("\n[STEP 1] DATA PREPARATION")
        train_df = pd.read_csv(train_path)
        train_df = train_df.drop('ID', axis=1)
        
        target_mapping = {'square': 0, 'circle': 1}
        y = train_df['y'].map(target_mapping).values
        X = train_df.drop('y', axis=1)
        
        preprocessor = MinimalPreprocessor()
        X_processed = preprocessor.fit_transform(X)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=config['val_ratio'], stratify=y, random_state=config['random_seed']
        )
        return X_train, X_val, y_train, y_val, preprocessor

class MinimalPreprocessor:
    def __init__(self):
        self.scaler = None; self.cat_encoder = None
    def fit_transform(self, X):
        X_processed = X.copy()
        if 'f11' in X.columns:
            self.cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_processed[['f11']] = self.cat_encoder.fit_transform(X_processed[['f11']])
        numerical_cols = [col for col in X.columns if col.startswith('f') and col != 'f11']
        self.scaler = StandardScaler()
        X_processed[numerical_cols] = self.scaler.fit_transform(X_processed[numerical_cols])
        return X_processed.values 
    def transform(self, X):
        X_processed = X.copy()
        if 'f11' in X.columns: X_processed[['f11']] = self.cat_encoder.transform(X_processed[['f11']])
        numerical_cols = [col for col in X.columns if col.startswith('f') and col != 'f11']
        X_processed[numerical_cols] = self.scaler.transform(X_processed[numerical_cols])
        return X_processed.values

# ==================== SETUP ====================
def setup_model(init_params, config):
    classifier = TabPFNClassifier(**init_params)
    classifier.softmax_temperature_ = classifier.softmax_temperature
    classifier._initialize_model_variables()
    
    model = classifier.models_[0]
    model.train()
    
    optimizer = Adam(
        model.parameters(), 
        lr=config['finetuning']['learning_rate'],
        weight_decay=config['finetuning']['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config['finetuning']['epochs'])
    
    return classifier, optimizer, scheduler, model

def evaluate(classifier, base_init_params, X_train_ctx, y_train_ctx, X_val, y_val):
    eval_config = base_init_params.copy()
    eval_config.pop('model_path', None)
    eval_config.update({
        'fit_mode': 'fit_preprocessors', 
        'inference_config': {'SUBSAMPLE_SAMPLES': FINETUNE_CONFIG['n_inference_context_samples']}
    })
    
    eval_clf = clone_model_for_evaluation(classifier, eval_config, TabPFNClassifier)
    eval_clf.fit(X_train_ctx, y_train_ctx)
    probs = eval_clf.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    return f1_score(y_val, preds, pos_label=1), log_loss(y_val, probs)

# ==================== OPTIMAL THRESHOLD FINDER ====================
def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    # Note: thresholds is 1 element shorter than f1_scores
    best_idx = np.argmax(f1_scores)
    
    # Safety check for index bounds
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
        
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_thresh, best_f1

# ==================== MAIN LOOP ====================
def run_finetuning(train_path='dataset-train-vf (2).csv'):
    X_train, X_val, y_train, y_val, preprocessor = FineTuneDataLoader.prepare(train_path, FINETUNE_CONFIG)
    
    # Clean Params
    tabpfn_init_params = {
        'device': FINETUNE_CONFIG['device'],
        'n_estimators': 1,
        'random_state': FINETUNE_CONFIG['random_seed'],
        'inference_precision': torch.float32,
        'fit_mode': 'batched',
        'softmax_temperature': 1.0
    }
    
    classifier, optimizer, scheduler, torch_model = setup_model(tabpfn_init_params, FINETUNE_CONFIG)
    
    # Save initial weights to verify updates later
    initial_weights = copy.deepcopy(list(torch_model.parameters())[0].data)

    # --- FOCAL LOSS ---
    print("\nUsing Focal Loss (alpha=0.75, gamma=2.0)")
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0).to(FINETUNE_CONFIG['device'])
    
    splitter = partial(train_test_split, test_size=0.25)
    training_datasets = classifier.get_preprocessed_datasets(
        X_train, y_train, splitter, FINETUNE_CONFIG['finetuning']['batch_size']
    )
    dataloader = DataLoader(
        training_datasets, batch_size=FINETUNE_CONFIG['finetuning']['meta_batch_size'], 
        collate_fn=meta_dataset_collator
    )

    # --- TRACKING ---
    best_f1 = -1.0
    best_loss = float('inf')
    best_epoch = -1
    best_model_state = copy.deepcopy(torch_model.state_dict())
    
    print("\n[STEP 3] STARTING FINE-TUNING")
    
    # Base Evaluation
    base_f1, base_loss = evaluate(classifier, tabpfn_init_params, X_train, y_train, X_val, y_val)
    print(f"Epoch 0 (Base): Val F1={base_f1:.4f} | Loss={base_loss:.4f}")
    best_f1 = base_f1
    best_loss = base_loss

    for epoch in range(1, FINETUNE_CONFIG['finetuning']['epochs'] + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            X_batch_train, X_batch_test, y_batch_train, y_batch_test, cat_ixs, confs = batch
            if len(np.unique(y_batch_train)) < 2: continue

            optimizer.zero_grad()
            classifier.fit_from_preprocessed(X_batch_train, y_batch_train, cat_ixs, confs)
            logits = classifier.forward(X_batch_test, return_logits=True)
            
            loss = loss_fn(logits, y_batch_test.to(FINETUNE_CONFIG['device']))
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(torch_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        
        # Evaluate
        val_f1, val_loss = evaluate(classifier, tabpfn_init_params, X_train, y_train, X_val, y_val)
        
        # === PEAK HUNTING LOGIC ===
        is_better_f1 = val_f1 > best_f1
        is_same_f1_better_loss = (val_f1 == best_f1) and (val_loss < best_loss)
        
        if is_better_f1 or is_same_f1_better_loss:
            best_f1 = val_f1
            best_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(torch_model.state_dict())
            print(f"Epoch {epoch}: Val F1={val_f1:.4f} | Loss={val_loss:.4f} (New Best!)")
        else:
            print(f"Epoch {epoch}: Val F1={val_f1:.4f} | Loss={val_loss:.4f}")

    print("\n" + "=" * 80)
    print(f"🏆 RESTORING BEST MODEL FROM EPOCH {best_epoch}")
    print(f"   Best F1: {best_f1:.4f}")
    print("=" * 80)
    
    # 1. LOAD THE PEAK MODEL
    torch_model.load_state_dict(best_model_state)
    
    # 2. VERIFY WEIGHTS CHANGED
    final_weights = list(torch_model.parameters())[0].data
    # FIX: Move both to CPU to avoid "Expected all tensors to be on the same device" error
    weight_diff = torch.norm(final_weights.cpu() - initial_weights.cpu()).item()
    print(f"Diagnostic: Weight Update Magnitude = {weight_diff:.6f}")
    if weight_diff == 0:
        print("⚠️ WARNING: Weights did not change! Check optimizer settings.")

    # 3. FIND OPTIMAL THRESHOLD ON VALIDATION
    # We need to find the specific probability cutoff that gives us the Peak F1.
    print("\n[STEP 4] OPTIMIZING THRESHOLD")
    
    # Create temp inference model to scan Validation set
    eval_config = tabpfn_init_params.copy()
    eval_config.update({
        'fit_mode': 'fit_preprocessors', 
        'inference_config': {'SUBSAMPLE_SAMPLES': FINETUNE_CONFIG['n_inference_context_samples']}
    })
    temp_clf = clone_model_for_evaluation(classifier, eval_config, TabPFNClassifier)
    temp_clf.fit(X_train, y_train)
    val_probs = temp_clf.predict_proba(X_val)[:, 1]
    
    optimal_threshold, optimal_f1 = find_best_threshold(y_val, val_probs)
    print(f"   Optimal Threshold found: {optimal_threshold:.4f} (Validation F1: {optimal_f1:.4f})")
    
    # 4. FINAL FIT ON ALL DATA
    print("\n[STEP 5] FITTING FINAL MODEL ON FULL DATA")
    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    
    final_inference_clf = TabPFNClassifier(
        device=FINETUNE_CONFIG['device'], n_estimators=1, fit_mode='fit_preprocessors',
        inference_config={'SUBSAMPLE_SAMPLES': FINETUNE_CONFIG['n_inference_context_samples']}
    )
    final_inference_clf._initialize_model_variables()
    final_inference_clf.models_[0].load_state_dict(best_model_state)
    final_inference_clf.fit(X_full, y_full)
    
    return final_inference_clf, preprocessor, optimal_threshold

if __name__ == "__main__":
    model, preprocessor, threshold = run_finetuning()
    
    print("\nGenerating Submission...")
    test_df = pd.read_csv('dataset-test-vf.csv')
    ids = test_df['ID']
    X_sub = preprocessor.transform(test_df.drop('ID', axis=1))
    
    # PREDICT WITH OPTIMAL THRESHOLD
    probs = model.predict_proba(X_sub)[:, 1]
    preds = (probs >= threshold).astype(int) # Using the learned threshold
    
    # Save
    filename = f'submission_peak_t{threshold:.2f}.csv'
    pd.DataFrame({'ID': ids, 'y': preds}).to_csv(filename, index=False)
    print(f"✓ {filename} saved (Threshold: {threshold:.4f})")
    
    # Visualization
    plt.figure(figsize=(10,4))
    sns.histplot(probs, bins=50, kde=True)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.title("Test Set Probability Distribution")
    plt.legend()
    plt.show()
