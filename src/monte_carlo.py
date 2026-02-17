import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

def run_simulation_gpu(probs_np, n_sims, steps, batch_size, device=None):
    """
    Runs a Monte Carlo simulation on the GPU to find the optimal decision threshold
    that maximizes the F1 score.
    
    Args:
        probs_np (np.array): Array of predicted probabilities.
        n_sims (int): Number of simulations to run.
        steps (int): Number of threshold steps to evaluate.
        batch_size (int): Batch size for processing simulations.
        device (torch.device, optional): Device to run on ('cuda' or 'cpu').
        
    Returns:
        tuple: (thresholds, mean_f1, std_f1)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Simulation running on: {device}")

    # 1. Move static data to GPU
    probs = torch.tensor(probs_np, dtype=torch.float32, device=device)
    thresholds = torch.linspace(0.01, 0.99, steps, device=device)

    # 2. Pre-calculate Predictions Matrix (n_samples x n_thresholds)
    #    We transpose it to (n_samples, n_thresholds) for easier matmul later
    preds_matrix = (probs.unsqueeze(1) >= thresholds.unsqueeze(0)).float()

    # Pre-calculate sum of positives for each threshold (for FP calculation)
    # Shape: (n_thresholds,)
    sum_preds = preds_matrix.sum(dim=0)

    # Accumulators for Mean and Variance (Welford's method or simple Sum/SumSq)
    # We use Sum/SumSq for speed here
    f1_sum = torch.zeros(steps, device=device)
    f1_sq_sum = torch.zeros(steps, device=device)

    n_batches = int(np.ceil(n_sims / batch_size))

    # 3. Batch Processing Loop
    for _ in tqdm(range(n_batches), desc="Processing Batches"):
        # Adjust batch size for last batch
        curr_batch = batch_size

        # A. Generate simulated reality for the whole batch at once
        # Shape: (batch_size, n_samples)
        rand_matrix = torch.rand((curr_batch, len(probs)), device=device)
        y_sim = (rand_matrix < probs).float()

        # B. Vectorized Confusion Matrix Calculation via Matrix Multiplication
        # TP = y_sim (B, S) @ preds (S, T) -> Result (B, T)
        tp = torch.matmul(y_sim, preds_matrix)

        # FP = Total Preds Positive - TP
        fp = sum_preds.unsqueeze(0) - tp

        # FN = Total True Positive (in simulation) - TP
        total_pos_sim = y_sim.sum(dim=1).unsqueeze(1)
        fn = total_pos_sim - tp

        # C. Calculate F1
        denominator = 2 * tp + fp + fn
        # Avoid division by zero
        f1_batch = torch.where(denominator > 0, (2 * tp) / denominator, torch.zeros_like(denominator))

        # D. Update Accumulators
        f1_sum += f1_batch.sum(dim=0)
        f1_sq_sum += (f1_batch ** 2).sum(dim=0)

    # 4. Calculate Final Stats
    mean_f1 = f1_sum / n_sims
    # Var = E[X^2] - (E[X])^2
    var_f1 = (f1_sq_sum / n_sims) - (mean_f1 ** 2)
    std_f1 = torch.sqrt(torch.clamp(var_f1, min=0)) # clamp to handle tiny precision errors

    return thresholds.cpu().numpy(), mean_f1.cpu().numpy(), std_f1.cpu().numpy()

def plot_threshold_analysis(thresholds, mean_f1, std_f1, best_thresh, best_f1, n_sims, save_path='threshold_analysis.png'):
    """
    Plots the results of the Monte Carlo threshold analysis.
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Plot Mean F1
    plt.plot(thresholds, mean_f1, color='#1f77b4', linewidth=2, label='Expected F1 Score')

    # Plot Confidence Interval (±1 Std Dev)
    plt.fill_between(thresholds,
                     mean_f1 - std_f1,
                     mean_f1 + std_f1,
                     color='#1f77b4', alpha=0.2, label='±1 Std Dev (Volatility)')

    # Mark Optimal Point
    plt.scatter(best_thresh, best_f1, color='red', s=100, zorder=5, label=f'Optimal: {best_thresh:.2f}')
    plt.axvline(best_thresh, color='red', linestyle='--', alpha=0.5)

    # Annotate
    plt.title(f'Monte Carlo Threshold Optimization ({n_sims} Simulations)', fontsize=14)
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(loc='lower center')
    plt.xlim(0, 1)

    # Save Plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved '{save_path}'")
