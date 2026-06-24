import numpy as np
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fetch_experiment_data import get_group_A, get_group_B
from app.classifier import classify_batch

# Global Variables:
MSP_THRESHHOLD = 0.6377
ISO_THRESHOLD = -0.0255

# Per-class MSP thresholds
MSP_THRESHOLDS_PER_CLASS = {
    "1": 0.8241,
    "2": 0.6537,
    "3": 0.6049,
    "4": 0.6771,
    "5": 0.5935,
    "6": 0.7984,
    "7": 0.6804,
    "8": 0.5589,
    "9": 0.5898,
}

load_dotenv()

# Set academic plotting style
sns.set_theme(style="whitegrid")

def evaluate_predictions(scores, labels, threshold, group_name):
    """Calculates basic stats and prepares data for bar charts."""
    
    passed = scores >= threshold
    rejected = scores < threshold
    
    total = len(scores)
    total_passed = passed.sum()
    total_rejected = rejected.sum()
    
    print(f"\n--- {group_name} Statistics ---")
    print(f"Total Samples: {total}")
    print(f"Passed (Normal): {total_passed} ({total_passed/total*100:.1f}%)")
    print(f"Rejected (Anomaly): {total_rejected} ({total_rejected/total*100:.1f}%)")
    
    # Create a DataFrame for easy grouping
    df = pd.DataFrame({"Label": labels, "Passed": passed})
    summary = df.groupby("Label")["Passed"].value_counts().unstack(fill_value=0)
    
    # Ensure columns exist even if one category is empty
    if True not in summary.columns: summary[True] = 0
    if False not in summary.columns: summary[False] = 0
    
    summary.columns = ["Rejected", "Passed"]
    return summary

def plot_stacked_bars(summary_A, summary_B):
    """Plots the exact pass/reject distribution for each label."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Group A Plot (OOD)
    summary_A.plot(kind="bar", stacked=True, ax=axes[0], color=["#e74c3c", "#2ecc71"])
    axes[0].set_title("Group A (True OOD) Pass/Reject Rates")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_xlabel("OOD Category")
    
    # Group B Plot (In-Distribution)
    summary_B.plot(kind="bar", stacked=True, ax=axes[1], color=["#e74c3c", "#2ecc71"])
    axes[1].set_title("Group B (Self-Drawn Digits) Pass/Reject Rates")
    axes[1].set_ylabel("Number of Samples")
    axes[1].set_xlabel("Digit Label")
    
    plt.tight_layout()
    plt.show()

def plot_auroc(scores_A, scores_B, title, threshhold):
    """Calculates and plots the AUROC curve."""
    # For AUROC: 1 = Normal/In-Distribution (Group B), 0 = Anomaly/OOD (Group A)
    y_true = np.concatenate([np.zeros(len(scores_A)), np.ones(len(scores_B))])
    y_scores = np.concatenate([scores_A, scores_B])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{title} ROC (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    # Plot our specific locked threshold point
    # We find the index of the threshold closest to our ISO_THRESHOLD
    idx = np.argmin(np.abs(thresholds - threshhold))
    plt.plot(fpr[idx], tpr[idx], marker='o', markersize=8, color="red", label=f"Chosen Threshold ({threshhold})")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (OOD mistakenly passed)')
    plt.ylabel('True Positive Rate (Normal digits passed)')
    plt.title('Receiver Operating Characteristic (AUROC)')
    plt.legend(loc="lower right")
    plt.show()

def isolation_forest_experiment():
    ####################################
    #-----Load the experiment data-----#
    ####################################
    # Group A is true OOD (90-> zeros; 90-> capital letters; 90-> random scribble)
    X_A, y_A = get_group_A()
    # Group B technically should be in distribution (270 self drawn digits: 30 for each digit (1 - 9))
    X_B, y_B = get_group_B()

    # Reshape the data for the isolation forest:
    X_A_flat = X_A.reshape(len(X_A), -1)
    X_B_flat = X_B.reshape(len(X_B), -1)

    iso_forest: IsolationForest = joblib.load(os.getenv("MODEL_PATH_ISO_FOREST"))

    ####################################
    #-------Actual Experiments---------#
    ####################################
    iso_scores_A = iso_forest.decision_function(X_A_flat)
    summary_A = evaluate_predictions(iso_scores_A, y_A, ISO_THRESHOLD, "Group A (OOD)")

    iso_scores_B = iso_forest.decision_function(X_B_flat)
    summary_B = evaluate_predictions(iso_scores_B, y_B, ISO_THRESHOLD, "Group B (ID)")
    
    # 2. Render Visualizations
    plot_stacked_bars(summary_A, summary_B)
    plot_auroc(iso_scores_A, iso_scores_B, "Isolation Forest", ISO_THRESHOLD)

def msp_experiment():
    ####################################
    #-----Load the experiment data-----#
    ####################################
    # Group A is true OOD (90-> zeros; 90-> capital letters; 90-> random scribble)
    X_A, y_A = get_group_A()
    # Group B technically should be in distribution (270 self drawn digits: 30 for each digit (1 - 9))
    X_B, y_B = get_group_B()

    # Reshape into the correct format
    X_A = X_A.reshape(-1, 28, 28).astype(np.uint8)
    X_B = X_B.reshape(-1, 28, 28).astype(np.uint8)

    ####################################
    #-------Actual Experiments---------#
    ####################################
    batch_summary_A = classify_batch(X_A)
    msp_scores_A = np.array([result["confidence"] for result in batch_summary_A])
    summary_A = evaluate_predictions(msp_scores_A, y_A, MSP_THRESHHOLD, "Group A (OOD)")

    batch_summary_B = classify_batch(X_B)
    msp_scores_B = np.array([result["confidence"] for result in batch_summary_B])
    summary_B = evaluate_predictions(msp_scores_B, y_B, MSP_THRESHHOLD, "Group B (ID)")
    
    # 2. Render Visualizations
    plot_stacked_bars(summary_A, summary_B)
    plot_auroc(msp_scores_A, msp_scores_B, "MSP", MSP_THRESHHOLD)

def evaluate_combined_predictions(passed_iso, passed_msp, labels, group_name):
    """Calculates absolute stats by combining both model filters."""
    # The image only passes if BOTH models say it is normal (Bitwise AND)
    final_passed = passed_iso & passed_msp
    final_rejected = ~final_passed # Invert the boolean arrays
    
    total = len(labels)
    total_passed = final_passed.sum()
    total_rejected = final_rejected.sum()
    
    # Calculate exactly which model caught what for a deeper breakdown
    caught_only_by_iso = (~passed_iso & passed_msp).sum()
    caught_only_by_msp = (passed_iso & ~passed_msp).sum()
    caught_by_both = (~passed_iso & ~passed_msp).sum()
    
    print(f"\n--- {group_name} Absolute Combined Statistics ---")
    print(f"Total Samples: {total}")
    print(f"Passed (Normal): {total_passed} ({total_passed/total*100:.1f}%)")
    print(f"Rejected Total: {total_rejected} ({total_rejected/total*100:.1f}%)")
    print(f"  -> Caught by Isolation Forest only: {caught_only_by_iso}")
    print(f"  -> Caught by MSP only: {caught_only_by_msp}")
    print(f"  -> Caught by BOTH models: {caught_by_both}")
    
    # Create DataFrame for the bar charts
    df = pd.DataFrame({"Label": labels, "Passed": final_passed})
    summary = df.groupby("Label")["Passed"].value_counts().unstack(fill_value=0)
    
    if True not in summary.columns: summary[True] = 0
    if False not in summary.columns: summary[False] = 0
    
    summary.columns = ["Rejected", "Passed"]
    return summary

def plot_combined_stacked_bars(summary_A, summary_B):
    """Plots the absolute pass/reject distribution for the dual-method system."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Group A Plot (OOD)
    summary_A.plot(kind="bar", stacked=True, ax=axes[0], color=["#e74c3c", "#2ecc71"])
    axes[0].set_title("Dual-Method Pipeline: Group A (True OOD)")
    axes[0].set_ylabel("Absolute Number of Samples")
    axes[0].set_xlabel("OOD Category")
    
    # Group B Plot (In-Distribution)
    summary_B.plot(kind="bar", stacked=True, ax=axes[1], color=["#e74c3c", "#2ecc71"])
    axes[1].set_title("Dual-Method Pipeline: Group B (Self-Drawn Digits)")
    axes[1].set_ylabel("Absolute Number of Samples")
    axes[1].set_xlabel("Digit Label")
    
    plt.tight_layout()
    plt.show()

def combined_ood_experiment():
    ####################################
    #-----Load the experiment data-----#
    ####################################
    X_A, y_A = get_group_A()
    X_B, y_B = get_group_B()

    ####################################
    #--First Filter: Isolation forest--#
    ####################################
    X_A_flat = X_A.reshape(len(X_A), -1)
    X_B_flat = X_B.reshape(len(X_B), -1)

    iso_forest = joblib.load(os.getenv("MODEL_PATH_ISO_FOREST"))

    iso_scores_A = iso_forest.decision_function(X_A_flat)
    iso_scores_B = iso_forest.decision_function(X_B_flat)

    # Create boolean arrays (True if passed, False if rejected)
    passed_iso_A = iso_scores_A >= ISO_THRESHOLD
    passed_iso_B = iso_scores_B >= ISO_THRESHOLD

    ####################################
    #-------Second Filter: MSP---------#
    ####################################
    # We evaluate the ENTIRE dataset again, not just the survivors
    X_A_img = X_A.reshape(-1, 28, 28).astype(np.uint8)
    X_B_img = X_B.reshape(-1, 28, 28).astype(np.uint8)

    batch_summary_A = classify_batch(X_A_img)
    msp_scores_A = np.array([result["confidence"] for result in batch_summary_A])
    passed_msp_A = msp_scores_A >= MSP_THRESHHOLD

    batch_summary_B = classify_batch(X_B_img)
    msp_scores_B = np.array([result["confidence"] for result in batch_summary_B])
    passed_msp_B = msp_scores_B >= MSP_THRESHHOLD

    ####################################
    #-------Final Combination----------#
    ####################################
    summary_A = evaluate_combined_predictions(passed_iso_A, passed_msp_A, y_A, "Group A (OOD)")
    summary_B = evaluate_combined_predictions(passed_iso_B, passed_msp_B, y_B, "Group B (ID)")

    # Render final absolute visualizations
    plot_combined_stacked_bars(summary_A, summary_B)

def evaluate_predictions_per_class(batch_results, labels, group_name):
    """Evaluates predictions using per-class thresholds."""
    passed = np.zeros(len(batch_results), dtype=bool)

    for idx, result in enumerate(batch_results):
        predicted_class = result["prediction"]
        confidence = result["confidence"]
        threshold = MSP_THRESHOLDS_PER_CLASS[predicted_class]
        passed[idx] = confidence >= threshold

    rejected = ~passed

    total = len(batch_results)
    total_passed = passed.sum()
    total_rejected = rejected.sum()

    print(f"\n--- {group_name} Statistics (Per-Class Thresholds) ---")
    print(f"Total Samples: {total}")
    print(f"Passed (Normal): {total_passed} ({total_passed/total*100:.1f}%)")
    print(f"Rejected (Anomaly): {total_rejected} ({total_rejected/total*100:.1f}%)")

    # Create a DataFrame for easy grouping
    df = pd.DataFrame({"Label": labels, "Passed": passed})
    summary = df.groupby("Label")["Passed"].value_counts().unstack(fill_value=0)

    # Ensure columns exist even if one category is empty
    if True not in summary.columns: summary[True] = 0
    if False not in summary.columns: summary[False] = 0

    summary.columns = ["Rejected", "Passed"]
    return summary, passed

def msp_per_class_experiment():
    ####################################
    #-----Load the experiment data-----#
    ####################################
    X_A, y_A = get_group_A()
    X_B, y_B = get_group_B()

    # Reshape into the correct format
    X_A = X_A.reshape(-1, 28, 28).astype(np.uint8)
    X_B = X_B.reshape(-1, 28, 28).astype(np.uint8)

    ####################################
    #-------Actual Experiments---------#
    ####################################
    batch_summary_A = classify_batch(X_A)
    summary_A, passed_A = evaluate_predictions_per_class(batch_summary_A, y_A, "Group A (OOD)")

    batch_summary_B = classify_batch(X_B)
    summary_B, passed_B = evaluate_predictions_per_class(batch_summary_B, y_B, "Group B (ID)")

    # Extract MSP scores for AUROC
    msp_scores_A = np.array([result["confidence"] for result in batch_summary_A])
    msp_scores_B = np.array([result["confidence"] for result in batch_summary_B])

    # 2. Render Visualizations
    plot_stacked_bars(summary_A, summary_B)
    plot_auroc_per_class(msp_scores_A, msp_scores_B, passed_A, passed_B, "MSP Per-Class Thresholds")

def plot_auroc_per_class(scores_A, scores_B, passed_A, passed_B, title):
    """Calculates and plots the AUROC curve using per-class thresholds."""
    # For AUROC: 1 = Normal/In-Distribution (Group B), 0 = Anomaly/OOD (Group A)
    y_true = np.concatenate([np.zeros(len(scores_A)), np.ones(len(scores_B))])
    y_scores = np.concatenate([scores_A, scores_B])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{title} ROC (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (OOD mistakenly passed)')
    plt.ylabel('True Positive Rate (Normal digits passed)')
    plt.title('Receiver Operating Characteristic (AUROC) - Per-Class Thresholds')
    plt.legend(loc="lower right")
    plt.show()

def evaluate_combined_predictions_per_class(passed_iso, batch_results, labels, group_name):
    """Calculates absolute stats by combining Isolation Forest with per-class MSP."""
    # Apply per-class MSP thresholds
    passed_msp = np.zeros(len(batch_results), dtype=bool)

    for idx, result in enumerate(batch_results):
        predicted_class = result["prediction"]
        confidence = result["confidence"]
        threshold = MSP_THRESHOLDS_PER_CLASS[predicted_class]
        passed_msp[idx] = confidence >= threshold

    # The image only passes if BOTH models say it is normal (Bitwise AND)
    final_passed = passed_iso & passed_msp
    final_rejected = ~final_passed

    total = len(labels)
    total_passed = final_passed.sum()
    total_rejected = final_rejected.sum()

    # Calculate exactly which model caught what for a deeper breakdown
    caught_only_by_iso = (~passed_iso & passed_msp).sum()
    caught_only_by_msp = (passed_iso & ~passed_msp).sum()
    caught_by_both = (~passed_iso & ~passed_msp).sum()

    print(f"\n--- {group_name} Absolute Combined Statistics (Per-Class MSP) ---")
    print(f"Total Samples: {total}")
    print(f"Passed (Normal): {total_passed} ({total_passed/total*100:.1f}%)")
    print(f"Rejected Total: {total_rejected} ({total_rejected/total*100:.1f}%)")
    print(f"  -> Caught by Isolation Forest only: {caught_only_by_iso}")
    print(f"  -> Caught by MSP (Per-Class) only: {caught_only_by_msp}")
    print(f"  -> Caught by BOTH models: {caught_by_both}")

    # Create DataFrame for the bar charts
    df = pd.DataFrame({"Label": labels, "Passed": final_passed})
    summary = df.groupby("Label")["Passed"].value_counts().unstack(fill_value=0)

    if True not in summary.columns: summary[True] = 0
    if False not in summary.columns: summary[False] = 0

    summary.columns = ["Rejected", "Passed"]
    return summary

def combined_ood_per_class_experiment():
    ####################################
    #-----Load the experiment data-----#
    ####################################
    X_A, y_A = get_group_A()
    X_B, y_B = get_group_B()

    ####################################
    #--First Filter: Isolation forest--#
    ####################################
    X_A_flat = X_A.reshape(len(X_A), -1)
    X_B_flat = X_B.reshape(len(X_B), -1)

    iso_forest = joblib.load(os.getenv("MODEL_PATH_ISO_FOREST"))

    iso_scores_A = iso_forest.decision_function(X_A_flat)
    iso_scores_B = iso_forest.decision_function(X_B_flat)

    # Create boolean arrays (True if passed, False if rejected)
    passed_iso_A = iso_scores_A >= ISO_THRESHOLD
    passed_iso_B = iso_scores_B >= ISO_THRESHOLD

    ####################################
    #-Second Filter: Per-Class MSP-----#
    ####################################
    # We evaluate the ENTIRE dataset again, not just the survivors
    X_A_img = X_A.reshape(-1, 28, 28).astype(np.uint8)
    X_B_img = X_B.reshape(-1, 28, 28).astype(np.uint8)

    batch_summary_A = classify_batch(X_A_img)
    batch_summary_B = classify_batch(X_B_img)

    ####################################
    #-------Final Combination----------#
    ####################################
    summary_A = evaluate_combined_predictions_per_class(passed_iso_A, batch_summary_A, y_A, "Group A (OOD)")
    summary_B = evaluate_combined_predictions_per_class(passed_iso_B, batch_summary_B, y_B, "Group B (ID)")

    # Render final absolute visualizations
    plot_combined_stacked_bars(summary_A, summary_B)

if __name__ == "__main__":
    # Available experiments:
    # isolation_forest_experiment()
    # msp_experiment()
    # combined_ood_experiment()
    # msp_per_class_experiment()
    # combined_ood_per_class_experiment()

    # Running per-class MSP with dual pipeline:
    combined_ood_per_class_experiment()