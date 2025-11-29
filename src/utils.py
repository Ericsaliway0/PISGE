## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python __pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type CPDB --score_threshold 0.99 --hidden_feats 1024 --learning_rate 0.001 --num_epochs 105
## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python __pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type HIPPIE --score_threshold 0.99 --in_feats 2048 --hidden_feats 256 --learning_rate 0.001 --num_epochs 105

## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python __pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --score_threshold 0.99 --learning_rate 0.001 --num_epochs 204
## p_value in average predicted score
## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python __pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type STRING --score_threshold 0.99 --learning_rate 0.001 --num_epochs 505
## python _gene_label_prediction_tsne_pertag.py --model_type ChebNet --net_type pathnet --score_threshold 0.4 --learning_rate 0.001 --num_epochs 65 
## (kg39) ericsali@erics-MBP-4 gnn_pathways % python _gene_label_prediction_tsne_sage.py --model_type EMOGI --net_type ppnet --score_threshold 0.5 --learning_rate 0.001 --num_epochs 100 
## (kg39) ericsali@erics-MBP-4 gnn_pathways % python _gene_label_prediction_tsne_pertag.py --model_type ATTAG --net_type ppnet --score_threshold 0.9 --learning_rate 0.001 --num_epochs 201

import json
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from torch_geometric.nn import GCNConv
from .models import ACGNN, HGDC, EMOGI, MTGCN, GCN, GAT, GraphSAGE, GIN, ChebNet, ChebNetII, DMGNN, FocalLoss
import pandas as pd
import torch.nn as nn

def save_model_details(model, args, model_csv_path, in_feats, hidden_feats, out_feats):
    """
    Extracts model details and saves them to a CSV file.

    Parameters:
    - model: The neural network model.
    - args: Arguments containing model configuration.
    - model_csv_path: File path to save the model details.
    - in_feats: Number of input features.
    - hidden_feats: Number of hidden layer features.
    - out_feats: Number of output features.
    """
    # Count layers and parameters
    num_layers = sum(1 for _ in model.children())  # Count layers
    total_params = sum(p.numel() for p in model.parameters())  # Count parameters

    # Detect attention layers
    attention_layer_nodes = None
    for layer in model.children():
        if hasattr(layer, 'heads'):  # Assuming attention layers have 'heads' attribute
            attention_layer_nodes = layer.heads

    # Detect residual connections
    has_residual = any(isinstance(layer, nn.Identity) for layer in model.modules())

    # Prepare data for CSV
    model_data = {
        "Method": [args.model_type],
        "Number of Layers": [num_layers],
        "Input Layer Nodes": [in_feats],
        "Hidden Layer Nodes": [hidden_feats],
        "Attention Layer Nodes": [attention_layer_nodes if attention_layer_nodes else "N/A"],
        "Output Layer Nodes": [out_feats],
        "Total Parameters": [total_params],
        "Residual Connection": ["Yes" if has_residual else "No"]
    }

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(model_data)
    df.to_csv(model_csv_path, index=False)
    print(f"Model architecture saved to {model_csv_path}")

def choose_model(model_type, in_feats, hidden_feats, out_feats):
    if model_type == 'GraphSAGE':
        return GraphSAGE(in_feats, hidden_feats, out_feats)
    elif model_type == 'GAT':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'GCN':
        return GCN(in_feats, hidden_feats, out_feats)
    elif model_type == 'GIN':
        return GIN(in_feats, hidden_feats, out_feats)
    elif model_type == 'HGDC':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'EMOGI':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'MTGCN':
        return GCN(in_feats, hidden_feats, out_feats)
    elif model_type == 'ChebNet':
        return ChebNet(in_feats, hidden_feats, out_feats)
    elif model_type == 'ChebNetII':
        return ChebNet(in_feats, hidden_feats, out_feats)    
    elif model_type == 'DMGNN':
        return DMGNN(
            in_feat_dim=in_feats,
            hidden_dim=hidden_feats,
            out_dim=out_feats,
            heads=4,
            dropout=0.5
        )
    elif model_type == 'ACGNN':
        return ACGNN(in_feats, hidden_feats, out_feats)
    else:
        raise ValueError("Invalid model type. Choose from ['GraphSAGE', 'GAT', 'EMOGI', 'HGDC', 'MTGCN', 'GCN', 'GIN', 'ChebNet', 'ChebNetII', 'DMGNN', 'ACGNN'].")

def save_and_plot_results_no_error_bar_pass(predicted_above, predicted_below, degrees_above, degrees_below, avg_above, avg_below, args):

    # Save predictions and degrees
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    def save_csv(data, filename, header):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerows(data)
        print(f"File saved: {filepath}")

    save_csv(predicted_above, f'{args.model_type}_above_threshold.csv', ['Gene', 'Score'])
    save_csv(predicted_below, f'{args.model_type}_below_threshold.csv', ['Gene', 'Score'])
    save_csv(degrees_above.items(), f'{args.model_type}_degrees_above.csv', ['Gene', 'Degree'])
    save_csv(degrees_below.items(), f'{args.model_type}_degrees_below.csv', ['Gene', 'Degree'])

    # Degree comparison barplot
    data = pd.DataFrame({
        'Threshold': ['Above', 'Below'],
        'Average Degree': [avg_above, avg_below]
    })
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x='Threshold', y='Average Degree', palette="viridis")
    plt.title('Average Degree Comparison')
    plt.savefig(os.path.join(output_dir, f'{args.model_type}_degree_comparison.png'))
    plt.show()

def plot_roc_curve(labels, scores, filename):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="blue")
    plt.plot([0, 1], [0, 1], color="salmon", linestyle="--")
    plt.title("Receiver Operating Characteristic Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    ##plt.grid(alpha=0.4)
    plt.savefig(filename)
    plt.show()
    print(f"ROC Curve saved to {filename}")

def plot_all_roc_curves_grid(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all ROC curves (one per cancer type) in a single figure.
    """
    plt.figure(figsize=(10, 8))
    
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc_score:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUROC Curves for All 16 Cancer Types")
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(False, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.show()
    print(f"Combined AUROC plot saved to {output_path}")

def plot_all_roc_curves_ori(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all AUROC curves (one per cancer type) with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} AUC = {auc_score:.3f}")

    # Random classifier diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2)

    # Axis labels and title
    plt.xlabel("Specificity", fontsize=14)
    plt.ylabel("Sensitivity", fontsize=14)
    plt.title("AUROC Curves for All Cancer Types", fontsize=16, weight='bold')

    # Custom AUC summary box (no frame)
    median_auc = round(np.median(all_aucs), 3)
    auc_range = (round(min(all_aucs), 3), round(max(all_aucs), 3))
    textstr = f"Median AUC = {median_auc}\nRange = {auc_range[0]}–{auc_range[1]}"
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    # Legend (no frame)
    plt.legend(loc='lower right', fontsize=9, ncol=2, frameon=False)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"AUROC plot saved to {output_path}")

def plot_all_roc_curves(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all AUROC curves (one per cancer type) with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} AUC = {auc_score:.4f}")

    # Random classifier diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2)

    # Axis labels and title
    plt.xlabel("Specificity", fontsize=26)
    plt.ylabel("Sensitivity", fontsize=26)
    plt.title("AUROC Curves for All Cancer Types", fontsize=14, weight='bold')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Custom AUC summary box (no frame)
    median_auc = round(np.median(all_aucs), 3)
    auc_range = (round(min(all_aucs), 3), round(max(all_aucs), 3))
    textstr = f"Median AUC = {median_auc}\nRange = {auc_range[0]}–{auc_range[1]}"
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=16, verticalalignment='top')

    # Legend (no frame)
    plt.legend(loc='lower right', fontsize=14, ncol=2, frameon=False)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"AUROC plot saved to {output_path}")

def plot_all_roc_curves_with_frame(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all AUROC curves (one per cancer type) with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} AUC = {auc_score:.3f}")

    # Random classifier diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2)

    # Axis labels and title
    plt.xlabel("Specificity", fontsize=14)
    plt.ylabel("Sensitivity", fontsize=14)
    plt.title("AUROC Curves for All Cancer Types", fontsize=16, weight='bold')

    # Custom AUC summary box
    median_auc = round(np.median(all_aucs), 3)
    auc_range = (round(min(all_aucs), 3), round(max(all_aucs), 3))
    textstr = f"Median AUC = {median_auc}\nRange = {auc_range[0]}–{auc_range[1]}"
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    # Legend
    plt.legend(loc='lower right', fontsize=9, ncol=2, frameon=True)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"AUROC plot saved to {output_path}")

def plot_all_roc_curves_mda_comparision_two_columns(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all AUROC curves (one per cancer type) with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} AUC = {auc_score:.4f}")

    # Random classifier diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2)

    # Axis labels and title
    # plt.xlabel("Specificity", fontsize=26)
    # plt.ylabel("Sensitivity", fontsize=26)
    plt.xlabel("False Positive Rate", fontsize=26)
    plt.ylabel("True Positive Rate", fontsize=26)
    plt.title("AUROC Curves for All Cancer Types", fontsize=14, weight='bold')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Custom AUC summary box (no frame)
    # median_auc = round(np.median(all_aucs), 3)
    # auc_range = (round(min(all_aucs), 3), round(max(all_aucs), 3))
    # textstr = f"Median AUC = {median_auc}\nRange = {auc_range[0]}–{auc_range[1]}"
    # plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
    #          fontsize=16, verticalalignment='top')

    # Legend (no frame)
    plt.legend(loc='lower right', fontsize=14, ncol=2, frameon=False)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"AUROC plot saved to {output_path}")

def plot_all_roc_curves_mda_comparision(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all AUROC curves (one per cancer type) with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC = {auc_score:.4f})")

    # Random classifier diagonal
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2)

    # Axis labels and title
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.title("AUROC Curves for TarBase", fontsize=20)#, weight='bold')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Legend in one column
    plt.legend(loc='lower right', fontsize=16, ncol=1, frameon=False)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"AUROC plot saved to {output_path}")

def plot_all_pr_curves_mda_comparision_(
    all_precisions, all_recalls, all_auprcs, model_labels, output_path
):
    """
    Plots all AUPRC curves (one per model) with scientific-style formatting, matching AUROC plot style.
    """
    plt.figure(figsize=(10, 8))

    # Plot each PR curve
    for precision, recall, auprc, label in zip(all_precisions, all_recalls, all_auprcs, model_labels):
        # Sort by recall to ensure correct plotting
        recall_sorted, precision_sorted = zip(*sorted(zip(recall, precision)))
        recall_sorted = np.array(recall_sorted)
        precision_sorted = np.array(precision_sorted)

        plt.plot(recall_sorted, precision_sorted, lw=1.5, label=f"{label} AUPRC = {auprc:.4f}")

    # Axis labels and title
    plt.xlabel("Recall", fontsize=26)
    plt.ylabel("Precision", fontsize=26)
    plt.title("AUPRC Curves for mirBase", fontsize=18)#, weight='bold')

    # Tick font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Legend
    plt.legend(loc='lower left', fontsize=14, ncol=1, frameon=False)

    # Clean plot
    plt.grid(False)
    plt.tight_layout()

    # Save and show
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"AUPRC plot saved to {output_path}")

def plot_all_pr_curves_mda_comparision(
    all_precisions, all_recalls, all_auprcs, model_labels, output_path
):
    """
    Plots Precision-Recall curves (AUPRC) for multiple models with legend at bottom left and no grid.
    """
    plt.figure(figsize=(10, 8))

    for precision, recall, auprc, label in zip(all_precisions, all_recalls, all_auprcs, model_labels):
        # Sort by recall to ensure a correct PR curve
        recall_sorted, precision_sorted = zip(*sorted(zip(recall, precision)))
        recall_sorted = np.array(recall_sorted)
        precision_sorted = np.array(precision_sorted)

        plt.plot(
            recall_sorted,
            precision_sorted,
            lw=1.2,
            label=f"{label} (AUPRC = {auprc:.4f})"
        )

    plt.title("AUPRC Curve for miRTarBase", fontsize=20)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.legend(loc="lower left", fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Precision-Recall Comparison Curve saved to {output_path}")

import matplotlib.pyplot as plt
import numpy as np

def plot_all_pr_curves_mda_comparision__(
    all_precisions, all_recalls, all_auprcs, model_labels, output_path
):
    """
    Plots Precision-Recall curves (AUPRC) for multiple models with proper sorting.

    Parameters:
        - all_precisions: list of arrays, precision values per model
        - all_recalls: list of arrays, recall values per model
        - all_auprcs: list of floats, AUPRC scores
        - model_labels: list of str, names of the models
        - output_path: str, where to save the figure
    """
    plt.figure(figsize=(8, 6))

    for precision, recall, auprc, label in zip(all_precisions, all_recalls, all_auprcs, model_labels):
        # Sort by recall to ensure proper curve
        recall_sorted, precision_sorted = zip(*sorted(zip(recall, precision)))

        # Convert to numpy arrays for safe handling
        recall_sorted = np.array(recall_sorted)
        precision_sorted = np.array(precision_sorted)

        plt.plot(
            recall_sorted,
            precision_sorted,
            lw=2,
            label=f"{label} (AUPRC = {auprc:.4f})"
        )

    plt.title("Precision-Recall Curve Comparison", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Precision-Recall Comparison Curve saved to {output_path}")

def plot_all_pr_curves_mda_comparision(
    all_precisions, all_recalls, all_auprcs, model_labels, output_path
):
    """
    Plots Precision-Recall curves (AUPRC) for multiple models with proper sorting.

    Parameters:
        - all_precisions: list of arrays, precision values per model
        - all_recalls: list of arrays, recall values per model
        - all_auprcs: list of floats, AUPRC scores
        - model_labels: list of str, names of the models
        - output_path: str, where to save the figure
    """
    plt.figure(figsize=(10, 8))

    for precision, recall, auprc, label in zip(all_precisions, all_recalls, all_auprcs, model_labels):
        # Sort by recall to ensure proper curve
        recall_sorted, precision_sorted = zip(*sorted(zip(recall, precision)))

        # Convert to numpy arrays for safe handling
        recall_sorted = np.array(recall_sorted)
        precision_sorted = np.array(precision_sorted)

        plt.plot(
            recall_sorted,
            precision_sorted,
            lw=1.2,
            label=f"{label} (AUPRC = {auprc:.4f})"
        )

    plt.title("AUPRC Curve for miRTarBase", fontsize=20)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.legend(loc="lower left", fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Precision-Recall Comparison Curve saved to {output_path}")

def plot_all_pr_curves(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all ROC curves (one per cancer type) in a single figure with scientific-style formatting.
    """
    plt.figure(figsize=(10, 8))

    # Plot each ROC curve
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} AUC = {auc_score:.3f}")

    # Plot diagonal (random)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='-', linewidth=1.2)

    # Axis labels and title
    plt.xlabel("1 - Specificity", fontsize=14)
    plt.ylabel("Sensitivity", fontsize=14)
    plt.title("Performance on Test Datasets", fontsize=16, weight='bold')

    # Invert X-axis to match the shared plot (Specificity on X, decreasing)
    plt.gca().invert_xaxis()

    # Custom AUC summary box
    median_auc = round(np.median(all_aucs), 3)
    auc_range = (round(min(all_aucs), 3), round(max(all_aucs), 3))
    textstr = f"Median AUC = {median_auc}\nRange = {auc_range[0]}–{auc_range[1]}"
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    # Legend
    plt.legend(loc='lower left', fontsize=9, ncol=2, frameon=True)

    # Remove grid, set tight layout
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Combined AUROC plot saved to {output_path}")

def plot_all_roc_curves_pa(all_fprs, all_tprs, all_aucs, cancer_names, output_path):
    """
    Plots all ROC curves (one per cancer type) in a single figure.
    """
    plt.figure(figsize=(10, 8))
    
    for fpr, tpr, auc_score, name in zip(all_fprs, all_tprs, all_aucs, cancer_names):
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc_score:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUROC Curves for All 16 Cancer Types")
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    # Grid line removed
    # plt.grid(False, linestyle='--', alpha=0.4)
    plt.grid(False)

    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.show()
    print(f"Combined AUROC plot saved to {output_path}")

def plot_pr_curve(labels, scores, filename):
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})", color="green")
    ##plt.plot([0, 1], [1, 0], color="salmon", linestyle="--")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    ##plt.grid(alpha=0.4)
    plt.savefig(filename)
    plt.show()
    print(f"Precision-Recall Curve saved to {filename}")

def load_graph_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    nodes = {}
    edges = []
    labels = []
    embeddings = []

    for entry in data:
        source = entry["source"]["properties"]
        target = entry["target"]["properties"]
        relation = entry["relation"]["type"]

        # Add source node
        if source["name"] not in nodes:
            nodes[source["name"]] = len(nodes)
            embeddings.append(source["embedding"])
            labels.append(source.get("label", -1) if source.get("label") is not None else -1)

        # Add target node
        if target["name"] not in nodes:
            nodes[target["name"]] = len(nodes)
            embeddings.append(target["embedding"])
            labels.append(target.get("label", -1) if target.get("label") is not None else -1)

        # Add edge
        edges.append((nodes[source["name"]], nodes[target["name"]]))

    # Convert embeddings and labels to tensors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return nodes, edges, embeddings_tensor, labels_tensor

def load_oncokb_genes(filepath):
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f)

def plot_and_analyze_ori(args):
    # File path for the saved predictions
    csv_file_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    results = []

    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            node_name, score, label = row
            score = float(score)
            label = int(label)
            results.append((node_name, score, label))
            
            # Check if label is 0 and print the row
            if label == 0:
                print(f"Node Name: {node_name}, Score: {score}, Label: {label}")


    # Extract scores and labels
    scores = np.array([row[1] for row in results])
    labels = np.array([row[2] for row in results])
    

    # Define group labels in the desired order
    group_labels = [1, 2, 0, 3]
    average_scores = []

    # Calculate average scores for each group
    for label in group_labels:
        group_scores = scores[labels == label]
        avg_score = np.mean(group_scores) if len(group_scores) > 0 else 0.0
        average_scores.append(avg_score)

    # Perform statistical tests to calculate p-values
    p_values = {}
    comparisons = [(1, 2), (1, 0), (1, 3)]  # Pairs to compare
    for group1, group2 in comparisons:
        scores1 = scores[labels == group1]
        scores2 = scores[labels == group2]
        if len(scores1) > 1 and len(scores2) > 1:
            _, p_value = ttest_ind(scores1, scores2, equal_var=False)
            p_values[(group1, group2)] = p_value
        else:
            p_values[(group1, group2)] = np.nan

    # Save average scores and p-values to another CSV
    avg_csv_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_pvalues_epo{args.num_epochs}_2048.csv'
    )
    os.makedirs(os.path.dirname(avg_csv_path), exist_ok=True)
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (group1, group2), p_value in p_values.items():
            writer.writerow(['', '', f'Group {group1} vs Group {group2}', p_value])
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(group_labels)), average_scores,
                   color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)
    for bar, avg_score in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg_score:.4f}',
                 ha='center', va='bottom', fontsize=12)
    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores for Each Gene Group', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    bar_plot_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_barplot_epo{args.num_epochs}_2048.png'
    )
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.show()

    '''
    
    # Convert gene_labels to NumPy array
    ##gene_labels = gene_labels.cpu().numpy()

    # Calculate average scores for each group in order of label 1, 2, 0, 3
    group_labels = [1, 2, 0, 3]  # Define the groups in the desired order
    average_scores = [np.mean(scores[gene_labels == label]) if (gene_labels == label).sum() > 0 else 0.0
                    for label in group_labels]

    p_values = {}
    for g1, g2 in [(1, 2), (1, 0), (1, 3)]:
        scores1 = scores[gene_labels == g1]
        scores2 = scores[gene_labels == g2]
        p_values[(g1, g2)] = ttest_ind(scores1, scores2, equal_var=False).pvalue if len(scores1) > 1 and len(scores2) > 1 else np.nan

    # Save averages and p-values
    avg_csv_path = os.path.join('results/gene_prediction/',
                                f'{args.model_type}_avg_scores_pvalues.csv')
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (g1, g2), p_val in p_values.items():
            writer.writerow(['', '', f'Group {g1} vs Group {g2}', p_val])
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot average scores
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(group_labels)), average_scores,
                   color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)
    for bar, avg in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg:.4f}',
                 ha='center', va='bottom', fontsize=12)
    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores by Gene Group', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    bar_plot_path = os.path.join('results/gene_prediction/',
                                 f'{args.model_type}_group_avg_scores.png')
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.show()'''

def plot_and_analyze_(args):
    # File path for the saved predictions
    csv_file_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    results = []

    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            node_name, score, label = row
            score = float(score)
            label = int(label)
            results.append((node_name, score, label))
    
    # Extract scores and labels as NumPy arrays
    scores = np.array([row[1] for row in results])
    labels = np.array([row[2] for row in results])
    
    # Define group labels in the desired order
    group_labels = [1, 2, 0, 3]
    average_scores = []

    # Calculate average scores for each group
    for label in group_labels:
        group_scores = scores[labels == label]
        avg_score = np.mean(group_scores) if len(group_scores) > 0 else 0.0
        average_scores.append(avg_score)

    # Perform statistical tests to calculate p-values
    p_values = {}
    comparisons = [(1, 2), (1, 0), (1, 3), (2, 3)]  # Added (2 vs. 3)
    for group1, group2 in comparisons:
        scores1 = scores[labels == group1]
        scores2 = scores[labels == group2]
        if len(scores1) > 1 and len(scores2) > 1:
            _, p_value = ttest_ind(scores1, scores2, equal_var=False)
            p_values[(group1, group2)] = p_value
        else:
            p_values[(group1, group2)] = np.nan

    # Save average scores and p-values to another CSV
    avg_csv_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_pvalues_threshold{args.score_threshold}_epo{args.num_epochs}_2048.csv'
    )
    os.makedirs(os.path.dirname(avg_csv_path), exist_ok=True)
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (group1, group2), p_value in p_values.items():
            writer.writerow(['', '', f'Group {group1} vs Group {group2}', p_value])
    
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(group_labels)), average_scores,
                   color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)
    for bar, avg_score in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg_score:.4f}',
                 ha='center', va='bottom', fontsize=12)
    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores for Each Gene Group', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the bar plot
    bar_plot_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_barplot_threshold{args.score_threshold}_epo{args.num_epochs}_2048.png'
    )
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.show()

def plot_and_analyze(args):
    # File path for the saved predictions
    csv_file_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    results = []

    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            node_name, score, label = row
            score = float(score)
            label = int(label)
            results.append((node_name, score, label))
    
    # Extract scores and labels as NumPy arrays
    scores = np.array([row[1] for row in results])
    labels = np.array([row[2] for row in results])
    
    # Define group labels in the desired order
    group_labels = [1, 2, 0, 3]
    average_scores = []

    # Calculate average scores for each group
    for label in group_labels:
        group_scores = scores[labels == label]
        avg_score = np.mean(group_scores) if len(group_scores) > 0 else 0.0
        average_scores.append(avg_score)

    # Perform statistical tests to calculate p-values
    p_values = {}
    comparisons = [(1, 2), (1, 0), (1, 3), (2, 3)]  # Added (2 vs. 3)
    for group1, group2 in comparisons:
        scores1 = scores[labels == group1]
        scores2 = scores[labels == group2]
        if len(scores1) > 1 and len(scores2) > 1:
            _, p_value = ttest_ind(scores1, scores2, equal_var=False)
            p_values[(group1, group2)] = p_value
        else:
            p_values[(group1, group2)] = np.nan

    # Save average scores and p-values to another CSV
    avg_csv_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_pvalues_threshold{args.score_threshold}_epo{args.num_epochs}_2048.csv'
    )
    os.makedirs(os.path.dirname(avg_csv_path), exist_ok=True)
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (group1, group2), p_value in p_values.items():
            writer.writerow(['', '', f'Group {group1} vs Group {group2}', p_value])
    
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    bar_width = 0.3  # Adjust the width (default is usually around 0.8)
    bars = plt.bar(range(len(group_labels)), average_scores, width=bar_width, 
                color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)

    # Add text labels on top of bars
    for bar, avg_score in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg_score:.4f}',
                ha='center', va='bottom', fontsize=12)

    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores for Each Gene Group', fontsize=16)
    ##plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the bar plot
    bar_plot_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_barplot_threshold{args.score_threshold}_epo{args.num_epochs}_2048.png'
    )
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.show()

def save_and_plot_results(predicted_above, predicted_below, degrees_above, degrees_below, avg_above, avg_below, avg_error_above, avg_error_below, args):

    # Save predictions and degrees
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    def save_csv(data, filename, header):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerows(data)
        print(f"File saved: {filepath}")

    save_csv(predicted_above, f'{args.model_type}_{args.net_type}_above_threshold.csv', ['Gene', 'Score'])
    save_csv(predicted_below, f'{args.model_type}_{args.net_type}_below_threshold.csv', ['Gene', 'Score'])
    save_csv(degrees_above.items(), f'{args.model_type}_{args.net_type}_degrees_above.csv', ['Gene', 'Degree'])
    save_csv(degrees_below.items(), f'{args.model_type}_{args.net_type}_degrees_below.csv', ['Gene', 'Degree'])

    # Degree comparison barplot with error bars
    data = pd.DataFrame({
        'Threshold': ['Above', 'Below'],
        'Average Degree': [avg_above, avg_below],
        'Error': [avg_error_above, avg_error_below]  # Add error values
    })
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(data['Threshold'], data['Average Degree'], yerr=data['Error'], capsize=5, color=['green', 'red'], edgecolor='black', alpha=0.8)

    # Add error bars explicitly (optional, can be done directly in the bar plot)
    for bar, error in zip(bars, data['Error']):
        plt.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=error, fmt='none', color='black', capsize=5, linestyle='--')

    plt.title('Average Degree Comparison with Error Bars')
    plt.savefig(os.path.join(output_dir, f'{args.model_type}_{args.net_type}_degree_comparison_with_error_bars.png'))
    plt.show()
    print(f"Degree comparison plot saved to {os.path.join(output_dir, f'{args.model_type}_{args.net_type}_degree_comparison_with_error_bars.png')}")
