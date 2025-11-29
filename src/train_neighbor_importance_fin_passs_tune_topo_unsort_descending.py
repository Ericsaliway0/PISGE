import os
import math
import csv
from collections import Counter

import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import psutil
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import entropy, ttest_ind, fisher_exact
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from sklearn.cluster import KMeans, SpectralBiclustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, confusion_matrix
)

from matplotlib import ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.cm import get_cmap
import dgl
import networkx as nx
from dgl.nn import GNNExplainer
from torch_geometric.nn import GCNConv
from .models import ACGNN, HGDC, EMOGI, MTGCN, GCN, GAT, GraphSAGE, GIN, ChebNet, FocalLoss
from src.utils import (
    choose_model, plot_roc_curve, plot_pr_curve, load_graph_data,
    load_oncokb_genes, plot_and_analyze, save_and_plot_results
)
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, ListedColormap, to_rgba, to_rgb
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
from venn import venn
import holoviews as hv
hv.extension('bokeh')
from holoviews import opts
from bokeh.io import output_notebook, export_png
from bokeh.plotting import show
from bokeh.io.export import get_screenshot_as_png
import plotly.graph_objects as go
from gprofiler import GProfiler
import glob
from pathlib import Path   
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
import numpy as np  # required for -log10
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import rbf_kernel
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d


def save_confirmed_predictions_to_csv(confirmed_predictions, output_dir, model_type, net_type, num_epochs):
    """
    Save confirmed predicted genes to a CSV file.
    
    Args:
    - confirmed_predictions: List of tuples (gene, score, sources).
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    confirmed_predictions_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_confirmed_predicted_genes_epo{num_epochs}_2048.csv')
    df_confirmed = pd.DataFrame(confirmed_predictions, columns=["Gene", "Score", "Source"])
    df_confirmed.to_csv(confirmed_predictions_csv_path, index=False)
    print(f"Confirmed predicted genes saved to {confirmed_predictions_csv_path}")

def analyze_sankey_structure(
    source,
    target,
    value,
    label_to_idx,
    node_names,
    cluster_to_genes,
    gene_to_neighbors,
    cluster_labels,
    name_to_index,
    scores
):
    # === Build Directed Graph
    G = nx.DiGraph()
    for s, t, v in zip(source, target, value):
        G.add_edge(s, t, weight=v)

    node_id_to_name = {v: k for k, v in label_to_idx.items()}

    # === 1. Degree Centrality of Genes
    gene_nodes = [label_to_idx[g] for g in node_names if g in label_to_idx]
    degree_centrality = nx.degree_centrality(G)
    gene_centrality_scores = {
        node_id_to_name[n]: degree_centrality[n]
        for n in gene_nodes if n in degree_centrality
    }

    # === 2. Gene-to-Cluster Participation Count
    gene_cluster_participation = defaultdict(set)
    for cluster_label, genes in cluster_to_genes.items():
        for gene in genes:
            gene_cluster_participation[gene].add(cluster_label)
    gene_cluster_counts = {gene: len(clusters) for gene, clusters in gene_cluster_participation.items()}

    # === 3. Entropy of Flow Distributions (per Confirmed Cluster)
    cluster_entropy = {}
    for cluster_label, genes in cluster_to_genes.items():
        scores_arr = np.array([scores[name_to_index[g]] for g in genes])
        probs = scores_arr / scores_arr.sum() if scores_arr.sum() > 0 else np.ones_like(scores_arr) / len(scores_arr)
        cluster_entropy[cluster_label] = entropy(probs)

    # === 4. Jaccard Similarity Between Clusters Based on Shared Downstream Clusters
    cluster_to_downstream = defaultdict(set)
    for gene, neighbors in gene_to_neighbors.items():
        if gene not in name_to_index:
            continue
        gene_idx = name_to_index[gene]
        cluster = f"Confirmed Cluster {cluster_labels[gene_idx]}"
        for neighbor_idx in neighbors:
            neighbor_cluster = f"Cluster {cluster_labels[neighbor_idx]}"
            cluster_to_downstream[cluster].add(neighbor_cluster)

    cluster_jaccard = {}
    for c1, c2 in combinations(cluster_to_downstream.keys(), 2):
        s1 = cluster_to_downstream[c1]
        s2 = cluster_to_downstream[c2]
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        if union > 0:
            cluster_jaccard[(c1, c2)] = intersection / union

    return {
        "gene_degree_centrality": gene_centrality_scores,
        "gene_cluster_counts": gene_cluster_counts,
        "cluster_entropy": cluster_entropy,
        "cluster_jaccard": cluster_jaccard,  # <- changed from cluster_jaccard_similarity
        "cluster_participation": gene_cluster_counts
    }

def save_metrics_to_csv(metrics_dict, output_dir='sankey_metrics'):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Degree Centrality
    pd.DataFrame.from_dict(metrics_dict['gene_degree_centrality'], orient='index', columns=['degree_centrality'])\
        .sort_values('degree_centrality', ascending=False)\
        .to_csv(f"{output_dir}/gene_degree_centrality.csv")

    # 2. Cluster Participation Count
    pd.DataFrame.from_dict(metrics_dict['gene_cluster_counts'], orient='index', columns=['cluster_count'])\
        .sort_values('cluster_count', ascending=False)\
        .to_csv(f"{output_dir}/gene_cluster_counts.csv")

    # 3. Entropy per Cluster
    pd.DataFrame.from_dict(metrics_dict['cluster_entropy'], orient='index', columns=['entropy'])\
        .sort_values('entropy', ascending=False)\
        .to_csv(f"{output_dir}/cluster_entropy.csv")

    # 4. Jaccard Similarity between Clusters
    # Jaccard Similarity between Clusters
    jaccard_df = pd.DataFrame([
        {'Cluster 1': k[0], 'Cluster 2': k[1], 'Jaccard Similarity': v}
        for k, v in metrics_dict['cluster_jaccard'].items()
    ])

    jaccard_df.sort_values(by='Jaccard Similarity', ascending=False)\
        .to_csv(f"{output_dir}/cluster_jaccard_similarity.csv", index=False)

def save_predictions_to_csv(predicted_genes, output_dir, model_type, net_type, num_epochs):
    """
    Save the predicted genes with their sources to a CSV file.
    
    Args:
    - predicted_genes: List of tuples (gene, score, sources) to save.
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    predicted_genes_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_predicted_driver_genes_epo{num_epochs}_2048.csv')
    df_predictions = pd.DataFrame(predicted_genes, columns=["Gene", "Score", "Confirmed Sources"])
    df_predictions.to_csv(predicted_genes_csv_path, index=False)
    print(f"Predicted driver genes with confirmed sources saved to {predicted_genes_csv_path}")

def save_predicted_known_drivers(predicted_driver_genes, output_dir, model_type, net_type, num_epochs):
    """
    Save predicted known cancer driver genes to a CSV file.
    
    Args:
    - predicted_driver_genes: List of predicted cancer driver genes.
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    predicted_drivers_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_predicted_known_drivers_epo{num_epochs}_2048.csv')
    df = pd.DataFrame(predicted_driver_genes, columns=["Gene"])
    df.to_csv(predicted_drivers_csv_path, index=False)
    print(f"Predicted known driver genes saved to {predicted_drivers_csv_path}")
        
def visualize_feature_relevance_heatmaps(relevance_df, clusters, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Merge relevance with cluster labels
    merged = relevance_df.copy()
    merged['gene'] = merged.index
    merged = pd.merge(merged, clusters, on='gene')

    for cluster_type in clusters.columns[1:]:
        for view in ['cancer', 'omics']:
            cluster_groups = merged.groupby(cluster_type)
            all_cluster_heatmaps = []
            fig, axes = plt.subplots(len(cluster_groups), 1, figsize=(12, 4 * len(cluster_groups)))
            if len(cluster_groups) == 1:
                axes = [axes]

            for ax, (label, group) in zip(axes, cluster_groups):
                data = group.drop(columns=['gene'] + list(clusters.columns[1:]))

                if view == 'cancer':
                    # Collapse omics features per cancer (max-over-omics per cancer)
                    cancer_types = list(set([col.split('_')[0] for col in data.columns]))
                    cancer_view = pd.DataFrame(index=data.index, columns=cancer_types)
                    for ct in cancer_types:
                        cols = [col for col in data.columns if col.startswith(ct + '_')]
                        cancer_view[ct] = data[cols].max(axis=1)
                    plot_data = cancer_view
                    title = f"{cluster_type.capitalize()} cluster {label} — Cancer view"
                    fname = f"{cluster_type.capitalize()}_cluster_{label}_cancer_heatmap.png"

                else:
                    # Collapse across cancer types for each omics type (max-over-cancers per omics)
                    omics_types = list(set([col.split('_')[-1] for col in data.columns]))
                    omics_view = pd.DataFrame(index=data.index, columns=omics_types)
                    for om in omics_types:
                        cols = [col for col in data.columns if col.endswith('_' + om)]
                        omics_view[om] = data[cols].max(axis=1)
                    plot_data = omics_view
                    title = f"{cluster_type.capitalize()} cluster {label} — Omics view"
                    fname = f"{cluster_type.capitalize()}_cluster_{label}_omics_heatmap.png"

                # Normalize per row for better heatmap contrast
                plot_data = pd.DataFrame(StandardScaler().fit_transform(plot_data.T).T,
                                         index=plot_data.index, columns=plot_data.columns)
                sns.heatmap(plot_data, cmap='viridis', ax=ax, cbar=True)
                ax.set_title(title)
                ax.set_xlabel('Features')
                ax.set_ylabel('Genes')

                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, fname))
                plt.close(fig)
 
CLUSTER_COLORS = {
    0: '#0077B6',  1: '#0000FF',  2: '#00B4D8',  3: '#48EAC4',
    4: '#F1C0E8',  5: '#B9FBC0',  6: '#32CD32',  7: '#bee1e6',
    8: '#8A2BE2',  9: '#E377C2'
}
'''CLUSTER_COLORS = {
    0: '#0077B6',  1: '#0000FF',  2: '#00B4D8',  3: '#48EAC4',
    4: '#F1C0E8',  5: '#B9FBC0',  6: '#32CD32',  7: '#bee1e6',
    8: '#8A2BE2',  9: '#E377C2', 10: '#8EECF5', 11: '#A3C4F3',
    12: '#FFB347', 13: '#FFD700', 14: '#FF69B4', 15: '#CD5C5C',
    16: '#7FFFD4', 17: '#FF7F50'
}'''

CLUSTER_COLORS_OMICS = {
    0: '#D62728',  1: '#1F77B4',  2: '#2CA02C',  3: '#9467BD'
}


def compute_lrp_scores(model, graph, features, node_indices=None):
    model.eval()
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    features.requires_grad_(True)


    with torch.enable_grad():
        logits = model(graph, features)
        probs = torch.sigmoid(logits.squeeze())

        # Select the nodes to analyze (e.g., predicted cancer genes)
        if node_indices is None:
            node_indices = torch.nonzero((probs > 0.5)).squeeze()

        relevance_scores = torch.zeros_like(features)

        for idx in node_indices:
            model.zero_grad()
            ##probs[idx].backward(retain_graph=True)
            probs[idx].backward(retain_graph=(idx != node_indices[-1]))

            relevance_scores[idx] = features.grad[idx].detach()

    return relevance_scores
def process_predictions(ranking, args, drivers_file_path, oncokb_file_path, ongene_file_path, ncg_file_path, intogen_file_path, node_names, non_labeled_nodes):
    """
    Process and save the predicted driver genes, confirmed sources, and known drivers.
    
    Args:
    - ranking: List of tuples (gene, score) representing ranked predictions.
    - args: Argument object containing model and network type, and score threshold.
    - drivers_file_path, oncokb_file_path, ongene_file_path, ncg_file_path, intogen_file_path: Paths to the confirmation gene files.
    - node_names, non_labeled_nodes: Information about node names and indices for matching.
    """
    # Load data from the confirmation files
    oncokb_genes = load_gene_set(oncokb_file_path)
    ongene_genes = load_gene_set(ongene_file_path)
    ncg_genes = load_gene_set(ncg_file_path)
    intogen_genes = load_gene_set(intogen_file_path)

    # Threshold for the score
    score_threshold = args.score_threshold

    confirmed_predictions = []
    predicted_genes = []

    for node, score in ranking:
        if score >= score_threshold:
            sources = []  # Accumulate sources confirming the gene
            if node in oncokb_genes:
                sources.append("OncoKB")
            if node in ongene_genes:
                sources.append("OnGene")
            if node in ncg_genes:
                sources.append("NCG")
            if node in intogen_genes:
                sources.append("IntOGen")
            if sources:  # If the gene is confirmed by at least one source
                confirmed_predictions.append((node, score, ", ".join(sources)))
            predicted_genes.append((node, score, ", ".join(sources) if sources else ""))

    # Save predictions to a CSV file
    save_predictions_to_csv(predicted_genes, 'results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)
    save_confirmed_predictions_to_csv(confirmed_predictions, 'results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)

    # Load known cancer driver genes
    with open(drivers_file_path, 'r') as f:
        known_drivers = set(line.strip() for line in f)

    # Collect predicted cancer driver genes that match the known drivers
    predicted_driver_genes = [node_names[i] for i in non_labeled_nodes if node_names[i] in known_drivers]

    # Save the predicted known cancer driver genes to a CSV file
    save_predicted_known_drivers(predicted_driver_genes, 'results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)

def load_gene_set(file_path):
    """
    Load a gene list from a file and return as a set.
    
    Args:
    - file_path: Path to the file containing genes, one per line.
    
    Returns:
    - Set of gene names.
    """
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def save_cluster_labels(cluster_labels, save_path):
    ##cluster_labels_path = os.path.join(os.path.dirname(save_path), "cluster_labels.npy")
    np.save(save_path, cluster_labels)
    print(f"Cluster labels saved to {save_path}")

# Save total genes per cluster
def save_total_genes_per_cluster(total_genes_per_cluster, save_path):
    ##total_genes_path = os.path.join(os.path.dirname(save_path), "total_genes_per_cluster.npy")
    np.save(save_path, total_genes_per_cluster)
    print(f"Total genes per cluster saved to {save_path}")

# Save predicted cancer genes per cluster
def save_predicted_counts(pred_counts, save_path):
    ##pred_counts_path = os.path.join(os.path.dirname(save_path), "predicted_counts.npy")
    np.save(save_path, pred_counts)
    print(f"Predicted cancer genes per cluster saved to {save_path}")

'''# Function to reload the saved graph data
def reload_spectral_biclustering_data(save_path):
    print(f"Loading graph data from {save_path}...")
    
    # Load the saved data
    checkpoint = torch.load(save_path)

    # Extract graph-related data
    edges = checkpoint['edges']
    features = checkpoint['features']
    labels = checkpoint['labels']
    cluster_labels = checkpoint['cluster']

    print("Graph data loaded successfully.")

    return edges, features, labels, cluster_labels
'''
# Function to reload cluster labels

def reload_cluster_labels(save_path):
    ##cluster_labels_path = os.path.join(os.path.dirname(save_path), "cluster_labels.npy")
    cluster_labels = np.load(save_path)
    print(f"Cluster labels loaded from {save_path}")
    return cluster_labels

# Function to reload total genes per cluster
def reload_total_genes_per_cluster(save_path):
    ##total_genes_path = os.path.join(os.path.dirname(save_path), "total_genes_per_cluster.npy")
    total_genes_per_cluster = np.load(save_path, allow_pickle=True).item()
    print(f"Total genes per cluster loaded from {save_path}")
    return total_genes_per_cluster

# Function to reload predicted cancer genes per cluster
def reload_predicted_counts(save_path):
    ##pred_counts_path = os.path.join(os.path.dirname(save_path), "predicted_counts.npy")
    pred_counts = np.load(save_path, allow_pickle=True).item()
    print(f"Predicted cancer genes per cluster loaded from {save_path}")
    return pred_counts

def load_bioclustered_graph(save_path):
    """
    Loads a previously saved DGL graph that includes features, labels, and cluster assignments.

    Args:
        save_path (str): Path to the saved graph file (.pth)

    Returns:
        dgl.DGLGraph: The reconstructed graph with restored node data.
    """
    data = torch.load(save_path)

    graph = dgl.graph(data['edges'])
    graph.ndata['feat'] = data['features']

    if data.get('label') is not None:
        graph.ndata['label'] = data['label']

    if data.get('cluster') is not None:
        graph.ndata['cluster'] = data['cluster']

    if 'degree' in data:
        graph.ndata['degree'] = data['degree']
    if 'train_mask' in data:
        graph.ndata['train_mask'] = data['train_mask']
    if 'test_mask' in data:
        graph.ndata['test_mask'] = data['test_mask']

    print("✅ Clustered graph loaded successfully.")
    return graph

def save_bioclustered_gene_info_csv(
    cluster_labels,
    total_genes_per_cluster,
    pred_counts,
    node_names,
    predicted_gene_indices,
    output_csv_path
):
    """
    Saves clustered gene information to a CSV file.

    Args:
        cluster_labels (np.ndarray): Cluster assignments.
        total_genes_per_cluster (dict): Total genes in each cluster.
        pred_counts (dict): Number of predicted cancer genes per cluster.
        node_names (list): List of all gene names by index.
        predicted_gene_indices (list): Indices of predicted cancer genes.
        output_csv_path (str): Path to save the CSV file.
    """
    gene_data = []
    for idx, name in enumerate(node_names):
        cluster = cluster_labels[idx]
        is_predicted = 1 if idx in predicted_gene_indices else 0
        gene_data.append({
            "Gene": name,
            "Cluster": cluster,
            "IsPredictedCancerGene": is_predicted,
            "TotalGenesInCluster": total_genes_per_cluster[cluster],
            "PredictedInCluster": pred_counts[cluster]
        })

    df = pd.DataFrame(gene_data)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Clustered gene information saved to {output_csv_path}")

def save_reduced_feature_relevance_scores(
    relevance_scores,      # Original 2048D matrix, shape [N, 2048]
    gene_names,            # List of gene symbols, length N
    output_csv_path        # File to save the reduced 64D matrix
):
    """
    Reduce 2048D LRP scores to 64D using max-over-16 logic,
    and save to CSV with columns like 'BRCA_mf', ..., 'KIRP_meth'.
    """
    import numpy as np
    import pandas as pd
    import os

    # -- Step 1: Reduce features to 64D
    reduced_scores = extract_summary_features_np_skip(relevance_scores)  # shape [N, 64]

    # -- Step 2: Define feature names in cancer x omics order
    cancer_types = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC']
    omics_types = ['cna', 'ge', 'meth', 'mf']
    column_labels = [f"{cancer}_{omics}" for omics in omics_types for cancer in cancer_types]

    # -- Step 3: Build DataFrame and save
    df = pd.DataFrame(reduced_scores, index=gene_names, columns=column_labels)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path)
    print(f"✅ Saved reduced feature relevance matrix to {output_csv_path}")

def plot_top_predicted_genes_tsne(graph, node_names, scores, output_path, top_k=1000):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores##.cpu().numpy()

    # Get top K predicted genes
    top_indices = np.argsort(scores)[-top_k:]
    top_embeddings = embeddings[top_indices]
    top_clusters = cluster_ids[top_indices]
    top_scores = scores[top_indices]
    top_names = [node_names[i] for i in top_indices]

    # t-SNE projection
    tsne = TSNE(n_components=2, random_state31)
    tsne_coords = tsne.fit_transform(top_embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    rcParams['pdf.fonttype'] = 42  # prevent font issues in vector graphics

    for c in np.unique(top_clusters):
        mask = top_clusters == c
        coords = tsne_coords[mask]
        plt.scatter(
            coords[:, 0], coords[:, 1],
            color=CLUSTER_COLORS.get(c, "#555555"),
            edgecolors='k',
            s=60,
            alpha=0.8
        )

        # Top 1 in this cluster (within top_k)
        cluster_scores = top_scores[mask]
        if cluster_scores.size > 0:
            top_idx_in_cluster = np.argmax(cluster_scores)
            name = np.array(top_names)[mask][top_idx_in_cluster]
            x, y = coords[top_idx_in_cluster]
            # Highlight the top 1 with a yellow circle (half the size of the original dot)
            plt.scatter(x, y, s=60, edgecolors='yellow', alpha=0.6, linewidth=1, marker='o', color='red')
            # Change label text color to red
            plt.text(x, y, name, fontsize=9, fontweight='bold', ha='center', va='center', color='black')

    plt.title("t-SNE of Top 1000 Predicted Genes by Cluster", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    # Remove legend
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Top predicted gene t-SNE plot saved to:\n{output_path}")

def plot_tsne_predicted_genes(graph, node_names, scores, output_path, args):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores##.cpu().numpy()

    predicted_mask = scores >= args.score_threshold
    predicted_indices = np.where(predicted_mask)[0]
    predicted_scores = scores[predicted_indices]
    predicted_clusters = cluster_ids[predicted_indices]
    predicted_embeddings = embeddings[predicted_indices]

    tsne = TSNE(n_components=2, random_state31)
    tsne_coords = tsne.fit_transform(predicted_embeddings)

    # Gather top 2 genes per cluster
    top_genes = []
    for c in np.unique(predicted_clusters):
        cluster_mask = predicted_clusters == c
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue
        top_indices = cluster_indices[np.argsort(predicted_scores[cluster_indices])[-2:]]  # top 2
        for idx in top_indices:
            top_genes.append((predicted_indices[idx], tsne_coords[idx], c))

    # Plot
    plt.figure(figsize=(10, 7))
    for idx, (node_idx, coord, cluster_id) in enumerate(top_genes):
        color = CLUSTER_COLORS.get(cluster_id, "#333333")
        plt.scatter(coord[0], coord[1], color=color, s=120, edgecolor='k')
        plt.text(coord[0]+1.5, coord[1], node_names[node_idx], fontsize=9, color=color)

    # Legend
    unique_predicted_clusters = np.unique(predicted_clusters)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=f"Cluster {c}", markersize=10)
        for c, color in CLUSTER_COLORS.items()
        if c in unique_predicted_clusters
    ]
    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE of Top 2 Predicted Genes per Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importance(
    relevance_vector,
    feature_names=None,
    node_name=None,
    top_k=9,
    gene_names=None,
    output_path="plots/feature_importance.png"):
    # Convert to NumPy array
    if isinstance(relevance_vector, torch.Tensor):
        relevance_vector = relevance_vector.detach().cpu().numpy()
    else:
        relevance_vector = np.array(relevance_vector)

    # --- 🔄 Min-Max Normalization ---
    min_val = relevance_vector.min()
    max_val = relevance_vector.max()
    norm_scores = (relevance_vector - min_val) / (max_val - min_val + 1e-8)

    # --- 🔝 Top-K selection ---
    top_indices = np.argsort(norm_scores)[-top_k:][::-1]
    top_scores = norm_scores[top_indices]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(relevance_vector))]

    # --- 🧬 Labeling ---
    if gene_names is not None:
        top_labels = [gene_names[i].capitalize() if i < len(gene_names) else f"Unknown {i}" for i in top_indices]
    else:
        top_labels = [feature_names[i] for i in top_indices]

    # Construct DataFrame
    df = pd.DataFrame({
        "feature": top_labels,
        "relevance": top_scores
    })

    # --- 📊 Plot ---
    plt.figure(figsize=(2.5, 2.5))
    sns.set_style("white")
    sns.barplot(
        data=df,
        x="feature",
        y="relevance",
        palette="Blues_d",
        dodge=False,
        legend=False,
        width=0.6
    )

    plt.ylabel("Relevance score", fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.title(f"{node_name}", fontsize=13)

    plt.xticks(rotation=90, ha='center', fontsize=11)
    plt.yticks(fontsize=11)

    sns.despine()
    plt.tight_layout()

    # --- 💾 Ensure directory exists and save ---
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.show()

def _plot_bar(omics_relevance, omics_colors, omics_order, output_path):
    plt.figure(figsize=(1.4, 2))
    sns.barplot(
        x=omics_relevance.index,
        y=omics_relevance.values,
        palette=[omics_colors[o] for o in omics_order],
        width=0.6
    )
    # Capitalize x-axis labels
    plt.xticks(
        ticks=range(len(omics_relevance.index)),
        labels=[label.upper() for label in omics_relevance.index],
        rotation=90,
        fontsize=9
    )
    plt.yticks(fontsize=10)
    plt.xlabel('', fontsize=10)
    plt.ylabel('', fontsize=10)
    plt.tick_params(axis='both', which='both', length=0)
    sns.despine()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.show()

def extract_summary_features_np_skip(features_np):
    num_nodes = features_np.shape[0]
    total_dim = features_np.shape[1]
    summary_features = []

    for o_idx in range(4):  # omics
        for c_idx in range(16):  # cancer
            base = o_idx * 16 * 16 + c_idx * 16
            if base + 16 > total_dim:
                continue  # skip invalid slice
            group = features_np[:, base:base + 16]  # [num_nodes, 16]
            max_vals = group.max(axis=1, keepdims=True)
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)

def compute_integrated_gradients(
    model, graph, features, node_indices=None, baseline=None, steps=50):
    model.eval()

    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)
    features = features.clone().detach()

    if baseline is None:
        baseline = torch.zeros_like(features)

    assert baseline.shape == features.shape, "Baseline must match feature shape"

    # Scale inputs and compute gradients
    scaled_inputs = [
        baseline + (float(i) / steps) * (features - baseline)
        for i in range(1, steps + 1)
    ]
    scaled_inputs = torch.stack(scaled_inputs)  # Shape: (steps, num_nodes, num_features)

    # Integrated gradients initialization
    integrated_grads = torch.zeros_like(features)

    for step_input in scaled_inputs:
        step_input.requires_grad_(True)
        logits = model(graph, step_input)
        probs = torch.sigmoid(logits.squeeze())

        if node_indices is None:
            node_indices = torch.nonzero(probs > 0.5, as_tuple=False).squeeze()
            if node_indices.ndim == 0:
                node_indices = node_indices.unsqueeze(0)

        grads = torch.autograd.grad(
            outputs=probs[node_indices],
            inputs=step_input,
            grad_outputs=torch.ones_like(probs[node_indices]),
            retain_graph=True,
            create_graph=False,
        )[0]

        integrated_grads += grads.detach()

    # Average the gradients and scale by the input difference
    avg_grads = integrated_grads / steps
    ig_attributions = (features - baseline) * avg_grads

    return ig_attributions

def get_two_hop_neighbors(graph, node_id):
    one_hop = set(graph.neighbors(node_id)) if node_id in graph else set()
    two_hop = set()
    
    for neighbor in one_hop:
        two_hop.update(graph.neighbors(neighbor))
    
    # Remove the original node and one-hop neighbors from two-hop set
    two_hop.difference_update(one_hop)
    two_hop.discard(node_id)
    
    return sorted(two_hop)

def save_cluster_legend(output_path_legend, cluster_colors, num_clusters=12):
    """
    Creates and saves a separate legend image for cluster colors in a single row.

    Args:
        output_path_legend (str): Path to save the legend image.
        cluster_colors (list): List of colors for each cluster.
        num_clusters (int): Number of clusters.
    """
    fig, ax = plt.subplots(figsize=(12, 1.5))  # Wider and shorter for single-row legend

    # Create legend handles labeled from Cluster 0 to Cluster 11
    legend_patches = [mpatches.Patch(color=cluster_colors[i], label=f"Cluster {i}") 
                      for i in range(num_clusters)]

    # Display legend with one row
    ax.legend(handles=legend_patches, loc='center', ncol=num_clusters,
              frameon=False, fontsize=14)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Save legend image
    plt.savefig(output_path_legend, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Legend saved to {output_path_legend}")

def compute_saliency_all_nodes(model, g, features, target_classes=None):
    """
    Compute saliency (relevance scores) for all nodes using gradients.

    Args:
        model: Trained GNN model
        g: DGL graph
        features: Input node features (torch.Tensor or np.ndarray)
        target_classes: Optional tensor/list of target classes per node. If None, use predicted class.

    Returns:
        relevance_scores: Tensor of shape [num_nodes, num_features] with saliency values.
    """
    model.eval()
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)
    
    features = features.clone().detach().requires_grad_(True)
    logits = model(g, features)

    num_nodes = features.shape[0]
    relevance_scores = torch.zeros_like(features)

    if target_classes is None:
        target_classes = torch.argmax(logits, dim=1)

    for node_id in range(num_nodes):
        model.zero_grad()
        if features.grad is not None:
            features.grad.zero_()

        score = logits[node_id, target_classes[node_id]]
        score.backward(retain_graph=True)

        relevance_scores[node_id] = features.grad[node_id].abs().detach()

    return relevance_scores

def plot_gene_feature_contributions_topo(gene_name, relevance_vector, feature_names, score, output_path=None):
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    barplot_path = output_path.replace(".png", "_omics_barplot.png") if output_path else None
    plot_omics_barplot_topo(df, barplot_path)

    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    heatmap_data = df.pivot(index='Omics', columns='Cancer', values='Relevance')
    omics_order = ['cna', 'ge', 'meth', 'mf']
    heatmap_data = heatmap_data.reindex(omics_order)

    plt.figure(figsize=(8, 2.8))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar=False, linewidths=0.3, linecolor='gray')
    ##plt.title(f"{gene_name} ({score:.3f})", fontsize=14)
    ##plt.title(f"{gene_name} ({float(score):.3f})", fontsize=14)
    if isinstance(score, np.ndarray):
        score = score.item()

    ##plt.title(f"{gene_name} ({score.item():.3f})", fontsize=14)
    plt.title(f"{gene_name}", fontsize=14)


    plt.yticks(rotation=0)
    plt.xticks(rotation=90, ha='right')
    plt.xlabel('')
    plt.ylabel('')
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_gene_feature_contributions_bio(gene_name, relevance_vector, feature_names, score, output_path=None):
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    barplot_path = output_path.replace(".png", "_omics_barplot.png") if output_path else None
    plot_omics_barplot_bio(df, barplot_path)

    df[['Omics', 'Cancer']] = df['Feature'].str.split(':', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    heatmap_data = df.pivot(index='Omics', columns='Cancer', values='Relevance')
    omics_order = ['cna', 'ge', 'meth', 'mf']
    heatmap_data = heatmap_data.reindex(omics_order)

    plt.figure(figsize=(8, 2.8))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar=False, linewidths=0.3, linecolor='gray')
    ##plt.title(f"{gene_name} ({score:.3f})", fontsize=14)
    ##plt.title(f"{gene_name} ({float(score):.3f})", fontsize=14)
    if isinstance(score, np.ndarray):
        score = score.item()
    ##plt.title(f"{gene_name} ({score.item():.3f})", fontsize=14)
    plt.title(f"{gene_name}", fontsize=14)


    plt.yticks(rotation=0)
    plt.xticks(rotation=90, ha='right')
    plt.xlabel('')
    plt.ylabel('')
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confirmed_neighbors_bio(
    args,
    graph,
    node_names,
    name_to_index,
    confirmed_genes,
    scores,
    cluster_labels,
    total_clusters,
    relevance_scores):
    # Build safe top-k index mapping
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list.")
            continue

        node_idx = topk_name_to_index[gene]
        gene_score = scores[node_idx]
        gene_cluster = graph.ndata["cluster_bio"][node_idx].item()
        print(f"{gene} → Node {node_idx} | Bio score: {gene_score:.4f} | Cluster: {gene_cluster}")

        neighbors = neighbors_dict.get(gene, [])

        # Filter only those neighbors present in top-k
        neighbor_scores_dict = {}
        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:
                    rel_score = relevance_scores[rel_idx].sum().item()
                    #if rel_score > 0.1:
                    neighbor_scores_dict[rel_idx] = rel_score

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors found for {gene}.")
            continue

        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        plot_path = os.path.join(
            "results/gene_prediction/bio_neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_bio_confirmed_neighbor_relevance_epo{args.num_epochs}.png"
        )

        plot_neighbor_relevance(
            neighbor_scores=top_neighbors,
            gene_name=f"{gene} (Cluster {gene_cluster})",
            node_id_to_name=node_id_to_name,
            output_path=plot_path,
            cluster_labels=cluster_labels,
            total_clusters=total_clusters,
            add_legend=False
        )

def plot_confirmed_neighbors_topo(
    args,
    graph,
    node_names,
    name_to_index,
    confirmed_genes,
    scores,
    relevance_scores,
    cluster_labels,
    total_clusters):
    # Only top-k names are passed in node_names
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list.")
            continue

        node_idx = topk_name_to_index[gene]
        gene_score = scores[node_idx]

        # ✅ Get cluster from graph
        gene_cluster = graph.ndata["cluster_topo"][node_idx].item()
        print(f"{gene} → Node {node_idx} | Topo score: {gene_score:.4f} | Cluster: {gene_cluster}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_scores_dict = {}

        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:  # bounds check
                    rel_score = relevance_scores[rel_idx].sum().item()
                    # if rel_score > 0.1:
                    neighbor_scores_dict[rel_idx] = rel_score

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors found for {gene}.")
            continue

        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        plot_path = os.path.join(
            "results/gene_prediction/topo_neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_topo_confirmed_neighbor_relevance_epo{args.num_epochs}.png"
        )

        plot_neighbor_relevance(
            neighbor_scores=top_neighbors,
            gene_name=f"{gene} (Cluster {gene_cluster})",
            node_id_to_name=node_id_to_name,
            output_path=plot_path,
            cluster_labels=cluster_labels,
            total_clusters=total_clusters,
            add_legend=False
        )

def plot_confirmed_neighbor_relevance(
    args,
    graph,
    node_names,
    name_to_index,
    confirmed_genes,
    scores,
    relevance_scores,
    mode="bio"):  # or "topo"):
    """
    Plots top-10 neighbor relevance scores for confirmed genes.
    Mode can be 'bio' or 'topo'.
    """


    assert mode in ("bio", "topo"), "Mode must be 'bio' or 'topo'"

    node_id_to_name = {i: name for i, name in enumerate(node_names)}
    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]
        print(f"{gene} → Node {node_idx} | {mode.capitalize()} score: {gene_score:.4f}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

        # Get relevance scores for neighbors (filtering by score > 0.1)
        neighbor_scores_dict = {
            i: relevance_scores[i].sum().item()
            for i in neighbor_indices
            if relevance_scores[i].sum().item() > 0.1
        }

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors found for {gene}.")
            continue

        # Sort and select top 10
        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        plot_path = os.path.join(
            "results/gene_prediction/neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_{mode}_neighbor_relevance_epo{args.num_epochs}.png"
        )

        plot_neighbor_relevance(
            neighbor_scores=top_neighbors,
            gene_name=f"{gene} (Cluster {gene_cluster})",
            node_id_to_name=node_id_to_name,
            output_path=plot_path,
            cluster_labels=cluster_labels,
            total_clusters=total_clusters,
            add_legend=False
        )

def plot_topo_clusterwise_feature_contributions(
    args,
    relevance_scores,           # 2D array (samples x features)
    cluster_labels,             # 1D array of cluster assignments
    feature_names,              # List of feature names (e.g., TOPO: BRCA, ...)
    per_cluster_feature_contributions_output_dir):  # Output folder
    ##omics_colors                # Dict of omics type colors (e.g., 'topo': '#1F77B4')):
    os.makedirs(per_cluster_feature_contributions_output_dir, exist_ok=True)

    '''def get_omics_color(feature_name):
        prefix = feature_name.split(":")[0].lower()
        return omics_colors.get(prefix, "#AAAAAA")'''

    unique_clusters = np.unique(cluster_labels)

    for cluster_id in sorted(unique_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_scores = relevance_scores[indices]
        avg_contribution = np.mean(cluster_scores, axis=0)
        total_score = np.sum(avg_contribution)

        fig, ax = plt.subplots(figsize=(10, 2.5))

        x = np.linspace(0, 1, len(feature_names))
        bar_width = 1 / len(feature_names) * 0.95

        bars = ax.bar(
            x,
            avg_contribution,
            width=bar_width,
            ##color=[get_omics_color(name) for name in feature_names],
            align='center'
        )

        ax.set_title(
            fr"Cluster {cluster_id} $\mathregular{{({len(indices)}\ genes,\ avg = {total_score:.2f})}}$",
            fontsize=14
        )

        clean_labels = [name.split(":")[1].strip() if ":" in name else name for name in feature_names]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels, rotation=90)

        '''for label, feature_name in zip(ax.get_xticklabels(), feature_names):
            label.set_color(get_omics_color(feature_name))'''

        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlim(-bar_width, 1 + bar_width)

        plt.tight_layout()
        save_path = os.path.join(
            per_cluster_feature_contributions_output_dir,
            f"{args.model_type}_{args.net_type}_TOPO_cluster_{cluster_id}_feature_contributions_epo{args.num_epochs}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved TOPO feature contribution barplot for Cluster {cluster_id} to {save_path}")

def plot_feature_importance_topo(
    relevance_vector,
    feature_names,
    node_name=None,
    output_path="plots"):


    if len(relevance_vector) != len(feature_names):
        raise ValueError("Length mismatch between relevance vector and feature names.")

    ##pretty_labels = [f"emb_{i}" for i in range(len(relevance_vector))]
    pretty_labels = [f"{i}" for i in range(len(relevance_vector))]

    df = pd.DataFrame({
        "feature": pretty_labels,
        "relevance": relevance_vector
    })

    plt.figure(figsize=(24, 5))
    sns.set_style("white")
    bars = plt.bar(df["feature"], df["relevance"], color="#607D8B")  # bluish-gray

    # Title (no mean)
    if node_name:
        plt.title(f"{node_name}", fontsize=16)

    # Axis labels and formatting
    plt.xlabel("Topology Embedding Dimension", fontsize=14)
    plt.ylabel("Relevance", fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.margins(x=0)

    num_bars = len(df)
    margin = 0.75
    ax = plt.gca()
    ax.set_xlim(-margin, num_bars - 1 + margin)

    ##ax.set_xlim(-bar_width, 1 + bar_width) 

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved TOPO feature importance plot to {output_path}")

def plot_bio_topo_saliency(bio_scores, topo_scores, title="", save_path=None):
    """
    Plot saliency relevance for bio and topo features with mean values in a color-patched legend.

    Parameters:
        bio_scores (np.ndarray): Relevance scores for bio features (length 1024).
        topo_scores (np.ndarray): Relevance scores for topo features (length 1024).
        title (str): Plot title.
        save_path (str, optional): Path to save the figure.
    """

    # 🔹 Normalize
    bio_scores_norm = (bio_scores - bio_scores.min()) / (bio_scores.max() - bio_scores.min() + 1e-8)
    topo_scores_norm = (topo_scores - topo_scores.min()) / (topo_scores.max() - topo_scores.min() + 1e-8)
    x = np.arange(0, 1024)

    # 🔹 Means
    mean_bio = bio_scores_norm.mean()
    mean_topo = topo_scores_norm.mean()

    # 🔹 Plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(x, bio_scores_norm, alpha=0.4, color="royalblue")
    plt.fill_between(x, topo_scores_norm, alpha=0.4, color="darkorange")

    # 🔹 Mean lines
    plt.axhline(mean_bio, color="royalblue", linestyle="--", linewidth=1.5)
    plt.axhline(mean_topo, color="darkorange", linestyle="--", linewidth=1.5)

    # 🔹 Custom legend with mean values
    legend_handles = [
        Patch(facecolor='royalblue', label=f'Bio Mean: {mean_bio:.2f}'),
        Patch(facecolor='darkorange', label=f'Topo Mean: {mean_topo:.2f}')
    ]
    plt.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=10)

    # 🔹 Formatting
    plt.xlim(0, 1023)
    plt.ylim(0, 1)
    plt.xlabel("Feature Index (0–1023)", fontsize=12)
    plt.ylabel("Normalized Relevance Score", fontsize=12)
    plt.title(title, fontsize=14)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved bio-topo saliency plot to {save_path}")
    else:
        plt.show()

def plot_bio_topo_saliency_cuberoot(bio_scores, topo_scores, title="", save_path=None):
    """
    Plot saliency relevance for bio and topo features using cube root transformed normalized values.
    This enhances low scores and compresses high spikes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import seaborn as sns

    # 🔹 Normalize to [0, 1]
    bio_scores_norm = (bio_scores - bio_scores.min()) / (bio_scores.max() - bio_scores.min() + 1e-8)
    topo_scores_norm = (topo_scores - topo_scores.min()) / (topo_scores.max() - topo_scores.min() + 1e-8)

    # 🔹 Apply cube root transform
    bio_scores_scaled = np.cbrt(bio_scores_norm)
    topo_scores_scaled = np.cbrt(topo_scores_norm)

    x = np.arange(0, len(bio_scores))

    # 🔹 Means (after transform)
    mean_bio = bio_scores_scaled.mean()
    mean_topo = topo_scores_scaled.mean()

    # 🔹 Plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(x, bio_scores_scaled, alpha=0.4, color="royalblue")
    plt.fill_between(x, topo_scores_scaled, alpha=0.4, color="darkorange")

    # 🔹 Mean lines
    plt.axhline(mean_bio, color="royalblue", linestyle="--", linewidth=1.5)
    plt.axhline(mean_topo, color="darkorange", linestyle="--", linewidth=1.5)

    # 🔹 Legend
    legend_handles = [
        Patch(facecolor='royalblue', label=f'Bio Mean: {mean_bio:.2f}'),
        Patch(facecolor='darkorange', label=f'Topo Mean: {mean_topo:.2f}')
    ]
    plt.legend(handles=legend_handles, loc='upper left', frameon=False, fontsize=10)

    # 🔹 Formatting
    plt.xlim(0, len(bio_scores) - 1)
    plt.ylim(0, 1)
    plt.xlabel("Feature Index (0–1023)", fontsize=12)
    plt.ylabel("Cube Root Transformed Score", fontsize=12)
    plt.title(title, fontsize=14)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved cube root bio-topo saliency plot to {save_path}")
    else:
        plt.show()

def plot_bio_topo_saliency_log(bio_scores, topo_scores, title="", save_path=None):
    """
    Plot saliency relevance for bio and topo features with log-transformed normalized values.
    Enhances small values and compresses high peaks.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import seaborn as sns

    # 🔹 Normalize
    bio_scores_norm = (bio_scores - bio_scores.min()) / (bio_scores.max() - bio_scores.min() + 1e-8)
    topo_scores_norm = (topo_scores - topo_scores.min()) / (topo_scores.max() - topo_scores.min() + 1e-8)

    # 🔹 Apply log transform
    bio_scores_scaled = np.log1p(bio_scores_norm) / np.log1p(1)  # log1p(x)/log1p(1) keeps range in [0,1]
    topo_scores_scaled = np.log1p(topo_scores_norm) / np.log1p(1)

    x = np.arange(0, len(bio_scores))

    # 🔹 Means (after transform)
    mean_bio = bio_scores_scaled.mean()
    mean_topo = topo_scores_scaled.mean()

    # 🔹 Plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(x, bio_scores_scaled, alpha=0.4, color="royalblue")
    plt.fill_between(x, topo_scores_scaled, alpha=0.4, color="darkorange")

    # 🔹 Mean lines
    plt.axhline(mean_bio, color="royalblue", linestyle="--", linewidth=1.5)
    plt.axhline(mean_topo, color="darkorange", linestyle="--", linewidth=1.5)

    # 🔹 Legend
    legend_handles = [
        Patch(facecolor='royalblue', label=f'Bio Mean: {mean_bio:.2f}'),
        Patch(facecolor='darkorange', label=f'Topo Mean: {mean_topo:.2f}')
    ]
    plt.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=10)

    # 🔹 Formatting
    plt.xlim(0, len(bio_scores) - 1)
    plt.ylim(0, 1)
    plt.xlabel("Feature Index (0–1023)", fontsize=12)
    plt.ylabel("Transformed Relevance Score (log1p)", fontsize=12)
    plt.title(title, fontsize=14)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved log-transformed bio-topo saliency plot to {save_path}")
    else:
        plt.show()

def extract_bio_summary_features_np(features_np):
    num_nodes = features_np.shape[0]
    total_dim = features_np.shape[1]
    summary_features = []

    num_omics = 4
    num_cancers = 16
    features_per_pair = 16

    for o_idx in range(num_omics):
        for c_idx in range(num_cancers):
            base = o_idx * num_cancers * features_per_pair + c_idx * features_per_pair
            if base + features_per_pair > total_dim:
                continue
            group = features_np[:, base:base + features_per_pair]
            max_vals = group.max(axis=1, keepdims=True)
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)  # ➜ (N, 64)

def extract_topo_summary_features_np(features_np):
    num_nodes = features_np.shape[0]
    total_dim = features_np.shape[1]
    summary_features = []

    num_topo_blocks = 4
    num_cancers = 16
    features_per_block = 16

    for t_idx in range(num_topo_blocks):
        for c_idx in range(num_cancers):
            base = t_idx * num_cancers * features_per_block + c_idx * features_per_block
            if base + features_per_block > total_dim:
                continue
            group = features_np[:, base:base + features_per_block]
            max_vals = group.max(axis=1, keepdims=True)
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)  # ➜ (N, 64)

def extract_summary_features_np(features_np):
    """
    Extracts summary features by computing the max of 16-dimensional segments across all (omics, cancer) pairs.
    This version only works with the first 1024 biological features.

    Args:
        features_np (np.ndarray): shape [num_nodes, 2048]

    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    num_nodes, num_features = features_np.shape
    summary_features = []

    assert num_features == 2048, f"Expected 2048 features, got {num_features}"

    # First 1024 for biological features (omics and cancer)
    for o_idx in range(4):  # 4 omics types
        for c_idx in range(16):  # 16 cancer types
            base = o_idx * 16 * 16 + c_idx * 16
            group = features_np[:, base:base + 16]  # [num_nodes, 16]
            max_vals = group.max(axis=1, keepdims=True)  # [num_nodes, 1]
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)  # shape: [num_nodes, 64]

def plot_neighbor_relevance_by_mode(
    gene,
    relevance_scores,
    mode,
    neighbor_scores,
    neighbors_dict,
    name_to_index,
    node_id_to_name,
    graph,
    cluster_labels,
    total_clusters,
    args,
    save_dir="results/gene_prediction/neighbor_feature_contributions/"):
    os.makedirs(save_dir, exist_ok=True)

    if gene not in name_to_index:
        print(f"⚠️ Gene {gene} not found in the graph.")
        return

    node_idx = name_to_index[gene]
    gene_score = neighbor_scores[node_idx]

    # ✅ Get gene cluster based on mode
    if mode == "bio":
        gene_cluster = graph.ndata["cluster_bio"][node_idx].item()
    elif mode == "topo":
        gene_cluster = graph.ndata["cluster_topo"][node_idx].item()
    else:
        gene_cluster = -1  # fallback if cluster type is unknown

    print(f"[{mode}] {gene} → Node {node_idx} | Predicted score: {gene_score:.4f} | Cluster: {gene_cluster}")

    neighbors = neighbors_dict.get(gene, [])
    neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

    relevance_vals = [relevance_scores[i].sum().item() for i in neighbor_indices]
    scores_dict = dict(zip(neighbor_indices, relevance_vals))

    output_path = os.path.join(
        save_dir,
        f"{args.model_type}_{args.net_type}_{gene}_{mode}_neighbor_relevance_epo{args.num_epochs}.png"
    )

    plot_neighbor_relevance(
        neighbor_scores=scores_dict,
        gene_name=f"{gene} (Cluster {gene_cluster})",
        node_id_to_name=node_id_to_name,
        output_path=output_path,
        cluster_labels=cluster_labels,
        total_clusters=total_clusters,
        add_legend=False
    )

def plot_saliency_for_gene(
    gene,
    relevance_scores,
    node_idx,
    save_dir,
    args,
    bio_feat_names,
    topo_feat_names):
    bio_1024 = relevance_scores[node_idx]["bio"].cpu().numpy().reshape(1, -1)
    topo_1024 = relevance_scores[node_idx]["topo"].cpu().numpy().reshape(1, -1)

    bio_64 = extract_summary_features_np_skip(bio_1024).squeeze()
    topo_64 = extract_summary_features_np_skip(topo_1024).squeeze()

    # BIO plot
    plot_feature_importance_bio(
        relevance_vector=bio_64,
        feature_names=bio_feat_names,
        node_name=gene,
        output_path=os.path.join(
            save_dir,
            f"{args.model_type}_{args.net_type}_{gene}_bio_feature_importance_epo{args.num_epochs}.png"
        )
    )

    # TOPO plot
    plot_feature_importance_topo(
        relevance_vector=topo_64,
        feature_names=topo_feat_names,
        node_name=gene,
        output_path=os.path.join(
            save_dir,
            f"{args.model_type}_{args.net_type}_{gene}_topo_feature_importance_epo{args.num_epochs}.png"
        )
    )

def plot_feature_importance_bio(relevance_vector, feature_names, node_name=None, output_path="plots"):
    """
    Plot biological feature importance in OMICS:CANCER format using alphabetical order of cancer names.

    Parameters:
        relevance_vector (array-like): Relevance scores.
        feature_names (list of str): Feature names in the format Cancer_Omics.
        node_name (str, optional): Name of the node (used for title).
        output_path (str): Path to save the plot.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Define mapping from TCGA codes to cancer names
    cancer_map = {
        'BLCA': 'Bladder', 'BRCA': 'Breast', 'CESC': 'Cervix', 'COAD': 'Colon',
        'ESCA': 'Esophagus', 'HNSC': 'HeadNeck', 'KIRC': 'Kidney', 'KIRP': 'KidneyPap',
        'LIHC': 'Liver', 'LUAD': 'LungAd', 'LUSC': 'LungSc', 'PRAD': 'Prostate',
        'READ': 'Rectum', 'STAD': 'Stomach', 'THCA': 'Thyroid', 'UCEC': 'Uterus'
    }

    # Sort cancer codes by alphabetical order of their full names
    sorted_cancer_codes = sorted(cancer_map.keys(), key=lambda k: cancer_map[k])
    omics_order = ['cna', 'ge', 'meth', 'mf']
    column_labels = [f"{omics.upper()}:{cancer}" for omics in omics_order for cancer in sorted_cancer_codes]

    # Validate input
    if len(relevance_vector) != len(feature_names):
        raise ValueError(f"Mismatch: {len(relevance_vector)} values vs {len(feature_names)} names")

    # Build DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })

    df["omics_type"] = df["feature"].apply(lambda x: x.split("_")[1].lower())
    df["cancer_type"] = df["feature"].apply(lambda x: x.split("_")[0])
    df["formatted_label"] = df.apply(lambda row: f"{row['omics_type'].upper()}:{row['cancer_type']}", axis=1)

    # Reorder using column_labels
    df["order"] = df["formatted_label"].apply(lambda x: column_labels.index(x) if x in column_labels else -1)
    df = df[df["order"] != -1].sort_values("order")

    # Color mapping
    omics_color = {
        'cna': '#1F77B4',
        'ge': '#9467BD',
        'meth': '#2CA02C',
        'mf': '#D62728'
    }
    df["bar_color"] = df["omics_type"].map(omics_color)

    # Plotting
    plt.figure(figsize=(24, 5))
    ax = sns.barplot(x="formatted_label", y="relevance", data=df, palette=df["bar_color"].tolist())

    num_bars = len(df)
    margin = 0.75
    ax.set_xlim(-margin, num_bars - 1 + margin)

    ax.set_title(node_name if node_name else "", fontsize=16)
    ax.set_ylabel("Saliency score", fontsize=14)
    ax.set_xlabel("Feature (Omics: Cancer)", fontsize=14)

    # Color tick labels
    for tick, omics in zip(ax.get_xticklabels(), df["omics_type"]):
        tick.set_color(omics_color.get(omics, "black"))

    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved BIO plot to {output_path}")

def plot_clusterwise_sorted_heatmaps(
    args,
    relevance_scores,          # [N, F] relevance scores for top genes
    cluster_labels,            # [N] cluster assignments for top genes
    gene_names,                # List[str], top-k gene names
    omics_splits,              # Dict[str, Tuple[int, int]], omics feature ranges
    output_dir,                # str, directory to save the heatmaps
    model_type,                # str, for filename
    net_type,                  # str, for filename
    epoch,                     # int, for filename
    tag="bio"                  # 'bio' or 'topo'
):
    os.makedirs(output_dir, exist_ok=True)
    num_clusters = len(set(cluster_labels))

    for cluster_id in range(num_clusters):
        # === Subset ===
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_scores = relevance_scores[cluster_indices]  # [n_genes_cluster, F]
        cluster_gene_names = [gene_names[i] for i in cluster_indices]

        # === Sort genes (rows) by total relevance ===
        gene_totals = cluster_scores.sum(axis=1)
        sorted_gene_idx = np.argsort(-gene_totals)
        cluster_scores_sorted = cluster_scores[sorted_gene_idx]
        sorted_gene_names = [cluster_gene_names[i] for i in sorted_gene_idx]

        # === Sort columns within each omics group ===
        sorted_columns = []
        for _, (start, end) in omics_splits.items():
            group_scores = cluster_scores_sorted[:, start:end]
            col_totals = group_scores.sum(axis=0)
            sorted_group_columns = np.argsort(-col_totals) + start
            sorted_columns.extend(sorted_group_columns)

        # === Apply column sorting ===
        final_scores = cluster_scores_sorted[:, sorted_columns]

        # === Plot ===
        plt.figure(figsize=(12, max(4, 0.25 * len(sorted_gene_names))))
        sns.heatmap(
            final_scores,
            cmap='YlGnBu',
            yticklabels=sorted_gene_names,
            xticklabels=False  # You can set True and relabel if you have feature names
        )
        plt.title(f"Cluster {cluster_id} ({tag}) - Relevance Heatmap")
        plt.xlabel("Sorted Features")
        plt.ylabel("Genes")

        save_path = os.path.join(
            output_dir,
            f"{model_type}_{net_type}_cluster_{cluster_id}_{tag}_sorted_heatmap_epo{epoch}.png"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

def get_cluster_color(cluster_id, total_clusters=10):
    """
    Get a color from a colormap based on the cluster ID.
    """
    cmap = cm.get_cmap('tab10') if total_clusters <= 10 else cm.get_cmap('tab20')
    return cmap(cluster_id % total_clusters)

def plot_neighbor_relevance(
    neighbor_scores,
    gene_name,
    node_id_to_name,
    output_path,
    cluster_labels=None,
    total_clusters=12,
    add_legend=False):
    """
    Plots the relevance score of neighbors for a confirmed gene, with optional coloring by cluster.
    Filters neighbors with relevance > 0.05, selects top 10, and normalizes scores.
    """
    # Filter and get top-10
    filtered = {k: v for k, v in neighbor_scores.items() if v > 0.05}
    if len(filtered) < 10:
        print(f"⚠️ Less than 10 neighbors > 0.05 for {gene_name}")
        return

    top_neighbors = dict(sorted(filtered.items(), key=lambda x: -x[1])[:10])
    neighbor_ids = list(top_neighbors.keys())
    neighbor_names = [node_id_to_name.get(nid, f"Node {nid}") for nid in neighbor_ids]
    raw_scores = list(top_neighbors.values())

    # Normalize scores to [0.025, 0.975]
    norm_scores = (np.array(raw_scores) - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores) + 1e-8)
    norm_scores = norm_scores * 0.95 + 0.025

    # Assign bar colors based on clusters
    colors = []
    cluster_ids = []
    for nid in neighbor_ids:
        if cluster_labels is not None:
            cluster_id = int(cluster_labels[nid])
            cluster_ids.append(cluster_id)
            colors.append(CLUSTER_COLORS.get(cluster_id % total_clusters, "gray"))
        else:
            colors.append("gray")

    # Plot
    plt.figure(figsize=(2.2, 2.2))
    sns.set_style("white")
    ax = sns.barplot(x=neighbor_names, y=norm_scores, palette=colors)

    plt.title(f"{gene_name}", fontsize=12)
    plt.ylabel("Relevance score", fontsize=10)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)

    sns.despine(left=False, bottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)

    if add_legend and cluster_labels is not None:
        unique_clusters = sorted(set(cluster_ids))
        legend_handles = [
            mpatches.Patch(color=CLUSTER_COLORS.get(cid % total_clusters, "gray"), label=f"Cluster {cid}")
            for cid in unique_clusters
        ]
        plt.legend(handles=legend_handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend().remove()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved neighbor relevance plot to {output_path}")

def plot_pcg_cancer_genes(
    clusters,
    predicted_cancer_genes_count,
    total_genes_per_cluster,
    node_names,
    cluster_labels,
    output_path):
    """
    Plots the percentage of predicted cancer genes per cluster.
    """

    # Convert to sorted array
    clusters = np.array(sorted(total_genes_per_cluster.keys()))
    total_genes_array = np.array([total_genes_per_cluster[c] for c in clusters])
    predicted_counts = np.array([predicted_cancer_genes_count.get(c, 0) for c in clusters])

    # Compute percentages
    percent_predicted = np.divide(predicted_counts, total_genes_array, where=total_genes_array > 0)

    # Prepare bar colors
    colors = [CLUSTER_COLORS.get(c, '#333333') for c in clusters]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(clusters, percent_predicted, color=colors, edgecolor='black')

    # Annotate with raw count
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        count = predicted_cancer_genes_count.get(cluster_id, 0)
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(count),
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Left margin space
    num_clusters = len(clusters)
    ax.set_xlim(-0.55, num_clusters - 0.65)

    # Labels and formatting
    ax.set_ylabel("Percent of PCGs", fontsize=20)
    plt.xlabel("")  # No xlabel here
    plt.xticks(clusters, fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_ylim(0, max(percent_predicted) + 0.1)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Plot saved to {output_path}")

def plot_kcg_cancer_genes(clusters, kcg_count, total_genes_per_cluster, node_names, cluster_labels, output_path):

    cluster_ids = sorted(clusters)
    total = [total_genes_per_cluster[c] for c in cluster_ids]
    kcgs = [kcg_count.get(c, 0) for c in cluster_ids]
    proportions = [k / t if t > 0 else 0 for k, t in zip(kcgs, total)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(cluster_ids, proportions, 
                   color=[CLUSTER_COLORS.get(c, '#333333') for c in cluster_ids],
                   edgecolor='black')

    # Annotate each bar with the raw KCG count
    for bar, cluster_id in zip(bars, cluster_ids):
        height = bar.get_height()
        count = kcg_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), 
                 ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax = plt.gca()
    num_clusters = len(cluster_ids)
    ax.set_xlim(-0.55, num_clusters - 0.65)

    # Formatting
    ##plt.xlabel("Cluster ID", fontsize=16)
    plt.ylabel("Percent of KCGs", fontsize=20)
    plt.xlabel("")
    plt.xticks(cluster_ids, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, max(proportions) + 0.1)

    sns.despine()  # 🔻 Remove top/right spines

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_interactions_with_pcgs(data, output_path):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with predicted cancer genes (PCGs).
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    sns.set_style("white")

    unique_clusters = sorted(data['Cluster'].unique())
    cluster_color_map = {cluster_id: CLUSTER_COLORS[cluster_id] for cluster_id in unique_clusters}

    ax = sns.boxplot(
        x='Cluster', y='Interactions', data=data, 
        hue='Cluster', palette=cluster_color_map, showfliers=False
    )

    sns.stripplot(
        x='Cluster', y='Interactions', data=data, 
        color='black', alpha=0.2, jitter=True, size=1.5
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # 👉 Adjust x-axis limits for more space on the left
    num_clusters = len(unique_clusters)
    ax.set_xlim(-0.55, num_clusters - 0.65)  # Or adjust -0.4, -0.5, etc. for more space

    plt.ylabel("Number of interactions with PCGs", fontsize=20)
    plt.xlabel("")
    plt.xticks(rotation=0, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 50)

    sns.despine()  # 🔻 Remove top/right spines

    plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.close()

    print(f"✅ Plot saved to {output_path}")

def plot_interactions_with_kcgs(data, output_path):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    sns.set_style("white")

    unique_clusters = sorted(data['Cluster'].unique())
    cluster_color_map = {cluster_id: CLUSTER_COLORS[cluster_id] for cluster_id in unique_clusters}

    ax = sns.boxplot(
        x='Cluster', y='Interactions', data=data, 
        hue='Cluster', palette=cluster_color_map, showfliers=False
    )

    sns.stripplot(
        x='Cluster', y='Interactions', data=data, 
        color='black', alpha=0.2, jitter=True, size=1.5
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # 👉 Adjust x-axis limits for more space on the left
    num_clusters = len(unique_clusters)
    ax.set_xlim(-0.55, num_clusters - 0.65)  # Or adjust -0.4, -0.5, etc. for more space

    plt.ylabel("Number of interactions with KCGs", fontsize=20)
    plt.xlabel("")
    plt.xticks(rotation=0, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 50)

    sns.despine()  # 🔻 Remove top/right spines

    plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.close()

    print(f"✅ Plot saved to {output_path}")

def plot_enriched_term_counts(enrichment_results, output_path, model_type, net_type, num_epochs, bio_color='#1f77b4', topo_color='#ff7f0e'):
    """
    Plot bar chart of the number of enriched terms per cluster for bio and topo clusters,
    with x-axis labels colored according to their type.

    Parameters:
        enrichment_results (dict): Dictionary with enrichment results for 'bio' and 'topo' clusters.
        output_path (str): Path to save the output plot.
        model_type (str): Model type for naming the file.
        net_type (str): Network type for naming the file.
        num_epochs (int): Number of epochs for naming the file.
        bio_color (str): Color for bio bars and labels.
        topo_color (str): Color for topo bars and labels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = []
    xtick_colors = []
    xtick_labels = []

    for cluster_type in ['bio', 'topo']:
        cluster_ids = list(enrichment_results[cluster_type].keys())
        term_counts = [len(res) for res in enrichment_results[cluster_type].values()]
        labels = [f"{cluster_type.capitalize()}_{i}" for i in cluster_ids]

        color = bio_color if cluster_type == 'bio' else topo_color
        bar_container = ax.bar(labels, term_counts, label=cluster_type, color=color)
        bars.extend(bar_container)

        xtick_labels.extend(labels)
        xtick_colors.extend([color] * len(labels))

    # Adjust x-axis limits
    ax.set_xlim(-0.65, len(bars) - 0.5)

    # Labeling
    ax.set_ylabel("Number of enriched terms", fontsize=28)
    ##ax.set_title("Functional Coherence: Enriched Term Counts per Cluster", fontsize=28)

    # Set custom x-tick labels and colors
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=20)
    for tick_label, color in zip(ax.get_xticklabels(), xtick_colors):
        tick_label.set_color(color)

    # Set y-tick font size
    ax.tick_params(axis='y', labelsize=20)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ##ax.legend(frameon=False, fontsize=24)

    plt.tight_layout()
    filename = f"{model_type}_{net_type}_term_counts_barplot_epo{num_epochs}.png"
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.close()

    print(f"Bar plot saved to {os.path.join(output_path, filename)}")

def plot_shared_enriched_pathways_venn(enrichment_results, output_path, model_type, net_type, num_epochs, bio_color='#1f77b4', topo_color='#ff7f0e'):
    """
    Plots a Venn diagram showing the overlap of enriched pathways between bio and topo clusters.

    Parameters:
        enrichment_results (dict): Dictionary with enrichment DataFrames under 'bio' and 'topo'.
        output_path (str): Path to save the output plot.
        model_type (str): Model type for naming the file.
        net_type (str): Network type for naming the file.
        num_epochs (int): Number of epochs for naming the file.
        bio_color (str): Color for the bio set in the Venn diagram.
        topo_color (str): Color for the topo set in the Venn diagram.
    """
    bio_terms = set(
        sum([df['name'].tolist() for df in enrichment_results['bio'].values() if not df.empty], [])
    )
    topo_terms = set(
        sum([df['name'].tolist() for df in enrichment_results['topo'].values() if not df.empty], [])
    )

    plt.figure(figsize=(8, 8))
    venn = venn2(
        [bio_terms, topo_terms],
        set_labels=('Bio', 'Topo'),
        set_colors=(bio_color, topo_color),
        alpha=0.7
    )

    # Set font size for all Venn labels and subset counts
    for text in venn.set_labels:
        if text:
            text.set_fontsize(28)
    for text in venn.subset_labels:
        if text:
            text.set_fontsize(28)

    plt.title("Overlap of enriched pathways", fontsize=16)

    filename = f"{model_type}_{net_type}_shared_pathways_venn_epo{num_epochs}.png"
    venn_path = os.path.join(output_path, filename)
    plt.savefig(venn_path, dpi=300)
    plt.close()

    print(f"Venn diagram saved to {venn_path}")

def plot_contingency_matrix(cluster_labels_bio_topk, cluster_labels_topo_topk, ari_score, nmi_score, output_dir, args):
    """
    Plots a contingency matrix comparing bio and topo cluster labels.

    Parameters:
    - cluster_labels_bio_topk: Cluster labels from bio features.
    - cluster_labels_topo_topk: Cluster labels from topo features.
    - ari_score: Adjusted Rand Index between the clusterings.
    - nmi_score: Normalized Mutual Information score.
    - output_dir: Directory to save the plot.
    - args: Arguments containing model type, net type, and epoch count.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    # === Contingency matrix ===
    contingency = confusion_matrix(cluster_labels_bio_topk, cluster_labels_topo_topk)

    # Plot heatmap of confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='BuPu', cbar=False, annot_kws={"fontsize": 20})

    plt.title(f"Contingency Matrix: Bio vs Topo\n(ARI={ari_score:.2f}, NMI={nmi_score:.2f})", fontsize=30)
    plt.xlabel("Topo clusters", fontsize=28)
    plt.ylabel("Bio clusters", fontsize=28)

    plt.xticks([])
    plt.yticks([])

    # Save plot
    contingency_plot_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_contingency_matrix_epo{args.num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(contingency_plot_path, dpi=300)
    plt.close()

    print(f"✅ Contingency matrix saved to {contingency_plot_path}")

def plot_omics_barplot_bio(df, output_path=None):
    """
    Plot omics relevance for biological features with format like 'MF:BRCA'.
    """
    omics_order = ['cna', 'ge', 'meth', 'mf']
    omics_colors = {
        'cna': '#9370DB',    # purple
        'ge': '#228B22',      # dark green
        'meth': '#00008B',   # dark blue
        'mf': '#b22222',     # dark red
    }

    # Extract 'Omics' and 'Cancer' from features like 'MF:BRCA'
    df[['Omics', 'Cancer']] = df['Feature'].str.split(':', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    _plot_bar(omics_relevance, omics_colors, omics_order, output_path)

def plot_omics_barplot_topo(df, output_path=None):
    """
    Plot omics relevance for topological features with format like 'BRCA_mf'.
    """
    omics_order = ['cna', 'ge', 'meth', 'mf']
    omics_colors = {
        'cna': '#9370DB',    # purple
        'ge': '#228B22',      # dark green
        'meth': '#00008B',   # dark blue
        'mf': '#b22222',     # dark red
    }

    # Extract 'Cancer' and 'Omics' from features like 'BRCA_mf'
    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    _plot_bar(omics_relevance, omics_colors, omics_order, output_path)

def plot_gene_feature_contributions_topo(gene_name, relevance_vector, feature_names, score, output_path=None):
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    # Barplot of all 64 topo features
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    barplot_path = output_path.replace(".png", "_omics_barplot.png") if output_path else None
    plot_omics_barplot_topo(df, barplot_path)

    # Prepare for heatmap
    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    heatmap_data = df.pivot(index='Cancer', columns='Omics', values='Relevance')
    heatmap_data = heatmap_data[['cna', 'ge', 'meth', 'mf']]  # Ensure column order

    # Plot vertical heatmap (Cancers as rows)
    plt.figure(figsize=(2.0, 5.0))
    # Capitalize omics column labels
    heatmap_data.columns = [col.upper() for col in heatmap_data.columns]
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar=False, linewidths=0.3, linecolor='gray')

    if isinstance(score, np.ndarray):
        score = score.item()
    plt.title(f"{gene_name}", fontsize=12)

    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.xlabel('')
    plt.ylabel('')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_and_plot_confirmed_genes_topo(
    args,
    node_names_topk,
    node_scores_topk,
    summary_feature_relevance,
    output_dir,
    confirmed_genes_save_path,
    cluster_labels_topk,
    tag="topo",
    confirmed_gene_path="data/ncg_8886.txt"):
    """
    Finds confirmed cancer genes and plots their topological feature contributions.
    """

    cancer_names = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC']
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{cancer}_{omics}" for cancer in cancer_names for omics in omics_order]

    with open(confirmed_gene_path) as f:
        known_cancer_genes = set(line.strip() for line in f if line.strip())

    confirmed_genes = [g for g in node_names_topk if g in known_cancer_genes]

    with open(confirmed_genes_save_path, "w") as f:
        for gene in confirmed_genes:
            f.write(f"{gene}\n")

    ##summary_feature_relevance = extract_summary_features_np_topo(summary_feature_relevance)

    plot_dir = os.path.join(output_dir, f"{tag}_confirmed_feature_contributions")
    os.makedirs(plot_dir, exist_ok=True)

    def get_scalar_score(score):
        if isinstance(score, np.ndarray):
            return score.item() if score.size == 1 else score[0]
        return float(score)

    for gene_name in confirmed_genes:
        idx = node_names_topk.index(gene_name)
        relevance_vector = summary_feature_relevance[idx]
        score = get_scalar_score(node_scores_topk[idx])
        plot_path = os.path.join(
            plot_dir,
            f"{args.model_type}_{args.net_type}_{gene_name}_{tag}_confirmed_feature_contributions_epo{args.num_epochs}.png"
        )
        cluster_id = cluster_labels_topk[idx].item()
        plot_gene_feature_contributions_topo(
            #gene_name=gene_name,
            gene_name=f"{gene_name} (Cluster {cluster_id})",
            relevance_vector=relevance_vector,
            feature_names=feature_names,
            output_path=plot_path,
            score=score
        )

def load_ground_truth_cancer_genes(file_path):
    """
    Load ground truth cancer genes from a file.
    
    Args:
        file_path (str): Path to the ground truth cancer gene file.
    
    Returns:
        set: A set containing ground truth cancer gene names.
    """
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def compute_node_saliency(model, graph, features, node_indices=None, use_abs=True, normalize=True):
    """
    Computes a fast gradient-based LRP approximation for selected nodes.

    Args:
        model: Trained GNN model
        graph: DGLGraph
        features: Node features (Tensor or numpy array)
        node_indices: List/Tensor of node indices to compute relevance for. If None, auto-select (probs > 0.5)
        use_abs: Whether to take absolute value of gradients (recommended)
        normalize: Whether to normalize relevance scores (per node)

    Returns:
        relevance_scores: Tensor of shape [num_nodes, num_features]
    """
    model.eval()

    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    features = features.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        logits = model(graph, features)
        probs = torch.sigmoid(logits.squeeze())

        if node_indices is None:
            node_indices = torch.nonzero(probs > 0.5, as_tuple=False).squeeze()
            if node_indices.ndim == 0:
                node_indices = node_indices.unsqueeze(0)

        relevance_scores = torch.zeros_like(features)

        for i, idx in enumerate(node_indices):
            model.zero_grad()
            if features.grad is not None:
                features.grad.zero_()

            probs[idx].backward(retain_graph=(i != len(node_indices) - 1))

            grads = features.grad[idx]
            relevance = grads.abs() if use_abs else grads

            if normalize:
                norm = relevance.norm(p=1)  # L1 norm
                if norm > 0:
                    relevance = relevance / norm

            relevance_scores[idx] = relevance.detach()

    return relevance_scores

def compute_neighbor_saliency(model, graph, features, node_idx):
    features = features.clone().detach().requires_grad_(True)
    output = model(graph, features)
    score = output[node_idx].max()  # pick the score you want
    model.zero_grad()
    score.backward()
    
    # Find neighbors
    neighbors = graph.successors(node_idx)  # if directed, or .neighbors(node_idx) for undirected

    neighbor_saliencies = {}
    for n in neighbors:
        neighbor_saliencies[n] = features.grad[n].abs().sum().item()
    
    return neighbor_saliencies

def plot_neighbor_saliency_heatmap(
    graph,
    confirmed_genes,
    node_names,
    name_to_index,
    relevance_scores,
    omics_splits,
    output_path="neighbor_saliency_heatmap.png"
    ):
    heatmap_data = []
    heatmap_labels = []

    for gene in confirmed_genes:
        if gene not in name_to_index:
            continue
        idx = name_to_index[gene]
        
        # Get relevance for the confirmed gene
        gene_relevance = relevance_scores[idx].cpu().numpy()
        heatmap_data.append(gene_relevance)
        heatmap_labels.append(f"{gene} (self)")
        
        # Get neighbors
        neighbors = graph.successors(idx).tolist()
        for neighbor_idx in neighbors:
            neighbor_name = node_names[neighbor_idx]
            neighbor_relevance = relevance_scores[neighbor_idx].cpu().numpy()
            heatmap_data.append(neighbor_relevance)
            heatmap_labels.append(f"{neighbor_name} (nbr)")
    
    # Stack into matrix
    heatmap_data = np.vstack(heatmap_data)
    
    # Optional: min-max normalization per row
    heatmap_data_norm = (heatmap_data - heatmap_data.min(axis=1, keepdims=True)) / \
                        (heatmap_data.max(axis=1, keepdims=True) - heatmap_data.min(axis=1, keepdims=True) + 1e-8)
    
    # Limit number of neighbors plotted
    MAX_NEIGHBORS = 100  # or whatever you want

    if heatmap_data_norm.shape[0] > MAX_NEIGHBORS:
        neighbor_importance = np.abs(heatmap_data_norm).mean(axis=1)
        top_indices = np.argsort(-neighbor_importance)[:MAX_NEIGHBORS]
        heatmap_data_norm = heatmap_data_norm[top_indices]
        heatmap_labels = [heatmap_labels[i] for i in top_indices]

    # ✅ NOW set figure size based on the reduced number of labels
    plt.figure(figsize=(14, max(8, len(heatmap_labels) * 0.3)))

    # Create heatmap
    sns.heatmap(heatmap_data_norm, cmap="viridis", yticklabels=heatmap_labels, xticklabels=False)
    plt.title("Neighbor Saliency Heatmap")
    plt.xlabel("Omics Feature Dimension")
    plt.ylabel("Confirmed Genes + Neighbors")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Neighbor saliency heatmap saved at {output_path}")

def build_subgraph(graph, target_idx, neighbors):
    G = nx.Graph()
    G.add_node(target_idx)

    for neighbor_idx in neighbors:
        G.add_node(neighbor_idx)
        G.add_edge(target_idx, neighbor_idx)

    return G

def plot_saliency_graph_multiomics(G, target_idx, bio_saliencies, topo_saliencies, node_names, save_path=None):
    pos = nx.spring_layout(G, seed=42)

    edges = list(G.edges())
    bio_scores = np.array([bio_saliencies.get(v, 0) for u, v in edges])
    topo_scores = np.array([topo_saliencies.get(v, 0) for u, v in edges])
    
    total_scores = bio_scores + topo_scores
    total_scores = np.clip(total_scores, 1e-6, None)  # Avoid zero division

    # Calculate color mixing
    bio_ratio = bio_scores / total_scores
    topo_ratio = topo_scores / total_scores

    edge_colors = [(topo_ratio[i], 0, bio_ratio[i]) for i in range(len(edges))]  # RGB: (red, 0, blue)
    edge_widths = total_scores / total_scores.max() * 5  # Max width = 5

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node == target_idx:
            node_colors.append('gold')
        else:
            node_colors.append('lightgray')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)

    # Draw edges
    for (u, v), color, width in zip(edges, edge_colors, edge_widths):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, edge_color=[color])

    # Draw labels
    labels = {node: node_names[node] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    plt.title(f"Neighbor Bio vs Topo Saliency for {node_names[target_idx]}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved multi-omics saliency graph to {save_path}")
    else:
        plt.show()

def plot_dynamic_sankey_topo_clusterlevel_each_gene(
    args,
    graph,
    node_names,
    name_to_index,
    confirmed_genes,
    scores,
    relevance_scores,
    cluster_labels,
    total_clusters,
    ):
    # Mapping node IDs and names
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    # Prepare folder to save
    output_dir = os.path.join("results/gene_prediction/topo_dynamic_sankey_clusterlevel")
    os.makedirs(output_dir, exist_ok=True)

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list.")
            continue

        node_idx = topk_name_to_index[gene]
        gene_score = scores[node_idx]

        # ✅ Get cluster label from graph
        gene_cluster = graph.ndata["cluster_topo"][node_idx].item()
        print(f"{gene} → Node {node_idx} | Topo score: {gene_score:.4f} | Cluster: {gene_cluster}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_scores_dict = {}

        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:  # bounds check
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores_dict[rel_idx] = rel_score

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors found for {gene}.")
            continue

        # Top neighbors (limit to 10)
        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        # Sankey nodes
        labels = [f"{gene} (Cluster {gene_cluster})"]
        colors = [f"rgba({(gene_cluster/total_clusters)*255},100,150,0.8)"]

        for idx in top_neighbors.keys():
            neighbor_name = node_id_to_name[idx]
            neighbor_cluster = cluster_labels[idx]
            labels.append(f"{neighbor_name} (Cluster {neighbor_cluster})")
            colors.append(f"rgba({(neighbor_cluster/total_clusters)*255},180,100,0.8)")

        # Sankey links
        source = [0] * len(top_neighbors)  # source = gene
        target = list(range(1, len(top_neighbors)+1))  # targets = neighbors
        value = list(top_neighbors.values())

        # Build Sankey figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
            ))])

        fig.update_layout(
            title_text=f"Topo Dynamic Sankey for {gene}",
            font_size=10,
            width=800,
            height=600
        )

        # Save HTML
        output_path = os.path.join(
            output_dir,
            f"{args.model_type}_{args.net_type}_{gene}_topo_confirmed_neighbor_sankey_epo{args.num_epochs}.html"
        )
        fig.write_html(output_path)
        fig.write_image(output_path.replace('.html', '.png'))

        print(f"✅ Sankey saved: {output_path}")

def plot_dynamic_sankey_bio_clusterlevel_not_fixed_colors(
    args,
    graph,
    node_names,
    name_to_index,
    confirmed_genes,
    scores,
    cluster_labels,
    total_clusters,
    relevance_scores
):
    # Build safe top-k index mapping
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    # Mapping cluster-to-cluster scores
    cluster_to_cluster_score = {}

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list. Skipping.")
            continue

        node_idx = topk_name_to_index[gene]
        gene_cluster = cluster_labels[node_idx].item()

        neighbors = neighbors_dict.get(gene, [])

        neighbor_scores_dict = {}
        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores_dict[rel_idx] = rel_score

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors for {gene}.")
            continue

        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        for rel_idx, rel_score in top_neighbors.items():
            neighbor_cluster = cluster_labels[rel_idx].item()

            key = (gene_cluster, neighbor_cluster)
            cluster_to_cluster_score[key] = cluster_to_cluster_score.get(key, 0) + rel_score

    if not cluster_to_cluster_score:
        print("⚠️ No cluster-to-cluster links to plot.")
        return

    # Now prepare Sankey inputs
    clusters_involved = set()
    for (src_c, tgt_c) in cluster_to_cluster_score.keys():
        clusters_involved.add(src_c)
        clusters_involved.add(tgt_c)
    clusters_involved = sorted(list(clusters_involved))

    cluster_id_to_label = {c: f"C{c}" for c in clusters_involved}
    label_to_index = {f"C{c}": i for i, c in enumerate(clusters_involved)}

    source = []
    target = []
    value = []
    label = [f"C{c}" for c in clusters_involved]
    color = [f"rgba({(c*37)%255}, {(c*83)%255}, {(c*131)%255}, 0.8)" for c in clusters_involved]

    for (src_c, tgt_c), score in cluster_to_cluster_score.items():
        source.append(label_to_index[f"C{src_c}"])
        target.append(label_to_index[f"C{tgt_c}"])
        value.append(score)

    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label,
            color=color
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])

    sankey_fig.update_layout(
        title_text=f"Confirmed Cluster → Neighbor Cluster (Bio) - {args.model_type}_{args.net_type}",
        font_size=10,
        width=1200,
        height=800
    )

    output_dir = "results/gene_prediction/bio_dynamic_sankey_clusterlevel/"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_bio_confirmed_dynamic_sankey_clusterlevel_epo{args.num_epochs}.html"
    )
    sankey_fig.write_html(plot_path)
    print(f"✅ Cluster-level Sankey diagram saved to {plot_path}")

def get_neighbors_gene_names(graph, node_names, name_to_index, genes):
    neighbors_dict = {}
    for gene in genes:
        if gene in name_to_index:
            idx = name_to_index[gene]
            neighbors = graph.successors(idx).tolist()
            neighbor_names = [node_names[n] for n in neighbors]
            neighbors_dict[gene] = neighbor_names
    return neighbors_dict

def plot_dynamic_sankey_bio_clusterlevel(
    args,
    graph,
    node_names,
    name_to_index,
    confirmed_genes,
    scores,
    cluster_labels,
    total_clusters,
    relevance_scores
):
    # Build safe top-k index mapping
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    # Mapping cluster-to-cluster scores
    cluster_to_cluster_score = {}

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list. Skipping.")
            continue

        node_idx = topk_name_to_index[gene]
        gene_cluster = cluster_labels[node_idx].item()

        neighbors = neighbors_dict.get(gene, [])

        neighbor_scores_dict = {}
        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores_dict[rel_idx] = rel_score

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors for {gene}.")
            continue

        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        for rel_idx, rel_score in top_neighbors.items():
            neighbor_cluster = cluster_labels[rel_idx].item()

            key = (gene_cluster, neighbor_cluster)
            cluster_to_cluster_score[key] = cluster_to_cluster_score.get(key, 0) + rel_score

    if not cluster_to_cluster_score:
        print("⚠️ No cluster-to-cluster links to plot.")
        return

    # Now prepare Sankey inputs
    clusters_involved = set()
    for (src_c, tgt_c) in cluster_to_cluster_score.keys():
        clusters_involved.add(src_c)
        clusters_involved.add(tgt_c)
    clusters_involved = sorted(list(clusters_involved))

    cluster_id_to_label = {c: f"C{c}" for c in clusters_involved}
    label_to_index = {f"C{c}": i for i, c in enumerate(clusters_involved)}

    source = []
    target = []
    value = []
    label = [f"C{c}" for c in clusters_involved]
    color = [CLUSTER_COLORS.get(c, "#CCCCCC") for c in clusters_involved]  # 🛠 fixed color

    for (src_c, tgt_c), score in cluster_to_cluster_score.items():
        source.append(label_to_index[f"C{src_c}"])
        target.append(label_to_index[f"C{tgt_c}"])
        value.append(score)

    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label,
            color=color
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])

    sankey_fig.update_layout(
        title_text=f"Confirmed Cluster → Neighbor Cluster (Bio) - {args.model_type}_{args.net_type}",
        font_size=10,
        width=1200,
        height=800
    )

    output_dir = "results/gene_prediction/bio_dynamic_sankey_clusterlevel/"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_bio_confirmed_dynamic_sankey_clusterlevel_epo{args.num_epochs}.html"
    )
    sankey_fig.write_html(plot_path)
    print(f"✅ Cluster-level Sankey diagram saved to {plot_path}")

def plot_dynamic_sankey_topo_clusterlevel(
    args,
    graph,
    node_names,
    name_to_index,
    confirmed_genes,
    scores,
    relevance_scores,
    cluster_labels,
    total_clusters
):
    # Mapping node IDs and names
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    # Mapping cluster-to-cluster scores
    cluster_to_cluster_score = {}

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list. Skipping.")
            continue

        node_idx = topk_name_to_index[gene]
        gene_score = scores[node_idx]

        # ✅ Get topo cluster from graph
        gene_cluster = graph.ndata["cluster_topo"][node_idx].item()

        neighbors = neighbors_dict.get(gene, [])
        neighbor_scores_dict = {}

        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:  # bounds check
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores_dict[rel_idx] = rel_score

        if not neighbor_scores_dict:
            print(f"⚠️ No valid neighbors for {gene}.")
            continue

        # Top neighbors (limit to 10)
        top_neighbors = dict(sorted(neighbor_scores_dict.items(), key=lambda x: -x[1])[:10])

        for rel_idx, rel_score in top_neighbors.items():
            neighbor_cluster = cluster_labels[rel_idx].item()

            key = (gene_cluster, neighbor_cluster)
            cluster_to_cluster_score[key] = cluster_to_cluster_score.get(key, 0) + rel_score

    if not cluster_to_cluster_score:
        print("⚠️ No cluster-to-cluster links to plot.")
        return

    # Prepare Sankey inputs
    clusters_involved = set()
    for (src_c, tgt_c) in cluster_to_cluster_score.keys():
        clusters_involved.add(src_c)
        clusters_involved.add(tgt_c)
    clusters_involved = sorted(list(clusters_involved))

    cluster_id_to_label = {c: f"C{c}" for c in clusters_involved}
    label_to_index = {f"C{c}": i for i, c in enumerate(clusters_involved)}

    source = []
    target = []
    value = []
    label = [f"C{c}" for c in clusters_involved]
    color = [CLUSTER_COLORS.get(c, "#CCCCCC") for c in clusters_involved]  # 🛠 fixed color

    for (src_c, tgt_c), score in cluster_to_cluster_score.items():
        source.append(label_to_index[f"C{src_c}"])
        target.append(label_to_index[f"C{tgt_c}"])
        value.append(score)

    # Build Sankey figure
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label,
            color=color
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
        ))])

    sankey_fig.update_layout(
        title_text=f"Confirmed Cluster → Neighbor Cluster (Topo) - {args.model_type}_{args.net_type}",
        font_size=10,
        width=1200,
        height=800
    )

    # Save
    output_dir = "results/gene_prediction/topo_dynamic_sankey_clusterlevel/"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_topo_confirmed_dynamic_sankey_clusterlevel_epo{args.num_epochs}.html"
    )
    sankey_fig.write_html(plot_path)
    print(f"✅ Cluster-level Sankey diagram saved to {plot_path}")

def plot_multilevel_sankey_bio_clusterlevel(
    args,
    graph,
    node_names,
    name_to_index,
    confirmed_genes,
    scores,
    cluster_labels,
    total_clusters,
    relevance_scores
):
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            print(f"⚠️ Gene {gene} not in top-k node list. Skipping.")
            continue

        node_idx = topk_name_to_index[gene]
        neighbors = neighbors_dict.get(gene, [])
        
        if not neighbors:
            print(f"⚠️ No neighbors found for {gene}.")
            continue

        neighbor_scores = {}
        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores[rel_idx] = rel_score

        if not neighbor_scores:
            print(f"⚠️ No valid neighbor relevance for {gene}.")
            continue

        # Limit to top 10 neighbors
        neighbor_scores = dict(sorted(neighbor_scores.items(), key=lambda x: -x[1])[:10])

        # Now build node list and mapping
        all_labels = []
        all_colors = []
        label_to_idx = {}

        # 1. Confirmed gene node
        confirmed_label = f"{gene}"
        all_labels.append(confirmed_label)
        all_colors.append("gray")
        label_to_idx[confirmed_label] = 0

        # 2. Neighbor nodes
        for neighbor_idx in neighbor_scores.keys():
            neighbor_name = node_id_to_name[neighbor_idx]
            neighbor_label = f"{neighbor_name}"
            if neighbor_label not in label_to_idx:
                label_to_idx[neighbor_label] = len(all_labels)
                all_labels.append(neighbor_label)
                all_colors.append("lightgray")

        # 3. Cluster nodes
        neighbor_clusters = set()
        for neighbor_idx in neighbor_scores.keys():
            neighbor_cluster = cluster_labels[neighbor_idx]
            neighbor_clusters.add(neighbor_cluster)

        for cluster_id in sorted(neighbor_clusters):
            cluster_label = f"Cluster {cluster_id}"
            if cluster_label not in label_to_idx:
                label_to_idx[cluster_label] = len(all_labels)
                all_labels.append(cluster_label)
                all_colors.append(CLUSTER_COLORS.get(cluster_id, "#000000"))

        # Build Sankey source-target-value-linkcolor
        source = []
        target = []
        value = []
        link_colors = []

        # Gene -> Neighbor
        for neighbor_idx, score in neighbor_scores.items():
            neighbor_name = node_id_to_name[neighbor_idx]
            source.append(label_to_idx[confirmed_label])
            target.append(label_to_idx[neighbor_name])
            value.append(score)
            link_colors.append("rgba(128,128,128,0.4)")  # gray links

        # Neighbor -> Cluster
        for neighbor_idx, score in neighbor_scores.items():
            neighbor_name = node_id_to_name[neighbor_idx]
            neighbor_cluster = cluster_labels[neighbor_idx]
            cluster_label = f"Cluster {neighbor_cluster}"

            source.append(label_to_idx[neighbor_name])
            target.append(label_to_idx[cluster_label])
            value.append(score)
            # Link colored by cluster
            cluster_color = CLUSTER_COLORS.get(neighbor_cluster, "#000000")
            link_colors.append(cluster_color.replace("#", "rgba(") + ",0.6)")  # lighter

        # Build figure
        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color=all_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors
            ))])

        fig.update_layout(
            title_text=f"Multi-Level Sankey: {gene} → Neighbors → Clusters (Bio)",
            font_size=10,
            width=1000,
            height=700
        )

        # Save
        output_dir = "results/gene_prediction/bio_multilevel_sankey/"
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(
            output_dir,
            f"{args.model_type}_{args.net_type}_{gene}_bio_multilevel_sankey_epo{args.num_epochs}.html"
        )
        fig.write_html(save_path)
        print(f"✅ Multi-level Sankey saved: {save_path}")

def compute_combined_relevance_scores(
    model,
    graph,
    features,
    node_indices=None,
    use_abs=True,
    normalize=False,
    prob_threshold=0.5
):
    """
    Computes gradient-based relevance (saliency) scores for selected nodes.

    Args:
        model: Trained GNN model
        graph: DGLGraph
        features: Node features (Tensor or numpy array)
        node_indices: List/Tensor of node indices to compute relevance for. If None, select using prob_threshold
        use_abs: Whether to use absolute value of gradients
        normalize: Whether to normalize relevance scores per node (optional)
        prob_threshold: Probability threshold for auto-selecting nodes

    Returns:
        relevance_scores: Tensor of shape [num_nodes, num_features]
    """
    model.eval()

    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    features = features.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        logits = model(graph, features)
        probs = torch.sigmoid(logits.squeeze())

        if node_indices is None:
            node_indices = torch.nonzero(probs > prob_threshold, as_tuple=False).squeeze()
            if node_indices.ndim == 0:
                node_indices = node_indices.unsqueeze(0)

        relevance_scores = torch.zeros_like(features)

        for i, idx in enumerate(node_indices):
            model.zero_grad()
            if features.grad is not None:
                features.grad.zero_()

            probs[idx].backward(retain_graph=(i != len(node_indices) - 1))

            grads = features.grad[idx]
            relevance = grads.abs() if use_abs else grads

            if normalize:
                norm = relevance.norm(p=1)  # L1 norm
                if norm > 0:
                    relevance = relevance / norm

            relevance_scores[idx] = relevance.detach()

    return relevance_scores

def saliency_to_color(saliency, min_saliency=0.0, max_saliency=1.0):
    saliency = np.clip((saliency - min_saliency) / (max_saliency - min_saliency), 0, 1)

    # Interpolate from blue (low) to red (high)
    r = int(0 + saliency * (255 - 0))    # Red from 0 to 255
    g = int(0)                           # Green stays 0
    b = int(255 - saliency * (255 - 0))  # Blue from 255 to 0

    return f"rgb({r},{g},{b})"

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

def saliency_to_grayscale(saliency, min_saliency=0.0, max_saliency=1.0):
    saliency = np.clip((saliency - min_saliency) / (max_saliency - min_saliency), 0, 1)
    gray_value = int(238 - saliency * (238 - 17))  # 238: #eeeeee (light gray), 17: #111111 (dark gray)
    return f"rgb({gray_value},{gray_value},{gray_value})"

def plot_top_confirmed_gene_neighbors_chord(
    graph,
    node_names,
    name_to_index,
    scores,
    confirmed_genes,
    output_html="top10_confirmed_gene_neighbors_chord.html",
    output_png="top10_confirmed_gene_neighbors_chord.png",
    top_k_genes=10,
    top_k_neighbors=10,
    min_edge_score=0.05
):
    """
    Plots a chord diagram of top confirmed genes and their neighbors.
    Saves both interactive HTML and static PNG versions.

    Parameters:
        graph: DGLGraph
        node_names: list mapping node indices to gene names
        name_to_index: dict mapping gene names to indices
        scores: numpy array of predicted scores
        confirmed_genes: list of confirmed gene names
        output_html: HTML path to save the interactive chord diagram
        output_png: PNG path to save the static image
        top_k_genes: number of confirmed genes to include
        top_k_neighbors: number of top neighbors per gene
        min_edge_score: minimum score threshold for edges
    """

    # Step 1: Filter top confirmed genes
    confirmed_gene_scores = [
        (gene, scores[name_to_index[gene]])
        for gene in confirmed_genes if gene in name_to_index
    ]
    top_confirmed_genes = sorted(confirmed_gene_scores, key=lambda x: x[1], reverse=True)[:top_k_genes]

    # Step 2: Build edges to top neighbors
    chord_links = []
    seen_edges = set()

    for gene, _ in top_confirmed_genes:
        idx = name_to_index[gene]
        neighbors = graph.successors(idx).tolist()
        neighbor_scores = [
            (node_names[n], scores[n]) for n in neighbors if node_names[n] != gene
        ]
        top_neighbors = sorted(neighbor_scores, key=lambda x: x[1], reverse=True)[:top_k_neighbors]

        for neighbor, score in top_neighbors:
            if score >= min_edge_score:
                edge_key = tuple(sorted((gene, neighbor)))
                if edge_key not in seen_edges:
                    chord_links.append((gene, neighbor, score))
                    seen_edges.add(edge_key)

    if not chord_links:
        print("[Chord Diagram] No valid edges found. Try lowering `min_edge_score`.")
        return

    # Step 3: Create Chord diagram
    chord = hv.Chord(chord_links).select(value=(min_edge_score, None))
    chord.opts(
        opts.Chord(
            cmap='Category20',
            edge_color='source',
            node_color='index',
            labels='name',
            edge_alpha=0.7,
            edge_line_width=hv.dim('value') * 5,
            width=900,
            height=900,
            title="Top Confirmed Genes and Their Neighbors"
        )
    )

    # Step 4: Save HTML
    hv.save(chord, output_html)
    print(f"[✔] HTML saved to: {output_html}")

    # Step 5: Save PNG using Bokeh backend
    try:
        from bokeh.io.export import export_png
        from bokeh.io import curdoc
        from holoviews.plotting.bokeh import render

        plot = render(chord)
        export_png(plot, filename=output_png)
        print(f"[✔] PNG saved to: {output_png}")
    except Exception as e:
        print(f"[⚠] PNG export failed: {e}")
        print("To enable PNG export, make sure you have installed: selenium, pillow, and a compatible web driver like chromedriver.")

def plot_top_confirmed_gene_neighbors_chord_not_gene_name(
    graph,
    node_names,
    name_to_index,
    scores,
    confirmed_genes,
    output_path="top10_confirmed_gene_neighbors_chord.html",
    top_k_genes=10,
    top_k_neighbors=10,
    min_edge_score=0.05  # filter out weak edges for clarity
):
    """
    Plots a chord diagram of top K confirmed genes and their top K neighbors by predicted cancer score.

    Parameters:
        graph: DGL graph
        node_names: list of all node names
        name_to_index: dict mapping names to indices
        scores: numpy array of cancer scores
        confirmed_genes: list of confirmed gene names
        output_path: where to save the HTML file
        top_k_genes: how many confirmed genes to show
        top_k_neighbors: how many neighbors per gene
        min_edge_score: minimum score threshold for showing edges
    """

    # Step 1: Rank confirmed genes by model score
    confirmed_gene_scores = [
        (gene, scores[name_to_index[gene]])
        for gene in confirmed_genes if gene in name_to_index
    ]
    top_confirmed_genes = sorted(confirmed_gene_scores, key=lambda x: x[1], reverse=True)[:top_k_genes]

    # Step 2: For each gene, get top neighbors by score
    chord_links = []
    seen_edges = set()

    for gene, _ in top_confirmed_genes:
        idx = name_to_index[gene]
        neighbors = graph.successors(idx).tolist()
        neighbor_scores = [
            (node_names[n], scores[n]) for n in neighbors if node_names[n] != gene
        ]
        top_neighbors = sorted(neighbor_scores, key=lambda x: x[1], reverse=True)[:top_k_neighbors]

        for neighbor, score in top_neighbors:
            if score >= min_edge_score:
                edge_key = tuple(sorted((gene, neighbor)))
                if edge_key not in seen_edges:
                    chord_links.append((gene, neighbor, score))
                    seen_edges.add(edge_key)

    if not chord_links:
        print("[Chord Diagram] No valid edges found. Try lowering `min_edge_score`.")
        return

    # Step 3: Create Chord Diagram
    chord = hv.Chord(chord_links).select(value=(min_edge_score, None))
    chord.opts(
        opts.Chord(
            cmap='Category20',
            edge_color='source',
            node_color='index',
            labels='name',
            edge_alpha=0.7,
            edge_line_width=hv.dim('value') * 5,
            width=900,
            height=900,
            title="Top Confirmed Genes and Neighbors"
        )
    )

    # Save
    hv.save(chord, output_path)
    print(f"[✔] Chord diagram saved to: {output_path}")

def plot_chord_diagram_topo(
    args,
    source_labels,
    target_labels,
    matrix,
    CLUSTER_COLORS
):
    """
    A stylized approximation of a chord diagram using Plotly's Sankey layout.
    """

    all_labels = list(set(source_labels) | set(target_labels))
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    
    source = []
    target = []
    value = []
    link_colors = []

    for i, src in enumerate(source_labels):
        for j, tgt in enumerate(target_labels):
            if matrix[i][j] > 0:
                source_idx = label_to_index[src]
                target_idx = label_to_index[tgt]
                source.append(source_idx)
                target.append(target_idx)
                value.append(matrix[i][j])

                # Get color from source if available
                color = CLUSTER_COLORS.get(i, "#999999")
                link_colors.append(hex_to_rgba(color, 0.5))

    node_colors = [CLUSTER_COLORS.get(i, "#888888") for i in range(len(all_labels))]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text="Chord Diagram (Approx) - Topological Gene Interactions",
        font_size=12,
        margin=dict(l=200, r=200, t=100, b=100),
        width=1200,
        height=1000,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Save output
    output_dir = "results/gene_prediction/topo_chord_diagram/"
    os.makedirs(output_dir, exist_ok=True)

    html_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_topo_chord_epo{args.num_epochs}.html"
    )
    fig.write_html(html_path)
    print(f"✅ Chord diagram saved as HTML: {html_path}")

    # Save PNG
    try:
        import plotly.io as pio
        png_path = os.path.join(
            output_dir,
            f"{args.model_type}_{args.net_type}_topo_chord_epo{args.num_epochs}.png"
        )
        fig.write_image(png_path, format="png", scale=2, width=1200, height=1000)
        print(f"🖼️ PNG also saved: {png_path}")
    except Exception as e:
        print(f"⚠️ Failed to save PNG: {e}")
        print("Tip: Install 'kaleido' to enable image export:\n  pip install kaleido")

def plot_top_confirmed_gene_neighbors_chord(
    graph,
    node_names,
    name_to_index,
    scores,
    confirmed_genes,
    cluster_labels=None,
    cluster_colors=None,
    output_html="results/chord/top10_confirmed_gene_neighbors_chord.html",
    output_png="results/chord/top10_confirmed_gene_neighbors_chord.png",
    top_k_genes=10,
    top_k_neighbors=10,
    min_edge_score=0.05
):
    """
    Plots a chord diagram of top confirmed genes and their neighbors.
    Saves both interactive HTML and static PNG versions.

    Parameters:
        graph: DGLGraph
        node_names: list mapping node indices to gene names
        name_to_index: dict mapping gene names to indices
        scores: numpy array of predicted scores
        confirmed_genes: list of confirmed gene names
        cluster_labels: list or array of cluster labels for each node (optional)
        cluster_colors: dict mapping cluster label to hex color (optional)
        output_html: path to save interactive HTML
        output_png: path to save static PNG
        top_k_genes: number of confirmed genes to include
        top_k_neighbors: number of top neighbors per gene
        min_edge_score: minimum score threshold for edges
    """
    # Ensure output dirs exist
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # Step 1: Filter top confirmed genes
    confirmed_gene_scores = [
        (gene, scores[name_to_index[gene]])
        for gene in confirmed_genes if gene in name_to_index
    ]
    top_confirmed_genes = sorted(confirmed_gene_scores, key=lambda x: x[1], reverse=True)[:top_k_genes]

    # Step 2: Build edges to top neighbors
    chord_links = []
    seen_edges = set()

    for gene, _ in top_confirmed_genes:
        idx = name_to_index[gene]
        neighbors = graph.successors(idx).tolist()
        neighbor_scores = [
            (node_names[n], scores[n]) for n in neighbors if node_names[n] != gene
        ]
        top_neighbors = sorted(neighbor_scores, key=lambda x: x[1], reverse=True)[:top_k_neighbors]

        for neighbor, score in top_neighbors:
            if score >= min_edge_score:
                edge_key = tuple(sorted((gene, neighbor)))
                if edge_key not in seen_edges:
                    chord_links.append((gene, neighbor, score))
                    seen_edges.add(edge_key)

    if not chord_links:
        print("[Chord Diagram] No valid edges found. Try lowering `min_edge_score`.")
        return

    # Step 3: Create node color mapping if cluster info is provided
    node_set = set([g for edge in chord_links for g in edge[:2]])
    df_nodes = pd.DataFrame({'name': list(node_set)})

    if cluster_labels is not None and cluster_colors is not None:
        df_nodes['cluster'] = df_nodes['name'].map(lambda g: cluster_labels[name_to_index[g]])
        df_nodes['color'] = df_nodes['cluster'].map(lambda c: cluster_colors.get(c, "#cccccc"))
    else:
        df_nodes['color'] = "#cccccc"

    name_to_color = dict(zip(df_nodes['name'], df_nodes['color']))

    # Step 4: Create Chord diagram
    chord = hv.Chord(chord_links).select(value=(min_edge_score, None))
    chord.opts(
        opts.Chord(
            cmap='Category20',
            edge_color='source',
            node_color=hv.dim('name').categorize(name_to_color, default="#cccccc"),
            labels='name',
            edge_alpha=0.7,
            edge_line_width=hv.dim('value') * 5,
            width=900,
            height=900,
            title="Top Confirmed Genes and Their Neighbors"
        )
    )

    # Step 5: Save HTML
    hv.save(chord, output_html)
    print(f"[✔] HTML saved to: {output_html}")

    # Step 6: Save PNG using Bokeh backend
    try:
        from bokeh.io.export import export_png
        from holoviews.plotting.bokeh import render
        plot = render(chord)
        export_png(plot, filename=output_png)
        print(f"[✔] PNG saved to: {output_png}")
    except Exception as e:
        print(f"[⚠] PNG export failed: {e}")
        print("To enable PNG export, install dependencies: `pip install selenium pillow` and configure a headless browser like `chromedriver`.")

def plot_enriched_pathways_heatmap(
    enrichment_results,
    output_dir,
    model_type,
    net_type,
    num_epochs,
    max_term_len=40,
    max_topo_term_len=60,
    max_rows=50,
    top_n_terms_per_cluster=None,
    return_data=False
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.cm import get_cmap
    from matplotlib.patches import Patch

    heatmap_data = pd.DataFrame()

    for cluster_type in ['bio', 'topo']:
        for cid, df in enrichment_results[cluster_type].items():
            if top_n_terms_per_cluster:
                df = df[df['p_value'] < 0.05].sort_values(by='p_value').head(top_n_terms_per_cluster)
            colname = f"{cluster_type.capitalize()}_{cid}"
            vals = {}
            for _, row in df.iterrows():
                p = row['p_value']
                name = row['name']
                if p < 0.05 and len(name) <= max_term_len:
                    term = f"{name} ({row['source']})"
                    vals[term] = -np.log10(p)
            heatmap_data[colname] = pd.Series(vals)

    heatmap_data = heatmap_data.fillna(0)
    heatmap_data = heatmap_data[heatmap_data.max(axis=1) > 1]

    enrichment_csv_path = os.path.join(
        output_dir,
        f"{model_type}_{net_type}_enrichment_matrix_epo{num_epochs}.csv"
    )
    heatmap_data.to_csv(enrichment_csv_path, index_label='Enriched Pathway')

    # Topo terms export
    topo_terms = []
    for cid, df in enrichment_results['topo'].items():
        if top_n_terms_per_cluster:
            df = df[df['p_value'] < 0.05].sort_values(by='p_value').head(top_n_terms_per_cluster)
        for _, row in df.iterrows():
            if row['p_value'] < 0.05 and len(row['name']) <= max_topo_term_len:
                topo_terms.append({
                    "Cluster": f"Topo_{cid}",
                    "Term": row['name'],
                    "Source": row['source'],
                    "p_value": row['p_value'],
                    "-log10(p)": -np.log10(row['p_value']),
                })

    topo_terms_df = pd.DataFrame(topo_terms)
    topo_terms_path = os.path.join(
        output_dir,
        f"{model_type}_{net_type}_topo_cluster_top_terms_epo{num_epochs}.csv"
    )
    topo_terms_df.to_csv(topo_terms_path, index=False)

    if heatmap_data.shape[0] > max_rows:
        step = max(1, heatmap_data.shape[0] // max_rows)
        selected_indices = heatmap_data.index[::step][:max_rows]
        heatmap_data = heatmap_data.loc[selected_indices]

    norm_data = heatmap_data / heatmap_data.max().replace(0, 1)

    colormaps = {
        'bio': get_cmap('Blues'),
        'topo': get_cmap('YlOrRd'),
    }

    colors = np.zeros((heatmap_data.shape[0], heatmap_data.shape[1], 4))
    col_types = []

    for i, col in enumerate(norm_data.columns):
        group = 'bio' if col.lower().startswith("bio") else 'topo'
        col_types.append(group)
        cmap = colormaps[group]
        colors[:, i, :] = cmap(norm_data[col].values)

    fig, ax = plt.subplots(figsize=(0.5 * len(norm_data.columns), 0.2 * len(norm_data)))

    ax.imshow(colors, aspect='auto')

    ax.set_xticks(np.arange(len(norm_data.columns)))
    ax.set_xticklabels(norm_data.columns, rotation=90, fontsize=14)
    ax.set_yticks(np.arange(len(norm_data.index)))
    ax.set_yticklabels(norm_data.index, fontsize=14)

    ax.set_ylabel("Enriched Pathway", fontsize=16, labelpad=20)

    for xtick, col in zip(ax.get_xticklabels(), col_types):
        xtick.set_color('darkblue' if col == 'bio' else 'darkred')

    ax.set_title("Top Enriched Pathways per Cluster (p < 0.05)", fontsize=15, pad=16)
    ax.set_xlabel("Cluster", fontsize=14)

    legend_patches = [
        Patch(color='cornflowerblue', label='Bio'),
        Patch(color='salmon', label='Topo')
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))

    sns.despine(ax=ax, trim=True)
    ax.tick_params(axis='both', which='both', length=0)
    plt.tight_layout(rect=[0, 0, 0.95, 0.93])

    enriched_terms_heatmap_path = os.path.join(
        output_dir,
        f"{model_type}_{net_type}_enriched_terms_heatmap_epo{num_epochs}.png"
    )
    plt.savefig(enriched_terms_heatmap_path, dpi=300)
    plt.close()

    if return_data:
        return heatmap_data, topo_terms_df

def save_and_plot_enriched_pathways(enrichment_results, args, output_dir):
    # === Prepare Data ===
    heatmap_data = pd.DataFrame()

    for cluster_type in ['bio', 'topo']:
        for cid, df in enrichment_results[cluster_type].items():
            colname = f"{cluster_type.capitalize()}_{cid}"
            vals = {}
            for _, row in df.iterrows():
                p = row['p_value']
                name = row['name']
                if p < 0.05 and len(name) <= 50:
                    term = f"{name} ({row['source']})"
                    vals[term] = -np.log10(p)
            heatmap_data[colname] = pd.Series(vals)

    # Clean and filter
    heatmap_data = heatmap_data.fillna(0)
    heatmap_data = heatmap_data[heatmap_data.max(axis=1) > 1]

    # Save full enrichment data to CSV
    enrichment_csv_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_enrichment_matrix_epo{args.num_epochs}.csv"
    )
    heatmap_data.to_csv(enrichment_csv_path, index_label='Enriched Pathway')

    # === Save Topo Cluster → Top Enriched Terms to CSV ===
    topo_terms = []
    for cid, df in enrichment_results['topo'].items():
        for _, row in df.iterrows():
            if row['p_value'] < 0.05 and len(row['name']) <= 60:
                topo_terms.append({
                    "Cluster": f"Topo_{cid}",
                    "Term": row['name'],
                    "Source": row['source'],
                    "p_value": row['p_value'],
                    "-log10(p)": -np.log10(row['p_value']),
                })
    topo_terms_df = pd.DataFrame(topo_terms)
    topo_terms_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_topo_cluster_top_terms_epo{args.num_epochs}.csv"
    )
    topo_terms_df.to_csv(topo_terms_path, index=False)

    # === Save Bio Cluster → Top Enriched Terms to CSV ===
    bio_terms = []
    for cid, df in enrichment_results['bio'].items():
        for _, row in df.iterrows():
            if row['p_value'] < 0.05 and len(row['name']) <= 60:
                bio_terms.append({
                    "Cluster": f"Bio_{cid}",
                    "Term": row['name'],
                    "Source": row['source'],
                    "p_value": row['p_value'],
                    "-log10(p)": -np.log10(row['p_value']),
                })
    bio_terms_df = pd.DataFrame(bio_terms)
    bio_terms_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_bio_cluster_top_terms_epo{args.num_epochs}.csv"
    )
    bio_terms_df.to_csv(bio_terms_path, index=False)

    # === Select 50 evenly spaced rows for plotting ===
    if heatmap_data.shape[0] > 50:
        step = max(1, heatmap_data.shape[0] // 50)
        selected_indices = heatmap_data.index[::step][:50]
        heatmap_data = heatmap_data.loc[selected_indices]

    # === Normalize for color contrast ===
    norm_data = heatmap_data.copy()
    norm_data = norm_data / norm_data.max().replace(0, 1)

    # === Apply group-wise colormaps ===
    colormaps = {
        'bio': get_cmap('Blues'),
        'topo': get_cmap('YlOrRd'),
    }

    colors = np.zeros((heatmap_data.shape[0], heatmap_data.shape[1], 4))  # RGBA
    col_types = []

    for i, col in enumerate(norm_data.columns):
        group = 'bio' if col.lower().startswith("bio") else 'topo'
        col_types.append(group)
        cmap = colormaps[group]
        colors[:, i, :] = cmap(norm_data[col].values)

    # === Plot ===
    fig, ax = plt.subplots(figsize=(0.5 * len(norm_data.columns), 0.2 * len(norm_data)))

    ax.imshow(colors, aspect='auto')
    ax.set_xticks(np.arange(len(norm_data.columns)))
    ax.set_xticklabels(norm_data.columns, rotation=90, fontsize=14)
    ax.set_yticks(np.arange(len(norm_data.index)))
    ax.set_yticklabels(norm_data.index, fontsize=14)
    ax.set_ylabel("Enriched Pathway", fontsize=16, labelpad=20)

    # Color x-axis labels
    for xtick, col in zip(ax.get_xticklabels(), col_types):
        xtick.set_color('darkblue' if col == 'bio' else 'darkred')

    ax.set_title("Top Enriched Pathways per Cluster (p < 0.05)", fontsize=15, pad=16)
    ax.set_xlabel("Cluster", fontsize=14)

    legend_patches = [
        Patch(color='cornflowerblue', label='Bio'),
        Patch(color='salmon', label='Topo')
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))

    sns.despine(ax=ax, trim=True)
    ax.tick_params(axis='both', which='both', length=0)
    plt.tight_layout(rect=[0, 0, 0.95, 0.93])

    # Save plot
    enriched_terms_heatmap_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_enriched_terms_heatmap_epo{args.num_epochs}.png"
    )
    plt.savefig(enriched_terms_heatmap_path, dpi=300)
    plt.close()

    # === Return DataFrames for downstream analysis ===
    return heatmap_data, topo_terms_df, bio_terms_df

def plot_collapsed_clusterfirst_multilevel_sankey_topo(
    args,
    graph,
    node_names,
    name_to_index,
    confirmed_genes,
    scores,
    cluster_labels,
    total_clusters,
    relevance_scores,
    CLUSTER_COLORS
    ): 

    # === Setup
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    # === Filter top 10 confirmed genes
    confirmed_genes = sorted(
        confirmed_genes,
        key=lambda g: scores[name_to_index[g]],
        reverse=True
    )[:10]

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    label_to_idx = {}
    all_labels = []
    all_colors = []
    font_sizes = []

    cluster_to_genes = {}
    gene_to_neighbors = {}

    highlight_node_indices = []
    highlight_node_saliency = []

    # Top 10 by model score
    top_scored_genes = sorted(
        confirmed_genes,
        key=lambda g: scores[name_to_index[g]],
        reverse=True
    )
        
    selected_genes = ["BRCA1", "TP53", "PIK3CA", "KRAS", "ALK"]

    # Combine and deduplicate while preserving order
    combined_genes = []
    seen = set()
    for g in selected_genes + top_scored_genes:
        if g in name_to_index and g not in seen:
            combined_genes.append(g)
            seen.add(g)
        if len(combined_genes) == 15:
            break

    confirmed_genes = combined_genes
    
    # === Build mappings
    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            continue

        node_idx = topk_name_to_index[gene]
        gene_cluster = cluster_labels[node_idx]
        cluster_label = f"Confirmed Cluster {gene_cluster}"

        if cluster_label not in cluster_to_genes:
            cluster_to_genes[cluster_label] = []
        cluster_to_genes[cluster_label].append(gene)

        neighbors = neighbors_dict.get(gene, [])
        neighbor_scores = {}

        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores[rel_idx] = rel_score

        if neighbor_scores:
            neighbor_scores = dict(sorted(neighbor_scores.items(), key=lambda x: -x[1])[:5])

        gene_to_neighbors[gene] = neighbor_scores

    source = []
    target = []
    value = []
    link_colors = []

    for cluster_label, genes in cluster_to_genes.items():
        if cluster_label not in label_to_idx:
            label_to_idx[cluster_label] = len(all_labels)
            all_labels.append(cluster_label)

            cluster_id = int(cluster_label.split()[-1])
            all_colors.append(CLUSTER_COLORS.get(cluster_id, "#000000"))
            font_sizes.append(18)

        cluster_idx = label_to_idx[cluster_label]

        genes_sorted = sorted(genes, key=lambda g: scores[name_to_index[g]], reverse=True)[:10]

        for gene in genes_sorted:
            if gene not in label_to_idx:
                label_to_idx[gene] = len(all_labels)
                all_labels.append(gene)

                rel_idx = topk_name_to_index[gene]
                saliency = relevance_scores[rel_idx].sum().item()
                gene_cluster = cluster_labels[rel_idx]
                color = CLUSTER_COLORS.get(gene_cluster, "#000000")
                all_colors.append(color)

                font_sizes.append(18 if saliency > 0.5 else 10)
                if saliency > 0.5:
                    highlight_node_indices.append(label_to_idx[gene])
                    highlight_node_saliency.append(saliency)

            gene_idx = label_to_idx[gene]

            source.append(cluster_idx)
            target.append(gene_idx)
            value.append(1)
            link_colors.append(hex_to_rgba(CLUSTER_COLORS.get(int(cluster_label.split()[-1]), "#000000"), 0.4))

            neighbors = gene_to_neighbors.get(gene, {})
            for neighbor_idx, neighbor_score in neighbors.items():
                neighbor_name = node_id_to_name[neighbor_idx]
                neighbor_cluster = cluster_labels[neighbor_idx]
                neighbor_cluster_label = f"Cluster {neighbor_cluster}"

                if neighbor_name not in label_to_idx:
                    label_to_idx[neighbor_name] = len(all_labels)
                    all_labels.append(neighbor_name)

                    saliency = relevance_scores[neighbor_idx].sum().item()
                    color = CLUSTER_COLORS.get(neighbor_cluster, "#000000")
                    all_colors.append(color)

                    font_sizes.append(16 if saliency > 0.5 else 10)
                    if saliency > 0.5:
                        highlight_node_indices.append(label_to_idx[neighbor_name])
                        highlight_node_saliency.append(saliency)

                neighbor_node_idx = label_to_idx[neighbor_name]

                if neighbor_cluster_label not in label_to_idx:
                    label_to_idx[neighbor_cluster_label] = len(all_labels)
                    all_labels.append(neighbor_cluster_label)
                    all_colors.append(CLUSTER_COLORS.get(neighbor_cluster, "#000000"))
                    font_sizes.append(18)

                neighbor_cluster_idx = label_to_idx[neighbor_cluster_label]

                # Gene → Neighbor
                source.append(gene_idx)
                target.append(neighbor_node_idx)
                value.append(neighbor_score)
                link_colors.append("rgba(160,160,160,0.5)")

                # Neighbor → Neighbor Cluster
                source.append(neighbor_node_idx)
                target.append(neighbor_cluster_idx)
                value.append(neighbor_score)
                link_colors.append(hex_to_rgba(CLUSTER_COLORS.get(neighbor_cluster, "#000000"), 0.6))

    # === Build Sankey
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=all_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    # === Add saliency highlight
    if highlight_node_indices:
        x_positions = []
        y_positions = []
        for idx in highlight_node_indices:
            x = 0.1 + (idx % 6) * 0.15
            y = 0.9 - (idx // 6) * 0.1
            x_positions.append(x)
            y_positions.append(y)

        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='none',
            marker=dict(
                size=[30 + 40 * (s-0.5) for s in highlight_node_saliency],
                color="rgba(255,0,0,0.3)",
                line=dict(width=2, color="rgba(255,0,0,0.7)"),
                sizemode='diameter'
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        
    fig.update_layout(
        title=None,  # Remove title text
        font_size=16,
        margin=dict(l=20, r=20, t=20, b=20), 
        width=1200,
        height=1200,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        )
    )

    # === Save
    output_dir = "results/gene_prediction/topo_collapsed_clusterfirst_multilevel_sankey/"
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_topo_collapsed_clusterfirst_multilevel_sankey_epo{args.num_epochs}.html"
    )
    fig.write_html(save_path)
    print(f"✅ Collapsed Cluster-First Multi-level Sankey (Topo) saved: {save_path}")

    # === Save PNG
    try:
        import plotly.io as pio
        png_save_path = os.path.join(
            output_dir,
            f"{args.model_type}_{args.net_type}_topo_collapsed_clusterfirst_multilevel_sankey_epo{args.num_epochs}.png"
        )
        fig.write_image(png_save_path, scale=2, width=1200, height=1200)
        print(f"🖼️ PNG also saved to: {png_save_path}")
    except Exception as e:
        print(f"⚠️ Failed to save PNG: {e}")
        print("Tip: Install 'kaleido' via pip to enable static image export: pip install kaleido")

def plot_collapsed_clusterfirst_multilevel_sankey_bio(
    args,
    graph,
    node_names,
    name_to_index,
    confirmed_genes,
    scores,
    cluster_labels,
    total_clusters,
    relevance_scores,
    CLUSTER_COLORS
):
    topk_name_to_index = {name: i for i, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    # Top 10 by model score
    top_scored_genes = sorted(
        confirmed_genes,
        key=lambda g: scores[name_to_index[g]],
        reverse=True
    )

    # Manually selected important genes
    selected_genes = ["BRCA1", "TP53", "PIK3CA", "KRAS", "ALK"]

    # Combine and deduplicate while preserving order
    combined_genes = []
    seen = set()
    for g in selected_genes + top_scored_genes:
        if g in name_to_index and g not in seen:
            combined_genes.append(g)
            seen.add(g)
        if len(combined_genes) == 15:
            break

    confirmed_genes = combined_genes



    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    label_to_idx = {}
    all_labels = []
    all_colors = []
    font_sizes = []

    cluster_to_genes = {}
    gene_to_neighbors = {}

    highlight_node_indices = []
    highlight_node_saliency = []

    for gene in confirmed_genes:
        if gene not in topk_name_to_index:
            continue

        node_idx = topk_name_to_index[gene]
        gene_cluster = cluster_labels[node_idx]
        cluster_label = f"Confirmed Cluster {gene_cluster}"

        cluster_to_genes.setdefault(cluster_label, []).append(gene)

        neighbors = neighbors_dict.get(gene, [])
        neighbor_scores = {}

        for n in neighbors:
            if n in topk_name_to_index:
                rel_idx = topk_name_to_index[n]
                if rel_idx < relevance_scores.shape[0]:
                    rel_score = relevance_scores[rel_idx].sum().item()
                    neighbor_scores[rel_idx] = rel_score

        if neighbor_scores:
            neighbor_scores = dict(sorted(neighbor_scores.items(), key=lambda x: -x[1])[:5])

        gene_to_neighbors[gene] = neighbor_scores

    source = []
    target = []
    value = []
    link_colors = []

    for cluster_label, genes in cluster_to_genes.items():
        if cluster_label not in label_to_idx:
            label_to_idx[cluster_label] = len(all_labels)
            all_labels.append(cluster_label)
            cluster_id = int(cluster_label.split()[-1])
            all_colors.append(CLUSTER_COLORS.get(cluster_id, "#000000"))
            font_sizes.append(24)

        cluster_idx = label_to_idx[cluster_label]
        genes_sorted = sorted(genes, key=lambda g: scores[name_to_index[g]], reverse=True)[:10]

        for gene in genes_sorted:
            if gene not in label_to_idx:
                label_to_idx[gene] = len(all_labels)
                all_labels.append(gene)

                rel_idx = topk_name_to_index[gene]
                saliency = relevance_scores[rel_idx].sum().item()
                gene_cluster = cluster_labels[rel_idx]
                color = CLUSTER_COLORS.get(gene_cluster, "#000000")
                all_colors.append(color)
                font_sizes.append(18 if saliency > 0.5 else 10)

                if saliency > 0.5:
                    highlight_node_indices.append(label_to_idx[gene])
                    highlight_node_saliency.append(saliency)

            gene_idx = label_to_idx[gene]

            source.append(cluster_idx)
            target.append(gene_idx)
            value.append(1)
            link_colors.append(hex_to_rgba(CLUSTER_COLORS.get(int(cluster_label.split()[-1]), "#000000"), 0.4))

            neighbors = gene_to_neighbors.get(gene, {})
            for neighbor_idx, neighbor_score in neighbors.items():
                neighbor_name = node_id_to_name[neighbor_idx]
                neighbor_cluster = cluster_labels[neighbor_idx]
                neighbor_cluster_label = f"Cluster {neighbor_cluster}"

                if neighbor_name not in label_to_idx:
                    label_to_idx[neighbor_name] = len(all_labels)
                    all_labels.append(neighbor_name)

                    saliency = relevance_scores[neighbor_idx].sum().item()
                    color = CLUSTER_COLORS.get(neighbor_cluster, "#000000")
                    all_colors.append(color)
                    font_sizes.append(16 if saliency > 0.5 else 10)

                    if saliency > 0.5:
                        highlight_node_indices.append(label_to_idx[neighbor_name])
                        highlight_node_saliency.append(saliency)

                neighbor_node_idx = label_to_idx[neighbor_name]

                if neighbor_cluster_label not in label_to_idx:
                    label_to_idx[neighbor_cluster_label] = len(all_labels)
                    all_labels.append(neighbor_cluster_label)
                    all_colors.append(CLUSTER_COLORS.get(neighbor_cluster, "#000000"))
                    font_sizes.append(18)

                neighbor_cluster_idx = label_to_idx[neighbor_cluster_label]

                source.append(gene_idx)
                target.append(neighbor_node_idx)
                value.append(neighbor_score)
                link_colors.append("rgba(160,160,160,0.5)")

                source.append(neighbor_node_idx)
                target.append(neighbor_cluster_idx)
                value.append(neighbor_score)
                link_colors.append(hex_to_rgba(CLUSTER_COLORS.get(neighbor_cluster, "#000000"), 0.6))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=all_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    if highlight_node_indices:
        x_positions = [0.1 + (idx % 6) * 0.15 for idx in highlight_node_indices]
        y_positions = [0.9 - (idx // 6) * 0.1 for idx in highlight_node_indices]

        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='none',
            marker=dict(
                size=[30 + 40 * (s-0.5) for s in highlight_node_saliency],
                color="rgba(255,0,0,0.3)",
                line=dict(width=2, color="rgba(255,0,0,0.7)"),
                sizemode='diameter'
            ),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title=None,
        font_size=16,
        margin=dict(l=20, r=20, t=20, b=20),
        width=1200,
        height=1200,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )

    output_dir = "results/gene_prediction/bio_collapsed_clusterfirst_multilevel_sankey/"
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_bio_collapsed_clusterfirst_multilevel_sankey_epo{args.num_epochs}.html"
    )
    fig.write_html(save_path)
    print(f"✅ Collapsed Cluster-First Multi-level Sankey saved: {save_path}")

    try:
        import plotly.io as pio
        png_save_path = os.path.join(
            output_dir,
            f"{args.model_type}_{args.net_type}_bio_collapsed_clusterfirst_multilevel_sankey_epo{args.num_epochs}.png"
        )
        fig.write_image(png_save_path, scale=2, width=1200, height=1200)
        print(f"🖼️ PNG also saved to: {png_save_path}")
    except Exception as e:
        print(f"⚠️ Failed to save PNG: {e}")
        print("Tip: Install 'kaleido' via pip to enable static image export: pip install kaleido")

    # === Run Structure Analysis
    sankey_stats = analyze_sankey_structure(
        source,
        target,
        value,
        label_to_idx,
        node_names,
        cluster_to_genes,
        gene_to_neighbors,
        cluster_labels,
        name_to_index,
        scores
    )

    # === Summary Plot: Entropy per Cluster
    entropy = sankey_stats["cluster_entropy"]

    entropy_df = pd.DataFrame(list(entropy.items()), columns=["Cluster", "Entropy"])
    entropy_df = entropy_df.sort_values("Entropy", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Entropy", y="Cluster", data=entropy_df, palette="coolwarm")
    plt.title("Cluster Entropy (Gene Participation Diversity)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{args.model_type}_{args.net_type}_entropy_bar_epo{args.num_epochs}.png"))
    plt.close()

    # === Summary Plot: Jaccard Heatmap
    jaccard = sankey_stats["cluster_jaccard"]
    ##jaccard_df = pd.DataFrame(jaccard).fillna(0)
    jaccard_df = pd.DataFrame([
        {"Cluster1": c1, "Cluster2": c2, "Jaccard": val}
        for (c1, c2), val in jaccard.items()
    ])

    pivot_df = jaccard_df.pivot(index="Cluster1", columns="Cluster2", values="Jaccard").fillna(0)


    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".2f", square=True, cbar_kws={'label': 'Jaccard Index'})
    plt.title("Jaccard Similarity Between Confirmed Gene Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{args.model_type}_{args.net_type}_jaccard_heatmap_epo{args.num_epochs}.png"))
    plt.close()

    # === (Optional) Summary Plot: Centrality per Gene
    #centrality_df = pd.DataFrame(list(sankey_stats["centrality"].items()), columns=["Gene", "Centrality"])
    centrality_df = pd.DataFrame(list(sankey_stats["gene_degree_centrality"].items()), columns=["Gene", "Centrality"])

    centrality_df = centrality_df.sort_values("Centrality", ascending=False)

    plt.figure(figsize=(10, 15))
    sns.barplot(x="Centrality", y="Gene", data=centrality_df, palette="magma")
    plt.title("Neighbor Centrality (Sum of Relevance)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{args.model_type}_{args.net_type}_centrality_bar_epo{args.num_epochs}.png"))
    plt.close()

    return sankey_stats

def plot_saliency_ridge_by_cluster_and_omics(
    saliency_scores,
    feature_names,
    cluster_assignments,
    output_path,
    omics_labels=["CNA", "GE", "METH", "MF"]
):
    # Map each feature to an omics type
    omics_map = {}
    for name in feature_names:
        for omics in omics_labels:
            if name.startswith(omics):
                omics_map[name] = omics
                break
        else:
            omics_map[name] = "Unknown"

    # Create DataFrame
    df = pd.DataFrame(saliency_scores, columns=feature_names)
    df["Cluster"] = cluster_assignments

    # Melt to long format
    df_long = df.melt(id_vars="Cluster", var_name="Feature", value_name="Saliency")
    df_long["Omics"] = df_long["Feature"].map(omics_map)

    # Convert cluster to string for better labeling
    df_long["Cluster"] = df_long["Cluster"].astype(str)

    # Plot style
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Create FacetGrid with rows = cluster, hue = omics
    g = sns.FacetGrid(
        df_long,
        row="Cluster",
        hue="Omics",
        aspect=15,
        height=0.6,
        palette="Set2",
        row_order=sorted(df_long["Cluster"].unique(), key=lambda x: int(x))
    )

    g.map(sns.kdeplot, "Saliency", bw_adjust=0.5, clip_on=False, fill=True, alpha=0.6, linewidth=1.5)
    g.map(sns.kdeplot, "Saliency", bw_adjust=0.5, clip_on=False, color="w", lw=2)

    '''def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.05, 0.2, f"Cluster {label}", fontweight="bold", color="black",
                ha="right", va="center", transform=ax.transAxes, fontsize=12)'''
                
    # Remove the old label function and add this instead:
    g.set_titles(row_template="{row_name}", size=12, weight='bold')

    # Optional: if cluster names should have "Cluster #" prefix
    g.set_titles(row_template="Cluster {row_name}", size=12, weight='bold')

    g.map(label, "Saliency")

    g.fig.subplots_adjust(hspace=-0.7)
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel="Saliency Score")
    g.despine(bottom=True, left=True)
    g.add_legend(title="Omics Type", label_order=omics_labels)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Ridge plot split by cluster and omics saved to {output_path}")

def plot_gene_expression_ridges_by_cluster(
    expression_matrix,  # shape: (num_genes, num_patients)
    node_names,
    cluster_labels,
    total_clusters,
    output_path_prefix="ridge_plot"
):
    # Convert to DataFrame for easier handling
    df_list = []
    for gene_idx, gene_name in enumerate(node_names):
        cluster = cluster_labels[gene_idx]
        for patient_idx, expr_value in enumerate(expression_matrix[gene_idx]):
            df_list.append({
                "Gene": gene_name,
                "Expression": expr_value,
                "Cluster": f"clus{cluster + 1}"  # for clus1, clus2, ...
            })

    df = pd.DataFrame(df_list)

    # For each cluster, plot ridge plot of gene expression
    clusters = sorted(df["Cluster"].unique())
    n_cols = 4
    n_rows = (len(clusters) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=False)
    axes = axes.flatten()

    palette = sns.color_palette("husl", n_colors=20)

    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        sub_df = df[df["Cluster"] == cluster]

        sns.violinplot(
            data=sub_df,
            x="Expression",
            y="Gene",
            ax=ax,
            palette=[to_hex(palette[i % len(palette)]) for i in range(len(sub_df["Gene"].unique()))],
            linewidth=1,
            scale="width",
            cut=0
        )
        ax.set_title(cluster)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    output_path = f"{output_path_prefix}_ridge_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Ridge plot saved to {output_path}")
    plt.close()

def plot_top_gene_expression_ridges_by_cluster(
    expression_matrix,         # shape: (num_genes, num_patients)
    node_names,                # list of gene names, len = num_genes
    cluster_labels,            # list/array of cluster assignments, len = num_genes
    total_clusters,            # int, number of unique clusters
    top_n=12,                  # number of top genes per cluster
    output_path_prefix="ridge_plot"
):
    print("Preparing data for ridge plots...")

    # Create a DataFrame with summary stats
    gene_df = pd.DataFrame(expression_matrix)
    gene_df["Gene"] = node_names
    gene_df["Cluster"] = cluster_labels
    gene_df["MeanExpression"] = gene_df.iloc[:, :-2].mean(axis=1)

    df_list = []

    for cluster in range(total_clusters):
        cluster_genes = gene_df[gene_df["Cluster"] == cluster]
        top_genes = cluster_genes.nlargest(top_n, "MeanExpression")

        for _, row in top_genes.iterrows():
            gene_name = row["Gene"]
            expression_values = row.iloc[:-3].values  # patient-level expression
            for patient_idx, expr_value in enumerate(expression_values):
                df_list.append({
                    "Gene": gene_name,
                    "Expression": expr_value,
                    "Cluster": f"clus{cluster + 1}"
                })

    df = pd.DataFrame(df_list)

    # Plot
    clusters = sorted(df["Cluster"].unique())
    n_cols = 4
    n_rows = (len(clusters) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), sharex=True)
    axes = axes.flatten()

    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        sub_df = df[df["Cluster"] == cluster]

        sns.violinplot(
            data=sub_df,
            x="Expression",
            y="Gene",
            ax=ax,
            linewidth=1,
            scale="width",
            cut=0,
            inner="quartile",
            orient="h"
        )
        ax.set_title(f"{cluster} (top {top_n})")
        ax.set_xlabel("Expression")
        ax.set_ylabel("")

    for j in range(len(clusters), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    output_path = f"{output_path_prefix}_top{top_n}_genes_ridge_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Top-{top_n} ridge plot saved to {output_path}")
    plt.close()

def plot_top_gene_omics_ridges_by_cluster(
    omics_data,             # dict of {omics_type: np.array of shape [num_genes, 256]}
    node_names,             # list of gene names
    cluster_labels,         # array/list of cluster assignments
    total_clusters,         # int
    top_n=12,               # top genes per cluster
    output_path_prefix="omics_ridge_plot"
):
    for omics, matrix in omics_data.items():
        print(f"Preparing ridge plots for omics type: {omics}")

        gene_df = pd.DataFrame(matrix)
        gene_df["Gene"] = node_names
        gene_df["Cluster"] = cluster_labels
        gene_df["MeanFeatureValue"] = gene_df.iloc[:, :-2].mean(axis=1)

        df_list = []

        for cluster in range(total_clusters):
            cluster_genes = gene_df[gene_df["Cluster"] == cluster]
            top_genes = cluster_genes.nlargest(top_n, "MeanFeatureValue")

            for _, row in top_genes.iterrows():
                gene_name = row["Gene"]
                feature_values = row.iloc[:-3].values  # 256 omics features
                for feature_idx, feat_value in enumerate(feature_values):
                    df_list.append({
                        "Gene": gene_name,
                        "FeatureValue": feat_value,
                        "Cluster": f"clus{cluster + 1}"
                    })

        df = pd.DataFrame(df_list)

        # Plotting
        clusters = sorted(df["Cluster"].unique())
        n_cols = 4
        n_rows = (len(clusters) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), sharex=True)
        axes = axes.flatten()

        for idx, cluster in enumerate(clusters):
            ax = axes[idx]
            sub_df = df[df["Cluster"] == cluster]

            sns.violinplot(
                data=sub_df,
                x="FeatureValue",
                y="Gene",
                ax=ax,
                linewidth=1,
                scale="width",
                cut=0,
                inner="quartile",
                orient="h"
            )
            ax.set_title(f"{omics} - {cluster} (top {top_n})")
            ax.set_xlabel("Omics Feature Value")
            ax.set_ylabel("")

        for j in range(len(clusters), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        output_path = f"{output_path_prefix}_{omics}_top{top_n}_genes.png"
        plt.savefig(output_path, dpi=300)
        print(f"✅ Ridge plot for {omics} saved to {output_path}")
        plt.close()

def plot_top_gene_ridge_all_clusters(
    saliency_scores,
    feature_names,
    cluster_assignments,
    output_path,
    top_n_genes=12
):
    # Create DataFrame
    df = pd.DataFrame(saliency_scores, columns=feature_names)
    df["Cluster"] = cluster_assignments

    # Melt to long format
    df_long = df.melt(id_vars="Cluster", var_name="Feature", value_name="Saliency")
    df_long["Gene"] = df_long["Feature"].apply(lambda x: x.split("_", 1)[-1])  # Strip omics prefix

    # Get mean saliency per gene per cluster (aggregating across all omics)
    mean_saliency = (
        df_long.groupby(["Cluster", "Gene"])["Saliency"]
        .mean()
        .reset_index()
    )

    # Select top N genes per cluster
    top_genes_per_cluster = (
        mean_saliency
        .sort_values(["Cluster", "Saliency"], ascending=[True, False])
        .groupby("Cluster")
        .head(top_n_genes)
    )

    # Filter original df_long to keep only features belonging to top genes
    df_filtered = df_long[df_long["Gene"].isin(top_genes_per_cluster["Gene"]) & 
                          df_long["Cluster"].isin(top_genes_per_cluster["Cluster"])]

    # Plot settings
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    os.makedirs(output_path, exist_ok=True)

    unique_clusters = sorted(df_filtered["Cluster"].unique(), key=lambda x: int(x))

    for cluster in unique_clusters:
        cluster_df = df_filtered[df_filtered["Cluster"] == cluster].copy()

        # Order genes by saliency
        gene_order = (
            cluster_df.groupby("Gene")["Saliency"]
            .mean()
            .sort_values()
            .index
        )
        cluster_df["Gene"] = pd.Categorical(cluster_df["Gene"], categories=gene_order, ordered=True)

        # Ridge plot
        g = sns.FacetGrid(
            cluster_df, row="Gene", hue="Gene", aspect=10, height=0.5,
            palette="Spectral", sharex=True
        )
        g.map(sns.kdeplot, "Saliency", bw_adjust=0.5, fill=True, alpha=0.8)
        g.map(sns.kdeplot, "Saliency", bw_adjust=0.5, color="black", lw=1.0)

        g.set_titles("")
        g.set(yticks=[], ylabel="", xlabel="Saliency Score")
        g.despine(bottom=True, left=True)
        g.fig.subplots_adjust(hspace=-0.9)
        plt.text(-0.01, 1.01, f"Cluster {cluster}", fontsize=14, fontweight='bold', transform=g.fig.transFigure)

        # Save
        fig_path = os.path.join(output_path, f"cluster_{cluster}_top_genes_ridge.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Ridge plot saved for Cluster {cluster} to {fig_path}")

def compute_relevance_scores(model, graph, features, node_indices=None, use_abs=True):
    """
    Computes gradient-based relevance scores (saliency) for selected nodes using sigmoid probabilities.

    Args:
        model: Trained GNN model
        graph: DGL graph
        features: Input node features
        node_indices: List/Tensor of node indices to compute relevance for. If None, auto-select using probs > 0.5
        use_abs: Whether to use absolute gradients (recommended for visualization)

    Returns:
        relevance_scores: Tensor of shape [num_nodes, num_features] (0s for nodes not analyzed)
    """
    model.eval()
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    features = features.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        logits = model(graph, features)
        probs = torch.sigmoid(logits.squeeze())

        # Auto-select nodes (e.g., predicted cancer genes)
        if node_indices is None:
            node_indices = torch.nonzero(probs > 0.5, as_tuple=False).squeeze()
            if node_indices.ndim == 0:
                node_indices = node_indices.unsqueeze(0)

        relevance_scores = torch.zeros_like(features)

        for i, idx in enumerate(node_indices):
            model.zero_grad()
            if features.grad is not None:
                features.grad.zero_()

            probs[idx].backward(retain_graph=(i != len(node_indices) - 1))

            grads = features.grad[idx]
            relevance_scores[idx] = grads.abs().detach() if use_abs else grads.detach()

    return relevance_scores

def extract_summary_features_np_bio(bio_embeddings_np):
    """
    Extracts summary features from just the 1024 biological features (bio only).

    Args:
        bio_embeddings_np (np.ndarray): shape [num_nodes, 1024]

    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    num_nodes, num_features = bio_embeddings_np.shape
    summary_features = []

    assert num_features == 1024, f"Expected 1024 bio features, got {num_features}"

    for o_idx in range(4):  # 4 omics types
        for c_idx in range(16):  # 16 cancer types
            base = o_idx * 16 * 16 + c_idx * 16
            group = bio_embeddings_np[:, base:base + 16]  # [num_nodes, 16]
            max_vals = group.max(axis=1, keepdims=True)
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)

def extract_summary_features_np_topo(topo_features_np):
    """
    Extracts summary features from the topological embedding section (features 1024–2047)
    by computing the max over each 16-dimensional segment.

    Args:
        features_np (np.ndarray): shape [num_nodes, 2048]

    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    num_nodes, num_features = topo_features_np.shape
    assert num_features == 1024, f"Expected 2048 features, got {num_features}"

    # Select topological features only
    ##topo_features = features_np[:, 1024:]  # shape: [num_nodes, 1024]
    ##topo_features = topo_features_np[:, 1024:2048]
    topo_features = topo_features_np  # already 1024 features


    summary_features = []

    # Pool over 64 chunks of 16 features
    for i in range(64):
        start = i * 16
        end = start + 16
        group = topo_features[:, start:end]  # shape: [num_nodes, 16]
        max_vals = group.max(axis=1, keepdims=True)  # shape: [num_nodes, 1]
        summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)  # shape: [num_nodes, 64]

def extract_specific_omics_cancer(bio_embeddings_np, omics_target='mf', cancer_target='BRCA'):
    """
    Extracts 16 features for a specific omics-cancer pair from the 1024 bio features.

    Args:
        bio_embeddings_np (np.ndarray): shape [num_nodes, 1024]
        omics_target (str): one of ['cna', 'ge', 'meth', 'mf']
        cancer_target (str): one of 16 cancer types like 'BRCA'

    Returns:
        np.ndarray: shape [num_nodes, 16]
    """
    omics_types = ['cna', 'ge', 'meth', 'mf']
    cancer_types = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD',
                    'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC']

    o_idx = omics_types.index(omics_target)
    c_idx = cancer_types.index(cancer_target)

    start = o_idx * 16 * 16 + c_idx * 16
    end = start + 16

    return bio_embeddings_np[:, start:end]

def plot_top_gene_ridge_from_bio_features(
    bio_embeddings_np,           # shape [num_genes, 1024]
    node_names,                # list of gene names
    cluster_labels,            # shape [num_genes]
    output_path, 
    top_n=12
):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    print("🔍 Extracting 64 summary features from 1024 bio features...")
    summary_features = extract_summary_features_np_bio(bio_embeddings_np)  # [num_genes, 64]

    # Prepare DataFrame
    df = pd.DataFrame(summary_features)
    df["Gene"] = node_names
    df["Cluster"] = cluster_labels
    df["MeanFeatureValue"] = df.iloc[:, :-2].mean(axis=1)

    os.makedirs(output_path, exist_ok=True)
    clusters = sorted(df["Cluster"].unique())

    for cluster in clusters:
        cluster_df = df[df["Cluster"] == cluster].copy()
        top_genes = cluster_df.nlargest(top_n, "MeanFeatureValue")

        # Prepare data for plotting
        plot_data = []
        for _, row in top_genes.iterrows():
            gene = row["Gene"]
            for i in range(64):
                plot_data.append({
                    "Gene": gene,
                    "FeatureIndex": i,
                    "Value": row[i]
                })

        plot_df = pd.DataFrame(plot_data)

        # Order genes by mean feature value
        gene_order = plot_df.groupby("Gene")["Value"].mean().sort_values().index
        plot_df["Gene"] = pd.Categorical(plot_df["Gene"], categories=gene_order, ordered=True)

        # Get global min/max for consistent x-axis
        xmin = plot_df["Value"].min() - 0.5
        xmax = plot_df["Value"].max() + 0.5

        # Pad each gene's feature values with xmin and xmax (adds negligible weight but stabilizes curve width)
        pad_df = []
        for gene in gene_order:
            pad_df.append({"Gene": gene, "FeatureIndex": -1, "Value": xmin})
            pad_df.append({"Gene": gene, "FeatureIndex": -1, "Value": xmax})

        plot_df = pd.concat([plot_df, pd.DataFrame(pad_df)], ignore_index=True)

        # Plot ridges
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(
            plot_df,
            row="Gene",
            hue="Gene",
            aspect=12,
            height=0.5,
            palette="Spectral",
            sharex=True
        )

        g.set(xlim=(xmin, xmax))
        
        g.map(sns.kdeplot, "Value", bw_adjust=0.6, fill=True, alpha=0.8)
        g.map(sns.kdeplot, "Value", bw_adjust=0.6, color="black", lw=1)

        # Remove internal titles
        g.set_titles("")

        # Set same x-axis range for all rows
        g.set(xlim=(xmin, xmax), xlabel="Feature Value (Max across 16 per Cancer)", ylabel="", yticks=[])

        # Set gene names as y-axis labels
        for ax, gene in zip(g.axes.flat, gene_order):
            ax.set_ylabel(gene, rotation=0, ha='right', va='center', fontsize=12, labelpad=10)

        # Remove spines and adjust layout
        g.despine(bottom=True, left=True)
        g.fig.subplots_adjust(hspace=0.3, left=0.3, top=0.93)  # top=0.95 leaves space for title
        g.fig.suptitle(f"Cluster {cluster}", x=0.55, y=0.995, fontsize=18, ha='center')


        # Save plot
        out_path = os.path.join(output_path, f"cluster_{cluster}_top{top_n}_genes_ridge.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Ridge plot saved: {out_path}")

def plot_all_omics_cancer_ridges(
    bio_embeddings_np,
    node_names,
    cluster_labels,
    output_base_path,
    top_n=12
):
    omics_list = ['cna', 'ge', 'meth', 'mf']
    ##cancer_list = ['BLCA', 'BRCA', 'LUAD', 'LUSC']
    cancer_list = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP',
                   'LIHC', 'LUAD', 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC']

    for omics in omics_list:
        for cancer in cancer_list:
            print(f"\n📊 Plotting ridge for {omics.upper()}-{cancer}...")
            subfolder = f"{omics.upper()}_{cancer}"
            output_path = os.path.join(output_base_path, subfolder)
            plot_top_gene_ridge_from_specific_omics(
                bio_embeddings_np=bio_embeddings_np,
                node_names=node_names,
                cluster_labels=cluster_labels,
                output_path=output_path,
                omics_target=omics,
                cancer_target=cancer,
                top_n=top_n
            )

def plot_top_gene_ridge_from_specific_omics(
    bio_embeddings_np,
    node_names,
    cluster_labels,
    output_path,
    omics_target='mf',
    cancer_target='BRCA',
    top_n=12
):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    # 1. Extract 16-dim features for the selected omics-cancer pair
    features_16 = extract_specific_omics_cancer(bio_embeddings_np, omics_target, cancer_target)

    # 2. Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # 3. Build DataFrame
    df = pd.DataFrame(features_16)
    df["Gene"] = node_names
    df["Cluster"] = cluster_labels
    df["MeanFeatureValue"] = df.iloc[:, :-2].mean(axis=1)

    clusters = sorted(df["Cluster"].unique())

    for cluster in clusters:
        cluster_df = df[df["Cluster"] == cluster].copy()
        top_genes = cluster_df.nlargest(top_n, "MeanFeatureValue")

        # Prepare melted long-form data for seaborn
        plot_data = []
        for _, row in top_genes.iterrows():
            gene = row["Gene"]
            for i in range(16):
                plot_data.append({
                    "Gene": gene,
                    "FeatureIndex": i,
                    "Value": row[i]
                })
        plot_df = pd.DataFrame(plot_data)

        # Sort gene order
        gene_order = plot_df.groupby("Gene")["Value"].mean().sort_values().index
        plot_df["Gene"] = pd.Categorical(plot_df["Gene"], categories=gene_order, ordered=True)

        # Compute proper x-axis limits with padding
        xmin = plot_df["Value"].min()
        xmax = plot_df["Value"].max()
        x_range = xmax - xmin
        xmin -= x_range * 0.5
        xmax += x_range * 0.5


        # Plot with ridge style
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(
            plot_df,
            row="Gene",
            hue="Gene",
            aspect=12,
            height=0.4,
            palette="Spectral",
            sharex=True
        )

        g.map(
            sns.kdeplot,
            "Value",
            bw_adjust=0.5,
            fill=True,
            alpha=0.8,
            cut=100,
            clip=(xmin, xmax)
        )
        g.map(
            sns.kdeplot,
            "Value",
            bw_adjust=0.5,
            color="black",
            lw=1,
            cut=100,
            clip=(xmin, xmax)
        )


        g.set_titles("")
        g.set(xlim=(xmin, xmax), xlabel="Feature Value", ylabel="", yticks=[])

        # Label genes on the left side
        for ax, gene in zip(g.axes.flat, gene_order):
            ax.set_ylabel(gene, rotation=0, ha='right', va='top', fontsize=18, labelpad=10)

        g.despine(bottom=True, left=True)
        g.fig.subplots_adjust(hspace=-0.3, left=0.3, right=0.95, top=0.93)
        g.fig.suptitle(f"Cluster {cluster}", x=0.6, fontsize=20)

        # After g.set(...) and before plt.savefig(...)
        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelsize=16)  # or 14, 16, etc.

        # Save
        out_path = os.path.join(
            output_path,
            f"{omics_target}_{cancer_target}_cluster{cluster}_top{top_n}_genes_ridge.png"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {out_path}")

def plot_gene_feature_contributions_bio(gene_name, relevance_vector, feature_names, score, output_path=None):
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    # Barplot of all 64 features
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    barplot_path = output_path.replace(".png", "_omics_barplot.png") if output_path else None
    plot_omics_barplot_bio(df, barplot_path)

    # Prepare for heatmap
    df[['Omics', 'Cancer']] = df['Feature'].str.split(':', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    heatmap_data = df.pivot(index='Cancer', columns='Omics', values='Relevance')
    heatmap_data = heatmap_data[['cna', 'ge', 'meth', 'mf']]  # Ensure column order

    # Plot vertical heatmap (Cancers as rows)
    plt.figure(figsize=(2.0, 5.0))
    # Capitalize omics column labels
    heatmap_data.columns = [col.upper() for col in heatmap_data.columns]
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar=False, linewidths=0.3, linecolor='gray')

    # Handle gene name and score
    if isinstance(score, np.ndarray):
        score = score.item()
    plt.title(f"{gene_name}", fontsize=12)

    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.xlabel('')
    plt.ylabel('')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_gene_feature_contributions_topo(gene_name, relevance_vector, feature_names, score, output_path=None):
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    # Barplot of all 64 topo features
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    barplot_path = output_path.replace(".png", "_omics_barplot.png") if output_path else None
    plot_omics_barplot_topo(df, barplot_path)

    # Prepare for heatmap
    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)
    df['Omics'] = df['Omics'].str.lower()

    heatmap_data = df.pivot(index='Cancer', columns='Omics', values='Relevance')
    heatmap_data = heatmap_data[['cna', 'ge', 'meth', 'mf']]  # Ensure column order

    # Plot vertical heatmap (Cancers as rows)
    plt.figure(figsize=(2.0, 5.0))
    # Capitalize omics column labels
    heatmap_data.columns = [col.upper() for col in heatmap_data.columns]
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar=False, linewidths=0.3, linecolor='gray')

    if isinstance(score, np.ndarray):
        score = score.item()
    plt.title(f"{gene_name}", fontsize=12)

    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.xlabel('')
    plt.ylabel('')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_and_plot_confirmed_genes_bio(
    args,
    node_names_topk,
    node_scores_topk,
    summary_feature_relevance,
    output_dir,
    confirmed_genes_save_path,
    cluster_labels_topk,
    tag="bio",
    confirmed_gene_path="data/ncg_8886.txt"):
    """
    Finds confirmed cancer genes and plots their biological feature contributions.
    """

    
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]

    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{omics}:{cancer}" for omics in omics_order for cancer in cancer_names]

    with open(confirmed_gene_path) as f:
        known_cancer_genes = set(line.strip() for line in f if line.strip())

    confirmed_genes = [g for g in node_names_topk if g in known_cancer_genes]

    with open(confirmed_genes_save_path, "w") as f:
        for gene in confirmed_genes:
            f.write(f"{gene}\n")

    plot_dir = os.path.join(output_dir, f"{tag}_confirmed_feature_contributions")
    os.makedirs(plot_dir, exist_ok=True)

    def get_scalar_score(score):
        if isinstance(score, np.ndarray):
            return score.item() if score.size == 1 else score[0]
        return float(score)

    for gene_name in confirmed_genes:
        idx = node_names_topk.index(gene_name)
        relevance_vector = summary_feature_relevance[idx]
        score = get_scalar_score(node_scores_topk[idx])
        plot_path = os.path.join(
            plot_dir,
            f"{args.model_type}_{args.net_type}_{gene_name}_{tag}_confirmed_feature_contributions_epo{args.num_epochs}.png"
        )
        cluster_id = cluster_labels_topk[idx].item()
        plot_gene_feature_contributions_bio(
            gene_name=f"{gene_name} (Cluster {cluster_id})",  # 👈 Title includes cluster
            relevance_vector=relevance_vector,
            feature_names=feature_names,
            output_path=plot_path,
            score=score
        )

def save_and_plot_confirmed_genes_topo(
    args,
    node_names_topk,
    node_scores_topk,
    summary_feature_relevance,
    output_dir,
    confirmed_genes_save_path,
    cluster_labels_topk,
    tag="topo",
    confirmed_gene_path="data/ncg_8886.txt"):
    """
    Finds confirmed cancer genes and plots their topological feature contributions.
    """

    cancer_names = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC']
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{cancer}_{omics}" for cancer in cancer_names for omics in omics_order]

    with open(confirmed_gene_path) as f:
        known_cancer_genes = set(line.strip() for line in f if line.strip())

    confirmed_genes = [g for g in node_names_topk if g in known_cancer_genes]

    with open(confirmed_genes_save_path, "w") as f:
        for gene in confirmed_genes:
            f.write(f"{gene}\n")

    ##summary_feature_relevance = extract_summary_features_np_topo(summary_feature_relevance)

    plot_dir = os.path.join(output_dir, f"{tag}_confirmed_feature_contributions")
    os.makedirs(plot_dir, exist_ok=True)

    def get_scalar_score(score):
        if isinstance(score, np.ndarray):
            return score.item() if score.size == 1 else score[0]
        return float(score)

    for gene_name in confirmed_genes:
        idx = node_names_topk.index(gene_name)
        relevance_vector = summary_feature_relevance[idx]
        score = get_scalar_score(node_scores_topk[idx])
        plot_path = os.path.join(
            plot_dir,
            f"{args.model_type}_{args.net_type}_{gene_name}_{tag}_confirmed_feature_contributions_epo{args.num_epochs}.png"
        )
        cluster_id = cluster_labels_topk[idx].item()
        plot_gene_feature_contributions_topo(
            #gene_name=gene_name,
            gene_name=f"{gene_name} (Cluster {cluster_id})",
            relevance_vector=relevance_vector,
            feature_names=feature_names,
            output_path=plot_path,
            score=score
        )

def plot_top_gene_ridge_across_omics(
    bio_embeddings_np,
    node_names,
    cluster_labels,
    output_path,
    cancer_target='BRCA',
    top_n=12
):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_path, exist_ok=True)

    # 1. Extract all 4 omics (64 features) for the cancer
    features_64 = extract_all_omics_for_cancer(bio_embeddings_np, cancer_target)

    # 2. Prepare DataFrame
    df = pd.DataFrame(features_64)
    df["Gene"] = node_names
    df["Cluster"] = cluster_labels
    df["MeanFeatureValue"] = df.iloc[:, :-2].mean(axis=1)

    clusters = sorted(df["Cluster"].unique())

    omics_types = ['cna', 'ge', 'meth', 'mf']
    omics_labels = []
    for omics in omics_types:
        omics_labels += [omics] * 16

    for cluster in clusters:
        cluster_df = df[df["Cluster"] == cluster].copy()
        top_genes = cluster_df.nlargest(top_n, "MeanFeatureValue")

        # Melt into long-form with omics and feature index
        plot_data = []
        for _, row in top_genes.iterrows():
            gene = row["Gene"]
            for i in range(64):
                plot_data.append({
                    "Gene": gene,
                    "FeatureIndex": i % 16,
                    "Omics": omics_labels[i],
                    "Value": row[i]
                })
        plot_df = pd.DataFrame(plot_data)

        # Sort genes
        gene_order = plot_df.groupby("Gene")["Value"].mean().sort_values().index
        plot_df["Gene"] = pd.Categorical(plot_df["Gene"], categories=gene_order, ordered=True)

        # X-axis limits
        xmin = plot_df["Value"].min()
        xmax = plot_df["Value"].max()
        x_range = xmax - xmin
        xmin -= x_range * 0.5
        xmax += x_range * 0.5

        # Plot
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(
            plot_df,
            row="Gene",
            hue="Omics",  # Color by omics
            aspect=12,
            height=0.4,
            palette="tab10",
            sharex=True
        )

        g.map(
            sns.kdeplot,
            "Value",
            bw_adjust=0.5,
            fill=True,
            alpha=0.8,
            cut=100,
            clip=(xmin, xmax)
        )
        g.map(
            sns.kdeplot,
            "Value",
            bw_adjust=0.5,
            color="black",
            lw=1,
            cut=100,
            clip=(xmin, xmax)
        )

        g.set_titles("")
        g.set(xlim=(xmin, xmax), xlabel="Feature Value", ylabel="", yticks=[])

        for ax, gene in zip(g.axes.flat, gene_order):
            ax.set_ylabel(gene, rotation=0, ha='right', va='top', fontsize=18, labelpad=10)

        g.despine(bottom=True, left=True)
        g.fig.subplots_adjust(hspace=-0.3, left=0.3, right=0.95, top=0.93)
        g.fig.suptitle(f"Cluster {cluster} — {cancer_target} (All Omics)", x=0.6, fontsize=20)

        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelsize=16)

        # Save
        out_path = os.path.join(output_path, f"{cancer_target}_allomics_cluster{cluster}_top{top_n}_genes_ridge.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {out_path}")

def run_gprofiler_enrichment(cluster_dict, cancer_type, tag):
    gp = GProfiler(return_dataframe=True)
    output_files = []
    for cluster, genes in cluster_dict.items():
        try:
            result = gp.profile(
                organism="hsapiens",
                query=genes,
                sources=["REAC", "KEGG", "GO:BP", "HP"],
                user_threshold=0.05,
                significance_threshold_method="fdr"
            )
            if not result.empty:
                for source in ["REAC", "KEGG", "GO:BP", "HP"]:
                    filtered = result[result["source"] == source]
                    path = f"results/gene_prediction/enrichment/{cancer_type}_{tag}_Cluster_{cluster}_{source}_enrichment.csv"
                    dir_path = os.path.dirname(path)
                    os.makedirs(dir_path, exist_ok=True)

                    filtered.to_csv(path, index=False)
                    output_files.append(path)
        except Exception as e:
            print(f"Enrichment failed for {cancer_type} {cluster}: {e}")
    return output_files

def plot_dot_enrichment_per_cluster(cancer_type, tag, source="REAC", top_n=10):
    files = glob.glob(f"results/gene_prediction/{cancer_type}_{tag}_Cluster_*_{source}_enrichment.csv")
    if not files:
        print(f"No enrichment files found for {cancer_type.upper()} [{tag}] and source {source}")
        return

    for f in files:
        try:
            cluster = Path(f).stem.split("_")[2]  # Cluster number
        except IndexError:
            print(f"Filename parsing failed: {f}")
            continue

        df = pd.read_csv(f)
        if df.empty or "intersection_size" not in df or "query_size" not in df:
            print(f"Invalid dataframe for {f}")
            continue

        df["gene_ratio"] = df["intersection_size"] / df["query_size"]
        df["-log10(FDR)"] = -np.log10(df["p_value"].clip(lower=1e-300))
        top_df = df.sort_values("p_value").head(top_n)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=top_df,
            x="gene_ratio",
            y="name",
            hue="-log10(FDR)",
            size="-log10(FDR)",
            sizes=(40, 200),
            palette="viridis",
            legend="brief"
        )
        plt.title(f"{source} Enrichment for {cancer_type.upper()} [{tag}] - Cluster {cluster}")
        plt.xlabel("Gene Ratio")
        plt.ylabel("Pathway")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"results/gene_prediction/{cancer_type}_{tag}_Cluster_{cluster}_{source}_dotplot.png", dpi=300)
        plt.close()

def plot_dot_enrichment_from_results(enrichment_results, top_n=10, source_filter=["REAC", "KEGG", "GO:BP", "HP"]):
    """
    Plot dot plots for enrichment results stored in memory.

    Args:
        enrichment_results (dict): Nested dictionary like {'bio': {0: df, 1: df, ...}, 'topo': {0: df, ...}}.
        top_n (int): Number of top enriched terms to display.
        source_filter (list): List of enrichment sources to include.
    """
    for cluster_type, cluster_data in enrichment_results.items():
        for cluster_id, df in cluster_data.items():
            if df.empty or "p_value" not in df or "intersection_size" not in df or "query_size" not in df:
                print(f"Skipping {cluster_type} cluster {cluster_id}: invalid or empty data")
                continue

            # Optional: filter by source
            if source_filter:
                df = df[df["source"].isin(source_filter)]

            # Compute additional metrics
            df["gene_ratio"] = df["intersection_size"] / df["query_size"]
            df["-log10(FDR)"] = -np.log10(df["p_value"].clip(lower=1e-300))
            df_plot = df.sort_values("p_value").head(top_n)

            if df_plot.empty:
                print(f"No significant enrichment to plot for {cluster_type} cluster {cluster_id}")
                continue

            # Plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=df_plot,
                x="gene_ratio",
                y="name",
                hue="-log10(FDR)",
                size="-log10(FDR)",
                sizes=(40, 200),
                palette="viridis",
                legend="brief"
            )
            plt.title(f"{cluster_type.capitalize()} Cluster {cluster_id} Enrichment")
            plt.xlabel("Gene Ratio")
            plt.ylabel("Pathway")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Save plot
            out_path = f"results/enrichment_dotplots/{cluster_type}_cluster_{cluster_id}_dotplot.png"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=300)
            plt.show()
            print(f"Saved: {out_path}")

def plot_dot_enrichment_per_cluster_all(
    cancer_types,
    tag="bio",
    source="REAC",
    top_n=10,
    n_clusters=10,
    base_path="results/gene_prediction/enrichment"
):
    base_path = Path(base_path)

    for cancer_type in cancer_types:
        print(f"\n🔍 Processing {cancer_type.upper()} [{tag}] enrichment dotplots...")
        
        for cluster_id in range(n_clusters):
            file_path = base_path / f"{cancer_type}_{tag}_Cluster_{cluster_id}_{source}_enrichment.csv"

            if not file_path.exists():
                print(f"  ✗ Missing: {file_path.name}")
                continue

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"  ✗ Error reading {file_path.name}: {e}")
                continue

            if df.empty or "intersection_size" not in df or "query_size" not in df:
                print(f"  ✗ Invalid content: {file_path.name}")
                continue

            # Calculate gene ratio and transformed FDR
            df["gene_ratio"] = df["intersection_size"] / df["query_size"]
            df["-log10(FDR)"] = -np.log10(df["p_value"].clip(lower=1e-300))
            top_df = df.sort_values("p_value").head(top_n)

            # Plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=top_df,
                x="gene_ratio",
                y="name",
                hue="-log10(FDR)",
                size="-log10(FDR)",
                sizes=(40, 200),
                palette="viridis",
                legend="brief"
            )
            plt.title(f"{source} Enrichment: {cancer_type.upper()} [{tag}] - Cluster {cluster_id}")
            plt.xlabel("Gene Ratio")
            plt.ylabel("Pathway")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Save plot
            out_path = base_path / f"{cancer_type}_{tag}_Cluster_{cluster_id}_{source}_dotplot.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"  ✓ Saved: {out_path.name}")

def apply_full_spectral_biclustering_cancer(cancer_feature, n_clusters):
    from sklearn.cluster import SpectralBiclustering
    import numpy as np

    print(f"Running Spectral Biclustering on 64-dim summary bio features with {n_clusters} clusters...")

    assert cancer_feature.shape[1] == 64, f"Expected 64 summary features, got {cancer_feature.shape[1]}"

    # Perform spectral biclustering
    bicluster = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state31)
    bicluster.fit(cancer_feature)

    row_labels = bicluster.row_labels_
    print("✅ Spectral Biclustering complete.")

    return row_labels

def plot_all_cancer_ridges_all_omics(
    bio_embeddings_np,
    node_names,
    #cluster_labels,
    best_k,
    output_base_path,
    top_n=12
):
    cancer_list = [
        'BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP',
        'LIHC', 'LUAD', 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC'
    ]

    for cancer in cancer_list:
        print(f"\n📊 Plotting ridge for ALL_OMICS — {cancer}...")
        output_path = os.path.join(output_base_path, f"ALL_OMICS_{cancer}")
        plot_top_gene_ridge_from_all_omics(
            bio_embeddings_np=bio_embeddings_np,
            node_names=node_names,
            #cluster_labels=cluster_labels,
            best_k=best_k,
            output_path=output_path,
            cancer_target=cancer,
            top_n=top_n
        )

def plot_top_gene_ridge_from_all_omics(
    bio_embeddings_np,
    node_names,
    #cluster_labels,
    best_k,
    output_path,
    cancer_target='BRCA',
    top_n=12
):
    # 1. Extract 64-dim features
    #features_64 = extract_all_omics_for_cancer(bio_embeddings_np, cancer_target)
    cancer_feature = extract_all_omics_for_cancer(bio_embeddings_np, cancer_target)
    row_labels = apply_full_spectral_biclustering_cancer(cancer_feature, n_clusters=best_k)
    
    # 2. Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # 3. Build DataFrame
    df = pd.DataFrame(cancer_feature)
    df["Gene"] = node_names
    df["Cluster"] = row_labels
    df["MeanFeatureValue"] = df.iloc[:, :-2].mean(axis=1)

    clusters = sorted(df["Cluster"].unique())
    cluster_dict = defaultdict(list)  # Save top genes for each cluster

    for cluster in clusters:
        cluster_df = df[df["Cluster"] == cluster].copy()
        top_genes = cluster_df.nlargest(top_n, "MeanFeatureValue")
        cluster_dict[cluster] = top_genes["Gene"].tolist()

        # Prepare long-form data for seaborn
        plot_data = []
        for _, row in top_genes.iterrows():
            gene = row["Gene"]
            for i in range(64):
                plot_data.append({
                    "Gene": gene,
                    "FeatureIndex": i,
                    "Value": row[i]
                })
        plot_df = pd.DataFrame(plot_data)

        gene_order = plot_df.groupby("Gene")["Value"].mean().sort_values().index
        plot_df["Gene"] = pd.Categorical(plot_df["Gene"], categories=gene_order, ordered=True)

        xmin = plot_df["Value"].min()
        xmax = plot_df["Value"].max()
        x_range = xmax - xmin
        xmin -= x_range * 0.5
        xmax += x_range * 0.5

        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(
            plot_df,
            row="Gene",
            hue="Gene",
            aspect=12,
            height=0.4,
            palette="Spectral",
            sharex=True
        )

        g.map(sns.kdeplot, "Value", bw_adjust=0.5, fill=True, alpha=0.8, cut=100, clip=(xmin, xmax))
        g.map(sns.kdeplot, "Value", bw_adjust=0.5, color="black", lw=1, cut=100, clip=(xmin, xmax))

        g.set_titles("")
        g.set(xlim=(xmin, xmax), xlabel="Feature Value", ylabel="", yticks=[])

        for ax, gene in zip(g.axes.flat, gene_order):
            ax.set_ylabel(gene, rotation=0, ha='right', va='top', fontsize=18, labelpad=10)

        g.despine(bottom=True, left=True)
        g.fig.subplots_adjust(hspace=-0.3, left=0.3, right=0.95, top=0.93)
        g.fig.suptitle(f"Cluster {cluster}", x=0.6, fontsize=20)

        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelsize=16)

        out_path = os.path.join(output_path, f"ALL_OMICS_{cancer_target}_cluster{cluster}_top{top_n}_genes_ridge.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {out_path}")

    # Save cluster_dict as JSON
    '''cluster_json_path = os.path.join(output_path, f"ALL_OMICS_{cancer_target}_cluster_genes.json")

    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert NumPy scalars to native Python types
        else:
            return obj

    cluster_dict_safe = make_json_safe(cluster_dict)

    with open(cluster_json_path, "w") as f:
        json.dump(cluster_dict_safe, f, indent=2)

    print(f"💾 Saved cluster gene dictionary: {cluster_json_path}")'''

def extract_all_omics_for_cancer(bio_embeddings_np, cancer_target='BRCA'):
    """
    Extracts 64 features (4 omics × 16 features) for a specific cancer type from the 1024 bio features.

    Args:
        bio_embeddings_np (np.ndarray): shape [num_nodes, 1024]
        cancer_target (str): one of 16 cancer types like 'BRCA'

    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    omics_types = ['cna', 'ge', 'meth', 'mf']
    cancer_types = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP',
                    'LIHC', 'LUAD', 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC']
    
    c_idx = cancer_types.index(cancer_target)

    feature_blocks = []
    for o_idx in range(len(omics_types)):
        start = o_idx * 16 * 16 + c_idx * 16
        end = start + 16
        feature_blocks.append(bio_embeddings_np[:, start:end])

    return np.concatenate(feature_blocks, axis=1)  # shape: [num_nodes, 64]

def make_cluster_dict(row_labels, node_names):
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(row_labels):
        cluster_dict[label].append(node_names[idx])
    return cluster_dict

def collect_top_enrichments(cancer_type, tag="bio", source="REAC", top_n=10):
    base_path = Path("results/gene_prediction/enrichment")
    base_path.mkdir(parents=True, exist_ok=True)
    terms = []

    for cluster_id in range(10):  # Clusters 0–9
        file = base_path / f"{cancer_type}_{tag}_Cluster_{cluster_id}_{source}_enrichment.csv"
        if not file.exists():
            continue

        df = pd.read_csv(file).sort_values("p_value").head(top_n)
        for _, row in df.iterrows():
            term = f"{row['name']} (C{cluster_id})"
            score = -np.log10(row["p_value"] + 1e-10)
            terms.append((term, score, cluster_id))

    return sorted(terms, key=lambda x: x[1], reverse=True)

def draw_horizontal_bar_plot(terms, cancer_type, source):
    if not terms:
        print(f"No enrichment terms for {cancer_type.upper()} ({source})")
        return

    # Sort and select top 20 terms by score
    terms_sorted = sorted(terms, key=lambda x: x[1], reverse=True)[:20]

    labels = [t[0] for t in terms_sorted]
    scores = [t[1] for t in terms_sorted]
    clusters = [t[2] for t in terms_sorted]
    colors = [CLUSTER_COLORS[c] for c in clusters]

    fig, ax = plt.subplots(figsize=(12, 0.5 * len(terms_sorted)))
    y_pos = np.arange(len(terms_sorted))

    ax.barh(y_pos, scores, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=18)  # Larger font size
    ax.invert_yaxis()  # Highest scores on top
    ax.set_xlabel("-log10(p-value)", fontsize=18)
    ax.set_title(f"Top Enriched Pathways — {cancer_type.upper()} ({source})", fontsize=18)
    ax.tick_params(axis='x', labelsize=16) 
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1) 

    out_path = Path(f"results/gene_prediction/{cancer_type}_{source}_bar_plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved bar plot: {out_path}")

def collect_enrichment_with_ratios(cancer_type, tag="bio", source="REAC", top_n=10):
    from pathlib import Path
    import numpy as np
    import pandas as pd

    base_path = Path("results/gene_prediction/enrichment")
    base_path.mkdir(parents=True, exist_ok=True)
    terms = []

    for cluster_id in range(10):  # Clusters 0–9
        file = base_path / f"{cancer_type}_{tag}_Cluster_{cluster_id}_{source}_enrichment.csv"
        if not file.exists():
            continue

        df = pd.read_csv(file).sort_values("p_value").head(top_n)
        for _, row in df.iterrows():
            raw_name = row['name']
            # Truncate long names
            if len(raw_name) <= 60:
                term = raw_name#[:50] + "..." if len(raw_name) > 50 else raw_name
            pval = row['p_value']
            intersection = row['intersection_size']
            input_size = row.get('effective_domain_size', 1)  # Prevent zero division
            gene_ratio = intersection / input_size if input_size > 0 else 0
            terms.append((term, gene_ratio, -np.log10(pval + 1e-10), cluster_id))

    return sorted(terms, key=lambda x: x[2], reverse=True)

def draw_dot_plot_with_ratio(terms, cancer_type, source, top_n=20):
    if not terms:
        print(f"No enrichment terms for {cancer_type.upper()} ({source})")
        return

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(terms, columns=["Term", "GeneRatio", "LogP", "Cluster"])
    df["Color"] = df["Cluster"].map(CLUSTER_COLORS)

    # 🔢 Select top N by LogP
    df = df.sort_values("LogP", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 0.4 * len(df)))
    scatter = sns.scatterplot(
        data=df,
        x="GeneRatio", y="Term",
        size="LogP", hue="Cluster",
        palette=CLUSTER_COLORS,
        sizes=(50, 300),
        edgecolor="black",
        linewidth=0.5
    )

    # Remove legend
    if scatter.legend_:
        scatter.legend_.remove()
        
    #plt.xlabel("Gene Ratio", fontsize=16)
    plt.ylabel("Enriched Pathway", fontsize=18)
    plt.title(f"Dot Plot (Ratio) — {cancer_type.upper()} ({source})", fontsize=18)
    plt.xscale("log")
    plt.xticks(fontsize=18)
    plt.xlabel("Gene Ratio (log scale)", fontsize=18)

    plt.yticks(fontsize=18)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = Path(f"results/gene_prediction/enrichment/{cancer_type}_{source}_dot_plot_ratio.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved dot plot with gene ratio: {out_path}")

def eigengap_analysis(feature_matrix, max_clusters=25, normalize=True, plot_path=None):
    # Step 1: Normalize features (optional)
    '''if normalize:
        from sklearn.preprocessing import StandardScaler
        feature_matrix = StandardScaler().fit_transform(feature_matrix)'''

    # Step 2: Similarity matrix (RBF kernel)
    similarity = rbf_kernel(feature_matrix, gamma=0.5)

    # Step 3: Compute Laplacian
    L, d = laplacian(similarity, normed=True, return_diag=True)

    # Step 4: Compute eigenvalues
    eigenvals, _ = eigh(L)

    # Print eigenvalues
    print("Eigenvalues:\n", eigenvals)

    # Save eigenvalues to CSV
    if plot_path:
        csv_path = os.path.splitext(plot_path)[0] + "_eigenvalues.csv"
        pd.DataFrame({"Eigenvalue Index": np.arange(len(eigenvals)), "Eigenvalue": eigenvals}).to_csv(csv_path, index=False)
        print(f"Eigenvalues saved to: {csv_path}")

    # Step 5: Optional plot
    if plot_path:
        x_vals = range(1, max_clusters + 1)
        y_vals = eigenvals[1:max_clusters + 1]
        
        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, color='blue', linestyle='-', label='Eigenvalues')
        plt.plot(x_vals, y_vals, color='red', marker='+', linestyle='None')

        gaps = np.diff(eigenvals[1:max_clusters + 1])
        best_k = np.argmax(gaps) + 1  # +1 because diff shifts index

        plt.axvline(
            x=best_k,
            color='pink',
            linestyle='--',
            label=f'Eigengap → k={best_k}'
        )

        plt.xlabel("Eigenvalue Index", fontsize=12)
        plt.ylabel("Eigenvalue", fontsize=12)
        plt.title("Eigengap Analysis", fontsize=14)

        # Set integer x-axis ticks
        plt.xticks(range(1, max_clusters + 1))

        # Remove grid
        plt.grid(False)

        # Remove legend frame
        legend = plt.legend()
        legend.get_frame().set_linewidth(0.0)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()
    else:
        # If no plot is saved, still compute best_k
        gaps = np.diff(eigenvals[1:max_clusters + 1])
        best_k = np.argmax(gaps) + 1

    return best_k, eigenvals

def plot_bio_clusterwise_feature_contributions(
    args,
    relevance_scores,           # 2D array (samples x features)
    cluster_labels,             # 1D array of cluster assignments
    feature_names,              # List of feature names (e.g., MF: BRCA, ...)
    per_cluster_feature_contributions_output_dir, 
    omics_colors):             # Dict of omics type colors (e.g., 'mf': '#D62728')
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    os.makedirs(per_cluster_feature_contributions_output_dir, exist_ok=True)

    def get_omics_color(feature_name):
        prefix = feature_name.split(":")[0].lower()
        return omics_colors.get(prefix, "#AAAAAA")

    def get_omics_prefix(feature_name):
        return feature_name.split(":")[0].lower()

    # Group features by omics type and preserve their indices
    omics_groups = defaultdict(list)
    for idx, fname in enumerate(feature_names):
        omics_groups[get_omics_prefix(fname)].append((idx, fname))

    # Follow omics_colors ordering if possible
    ordered_features = []
    for omics in omics_colors:
        ordered_features.extend(omics_groups.get(omics, []))
    for omics in omics_groups:
        if omics not in omics_colors:
            ordered_features.extend(omics_groups[omics])

    ordered_indices = [idx for idx, _ in ordered_features]
    ordered_feature_names = [name for _, name in ordered_features]

    unique_clusters = np.unique(cluster_labels)

    for cluster_id in sorted(unique_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_scores = relevance_scores[indices]
        avg_contribution = np.mean(cluster_scores, axis=0)
        total_score = np.sum(avg_contribution)

        fig, ax = plt.subplots(figsize=(10, 2.5))

        x = np.linspace(0, 1, len(ordered_feature_names))
        bar_width = 1 / len(ordered_feature_names) * 0.95

        bars = ax.bar(
            x,
            avg_contribution[ordered_indices],
            width=bar_width,
            color=[get_omics_color(name) for name in ordered_feature_names],
            align='center'
        )

        ax.set_title(
            fr"Cluster {cluster_id} $\mathregular{{({len(indices)}\ genes,\ avg = {total_score:.2f})}}$",
            fontsize=14
        )

        clean_labels = [name.split(":")[1].strip() if ":" in name else name for name in ordered_feature_names]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels, rotation=90)

        for label, feature_name in zip(ax.get_xticklabels(), ordered_feature_names):
            label.set_color(get_omics_color(feature_name))

        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlim(-bar_width, 1 + bar_width)

        plt.tight_layout()
        save_path = os.path.join(
            per_cluster_feature_contributions_output_dir,
            f"{args.model_type}_{args.net_type}_BIO_cluster_{cluster_id}_feature_contributions_epo{args.num_epochs}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved BIO feature contribution barplot for Cluster {cluster_id} to {save_path}")

def save_graph_with_clusters(graph, save_path):
    torch.save({
        'edges': graph.edges(),
        'features': graph.ndata['feat'],
        'labels': graph.ndata.get('label', None),
        'cluster_bio_summary': graph.ndata['cluster_bio_summary']
    }, save_path)

def compute_total_genes_per_cluster(row_labels, n_clusters):
    return {i: np.sum(row_labels == i) for i in range(n_clusters)}

def compute_relevance_scores_norm(
    model,
    graph,
    features,
    node_indices,
    normalize=True,
    feature_groups=None):  # e.g., {"bio": (0, 1024), "topo": (1024, 2048)}):
    """
    Compute saliency-based relevance scores with optional normalization and feature group selection.

    Args:
        model (torch.nn.Module): Your trained GNN model.
        graph (DGLGraph): The graph structure.
        features (torch.Tensor): Node feature matrix (N x F).
        node_indices (list[int]): Node indices to compute relevance for.
        normalize (bool): Whether to normalize saliency per node.
        feature_groups (dict): Dict of feature group name → (start, end) slice indices.

    Returns:
        dict: node_idx → dict of {group_name: relevance_tensor} or "all": full saliency
    """
    model.eval()
    features = features.clone().detach().requires_grad_(True)

    output = model(graph, features)
    relevance_dict = {}

    for node_idx in node_indices:
        model.zero_grad()
        node_score = output[node_idx].squeeze()
        node_score.backward(retain_graph=True)

        saliency = features.grad[node_idx].detach().abs()  # (F,)
        
        if normalize:
            saliency = saliency / (saliency.sum() + 1e-9)

        if feature_groups:
            group_scores = {}
            for group_name, (start, end) in feature_groups.items():
                group_scores[group_name] = saliency[start:end]
            relevance_dict[node_idx] = group_scores
        else:
            relevance_dict[node_idx] = {"all": saliency}

        features.grad.zero_()

    return relevance_dict

def count_predicted_genes_per_cluster(row_labels, node_names, predicted_cancer_genes, n_clusters):
    pred_counts = {i: 0 for i in range(n_clusters)}
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    for idx in predicted_indices:
        if 0 <= idx < len(row_labels):
            pred_counts[row_labels[idx]] += 1
    return pred_counts, predicted_indices

def plot_bio_heatmap_unsort_no_legend_patches(summary_bio_features, row_labels, col_labels, predicted_indices, output_path):
    from matplotlib.colors import LinearSegmentedColormap
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    bio_feat_names_64 = [f"{omics}_{cancer}" for omics in ['cna', 'ge', 'meth', 'mf'] for cancer in cancer_names]
    topk = 1000
    predicted_indices = predicted_indices[:topk]
    summary_topk = summary_bio_features[predicted_indices]
    row_labels_topk = row_labels[predicted_indices]
    sorted_indices = np.argsort(row_labels_topk)
    sorted_matrix = summary_topk[sorted_indices, :]
    sorted_labels = row_labels_topk[sorted_indices]

    # === Heatmap layout with narrower cluster bar and no gap
    fig = plt.figure(figsize=(17, 17))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.25, 13.7], wspace=0)

    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[0, 1])

    # === Cluster color bar
    cluster_colors = [to_rgb(CLUSTER_COLORS[label]) for label in sorted_labels]
    cluster_colors_array = np.array(cluster_colors).reshape(-1, 1, 3)
    ax_cluster.imshow(cluster_colors_array, aspect='auto')
    ax_cluster.axis("off")

    cluster_counts = Counter(sorted_labels)
    start_idx = 0
    for cluster_id in sorted(cluster_counts):
        count = cluster_counts[cluster_id]
        mid_idx = start_idx + count // 2
        ax_cluster.text(-0.75, mid_idx, f'{count}', va='center', ha='right',
                        fontsize=14, fontweight='bold', color='black')  # larger cluster number
        start_idx += count

        # === Heatmap
        bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
        sns.heatmap(sorted_matrix, cmap=bluish_gray_gradient, center=0, vmin=-2, vmax=2,
                    ax=ax_heatmap, cbar=False)
        ax_heatmap.set_title("Heatmap of Summary Bio Features Sorted by Spectral Biclusters", fontsize=18)
        ax_heatmap.set_yticks([])

        xticks = np.arange(len(bio_feat_names_64)) + 0.5
        ax_heatmap.set_xticks(xticks)

        # Set only cancer names, big and bold
        cancer_labels_only = [name.split('_')[1] for name in bio_feat_names_64]
        ax_heatmap.set_xticklabels(cancer_labels_only, rotation=90, fontsize=18)#, weight='bold')

        # Color each tick label based on omics type
        omics_colors = {
            'cna': '#9370DB', 'ge': '#228B22', 'meth': '#00008B', 'mf': '#b22222',
        }
        for tick_label, name in zip(ax_heatmap.get_xticklabels(), bio_feat_names_64):
            omics = name.split('_')[0]
            tick_label.set_color(omics_colors.get(omics, 'black'))

        ax_heatmap.set_xlabel("Summary Bio Features (Grouped by Omics)", fontsize=18)
        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"✅ Summary bio heatmap saved to {output_path}")

def plot_tsne(features, row_labels, predicted_indices, n_clusters, output_path):
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1), random_state31)
    reduced_embeddings = tsne.fit_transform(features)
    plt.figure(figsize=(12, 10))
    for cluster_id in range(n_clusters):
        idx = np.where(row_labels == cluster_id)[0]
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1],
                    color=CLUSTER_COLORS.get(cluster_id, "#777777"),
                    edgecolor='k', s=100, alpha=0.8)
    for idx in predicted_indices:
        x, y = reduced_embeddings[idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=50, linewidths=2)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.title("t-SNE of Spectral Biclustering (Summary Bio Features)")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ t-SNE visualization saved to {output_path}")

def plot_bio_heatmap_raw(summary_bio_features, predicted_indices, save_path):
    """
    Plot raw (unclustered) heatmap of summary bio features for predicted genes,
    with omics bar and consistent styling.

    Parameters:
    - summary_bio_features: np.ndarray, shape (num_nodes, 64)
    - predicted_indices: list of indices for predicted cancer genes
    - save_path: path to save the heatmap PNG
    """
    print("📊 Plotting raw bio heatmap (unclustered)...")
    assert summary_bio_features.shape[1] == 64, "Expected 64 summary bio features."

    # === Select only predicted cancer genes
    data = summary_bio_features[predicted_indices]

    # === Omics group boundaries for 64-dim summary bio features
    omics_group_sizes = {
        "expression": 16,
        "methylation": 16,
        "mutation": 16,
        "copy_number": 16
    }
    omics_colors = {
        "expression": "#76D7C4",
        "methylation": "#F7DC6F",
        "mutation": "#F5B7B1",
        "copy_number": "#C39BD3"
    }

    # === Feature names
    feature_names = [f"{omics}_{i}" for omics, size in omics_group_sizes.items() for i in range(size)]
    omics_color_bar = []
    for group, size in omics_group_sizes.items():
        omics_color_bar.extend([omics_colors[group]] * size)

    # === Colormap
    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient",
        ["#F0F3F4", "#85929e"]
    )

    # === Plotting
    fig = plt.figure(figsize=(14, 10))
    grid_spec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.15, 0.85])
    ax_contrib = fig.add_subplot(grid_spec[0])
    ax_heatmap = fig.add_subplot(grid_spec[1])

    # === Column contribution bar
    contrib = data.sum(axis=0)
    ax_contrib.bar(np.arange(data.shape[1]), contrib, color="#85929e", width=1.0)
    ax_contrib.axis("off")

    # === Heatmap
    sns.heatmap(
        data,
        cmap=bluish_gray_gradient,
        ax=ax_heatmap,
        cbar_kws={"label": "Feature Relevance"},
        xticklabels=feature_names,
        yticklabels=[f"Gene{i}" for i in range(data.shape[0])],
        linewidths=0.2,
        linecolor='gray'
    )
    ax_heatmap.set_xlabel("Biological Features")
    ax_heatmap.set_ylabel("Predicted Cancer Genes")
    ax_heatmap.tick_params(axis='x', rotation=90)

    # === Omics group bar
    for x, color in enumerate(omics_color_bar):
        ax_heatmap.add_patch(plt.Rectangle((x, -1), 1, 0.5, color=color, transform=ax_heatmap.transData, clip_on=False))

    # === Omics legend
    handles = [Patch(facecolor=color, label=label) for label, color in omics_colors.items()]
    ax_heatmap.legend(
        handles=handles,
        title="Omics Group",
        loc='upper right',
        bbox_to_anchor=(1.15, 1.0),
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Raw bio heatmap saved to: {save_path}")

def plot_bio_heatmap_raw_unsorted(summary_bio_features, predicted_indices, output_path):
    """
    Plot unclustered raw heatmap of 64-dim summary bio features for top predicted genes.
    Features are grouped into 4 omics types x 16 cancers and are color-coded on x-axis.

    Parameters:
    - summary_bio_features: np.ndarray of shape (num_nodes, 64)
    - predicted_indices: list or array of top predicted gene indices
    - output_path: str, path to save the heatmap image
    """
    print("📊 Plotting raw bio heatmap without clustering...")

    # === Top-K predicted genes to show
    topk = 1000
    predicted_indices = predicted_indices[:topk]
    summary_topk = summary_bio_features[predicted_indices]

    # === Bio feature labels (4 omics × 16 cancers)
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    bio_feat_names_64 = [f"{omics}_{cancer}" for omics in ['cna', 'ge', 'meth', 'mf'] for cancer in cancer_names]

    # === Figure layout
    fig = plt.figure(figsize=(17, 18))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[17, 2], width_ratios=[0.25, 13.7], hspace=0.3, wspace=0.00)

    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    # === Fake cluster panel (empty) for visual consistency
    ax_cluster.axis("off")

    # === Colormap and heatmap
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    sns.heatmap(summary_topk, cmap=bluish_gray_gradient, center=0, vmin=-2, vmax=2,
                ax=ax_heatmap, cbar=False)

    ax_heatmap.set_title("Raw Heatmap of Summary Bio Features (Top Predicted Genes)", fontsize=18)
    ax_heatmap.set_yticks([])

    # === X-axis feature names (with omics coloring)
    xticks = np.arange(len(bio_feat_names_64)) + 0.5
    ax_heatmap.set_xticks(xticks)
    cancer_labels_only = [name.split('_')[1] for name in bio_feat_names_64]
    ax_heatmap.set_xticklabels(cancer_labels_only, rotation=90, fontsize=18)

    # === Omics coloring for x-tick labels
    omics_colors = {'cna': '#9370DB', 'ge': '#228B22', 'meth': '#00008B', 'mf': '#b22222'}
    for tick_label, name in zip(ax_heatmap.get_xticklabels(), bio_feat_names_64):
        omics = name.split('_')[0]
        tick_label.set_color(omics_colors.get(omics, 'black'))

    # === Legend bar for omics types
    omics_patches = [Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()]
    ax_legend.legend(handles=omics_patches, loc='center', ncol=len(omics_patches),
                     fontsize=18, frameon=False)

    # === Final adjustments and save
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.03, hspace=0.2, wspace=0.01)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Raw bio heatmap saved to {output_path}")

def plot_topo_heatmap_raw_unsorted(summary_topo_features, predicted_indices, output_path):
    """
    Plot unclustered raw heatmap of 64-dim topological summary features for top predicted genes.

    Parameters:
    - summary_topo_features: np.ndarray of shape (num_nodes, 64)
    - predicted_indices: list or array of top predicted gene indices
    - output_path: str, path to save the heatmap image
    """
    print("📊 Plotting raw topo heatmap without clustering...")

    # === Top-K predicted genes to show
    topk = 1000
    predicted_indices = predicted_indices[:topk]
    summary_topk = summary_topo_features[predicted_indices]

    # === Topo feature labels: "00" to "63"
    topo_feat_names_64 = [f"{i:02d}" for i in range(64)]

    # === Layout
    fig = plt.figure(figsize=(17, 18))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[17, 2], width_ratios=[0.25, 13.7], hspace=0.3, wspace=0.00)
    
    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    # === Empty cluster column for layout consistency
    ax_cluster.axis("off")

    # === Heatmap
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    sns.heatmap(
        summary_topk,
        cmap=bluish_gray_gradient,
        center=0,
        vmin=-2, vmax=2,
        ax=ax_heatmap,
        cbar=False
    )
    ax_heatmap.set_title("Raw Topological Feature Heatmap (Top Predicted Genes)", fontsize=18, pad=12)
    ax_heatmap.set_yticks([])

    xticks = np.arange(len(topo_feat_names_64)) + 0.5
    ax_heatmap.set_xticks(xticks)
    ax_heatmap.set_xticklabels(topo_feat_names_64, rotation=90, fontsize=12)

    # === Save
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.05)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Raw topo heatmap saved to {output_path}")

def apply_full_spectral_biclustering_bio(graph, summary_bio_features, summary_bio_features_norm, node_names,
                                         predicted_cancer_genes, n_clusters,
                                         save_path, save_cluster_labels_path,
                                         save_total_genes_per_cluster_path, save_predicted_counts_path,
                                         output_path_genes_clusters, output_path_heatmap):

    print(f"Running Spectral Biclustering on 64-dim summary bio features with {n_clusters} clusters...")
    assert summary_bio_features.shape[1] == 64, f"Expected 64 summary features, got {summary_bio_features.shape[1]}"

    # === Normalize features
    summary_bio_features = StandardScaler().fit_transform(summary_bio_features)

    # === Spectral biclustering
    bicluster = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state31)
    bicluster.fit(summary_bio_features)
    row_labels = bicluster.row_labels_
    col_labels = bicluster.column_labels_
    graph.ndata['cluster_bio_summary'] = torch.tensor(row_labels, dtype=torch.long)
    print("✅ Spectral Biclustering (summary bio) complete.")

    # === Save results
    save_graph_with_clusters(graph, save_path)
    save_cluster_labels(row_labels, save_cluster_labels_path)
    total_genes_per_cluster = compute_total_genes_per_cluster(row_labels, n_clusters)
    save_total_genes_per_cluster(total_genes_per_cluster, save_total_genes_per_cluster_path)

    pred_counts, predicted_indices = count_predicted_genes_per_cluster(row_labels, node_names, predicted_cancer_genes, n_clusters)
    save_predicted_counts(pred_counts, save_predicted_counts_path)

    # === Save original (unclustered) heatmap before biclustering
    output_path_unclustered_heatmap = output_path_heatmap.replace(".png", "_unclustered.png")
    #plot_bio_heatmap_raw(summary_bio_features, predicted_indices, output_path_unclustered_heatmap)
    plot_bio_heatmap_raw_unsorted(summary_bio_features, predicted_indices, output_path_unclustered_heatmap)    
    plot_tsne(summary_bio_features, row_labels, predicted_indices, n_clusters, output_path_genes_clusters)
    plot_bio_heatmap_unsort(summary_bio_features, summary_bio_features_norm, row_labels, col_labels, predicted_indices, output_path_heatmap)


    return graph, row_labels, col_labels, total_genes_per_cluster, pred_counts

def apply_full_spectral_biclustering_topo(graph, summary_topo_features, node_names,
                                          predicted_cancer_genes, n_clusters,
                                          save_path, save_cluster_labels_path,
                                          save_total_genes_per_cluster_path, save_predicted_counts_path,
                                          output_path_genes_clusters, output_path_heatmap):

    print(f"Running Spectral Biclustering on topo features with {n_clusters} clusters...")
    assert summary_topo_features.shape[1] == 64, f"Expected 64 summary features, got {summary_topo_features.shape[1]}"

    # === Normalize topo features
    summary_topo_features = StandardScaler().fit_transform(summary_topo_features)

    # === Spectral Biclustering
    bicluster = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state31)
    bicluster.fit(summary_topo_features)
    row_labels = bicluster.row_labels_
    col_labels = bicluster.column_labels_
    graph.ndata['cluster_topo'] = torch.tensor(row_labels, dtype=torch.long)
    print("✅ Spectral Biclustering (topo) complete.")

    # === Save results
    save_graph_with_clusters(graph, save_path)
    save_cluster_labels(row_labels, save_cluster_labels_path)
    total_genes_per_cluster = compute_total_genes_per_cluster(row_labels, n_clusters)
    save_total_genes_per_cluster(total_genes_per_cluster, save_total_genes_per_cluster_path)

    pred_counts, predicted_indices = count_predicted_genes_per_cluster(row_labels, node_names, predicted_cancer_genes, n_clusters)
    save_predicted_counts(pred_counts, save_predicted_counts_path)
    
    # === Save original (unclustered) heatmap before biclustering
    output_path_unclustered_heatmap = output_path_heatmap.replace(".png", "_unclustered.png")
    plot_topo_heatmap_raw_unsorted(summary_topo_features, predicted_indices, output_path_unclustered_heatmap) 
    
    plot_tsne(summary_topo_features, row_labels, predicted_indices, n_clusters, output_path_genes_clusters)
    plot_topo_heatmap_unsort(summary_topo_features, row_labels, col_labels, predicted_indices, output_path_heatmap)

    return graph, row_labels, col_labels, total_genes_per_cluster, pred_counts

def plot_topo_biclustering_heatmap_not_sorted(
    args,
    relevance_scores,
    cluster_labels,
    output_path,
    gene_names=None,
    col_labels=None
    ):

    # 🔹 Extract 64D summary of topological features
    relevance_scores = extract_summary_features_np_topo(relevance_scores)
    # Normalize features-----------------------------------------------------------------------------------------------------------------
    #relevance_scores = StandardScaler().fit_transform(relevance_scores)*10
    
    # 🔹 Create topo feature names (01–64)
    feature_names = [f"{i+1:02d}" for i in range(relevance_scores.shape[1])]

    # 🔹 Sort by cluster → then by gene-wise relevance within cluster
    sorted_indices = []
    cluster_labels = np.array(cluster_labels)
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        cluster_idx = np.where(cluster_labels == cluster)[0]
        cluster_scores = relevance_scores[cluster_idx]
        cluster_gene_sums = cluster_scores.sum(axis=1)
        sorted_cluster = cluster_idx[np.argsort(-cluster_gene_sums)]
        sorted_indices.extend(sorted_cluster)

    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]
    if gene_names is not None:
        gene_names = [gene_names[i] for i in sorted_indices]

    # 🔹 Compute cluster boundaries and centers
    _, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Apply log transformation to enhance low-intensity features
    sorted_scores = np.log1p(sorted_scores)  # This will emphasize smaller values

    # 🔹 Normalize scores (optional but improves contrast)
    #sorted_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min())
    
    # 🔹 Set colormap
    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient", ["#F0F3F4", "#85929e"]
    )

    # 🔹 Setup figure layout
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50)

    ax_bar = fig.add_subplot(gs[0, 2:45])
    ax = fig.add_subplot(gs[1:13, 2:45])
    ax_curve = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[5:9, 49])
    ax_legend = fig.add_subplot(gs[14, 2:45])

    # 🔹 Compute dynamic vmax
    vmin = np.percentile(sorted_scores, 5)
    vmax = np.percentile(sorted_scores, 99)


    feature_means = sorted_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min()) * 0.04

    ax_bar.bar(
        np.arange(len(feature_means)) + 0.5, 
        feature_means,
        width=1.0,
        color="#B0BEC5",
        linewidth=0,
        alpha=0.6
    )

    ax_bar.set_xticks([0, len(feature_means)])
    ax_bar.set_xticklabels(['0', '1'], fontsize=16)
    ax_bar.tick_params(axis='x', direction='out', pad=1)
        
    ax_bar.set_xlim(0, len(feature_means))  # align with heatmap width
    ax_bar.set_ylim(0, 0.04)
    ax_bar.set_yticks([])
    ax_bar.set_yticklabels([])
    ax_bar.tick_params(axis='y', length=0)  # removes tick marks
    ax_bar.set_xticks([])


    for spine in ['left', 'bottom', 'top', 'right']:
        ax_bar.spines[spine].set_visible(False)
    '''for spine in ['left', 'bottom']:
        ax_bar.spines[spine].set_visible(True)
        ax_bar.spines[spine].set_linewidth(1.0)
        ax_bar.spines[spine].set_color("black")'''


    # 🔹 Apply log transformation to enhance low-intensity features
    sorted_scores = np.log1p(sorted_scores)  # This will emphasize smaller values

    # 🔹 Normalize scores (optional but improves contrast)
    sorted_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min())

    # 🔹 Compute vmin and vmax dynamically
    vmin = 0#np.percentile(sorted_scores, 1)   # Stretch the color range from low values
    vmax = np.percentile(sorted_scores, 99)  # Cap extreme values

    # 🔹 Choose a perceptually clear colormap
    #colormap = "mako"  # or try "viridis", "plasma", "rocket", etc.

    # 🔹 Plot heatmap with new settings
    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Log-Scaled Relevance",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )

    # 🔹 Plot heatmap
    '''sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=15,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Relevance Score",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )'''
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=16)
    ax_cbar.yaxis.label.set_size(18)

    # 🔹 Add cluster color stripes
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # 🔹 Cluster size labels
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=18, fontweight='bold'
        )

    # 🔹 X-tick labels below heatmap
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=18)
    ax.tick_params(axis='x', bottom=True, labelbottom=True)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Omics + LRP Legend
    ax_legend.axis("off")
    lrp_patch = Patch(facecolor='#a9cce3', alpha=0.8, label='Saliency Sum')
    ax_legend.legend(
        handles=[lrp_patch],
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=16,
        handleheight=1.5,
        handlelength=3
    )

    # 🔹 Saliency Sum curve
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())
    y = np.arange(len(lrp_sums))

    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='#a9cce3',
        alpha=0.8,
        linewidth=3
    )

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.hlines(
        y=1.05, xmin=0, xmax=1,
        color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform()
    )
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Final layout + save
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

    # 🔹 Cluster-wise contribution breakdown
    plot_topo_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,  # Not sorted for per-cluster breakdown
        cluster_labels=cluster_labels,
        feature_names=[f"{i+1:02d}" for i in range(relevance_scores.shape[1])],
        per_cluster_feature_contributions_output_dir=os.path.join(
            os.path.dirname(output_path), "per_cluster_feature_contributions_topo"
        )
    )

    return pd.DataFrame(sorted_scores, index=gene_names, columns=feature_names)

def plot_topo_heatmap_unsort(summary_topo_features, row_labels, col_labels, predicted_indices, output_path):

    topo_feat_names_64 = [f"{i:02d}" for i in range(64)]
    topk = 1000
    predicted_indices = predicted_indices[:topk]
    summary_topk = summary_topo_features[predicted_indices]
    row_labels_topk = row_labels[predicted_indices]
    sorted_indices = np.argsort(row_labels_topk)
    sorted_matrix = summary_topk[sorted_indices, :]
    sorted_labels = row_labels_topk[sorted_indices]

    fig = plt.figure(figsize=(17, 18))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[17, 2], width_ratios=[0.25, 13.7], hspace=0.3, wspace=0.00)
    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    cluster_colors = [to_rgb(CLUSTER_COLORS[label]) for label in sorted_labels]
    cluster_colors_array = np.array(cluster_colors).reshape(-1, 1, 3)
    ax_cluster.imshow(cluster_colors_array, aspect='auto')
    ax_cluster.axis("off")

    # Add cluster size labels
    cluster_counts = Counter(sorted_labels)
    start_idx = 0
    for cluster_id in sorted(cluster_counts):
        count = cluster_counts[cluster_id]
        mid_idx = start_idx + count // 2
        ax_cluster.text(-0.75, mid_idx, f'{count}', va='center', ha='right',
                        fontsize=14, fontweight='bold', color='black')
        start_idx += count

    # Heatmap
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    sns.heatmap(
        sorted_matrix,
        cmap=bluish_gray_gradient,
        center=0,
        vmin=-2, vmax=2,
        ax=ax_heatmap,
        cbar=False
    )
    ax_heatmap.set_title("Topological Feature Heatmap (Sorted by Cluster)", fontsize=18, pad=12)
    ax_heatmap.set_yticks([])

    xticks = np.arange(len(topo_feat_names_64)) + 0.5
    ax_heatmap.set_xticks(xticks)
    ax_heatmap.set_xticklabels(topo_feat_names_64, rotation=90, fontsize=12)

    # Save
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.05)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Summary topo heatmap saved to {output_path}")

def plot_bio_biclustering_heatmap_not_sorted(
    args,
    relevance_scores,
    cluster_labels,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None,
    col_labels=None
    ):  
    
    relevance_scores = extract_summary_features_np_bio(relevance_scores)
    # 🔹 Normalize relevance scores row-wise (per gene)
    '''relevance_scores = (relevance_scores - relevance_scores.min(axis=1, keepdims=True)) / \
                    (relevance_scores.max(axis=1, keepdims=True) - relevance_scores.min(axis=1, keepdims=True) + 1e-6)'''
    # Normalize features-----------------------------------------------------------------------------------------------------------------
    relevance_scores = StandardScaler().fit_transform(relevance_scores)*20
    
    # 🔹 Set default omics colors
    if omics_colors is None:
        omics_colors = {
            'cna': '#9370DB',    # purple
            'ge': '#228B22',      # dark green
            'meth': '#00008B',   # dark blue
            'mf': '#b22222',     # dark red
        }

    # 🔹 Cancer types and omics order
    
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]

    omics_order = ['cna', 'ge', 'meth', 'mf']

    # 🔹 Build feature names internally
    feature_names = [f"{omics.upper()}: {cancer}" for omics in omics_order for cancer in cancer_names]

    sorted_indices = np.argsort(cluster_labels)
    relevance_scores = relevance_scores[sorted_indices]
    cluster_labels = cluster_labels[sorted_indices]

    # 🔹 Then sort within each cluster (optional: by LRP sum descending)
    new_order = []
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        
        # Sort within this cluster, for example by total relevance score (descending)
        cluster_relevance_sums = relevance_scores[cluster_indices].sum(axis=1)
        sorted_within = cluster_indices[np.argsort(-cluster_relevance_sums)]  # Descending
        new_order.extend(sorted_within)

    # 🔹 Apply new sorted order
    sorted_scores = relevance_scores[new_order]
    sorted_clusters = cluster_labels[new_order]


    # 🔹 Reorder columns using col_labels (if provided)
    if col_labels is not None:
        sorted_order = np.argsort(col_labels)
        sorted_scores = sorted_scores[:, sorted_order]
        feature_names = [feature_names[i] for i in sorted_order]

            
    # 🔹 Feature color mapping
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    # 🔹 Setup figure
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50)
    ax_bar    = fig.add_subplot(gs[0, 2:45])       # new bar on top
    ax        = fig.add_subplot(gs[1:13, 2:45])    # shift heatmap down
    ax_curve  = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar   = fig.add_subplot(gs[5:9, 49])
    #ax_legend = fig.add_subplot(gs[14, 2:45])
    ax_bar.axis("off")  # Hide regular axis stuff
    ax_bar.set_xlim(0, len(feature_names))
    ax_bar.set_ylim(0, 1.1)
        
    # 🔹 Compute cluster boundaries
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Use dynamic vmax for contrast
    vmin = 0#np.percentile(sorted_scores, 1)
    vmax = np.percentile(sorted_scores, 99)


    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient",
        ["#F0F3F4", "#85929e"]
    )

    # 🔹 Compute mean relevance per omics group
    omics_means = {}
    for omics, (start, end) in omics_splits.items():
        group_scores = sorted_scores[:, start:end+1]
        omics_means[omics] = group_scores.mean()

    # 🔹 Sort omics groups by descending mean relevance
    sorted_omics = sorted(omics_means.keys(), key=lambda x: omics_means[x], reverse=True)

    # 🔹 Reorder columns: first by omics group, then within-group by mean column relevance
    sorted_col_indices = []
    sorted_feature_names = []
    feature_colors = []

    for omics in sorted_omics:
        start, end = omics_splits[omics]
        group_indices = list(range(start, end + 1))

        # Get mean relevance per feature in this omics group
        col_means = sorted_scores[:, group_indices].mean(axis=0)
        group_sorted_indices = [group_indices[i] for i in np.argsort(-col_means)]  # descending

        sorted_col_indices.extend(group_sorted_indices)
        sorted_feature_names.extend([feature_names[i] for i in group_sorted_indices])
        feature_colors.extend([omics_colors[omics]] * len(group_sorted_indices))

    # 🔹 Apply sorted column order
    sorted_scores = sorted_scores[:, sorted_col_indices]
    feature_names = sorted_feature_names

    # 🔹 Compute per-feature mean relevance (column-wise)
    feature_means = sorted_scores.mean(axis=0)
    normed_feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-6)

    # 🔹 Plot one bar per feature (aligned to heatmap)
    # 🔹 Ensure cancer types within each omics group are treated as units
    feature_index = 0
    for omics in sorted_omics:
        start, end = omics_splits[omics]
        num_cancers = end - start + 1

        # Get the subset of sorted column indices and colors for this omics
        group_indices = [i for i in sorted_col_indices if start <= i <= end]
        group_means = feature_means[group_indices]
        group_colors = [omics_colors[omics]] * len(group_indices)

        for mean_val, color in zip(group_means, group_colors):
            norm_val = (mean_val - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-6)
            ax_bar.bar(
                x=feature_index + 0.5,
                height=norm_val,
                width=1.0,
                bottom=0,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.3 + 0.7 * norm_val
            )
            feature_index += 1


    # 🔹 Plot heatmap
    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Relevance Score",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=18)
    ax_cbar.yaxis.label.set_size(18)

    # 🔹 Add cluster color stripes
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # 🔹 Add cluster size labels
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=18, fontweight='bold'
        )

    # 🔹 Color x-tick labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels([c.split(": ")[1] for c in feature_names], rotation=90, fontsize=14)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 LRP curve
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='#a9cce3',
        alpha=0.8,
        linewidth=3
    )

    # Compute mean relevance per omics group (column-wise)
    omics_means = {}
    for omics, (start, end) in omics_splits.items():
        group_scores = sorted_scores[:, start:end+1]  # inclusive end
        omics_means[omics] = group_scores.mean()

    # Map omics to x-center positions
    group_centers = {
        omics: (omics_splits[omics][0] + omics_splits[omics][1]) / 2 + 0.5
        for omics in omics_order
    }

    # Normalize mean values for alpha mapping
    mean_vals = np.array([omics_means[om] for om in omics_order])
    min_mean, max_mean = mean_vals.min(), mean_vals.max()
    normalized_means = (mean_vals - min_mean) / (max_mean - min_mean + 1e-6)  # prevent zero division

    # Plot bars with darkness mapped to mean relevance
    for i, omics in enumerate(omics_order):
        ax.bar(
            x=group_centers[omics],
            height=0.15,  # small bar height
            width=(omics_splits[omics][1] - omics_splits[omics][0] + 1),
            bottom=len(sorted_scores) + 1.5,
            color=omics_colors[omics],
            edgecolor='black',
            linewidth=1,
            alpha=0.3 + 0.7 * normalized_means[i]  # range alpha from 0.3 (light) to 1.0 (dark)
        )


    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.hlines(
        y=1.01, xmin=0, xmax=1,
        color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform()
    )
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Layout and save
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

    # 🔹 Optional: Cluster-wise contributions
    plot_bio_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,
        cluster_labels=cluster_labels,
        feature_names=feature_names,
        per_cluster_feature_contributions_output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions_bio"),
        omics_colors=omics_colors
    )

    # 🔹 Optional: Cluster-wise contributions
    bio_clusterwise_df = plot_bio_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,
        cluster_labels=cluster_labels,
        feature_names=feature_names,
        per_cluster_feature_contributions_output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions_bio/"),
        omics_colors=omics_colors
    )

    '''plot_all_clusterwise_contributions(
        feature_contributions_df=bio_clusterwise_df,
        feature_groups=feature_groups_dict,
        output_pdf_path=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions_bio/bio_clusterwise_feature_contributions.pdf"),
    )'''

    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def plot_all_clusterwise_contributions(feature_contributions_df, feature_groups, output_pdf_path, figsize=(18, 12), cols=3):
    """
    Plots all clusterwise feature contributions into a multi-panel PDF.
    
    Parameters:
    - feature_contributions_df: DataFrame (cluster x feature) of relevance values
    - feature_groups: dict mapping feature names to group labels (e.g., omics types)
    - output_pdf_path: path to save the final PDF
    - figsize: figure size for the whole panel
    - cols: number of columns per page
    """
    n_clusters = feature_contributions_df.shape[0]
    cluster_ids = feature_contributions_df.index.tolist()
    features = feature_contributions_df.columns.tolist()

    # Assign group colors
    group_labels = list(set(feature_groups.values()))
    cmap = plt.get_cmap('tab10')
    group_color_map = {group: cmap(i % 10) for i, group in enumerate(group_labels)}
    feature_colors = [group_color_map[feature_groups[f]] for f in features]

    # Define layout
    rows = int(np.ceil(n_clusters / cols))

    with PdfPages(output_pdf_path) as pdf:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()

        for i, cluster_id in enumerate(cluster_ids):
            ax = axes[i]
            contributions = feature_contributions_df.loc[cluster_id]

            bars = ax.bar(range(len(features)), contributions.values, color=feature_colors)
            ax.set_title(f"Cluster {cluster_id}", fontsize=10)
            ax.set_xticks([])
            ax.set_ylim(0, max(1.0, contributions.max() * 1.1))  # normalize for better visual spacing

        # Hide unused axes if any
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=group_color_map[g]) for g in group_labels]
        labels = group_labels
        fig.legend(handles, labels, title="Omics Type", loc="lower center", ncol=len(group_labels), fontsize=10)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

def plot_bio_heatmap_unsort(
    summary_bio_features, 
    summary_bio_features_norm, 
    row_labels, 
    col_labels, 
    predicted_indices, 
    output_path
):
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['cna', 'ge', 'meth', 'mf']
    bio_feat_names_64 = [f"{omics}_{cancer}" for omics in omics_order for cancer in cancer_names]

    topk = 1000
    predicted_indices = predicted_indices[:topk]
    row_labels_topk = row_labels[predicted_indices]

    # Sort within each cluster by descending LRP sum
    lrp_sums = summary_bio_features_norm[predicted_indices].sum(axis=1)
    cluster_sorted_indices = []

    for cluster_id in sorted(np.unique(row_labels_topk)):
        cluster_mask = (row_labels_topk == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_lrp = lrp_sums[cluster_indices]
        sorted_within = cluster_indices[np.argsort(-cluster_lrp)]
        cluster_sorted_indices.extend(sorted_within)

    sorted_row_indices = cluster_sorted_indices

    summary_topk = summary_bio_features[predicted_indices]
    summary_topk_norm = summary_bio_features_norm[predicted_indices]
    sorted_scores = summary_topk_norm[sorted_row_indices]
    
    sorted_matrix = summary_topk[sorted_row_indices, :]
    sorted_labels = row_labels_topk[sorted_row_indices]

    # Compute mean importance per feature
    col_mean_importance = np.abs(sorted_matrix).mean(axis=0)

    # Sort omics types by total relevance
    omics_group_scores = {}
    omics_group_indices = {}
    for omics in omics_order:
        indices = [i for i, name in enumerate(bio_feat_names_64) if name.startswith(omics)]
        group_score = np.mean(col_mean_importance[indices])
        omics_group_scores[omics] = group_score
        omics_group_indices[omics] = indices

    sorted_omics_order = sorted(omics_group_scores, key=omics_group_scores.get, reverse=True)

    # Sort features within each omics group
    omics_grouped_indices = []
    for omics in sorted_omics_order:
        indices = omics_group_indices[omics]
        
        cancer_to_score = {}
        for i in indices:
            cancer = bio_feat_names_64[i].split('_')[1]
            cancer_to_score[cancer] = col_mean_importance[i]
        
        sorted_cancers = sorted(cancer_to_score, key=cancer_to_score.get, reverse=True)
        sorted_within = [i for cancer in sorted_cancers
                         for i in indices if bio_feat_names_64[i].split('_')[1] == cancer]
        
        omics_grouped_indices.extend(sorted_within)

    # Apply sorting
    sorted_matrix = sorted_matrix[:, omics_grouped_indices]
    sorted_scores = sorted_scores[:, omics_grouped_indices]
    sorted_feat_names = [bio_feat_names_64[i] for i in omics_grouped_indices]

    # Layout
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1, 17], width_ratios=[0.25, 13.7, 1.5], hspace=0.0, wspace=0.00)

    ax_bar = fig.add_subplot(gs[0, 1])
    ax_cluster = fig.add_subplot(gs[1, 0])
    ax_heatmap = fig.add_subplot(gs[1, 1])
    ax_curve = fig.add_subplot(gs[1, 2])

    omics_colors = {'cna': '#9370DB', 'ge': '#228B22', 'meth': '#00008B', 'mf': '#b22222'}
    bar_colors = [omics_colors[name.split('_')[0]] for name in sorted_feat_names]

    # Feature bar plot
    bar_heights = np.abs(sorted_scores).mean(axis=0)
    x_vals = np.arange(len(sorted_feat_names)) + 0.5
    #ax_bar.bar(x_vals, bar_heights, color=bar_colors, edgecolor='black', linewidth=0.1, width=0.9)
    ax_bar.bar(x_vals, bar_heights, color=bar_colors, edgecolor='black', linewidth=0.1, width=1.0)
    ax_bar.set_xlim(0, len(sorted_feat_names))
    ax_bar.set_ylim(0, np.max(bar_heights) * 1.1)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.axis('off')
    ax_bar.set_title("Per-Feature Importance (Color = Omics)", fontsize=14)
    ax_bar.margins(x=0)

    # Heatmap
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])
    sns.heatmap(sorted_matrix, cmap=bluish_gray_gradient, center=0, vmin=-2, vmax=2,
                ax=ax_heatmap, cbar=False)
    ax_heatmap.set_yticks([])

    xticks = np.arange(len(sorted_feat_names)) + 0.5
    cancer_labels_only = [name.split('_')[1] for name in sorted_feat_names]
    ax_heatmap.set_xticks(xticks)
    ax_heatmap.set_xticklabels(cancer_labels_only, rotation=90, fontsize=14)

    for tick_label, name in zip(ax_heatmap.get_xticklabels(), sorted_feat_names):
        omics = name.split('_')[0]
        tick_label.set_color(omics_colors.get(omics, 'black'))

    # Cluster strip
    cluster_colors = [to_rgb(CLUSTER_COLORS[label]) for label in sorted_labels]
    cluster_colors_array = np.array(cluster_colors).reshape(-1, 1, 3)
    ax_cluster.imshow(cluster_colors_array, aspect='auto')
    ax_cluster.axis("off")

    cluster_counts = Counter(sorted_labels)
    start_idx = 0
    for cluster_id in sorted(cluster_counts):
        count = cluster_counts[cluster_id]
        mid_idx = start_idx + count // 2
        ax_cluster.text(-0.75, mid_idx, f'{count}', va='center', ha='right',
                        fontsize=14, fontweight='bold', color='black')
        start_idx += count

    # LRP sum curve per cluster (reversed cluster order, and sorted within each cluster)
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min() + 1e-6)
    y = np.arange(len(lrp_sums))

    cluster_to_indices = {}
    for idx, label in enumerate(sorted_labels):
        cluster_to_indices.setdefault(label, []).append(idx)

    sorted_cluster_ids = sorted(cluster_to_indices.keys(), reverse=True)
    y_offset = 0

    for cluster_id in sorted_cluster_ids:
        indices = cluster_to_indices[cluster_id]
        cluster_lrp = lrp_sums[indices]
        sorted_within = np.argsort(cluster_lrp)
        sorted_indices = [indices[i] for i in sorted_within]

        y_cluster = np.arange(y_offset, y_offset + len(sorted_indices))
        lrp_cluster_sorted = lrp_sums[sorted_indices]

        ax_curve.fill_betweenx(
            y_cluster,
            0,
            lrp_cluster_sorted,
            color=CLUSTER_COLORS.get(cluster_id, "#CCCCCC"),
            alpha=0.8,
            linewidth=0
        )
        y_offset += len(sorted_indices)

    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.axis('off')

    # Save figure
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.03, hspace=0.2, wspace=0.01)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Summary bio heatmap saved to {output_path}")

def plot_topo_biclustering_heatmap_unsorted(
    args,
    relevance_scores,
    cluster_labels,
    output_path,
    gene_names=None,
    col_labels=None
):
    """
    Unsorted topo heatmap with connected cluster bar and gene count labels.
    """
    from matplotlib.patches import Rectangle

    # 🔹 Normalize summary topo features (0-1)
    relevance_scores = extract_summary_features_np_topo(relevance_scores)
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())

    # 🔹 Generate feature names (01 to 64)
    feature_names = [f"{i+1:02d}" for i in range(relevance_scores.shape[1])]
    cluster_labels = np.array(cluster_labels)

    sorted_indices = np.argsort(cluster_labels)
    cluster_labels = cluster_labels[sorted_indices]
    
    # 🔹 Compute cluster stats
    _, counts = np.unique(cluster_labels, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Setup figure layout
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)

    ax_bar = fig.add_subplot(gs[0, 2:45])
    ax = fig.add_subplot(gs[1:13, 2:45])
    ax_curve = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[5:9, 49])

    # 🔹 Log-transform scores
    scores = np.log1p(relevance_scores)

    # 🔹 Color map
    bluish_gray_gradient = LinearSegmentedColormap.from_list("bluish_gray_gradient", ["#F0F3F4", "#85929e"])

    # 🔹 Feature mean bar
    feature_means = scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min()) * 0.04
    ax_bar.bar(np.arange(len(feature_means)) + 0.5, feature_means, width=1.0, color="#B0BEC5", alpha=0.6)
    ax_bar.set_xticks([]), ax_bar.set_yticks([]), ax_bar.set_xlim(0, len(feature_means)), ax_bar.set_ylim(0, 0.04)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # 🔹 Heatmap
    vmin, vmax = 0, np.percentile(scores, 99)
    sns.heatmap(
        scores, cmap=bluish_gray_gradient, vmin=vmin, vmax=vmax,
        xticklabels=False, yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Log-Scaled Relevance",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )

    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=16)
    ax_cbar.yaxis.label.set_size(18)

    # 🔹 Add cluster bar
    for i, cluster in enumerate(cluster_labels):
        ax.add_patch(Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # 🔹 Add cluster counts
    for cluster_id, center_y, count in zip(np.unique(cluster_labels), cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=18, fontweight='bold'
        )

    # 🔹 X-axis labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=14)
    ax.tick_params(axis='x', bottom=True, labelbottom=True)
    ax.set_xlabel(""), ax.set_ylabel(""), ax.set_title("")

    # 🔹 Saliency sum curve
    saliency_sums = scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    y = np.arange(len(saliency_sums))

    ax_curve.fill_betweenx(y, 0, saliency_sums, color='#a9cce3', alpha=0.8, linewidth=3)
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.hlines(y=1.05, xmin=0, xmax=1,
                    color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform())
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(saliency_sums))
    for spine in ax_curve.spines.values():
        spine.set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Save plot
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(rect=[0.02, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved unsorted topo clustering heatmap to {output_path}")

    return pd.DataFrame(scores, index=gene_names, columns=feature_names)

def plot_bio_biclustering_heatmap_unsort(
    args,
    relevance_scores,
    cluster_labels,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None,
    col_labels=None
):  
    
    # 🔹 Extract and normalize relevance scores
    relevance_scores = extract_summary_features_np_bio(relevance_scores)
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min()) * 20

    if omics_colors is None:
        omics_colors = {
            'cna': '#9370DB',    # purple
            'ge': '#228B22',     # dark green
            'meth': '#00008B',   # dark blue
            'mf': '#b22222',     # dark red
        }

    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]
    omics_order = ['cna', 'ge', 'meth', 'mf']
    feature_names = [f"{omics.upper()}: {cancer}" for omics in omics_order for cancer in cancer_names]

    # If col_labels provided, reorder columns accordingly
    # if col_labels is not None:
    #     sorted_order = np.argsort(col_labels)
    #     relevance_scores = relevance_scores[:, sorted_order]
    #     feature_names = [feature_names[i] for i in sorted_order]

    # Build feature color bar
    feature_colors = []
    for i in range(len(feature_names)):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")  # fallback color

    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)

    ax_bar    = fig.add_subplot(gs[0, 2:45])      
    ax        = fig.add_subplot(gs[1:13, 2:45])    
    ax_curve  = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar   = fig.add_subplot(gs[5:9, 49])

    ax_bar.axis("off")
    ax_bar.set_xlim(0, len(feature_names))
    ax_bar.set_ylim(0, 1.1)

    # Normalize per-feature means
    feature_means = relevance_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-6)

    for i, (mean_val, color) in enumerate(zip(feature_means, feature_colors)):
        ax_bar.bar(
            x=i + 0.5,
            height=mean_val,
            width=1.0,
            bottom=0,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.3 + 0.7 * mean_val
        )

    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient",
        ["#F0F3F4", "#85929e"]
    )

    vmin = 0
    vmax = np.percentile(relevance_scores, 99)

    sns.heatmap(
        relevance_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Relevance Score",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )

    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=18)
    ax_cbar.yaxis.label.set_size(18)

    # Sort by cluster only (no intra-cluster sorting)
    cluster_order = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[cluster_order]
    sorted_clusters = cluster_labels[cluster_order]

    # Cluster stripe
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle((-1.5, i), 1.5, 1, linewidth=0, facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')), clip_on=False))

    # Cluster size text
    unique_clusters, cluster_sizes = np.unique(sorted_clusters, return_counts=True)
    start_idx = 0
    for cluster, size in zip(unique_clusters, cluster_sizes):
        center_y = start_idx + size / 2
        ax.text(-2.0, center_y, f"{size}", va='center', ha='right', fontsize=18, fontweight='bold')
        start_idx += size

    # Add xtick labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels([c.split(": ")[1] for c in feature_names], rotation=90, fontsize=14)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # LRP curve
    saliency_sums = relevance_scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    y = np.arange(len(saliency_sums))
    ax_curve.fill_betweenx(
        y, 0, saliency_sums,
        color='#a9cce3',
        alpha=0.8,
        linewidth=3
    )

    # Omics bar below
    omics_means = {}
    for omics, (start, end) in omics_splits.items():
        group_scores = relevance_scores[:, start:end+1]
        omics_means[omics] = group_scores.mean()

    group_centers = {
        omics: (omics_splits[omics][0] + omics_splits[omics][1]) / 2 + 0.5
        for omics in omics_order
    }

    mean_vals = np.array([omics_means[om] for om in omics_order])
    min_mean, max_mean = mean_vals.min(), mean_vals.max()
    normalized_means = (mean_vals - min_mean) / (max_mean - min_mean + 1e-6)

    for i, omics in enumerate(omics_order):
        ax.bar(
            x=group_centers[omics],
            height=0.15,
            width=(omics_splits[omics][1] - omics_splits[omics][0] + 1),
            bottom=len(relevance_scores) + 1.5,
            color=omics_colors[omics],
            edgecolor='black',
            linewidth=1,
            alpha=0.3 + 0.7 * normalized_means[i]
        )

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.hlines(
        y=1.01, xmin=0, xmax=1,
        color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform()
    )
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.set_yticks([])
    ax_curve.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_bio_biclustering_heatmap(
    args,
    relevance_scores,
    cluster_labels,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None,
    col_labels=None
    ):  
    
    relevance_scores = extract_summary_features_np_bio(relevance_scores)
    # 🔹 Normalize relevance scores row-wise (per gene)
    '''relevance_scores = (relevance_scores - relevance_scores.min(axis=1, keepdims=True)) / \
                    (relevance_scores.max(axis=1, keepdims=True) - relevance_scores.min(axis=1, keepdims=True) + 1e-6)'''
    # Normalize features-----------------------------------------------------------------------------------------------------------------
    ##relevance_scores = StandardScaler().fit_transform(relevance_scores)*20
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())*20
    
    # 🔹 Set default omics colors
    if omics_colors is None:
        omics_colors = {
            'cna': '#9370DB',    # purple
            'ge': '#228B22',      # dark green
            'meth': '#00008B',   # dark blue
            'mf': '#b22222',     # dark red
        }

    # 🔹 Cancer types and omics order
    
    cancer_names = [
        'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'KidneyPap',
        'Liver', 'LungAd', 'LungSc', 'Prostate', 'Rectum', 'Stomach', 'Thyroid', 'Uterus'
    ]

    omics_order = ['cna', 'ge', 'meth', 'mf']

    # 🔹 Build feature names internally
    feature_names = [f"{omics.upper()}: {cancer}" for omics in omics_order for cancer in cancer_names]

    sorted_indices = np.argsort(cluster_labels)
    relevance_scores = relevance_scores[sorted_indices]
    cluster_labels = cluster_labels[sorted_indices]

    # 🔹 Then sort within each cluster (optional: by LRP sum descending)
    new_order = []
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        
        # Sort within this cluster, for example by total relevance score (descending)
        cluster_relevance_sums = relevance_scores[cluster_indices].sum(axis=1)
        sorted_within = cluster_indices[np.argsort(-cluster_relevance_sums)]  # Descending
        new_order.extend(sorted_within)

    # 🔹 Apply new sorted order
    sorted_scores = relevance_scores[new_order]
    sorted_clusters = cluster_labels[new_order]


    # 🔹 Reorder columns using col_labels (if provided)
    if col_labels is not None:
        sorted_order = np.argsort(col_labels)
        sorted_scores = sorted_scores[:, sorted_order]
        feature_names = [feature_names[i] for i in sorted_order]

            
    # 🔹 Feature color mapping
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    # 🔹 Setup figure
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50, wspace=0.0, hspace=0.0)

    ax_bar    = fig.add_subplot(gs[0, 2:45])      
    ax        = fig.add_subplot(gs[1:13, 2:45])    
    ax_curve  = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar   = fig.add_subplot(gs[5:9, 49])
    #ax_legend = fig.add_subplot(gs[14, 2:45])
    ax_bar.axis("off")  # Hide regular axis stuff
    ax_bar.set_xlim(0, len(feature_names))
    ax_bar.set_ylim(0, 1.1)
        
    # 🔹 Add omics bar plot axis above the heatmap
    '''ax = fig.add_subplot(gs[1:13, 2:45])
    ax_curve = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[5:9, 49])
    ax_legend = fig.add_subplot(gs[13, 2:45])
    ax_bar.axis("off")  # Hide regular axis stuff
    ax_bar.set_xlim(0, len(feature_names))
    ax_bar.set_ylim(0, 1.1)'''

        
    # 🔹 Compute cluster boundaries
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Use dynamic vmax for contrast
    vmin = 0#np.percentile(sorted_scores, 1)
    vmax = np.percentile(sorted_scores, 99)


    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient",
        ["#F0F3F4", "#85929e"]
    )

    # 🔹 Compute mean relevance per omics group
    omics_means = {}
    for omics, (start, end) in omics_splits.items():
        group_scores = sorted_scores[:, start:end+1]
        omics_means[omics] = group_scores.mean()

    # 🔹 Sort omics groups by descending mean relevance
    sorted_omics = sorted(omics_means.keys(), key=lambda x: omics_means[x], reverse=True)

    # 🔹 Reorder columns: first by omics group, then within-group by mean column relevance
    sorted_col_indices = []
    sorted_feature_names = []
    feature_colors = []

    for omics in sorted_omics:
        start, end = omics_splits[omics]
        group_indices = list(range(start, end + 1))

        # Get mean relevance per feature in this omics group
        col_means = sorted_scores[:, group_indices].mean(axis=0)
        group_sorted_indices = [group_indices[i] for i in np.argsort(-col_means)]  # descending

        sorted_col_indices.extend(group_sorted_indices)
        sorted_feature_names.extend([feature_names[i] for i in group_sorted_indices])
        feature_colors.extend([omics_colors[omics]] * len(group_sorted_indices))

    # 🔹 Apply sorted column order
    sorted_scores = sorted_scores[:, sorted_col_indices]
    feature_names = sorted_feature_names

    # 🔹 Compute per-feature mean relevance (column-wise)
    feature_means = sorted_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-6)

    # 🔹 Plot one bar per feature (aligned to heatmap)
    for i, (mean_val, color) in enumerate(zip(feature_means, feature_colors)):
        ax_bar.bar(
            x=i + 0.5,  # center over column
            height=mean_val,
            width=1.0,
            bottom=0,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.3 + 0.7 * mean_val  # dimmer for low values
        )

    # 🔹 Plot heatmap
    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Relevance Score",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=18)
    ax_cbar.yaxis.label.set_size(18)

    # 🔹 Add cluster color stripes
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # 🔹 Add cluster size labels
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=18, fontweight='bold'
        )

    # 🔹 Color x-tick labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels([c.split(": ")[1] for c in feature_names], rotation=90, fontsize=14)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Omics + LRP curve legend
    '''ax_legend.axis("off")

    # Create omics patches
    omics_patches = [
        Patch(color=color, label=omics.upper())
        for omics, color in omics_colors.items()
    ]

    # Create LRP patch
    lrp_patch = Patch(facecolor='#a9cce3', alpha=0.8, label='Saliency Sum')

    # Combine and render
    ax_legend.legend(
        handles=omics_patches + [lrp_patch],
        loc="center",
        ncol=len(omics_patches) + 1,
        frameon=False,
        fontsize=18,
        handleheight=1.5,
        handlelength=3
    )'''


    # 🔹 LRP curve
    saliency_sums = sorted_scores.sum(axis=1)
    saliency_sums = (saliency_sums - saliency_sums.min()) / (saliency_sums.max() - saliency_sums.min())
    y = np.arange(len(saliency_sums))
    ax_curve.fill_betweenx(
        y, 0, saliency_sums,
        color='#a9cce3',
        alpha=0.8,
        linewidth=3
    )

    # Compute mean relevance per omics group (column-wise)
    omics_means = {}
    for omics, (start, end) in omics_splits.items():
        group_scores = sorted_scores[:, start:end+1]  # inclusive end
        omics_means[omics] = group_scores.mean()

    # Map omics to x-center positions
    group_centers = {
        omics: (omics_splits[omics][0] + omics_splits[omics][1]) / 2 + 0.5
        for omics in omics_order
    }

    # Normalize mean values for alpha mapping
    mean_vals = np.array([omics_means[om] for om in omics_order])
    min_mean, max_mean = mean_vals.min(), mean_vals.max()
    normalized_means = (mean_vals - min_mean) / (max_mean - min_mean + 1e-6)  # prevent zero division

    # Plot bars with darkness mapped to mean relevance
    for i, omics in enumerate(omics_order):
        ax.bar(
            x=group_centers[omics],
            height=0.15,  # small bar height
            width=(omics_splits[omics][1] - omics_splits[omics][0] + 1),
            bottom=len(sorted_scores) + 1.5,
            color=omics_colors[omics],
            edgecolor='black',
            linewidth=1,
            alpha=0.3 + 0.7 * normalized_means[i]  # range alpha from 0.3 (light) to 1.0 (dark)
        )


    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.hlines(
        y=1.01, xmin=0, xmax=1,
        color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform()
    )
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(saliency_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Create legend patches
    '''lrp_patch = Patch(facecolor='#a9cce3', alpha=0.8, label='Saliency Sum')

    # Optional: Also create cluster color legend (if not already somewhere else)
    cluster_patches = [
        Patch(facecolor=color, label=f'Cluster {cid}')
        for cid, color in CLUSTER_COLORS.items()
    ]

    # 🔹 Place legend near the X-axis label area (or wherever appropriate)
    legend_ax = fig.add_subplot(gs[11, 2:48])
    legend_ax.axis('off')  # Hide the axis

    legend_ax.legend(
        handles=[lrp_patch],  # add cluster_patches + [lrp_patch] if you want both
        loc='center',
        ncol=1,
        frameon=False,
        fontsize=12
    )'''

    # 🔹 Layout and save
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

    # 🔹 Optional: Cluster-wise contributions
    plot_bio_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,
        cluster_labels=cluster_labels,
        feature_names=feature_names,
        per_cluster_feature_contributions_output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions_bio"),
        omics_colors=omics_colors
    )

    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def plot_topo_biclustering_heatmap(
    args,
    relevance_scores,
    cluster_labels,
    output_path,
    gene_names=None,
    col_labels=None
    ):

    # 🔹 Extract 64D summary of topological features
    relevance_scores = extract_summary_features_np_topo(relevance_scores)
    # Normalize features-----------------------------------------------------------------------------------------------------------------
    #relevance_scores = StandardScaler().fit_transform(relevance_scores)*10
    
    # 🔹 Create topo feature names (01–64)
    feature_names = [f"{i+1:02d}" for i in range(relevance_scores.shape[1])]

    # 🔹 Sort by cluster → then by gene-wise relevance within cluster
    sorted_indices = []
    cluster_labels = np.array(cluster_labels)
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        cluster_idx = np.where(cluster_labels == cluster)[0]
        cluster_scores = relevance_scores[cluster_idx]
        cluster_gene_sums = cluster_scores.sum(axis=1)
        sorted_cluster = cluster_idx[np.argsort(-cluster_gene_sums)]
        sorted_indices.extend(sorted_cluster)

    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]
    if gene_names is not None:
        gene_names = [gene_names[i] for i in sorted_indices]

    # 🔹 Compute cluster boundaries and centers
    _, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Apply log transformation to enhance low-intensity features
    sorted_scores = np.log1p(sorted_scores)  # This will emphasize smaller values

    # 🔹 Normalize scores (optional but improves contrast)
    #sorted_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min())
    
    # 🔹 Set colormap
    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient", ["#F0F3F4", "#85929e"]
    )

    # 🔹 Setup figure layout
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(nrows=15, ncols=50)

    ax_bar = fig.add_subplot(gs[0, 2:45])
    ax = fig.add_subplot(gs[1:13, 2:45])
    ax_curve = fig.add_subplot(gs[1:13, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[5:9, 49])
    ax_legend = fig.add_subplot(gs[14, 2:45])

    # 🔹 Compute dynamic vmax
    vmin = np.percentile(sorted_scores, 5)
    vmax = np.percentile(sorted_scores, 99)


    feature_means = sorted_scores.mean(axis=0)
    feature_means = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min()) * 0.04

    ax_bar.bar(
        np.arange(len(feature_means)) + 0.5, 
        feature_means,
        width=1.0,
        color="#B0BEC5",
        linewidth=0,
        alpha=0.6
    )

    ax_bar.set_xticks([0, len(feature_means)])
    ax_bar.set_xticklabels(['0', '1'], fontsize=16)
    ax_bar.tick_params(axis='x', direction='out', pad=1)
        
    ax_bar.set_xlim(0, len(feature_means))  # align with heatmap width
    ax_bar.set_ylim(0, 0.04)
    ax_bar.set_yticks([])
    ax_bar.set_yticklabels([])
    ax_bar.tick_params(axis='y', length=0)  # removes tick marks
    ax_bar.set_xticks([])


    for spine in ['left', 'bottom', 'top', 'right']:
        ax_bar.spines[spine].set_visible(False)
    '''for spine in ['left', 'bottom']:
        ax_bar.spines[spine].set_visible(True)
        ax_bar.spines[spine].set_linewidth(1.0)
        ax_bar.spines[spine].set_color("black")'''


    # 🔹 Apply log transformation to enhance low-intensity features
    sorted_scores = np.log1p(sorted_scores)  # This will emphasize smaller values

    # 🔹 Normalize scores (optional but improves contrast)
    sorted_scores = (sorted_scores - sorted_scores.min()) / (sorted_scores.max() - sorted_scores.min())

    # 🔹 Compute vmin and vmax dynamically
    vmin = 0#np.percentile(sorted_scores, 1)   # Stretch the color range from low values
    vmax = np.percentile(sorted_scores, 99)  # Cap extreme values

    # 🔹 Choose a perceptually clear colormap
    #colormap = "mako"  # or try "viridis", "plasma", "rocket", etc.

    # 🔹 Plot heatmap with new settings
    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Log-Scaled Relevance",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )

    # 🔹 Plot heatmap
    '''sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=15,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Relevance Score",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )'''
    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=16)
    ax_cbar.yaxis.label.set_size(18)

    # 🔹 Add cluster color stripes
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # 🔹 Cluster size labels
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=18, fontweight='bold'
        )

    # 🔹 X-tick labels below heatmap
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=14)
    ax.tick_params(axis='x', bottom=True, labelbottom=True)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Omics + LRP Legend
    ax_legend.axis("off")
    lrp_patch = Patch(facecolor='#a9cce3', alpha=0.8, label='Saliency Sum')
    ax_legend.legend(
        handles=[lrp_patch],
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=16,
        handleheight=1.5,
        handlelength=3
    )

    # 🔹 Saliency Sum curve
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())
    y = np.arange(len(lrp_sums))

    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='#a9cce3',
        alpha=0.8,
        linewidth=3
    )

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=16)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.hlines(
        y=1.05, xmin=0, xmax=1,
        color='black', linewidth=1.5, transform=ax_curve.get_xaxis_transform()
    )
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Final layout + save
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

    # 🔹 Cluster-wise contribution breakdown
    plot_topo_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,  # Not sorted for per-cluster breakdown
        cluster_labels=cluster_labels,
        feature_names=[f"{i+1:02d}" for i in range(relevance_scores.shape[1])],
        per_cluster_feature_contributions_output_dir=os.path.join(
            os.path.dirname(output_path), "per_cluster_feature_contributions_topo"
        )
    )

    return pd.DataFrame(sorted_scores, index=gene_names, columns=feature_names)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    omics_types = ['cna', 'ge', 'meth', 'mf']
    cancer_types = ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'READ', 'STAD', 'THCA', 'UCEC']

    bio_feat_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]    
    
    topo_feat_names = [f"Topo_{i}" for i in range(64)]

    # Define feature groups
    feature_groups = {
        "bio": (0, 1024),
        "topo": (1024, 2048)
    }
    
    omics_splits = {
        'cna': (0, 15),
        'ge': (16, 31),
        'meth': (32, 47),
        'mf': (48, 63),
    }
    
    epoch_times, cpu_usages, gpu_usages = [], [], []

    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_graph_2048.json')
    ##data_path = os.path.join('../gat/data/json_graphs_omics_mf/', f'{cancer_type}_graph.json')
    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    node_names = list(nodes.keys())

    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    node_id_to_name = {i: name for i, name in enumerate(node_names)}
    graph = dgl.graph(edges, num_nodes=len(nodes))

    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"

    # ✅ Load ground truth cancer gene names
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    
    in_feats = embeddings.shape[1]
    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    if hasattr(model, 'set_graph'):
        model.set_graph(graph)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features = embeddings.to(device)
    labels = labels.to(device).float()
    train_mask = graph.ndata['train_mask'].to(device)


    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048**2 if torch.cuda.is_available() else 0.0

        model.train()
        logits = model(graph, features).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_times.append(time.time() - epoch_start)
        cpu_usages.append(cpu_usage)
        gpu_usages.append(gpu_usage)

        tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}, CPU: {cpu_usage}%, GPU: {gpu_usage:.2f} MB")

    model.eval()
    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()

    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    process_predictions(
        ranking, args,
        "data/796_drivers.txt",
        "data/oncokb_1172.txt",
        "data/ongene_803.txt",
        "data/ncg_8886.txt",
        "data/intogen_23444.txt",
        node_names,
        non_labeled_nodes
    )

    predicted_cancer_genes = [i for i, _ in ranking[:1000]]
    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top predicted nodes: {avg_degree:.2f}")
    else:
        print("No top nodes predicted above the threshold.")


    ###########################################################################################################################################
    #
    # Biclustering for Bio and Topo 
    # 
    ###########################################################################################################################################
    
    # Get neighbors of top predicted genes
    neighbor_indices = set()
    for idx in top_gene_indices:
        neighbors = graph.successors(idx).tolist()
        neighbor_indices.update(neighbors)

    print("Computing relevance scores ...")
        
    ##gene_indices = [name_to_index[gene] for gene in node_names if gene in name_to_index]
    gene_indices = [name_to_index[name] for name in node_names if name in name_to_index]
    relevance_scores = compute_relevance_scores(model, graph, features)

    # Slice relevance scores for biological and topological embeddings
    relevance_scores_bio = relevance_scores[:, :1024]
    relevance_scores_topo = relevance_scores[:, 1024:]
    
    ##relevance_scores = extract_summary_features_np_bio(relevance_scores)
    
        
    print("Generating feature importance plots...")

    top_node_features = embeddings[top_gene_indices].cpu().numpy()  # shape: [num_top_genes, feature_dim]

    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")

    # Make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('data/', exist_ok=True)

    # Extract node names of top-k predicted genes
    node_names_topk = [node_names[i] for i in top_gene_indices]

    # Use dynamic naming based on model/net type and epochs
    bio_output_img = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_BIO_clusters_epo{args.num_epochs}.png")
    topo_output_img = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_TOPO_clusters_epo{args.num_epochs}.png")

    bio_output_path_heatmap = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_BIO_heatmap_epo{args.num_epochs}.png")
    topo_output_path_heatmap = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_TOPO_heatmap_epo{args.num_epochs}.png")
    
    bio_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_BIO_clustered_graph.pth")
    topo_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_TOPO_clustered_graph.pth")

    bio_cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_BIO_cluster_labels.npy")
    topo_cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_TOPO_cluster_labels.npy")

    bio_total_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_BIO_total_genes_per_cluster.npy")
    topo_total_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_TOPO_total_genes_per_cluster.npy")

    bio_pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_BIO_pred_counts.npy")
    topo_pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_TOPO_pred_counts.npy")

    plot_silhouette_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")
    #best_k = 10#find_optimal_k(top_node_features, k_range=(5, 20), plot_path=plot_silhouette_path, save_best_k_path=best_k_path)
    # Automatically determine best_k via eigengap
    plot_eigengap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_eigengap_epo{args.num_epochs}.png")
    #best_k=eigengap_analysis(top_node_features, max_clusters=25, normalize=True, plot_path=plot_eigengap_path)
    
    best_k=10


    bio_feats = graph.ndata['feat'][:, :1024]  # biological features
    topo_feats = graph.ndata['feat'][:, 1024:] # topology features

    bio_embeddings_np = bio_feats.cpu().numpy()
    summary_bio_features = extract_summary_features_np_bio(bio_embeddings_np)  # shape [num_nodes, 64]
    assert summary_bio_features.shape[1] == 64, f"Expected 64 summary features, got {summary_bio_features.shape[1]}"

    # === Normalize features without altering the original input
    #summary_bio_features_norm = StandardScaler().fit_transform(summary_bio_features)
    summary_bio_features_norm = summary_bio_features
    
    
    topo_embeddings_np = topo_feats.cpu().numpy()
    summary_topo_features = extract_summary_features_np_topo(topo_embeddings_np)  # shape [num_nodes, 64]

    # === BIO BICLUSTERING ===
    graph, cluster_labels_bio, col_labels_bio, bio_total_counts, bio_pred_counts = apply_full_spectral_biclustering_bio(
        graph=graph,
        #bio_embeddings=bio_feats,
        summary_bio_features=summary_bio_features,
        summary_bio_features_norm=summary_bio_features_norm,
        node_names=node_names,
        #feature_names=None,
        predicted_cancer_genes=node_names_topk,
        n_clusters=best_k,
        save_path=bio_graph_path,
        save_cluster_labels_path=bio_cluster_labels_path,
        save_total_genes_per_cluster_path=bio_total_per_cluster_path,
        save_predicted_counts_path=bio_pred_counts_path,
        output_path_genes_clusters=bio_output_img,
        output_path_heatmap=bio_output_path_heatmap
    )

    # Assign cluster labels
    graph.ndata['cluster'] = torch.tensor(cluster_labels_bio)

    # === TOPO BICLUSTERING ===
    graph, cluster_labels_topo, col_labels_topo, topo_total_counts, topo_pred_counts = apply_full_spectral_biclustering_topo(
        graph=graph,
        #topo_embeddings=topo_feats,
        summary_topo_features=summary_topo_features,
        node_names=node_names,
        #feature_names=None,
        predicted_cancer_genes=node_names_topk,
        n_clusters=best_k,
        save_path=topo_graph_path,
        save_cluster_labels_path=topo_cluster_labels_path,
        save_total_genes_per_cluster_path=topo_total_per_cluster_path,
        save_predicted_counts_path=topo_pred_counts_path,
        output_path_genes_clusters=topo_output_img,
        output_path_heatmap=topo_output_path_heatmap
    )

    # Assign cluster labels
    graph.ndata['cluster'] = torch.tensor(cluster_labels_topo)

    ###########################################################################################################################################
    #
    # relevance scores
    # 
    ###########################################################################################################################################


    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.csv")
     
    for cluster_type, cluster_labels, relevance_scores_subset, tag in [
        ("bio", cluster_labels_bio, relevance_scores_bio, "bio"),
        ("topo", cluster_labels_topo, relevance_scores_topo, "topo")
        ]:
        # Assign cluster labels to graph
        graph.ndata[f'cluster_{tag}'] = torch.tensor(cluster_labels)

        # Build cluster-to-gene mappings
        predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)
        cluster_to_genes = {i: [] for i in range(best_k)}
        for idx in predicted_cancer_genes_indices:
            cluster_id = cluster_labels[idx]
            cluster_to_genes[cluster_id].append(node_names[idx])

        # Prepare cluster-degrees dataframe for known and predicted genes
        cluster_labels_np = graph.ndata[f'cluster_{tag}'].cpu().numpy()
        degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

        # Known cancer genes
        kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
        kcg_data = pd.DataFrame({
            "Cluster": cluster_labels_np[kcg_nodes],
            "Interactions": degrees_np[kcg_nodes]
        })

        # Predicted cancer genes
        pcg_nodes = top_gene_indices
        pcg_data = pd.DataFrame({
            "Cluster": cluster_labels_np[pcg_nodes],
            "Interactions": degrees_np[pcg_nodes]
        })

        # Save path for confirmed gene list
        confirmed_genes_save_path = os.path.join(
            output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_{tag}_epo{args.num_epochs}.txt"
        )

        # Prepare for heatmap
        topk_node_indices_tensor = torch.tensor(top_gene_indices, dtype=torch.int64).view(-1)
        relevance_scores_topk = relevance_scores_subset[topk_node_indices_tensor]
        cluster_labels_topk = graph.ndata[f'cluster_{tag}'][topk_node_indices_tensor]
        summary_bio_topk = extract_summary_features_np_bio(relevance_scores_topk.detach().cpu().numpy())
         

        heatmap_path = os.path.join(
            output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_{tag}_epo{args.num_epochs}.png"
        )

        heatmap_path_unsort = os.path.join(
            output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_{tag}_epo{args.num_epochs}_unsort.png"
        )
        
        saliency_ridge_path = os.path.join(
            output_dir, f"{args.model_type}_{args.net_type}_saliency_ridge_{tag}_epo{args.num_epochs}.png"
        )

        ridge_all_clusters_path = os.path.join(
            output_dir, f"{args.model_type}_{args.net_type}_ridge_all_clusters_{tag}_epo{args.num_epochs}.png"
        )

        ridge_from_bio_path_across = os.path.join(
            output_dir, f"{args.model_type}_{args.net_type}_ridge_from_{tag}_across_epo{args.num_epochs}"
        )        

        if tag == "bio":
            plot_bio_biclustering_heatmap(
                args=args,
                relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
                cluster_labels=cluster_labels_topk.cpu().numpy(),
                gene_names=node_names_topk,
                omics_splits=omics_splits,
                output_path=heatmap_path,
                col_labels=col_labels_bio
            )
            
            plot_bio_biclustering_heatmap_unsort(
                args=args,
                relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
                cluster_labels=cluster_labels_topk.cpu().numpy(),
                gene_names=node_names_topk,
                omics_splits=omics_splits,
                output_path=heatmap_path_unsort,
                col_labels=col_labels_bio
            )

            '''plot_bio_clusterwise_feature_contributions(
                args,
                relevance_scores,
                cluster_labels,
                feature_names,
                output_dir
            )'''

            '''cluster_dict = cluster_to_genes  # from earlier code
            
            plot_all_cancer_ridges_all_omics(
                bio_embeddings_np=bio_embeddings_np,
                node_names=node_names,
                #cluster_labels=row_labels,
                best_k=best_k,
                output_base_path=ridge_from_bio_path_across,
                top_n=12
            )
                        
            for cancer_type in cancer_types:
                cancer_feature = extract_all_omics_for_cancer(bio_embeddings_np, cancer_type)
                row_labels = apply_full_spectral_biclustering_cancer(cancer_feature, n_clusters=best_k)
                cluster_dict = make_cluster_dict(row_labels, node_names)
                enrichment_files = run_gprofiler_enrichment(cluster_dict, cancer_type, tag)

                terms = collect_enrichment_with_ratios(cancer_type=cancer_type, tag="bio", source="REAC", top_n=20)
                draw_dot_plot_with_ratio(terms, cancer_type=cancer_type, source="REAC")
                
                ##plot_dot_enrichment_per_cluster(cancer_type, tag, source="REAC", top_n=20)
            
            # Extract 64D summary feature vectors for top-k only
            #summary_bio_topk = extract_summary_features_np_bio(relevance_scores_topk.detach().cpu().numpy())

            save_and_plot_confirmed_genes_bio(
                args=args,
                node_names_topk=node_names_topk,
                node_scores_topk=relevance_scores_topk.detach().cpu().numpy(),
                summary_feature_relevance=summary_bio_topk,
                output_dir=output_dir,
                confirmed_genes_save_path=confirmed_file,
                cluster_labels_topk=cluster_labels_topk,
                tag="bio",
                confirmed_gene_path="data/ncg_8886.txt"
            )

            with open(confirmed_file, "r") as f:
                confirmed_genes = [line.strip() for line in f if line.strip()]
                
            #sankey_bio_output_path=os.path.join(output_dir, f"{args.model_type}_{args.net_type}_dynamic_sankey_bio_epo{args.num_epochs}.html")
            ##selected_genes = ["PLK1", "SKP2", "SRC", "TRAF2"]
            plot_collapsed_clusterfirst_multilevel_sankey_bio(
                args,
                graph,
                node_names,
                name_to_index,
                confirmed_genes,
                scores,
                cluster_labels,
                best_k,
                relevance_scores,
                CLUSTER_COLORS
            )

            chord_bio_output_path=os.path.join(output_dir, f"{args.model_type}_{args.net_type}_gene_neighbors_chord_bio_epo{args.num_epochs}.html")
            chord_bio_output_path_png=os.path.join(output_dir, f"{args.model_type}_{args.net_type}_gene_neighbors_chord_bio_epo{args.num_epochs}.png")
            ##selected_genes = ["PLK1", "SKP2", "SRC", "TRAF2"]
            plot_top_confirmed_gene_neighbors_chord(
                graph=graph,
                node_names=node_names,
                name_to_index=name_to_index,
                scores=scores,
                confirmed_genes=confirmed_genes,
                cluster_labels=cluster_labels,
                cluster_colors=CLUSTER_COLORS,
                output_html=chord_bio_output_path,
                output_png=chord_bio_output_path_png
            )

            # ✅ Plot bio neighbor relevance for confirmed genes
            plot_confirmed_neighbors_bio(
                args=args,
                graph=graph,
                node_names=node_names,
                name_to_index=name_to_index,
                confirmed_genes=confirmed_genes,
                scores=scores,
                cluster_labels=cluster_labels_topk,
                total_clusters=best_k,
                relevance_scores=relevance_scores_topk
            )     

            neighbor_saliency_output = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_neighbor_saliency_heatmap_epo{args.num_epochs}.png")

            plot_neighbor_saliency_heatmap(
                graph=graph,
                confirmed_genes=confirmed_genes,
                node_names=node_names,
                name_to_index=name_to_index,
                relevance_scores=relevance_scores, # Full relevance scores (before slicing to topk)
                omics_splits=omics_splits,
                output_path=neighbor_saliency_output
            )

            target_gene_name = "BRCA1"
            target_idx = node_names.index(target_gene_name)

            neighbor_saliencies = compute_neighbor_saliency(model, graph, features, target_idx)
            top_neighbors = sorted(neighbor_saliencies.items(), key=lambda x: x[1], reverse=True)[:5]
            top_neighbor_indices = [n_idx for n_idx, _ in top_neighbors]

            G_sub = build_subgraph(graph, target_idx, top_neighbor_indices)

            saliency_graph_save_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_saliency_graph_{target_gene_name}_epo{args.num_epochs}.png")
            '''                
        else:
            plot_topo_biclustering_heatmap(
                args=args,
                relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
                cluster_labels=cluster_labels_topk.cpu().numpy(),
                gene_names=node_names_topk,
                output_path=heatmap_path,
                col_labels=col_labels_topo
            )

            plot_topo_biclustering_heatmap_unsorted(
                args=args,
                relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
                cluster_labels=cluster_labels_topk.cpu().numpy(),
                gene_names=node_names_topk,
                output_path=heatmap_path_unsort,
                col_labels=col_labels_topo
            )
            
            # Extract 64D summary feature vectors for top-k only
            '''summary_topo_topk = extract_summary_features_np_topo(relevance_scores_topk.detach().cpu().numpy())
            
            save_and_plot_confirmed_genes_topo(
                args=args,
                node_names_topk=node_names_topk,
                node_scores_topk=relevance_scores_topk.detach().cpu().numpy(),
                summary_feature_relevance=summary_topo_topk,  # full 1024D, will be reduced inside
                output_dir=output_dir,
                confirmed_genes_save_path=confirmed_file,
                cluster_labels_topk=cluster_labels_topk,
                tag="topo",
                confirmed_gene_path="data/ncg_8886.txt"
            )

            with open(confirmed_file, "r") as f:
                confirmed_genes = [line.strip() for line in f if line.strip()]

            #plot_collapsed_clusterfirst_multilevel_sankey_topo_sorted(
            plot_collapsed_clusterfirst_multilevel_sankey_topo(
                args=args,
                graph=graph,
                node_names=node_names,
                name_to_index=name_to_index,
                confirmed_genes=confirmed_genes,
                scores=scores,
                cluster_labels=cluster_labels,
                total_clusters=best_k,
                relevance_scores=relevance_scores,
                CLUSTER_COLORS=CLUSTER_COLORS                 
            )
                            
            # ✅ Plot topo neighbor relevance for confirmed genes
            plot_confirmed_neighbors_topo(
                args=args,
                graph=graph,
                node_names=node_names,
                name_to_index=name_to_index,
                confirmed_genes=confirmed_genes,
                scores=scores,
                cluster_labels=cluster_labels_topk,
                total_clusters=best_k,
                relevance_scores=relevance_scores_topk
            )
            '''
    ###########################################################################################################################################
    #
    # pcg, kcg interaction
    # 
    ###########################################################################################################################################
    
    # Plot PCG cluster percentages
    plot_pcg_cancer_genes(
        clusters=range(best_k),
        predicted_cancer_genes_count=bio_pred_counts,
        total_genes_per_cluster=bio_total_counts,
        node_names=node_names_topk,
        cluster_labels=cluster_labels_bio,
        output_path=os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_bio_epo{args.num_epochs}.png')
    )

    # Plot known cancer gene percentages
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)
    kcg_counts_bio = {
        i: sum((np.array(cluster_labels_bio) == i) & np.isin(range(len(cluster_labels_bio)), list(gt_indices)))
        for i in range(best_k)
    }
    
    plot_kcg_cancer_genes(
        clusters=range(best_k),
        kcg_count=kcg_counts_bio,
        total_genes_per_cluster=bio_total_counts,
        node_names=node_names_topk,
        cluster_labels=cluster_labels_bio,
        output_path=os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_bio_epo{args.num_epochs}.png')
    )

    # === BIO INTERACTION PLOTS ===
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    cluster_labels_np_bio = np.array(cluster_labels_bio)

    kcg_nodes = [i for i, name in enumerate(node_names_topk) if name in ground_truth_cancer_genes]
    kcg_data_bio = pd.DataFrame({
        "Cluster": cluster_labels_np_bio[kcg_nodes],
        "Interactions": degrees_np[kcg_nodes]
    })

    pcg_data_bio = pd.DataFrame({
        "Cluster": cluster_labels_np_bio[top_gene_indices],
        "Interactions": degrees_np[top_gene_indices]
    })

    suffix = 'bio'
    output_path_interactions_kcgs_bio = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_{suffix}_epo{args.num_epochs}.png')
    output_path_interactions_pcgs_bio = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_{suffix}_epo{args.num_epochs}.png')

    plot_interactions_with_kcgs(kcg_data_bio, output_path_interactions_kcgs_bio)
    plot_interactions_with_pcgs(pcg_data_bio, output_path_interactions_pcgs_bio)

    # Plot PCG cluster percentages
    plot_pcg_cancer_genes(
        clusters=range(best_k),
        predicted_cancer_genes_count=topo_pred_counts,
        total_genes_per_cluster=topo_total_counts,
        node_names=node_names_topk,
        cluster_labels=cluster_labels_topo,
        output_path=os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_topo_epo{args.num_epochs}.png')
    )

    # Plot known cancer gene percentages
    kcg_counts_topo = {
        i: sum((np.array(cluster_labels_topo) == i) & np.isin(range(len(cluster_labels_topo)), list(gt_indices)))
        for i in range(best_k)
    }
    
    plot_kcg_cancer_genes(
        clusters=range(best_k),
        kcg_count=kcg_counts_topo,
        total_genes_per_cluster=topo_total_counts,
        node_names=node_names_topk,
        cluster_labels=cluster_labels_topo,
        output_path=os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_topo_epo{args.num_epochs}.png')
    )

    # === TOPO INTERACTION PLOTS ===
    cluster_labels_np_topo = np.array(cluster_labels_topo)

    kcg_data_topo = pd.DataFrame({
        "Cluster": cluster_labels_np_topo[kcg_nodes],
        "Interactions": degrees_np[kcg_nodes]
    })

    pcg_data_topo = pd.DataFrame({
        "Cluster": cluster_labels_np_topo[top_gene_indices],
        "Interactions": degrees_np[top_gene_indices]
    })

    suffix = 'topo'
    output_path_interactions_kcgs_topo = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_{suffix}_epo{args.num_epochs}.png')
    output_path_interactions_pcgs_topo = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_{suffix}_epo{args.num_epochs}.png')

    plot_interactions_with_kcgs(kcg_data_topo, output_path_interactions_kcgs_topo)
    plot_interactions_with_pcgs(pcg_data_topo, output_path_interactions_pcgs_topo)

    ###########################################################################################################################################
    #
    # pathway enrichment
    # 
    ###########################################################################################################################################
    
    # === Compare BIO and TOPO cluster assignments ===
    print("Comparing BIO vs TOPO cluster assignments on top predicted genes...")

    # Ensure tensors are on CPU
    cluster_labels_bio_topk = graph.ndata['cluster_bio'][topk_node_indices_tensor].cpu().numpy()
    cluster_labels_topo_topk = graph.ndata['cluster_topo'][topk_node_indices_tensor].cpu().numpy()

    # Compute ARI and NMI
    ari_score = adjusted_rand_score(cluster_labels_bio_topk, cluster_labels_topo_topk)
    nmi_score = normalized_mutual_info_score(cluster_labels_bio_topk, cluster_labels_topo_topk)

    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

    # === Contingency matrix ===
    # Plot contingency matrix using modular function
    plot_contingency_matrix(
        cluster_labels_bio_topk,
        cluster_labels_topo_topk,
        ari_score,
        nmi_score,
        output_dir,
        args
    )
    
    #######################################################################################################################################
    
    # ==== Cluster-wise Functional Enrichment ====

    def get_marker_genes_by_cluster(node_names_topk, cluster_labels_topk):
        cluster_to_genes = defaultdict(list)
        for name, label in zip(node_names_topk, cluster_labels_topk):
            cluster_to_genes[label.item()].append(name)
        return cluster_to_genes

    def run_enrichment(cluster_to_genes, organism='hsapiens'):
        gp = GProfiler(return_dataframe=True)
        enrichment_results = {}
        for cluster_id, genes in cluster_to_genes.items():
            if len(genes) >= 5:  # Skip tiny clusters
                enriched = gp.profile(
                    organism=organism,
                    query=genes,
                    sources=['KEGG', 'REAC', 'HP', 'GO:BP'],
                    user_threshold=0.05
                )
                enrichment_results[cluster_id] = enriched
        return enrichment_results

    def summarize_enrichment(enrichment_dict, label):
        for cluster_id, df in enrichment_dict.items():
            top_terms = df[['name', 'p_value']].sort_values('p_value').head(5)
            print(f"\n{label} Cluster {cluster_id} — Top Enriched Terms:")
            print(top_terms.to_string(index=False))

    # Enrichment for bio clusters
    bio_clusters = get_marker_genes_by_cluster(node_names_topk, cluster_labels_bio)
    bio_enrichment = run_enrichment(bio_clusters)
    summarize_enrichment(bio_enrichment, label='Bio')

    # Enrichment for topo clusters
    topo_clusters = get_marker_genes_by_cluster(node_names_topk, cluster_labels_topo)
    topo_enrichment = run_enrichment(topo_clusters)
    summarize_enrichment(topo_enrichment, label='Topo')
    
    ###########################################################################################################    

    # Convert defaultdict to DataFrame
    bio_clusters_df = pd.DataFrame(
        [(cluster, gene) for cluster, genes in bio_clusters.items() for gene in genes],
        columns=["bio_cluster", "gene"]
    )

    topo_clusters_df = pd.DataFrame(
        [(cluster, gene) for cluster, genes in topo_clusters.items() for gene in genes],
        columns=["topo_cluster", "gene"]
    )

    # Then build cluster_gene_map
    cluster_gene_map = {
        'bio': bio_clusters_df.groupby('bio_cluster')['gene'].apply(list).to_dict(),
        'topo': topo_clusters_df.groupby('topo_cluster')['gene'].apply(list).to_dict()
    }

    # Initialize g:Profiler client
    gp = GProfiler(return_dataframe=True)

    # Setup: assuming enrichment_results = {}
    enrichment_results = {}

    for cluster_type, cluster_map in cluster_gene_map.items():
        enrichment_results[cluster_type] = {}
        for cluster_id, genes in cluster_map.items():
            # Skip empty clusters
            if not genes:
                continue
            
            # Perform real enrichment query
            try:
                result_df = gp.profile(
                    organism='hsapiens',
                    query=genes,
                    #sources=['GO:BP', 'REAC', 'KEGG', 'HP'],
                    sources=['KEGG', 'GO:BP', 'REAC', 'HP'],
                    user_threshold=0.05,
                    significance_threshold_method="fdr"
                )

                # Filter and store only significant results
                sig_results = result_df[result_df['p_value'] < 0.05]

                enrichment_results[cluster_type][cluster_id] = sig_results

            except Exception as e:
                print(f"Enrichment failed for {cluster_type.capitalize()} cluster {cluster_id}: {e}")
                enrichment_results[cluster_type][cluster_id] = pd.DataFrame()

    plot_enriched_term_counts(
        enrichment_results=enrichment_results,
        output_path=output_dir,
        model_type=args.model_type,
        net_type=args.net_type,
        num_epochs=args.num_epochs,
        #bio_color='#1f77b4', 
        #topo_color='#ff7f0e'
        #bio_color = '#0077B6',
        #topo_color = '#F15BB5'
        #bio_color = '#F08080',
        #topo_color = '#006400'
        ##topo_color = '#90EE90'
        bio_color='#1f77b4', 
        topo_color='#ff7f0e'
    )
    
    plot_shared_enriched_pathways_venn(
        enrichment_results=enrichment_results,
        output_path=output_dir,
        model_type=args.model_type,
        net_type=args.net_type,
        num_epochs=args.num_epochs,
        #bio_color = '#0077B6',
        #topo_color = '#F15BB5'
        #bio_color = '#F08080',
        #topo_color = '#90EE90'
        #topo_color = '#006400'
        bio_color='#1f77b4', 
        topo_color='#ff7f0e'
    )

    ########################################################################################################################################    

    #save_and_plot_enriched_pathways(enrichment_results, args, output_dir)
    heatmap_df, topo_terms_df, bio_terms_df = save_and_plot_enriched_pathways(enrichment_results, args, output_dir)

    ########################################################################################################################################    

    top_node_embeddings_bio = bio_feats[top_gene_indices].cpu().numpy()

    # Assuming embedding is of shape [N, 1024] or similar
    feature_dim_bio = top_node_embeddings_bio.shape[1]
    top_node_feature_names_bio = [f'feat_{i}' for i in range(feature_dim_bio)]

    relevance_df_bio = pd.DataFrame(
        relevance_scores_topk.numpy(),
        index=node_names_topk,                   # list of gene names
        columns=top_node_feature_names_bio           # list of string feature names
    )

    top_node_embeddings_topo = topo_feats[top_gene_indices].cpu().numpy()

    # Assuming embedding is of shape [N, 1024] or similar
    feature_dim_topo = top_node_embeddings_topo.shape[1]
    top_node_feature_names_topo = [f'feat_{i}' for i in range(feature_dim_topo)]

    relevance_df_topo = pd.DataFrame(
        relevance_scores_topk.numpy(),
        index=node_names_topk,                   # list of gene names
        columns=top_node_feature_names_topo           # list of string feature names
    )
    

    def cluster_dict_to_df(cluster_dict, cluster_label_name):
        rows = []
        for cluster_id, genes in cluster_dict.items():
            rows.extend([(gene, cluster_id) for gene in genes])
        return pd.DataFrame(rows, columns=['gene', cluster_label_name])

    bio_clusters = get_marker_genes_by_cluster(node_names_topk, cluster_labels_bio)
    topo_clusters = get_marker_genes_by_cluster(node_names_topk, cluster_labels_topo)

    bio_clusters_df = cluster_dict_to_df(bio_clusters, 'bio_cluster')
    topo_clusters_df = cluster_dict_to_df(topo_clusters, 'topo_cluster')
    
    # Create heatmaps per cluster set

    visualize_feature_relevance_heatmaps(relevance_df_bio, bio_clusters_df, "results/bio_heatmaps")
    visualize_feature_relevance_heatmaps(relevance_df_topo, topo_clusters_df, "results/topo_heatmaps")

    print("✅ Clustered feature relevance heatmaps saved in 'results/bio_heatmaps' and 'results/topo_heatmaps'")

    ###########################################################################################################################################
    #
    # neighbor relevance
    # 
    ###########################################################################################################################################
    
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    node_id_to_name = {i: name for i, name in enumerate(node_names)}
    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    for gene in confirmed_genes:
        plot_neighbor_relevance_by_mode(
            gene=gene,
            relevance_scores=relevance_scores_bio,
            mode='BIO',
            neighbor_scores=scores,
            neighbors_dict=neighbors_dict,
            name_to_index=name_to_index,
            node_id_to_name=node_id_to_name,
            graph=graph,
            cluster_labels=graph.ndata["cluster_bio"], 
            total_clusters=best_k,            
            args=args
        )

        plot_neighbor_relevance_by_mode(
            gene=gene,
            relevance_scores=relevance_scores_topo,
            mode='TOPO',
            neighbor_scores=scores,
            neighbors_dict=neighbors_dict,
            name_to_index=name_to_index,
            node_id_to_name=node_id_to_name,
            graph=graph,
            cluster_labels=graph.ndata["cluster_topo"], 
            total_clusters=best_k,   
            args=args
        )

    ###########################################################################################################################################
    #
    # feature importance
    # 
    ###########################################################################################################################################
    
    gene_indices = [name_to_index[gene] for gene in confirmed_genes if gene in name_to_index]

    # Compute relevance scores using saliency (Integrated Gradients)
    relevance_scores = compute_relevance_scores_norm(
        model=model,
        graph=graph,
        features=features,
        node_indices=gene_indices,
        normalize=True,
        feature_groups=feature_groups
    )
            
    save_dir = os.path.join(output_dir, "gnnexplainer/")
    os.makedirs(save_dir, exist_ok=True)

    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]

        if node_idx not in relevance_scores:
            print(f"⚠️ No relevance found for {gene} (node {node_idx})")
            continue

        if not isinstance(relevance_scores[node_idx], dict) or \
        "bio" not in relevance_scores[node_idx] or "topo" not in relevance_scores[node_idx]:
            print(f"⚠️ Relevance score for {gene} (node /{node_idx}) is not in expected format.")
            continue

        # Get relevance vectors and reduce
        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]
        print(f"{gene} → Node {node_idx} | Predicted score = {gene_score:.4f}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

        if node_idx not in relevance_scores:
            print(f"⚠️ No relevance found for {gene} (node {node_idx})")
            continue

        plot_saliency_for_gene(
            gene=gene,
            relevance_scores=relevance_scores,
            node_idx=node_idx,
            save_dir=save_dir,
            args=args,
            bio_feat_names=bio_feat_names,
            topo_feat_names=topo_feat_names
        )

    ###########################################################################################################################################
    #
    # Bio vs Topo contribution comaprison
    # 
    ###########################################################################################################################################
    
    # This is the correct dictionary for group-wise aggregation
    groupwise_saliency = defaultdict(list)

    for node_id, group_saliency in relevance_scores.items():
        for group_name, saliency_tensor in group_saliency.items():
            groupwise_saliency[group_name].append(saliency_tensor.cpu().numpy())

    for group_name in groupwise_saliency:
        groupwise_saliency[group_name] = np.stack(groupwise_saliency[group_name], axis=0)  # shape: [N, F]

    bio_scores = groupwise_saliency["bio"]
    topo_scores = groupwise_saliency["topo"]

    if len(bio_scores) > 0 and len(topo_scores) > 0:
        bio_means = np.mean(bio_scores, axis=0)
        topo_means = np.mean(topo_scores, axis=0)

        plot_bio_topo_saliency(
            bio_means,
            topo_means,
            title="Average Feature Relevance for Confirmed Genes (Bio vs Topo)",
            save_path=os.path.join(save_dir, f"{args.model_type}_{args.net_type}_bio_topo_comparison_lineplot_square_{args.num_epochs}.png")
        )

        plot_bio_topo_saliency_cuberoot(
            bio_means,
            topo_means,
            title="Average Feature Relevance for Confirmed Genes (Bio vs Topo)",
            save_path=os.path.join(save_dir, f"{args.model_type}_{args.net_type}_bio_topo_comparison_cuberoot_{args.num_epochs}.png")
        )
