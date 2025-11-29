
import dgl
import torch
import torch.nn as nn
import numpy as np
import os
import time
import psutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import scipy.stats
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
from torch_geometric.nn import GCNConv
from .models import ACGNN, HGDC, EMOGI, MTGCN, GCN, GAT, GraphSAGE, GIN, ChebNet, ChebNetII, DMGNN, FocalLoss
from src.utils import (choose_model, plot_roc_curve, plot_pr_curve, load_graph_data, 
                       load_oncokb_genes, plot_and_analyze, save_and_plot_results)
from venn import venn
from matplotlib_venn import venn3, venn2, venn3_circles, venn2_circles
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from itertools import combinations
import matplotlib.patches as mpatches
from captum.attr import IntegratedGradients
from torch.nn.functional import sigmoid
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
from sklearn.cluster import SpectralBiclustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import minmax_scale
from captum.attr import IntegratedGradients
from dgl.nn import GNNExplainer
import umap
from matplotlib.colors import to_rgba, LinearSegmentedColormap

 

CLUSTER_COLORS = {
    0: '#0077B6',  1: '#0000FF',  2: '#00B4D8',  3: '#48EAC4',
    4: '#F1C0E8',  5: '#B9FBC0',  6: '#32CD32',  7: '#bee1e6',
    8: '#8A2BE2',  9: '#E377C2', 10: '#8EECF5', 11: '#A3C4F3'
}
CLUSTER_COLORS_OMICS = {
    0: '#D62728',  1: '#1F77B4',  2: '#2CA02C',  3: '#9467BD'
}

# ✅ 9 fixed cluster colors
'''CLUSTER_COLORS = {
    0: '#0077B6',  1: '#00B4D8',  2: '#F1C0E8',
    3: '#B9FBC0',  4: '#32CD32', 5: '#8A2BE2',
    6: '#E377C2',  7: '#8EECF5', 8: '#A3C4F3'
}'''
def cluster_and_visualize_predicted_genes(graph, predicted_cancer_genes, node_names, 
                                          output_path_genes_clusters, num_clusters=10):
    """
    Clusters gene embeddings into groups using KMeans, visualizes them with t-SNE, 
    and marks predicted cancer genes with red circles (half the size of non-cancer dots).

    Returns:
        cluster_labels (np.ndarray): Cluster assignments for each gene.
        total_genes_per_cluster (dict): Total number of genes per cluster.
        pred_counts (dict): Number of predicted cancer genes per cluster.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    # Define 10 distinct colors for clusters (excluding red)
    cluster_colors = [
        '#0077B6', '#0000FF', '#00B4D8', '#48EAC4', '#F1C0E8', 
        '#B9FBC0', '#32CD32', '#FFD700', '#8A2BE2', '#E377C2'
    ]
    # Store cluster labels in graph
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)

    # Calculate the total number of genes per cluster
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(num_clusters)}

    # Calculate the number of predicted cancer genes per cluster
    pred_counts = {i: 0 for i in range(num_clusters)}
    '''for gene_idx in predicted_cancer_genes:
        print(f"gene_idx type: {type(gene_idx)}, value: {gene_idx}")
        print(f"cluster_labels type: {type(cluster_labels)}, length: {len(cluster_labels)}")

        cluster_id = cluster_labels[gene_idx]
        pred_counts[cluster_id] += 1'''

    # Convert gene names to their corresponding indices
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_cancer_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    # Ensure valid indices before using them
    for gene_idx in predicted_cancer_gene_indices:
        if 0 <= gene_idx < len(cluster_labels):  
            cluster_id = cluster_labels[gene_idx]
            pred_counts[cluster_id] += 1
        else:
            print(f"Skipping invalid index: {gene_idx}")


    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))

    # Plot clusters (Non-cancer genes)
    non_cancer_dot_size = 100  # Default dot size
    red_circle_size = non_cancer_dot_size / 2  # Half the size

    for cluster_id in range(num_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=CLUSTER_COLORS[cluster_id], 
                    edgecolor='k', s=non_cancer_dot_size, alpha=0.8)

    # Mark predicted cancer genes with red circles (⚪, half the size)
    for gene_idx in predicted_cancer_gene_indices:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=red_circle_size, linewidths=2)


    # Labels and title
    ##plt.title("Gene clustering with predicted cancer genes")
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)

    # Save plot
    plt.savefig(output_path_genes_clusters, bbox_inches="tight")  # Ensure proper cropping
    plt.close()

    print(f"Cluster visualization saved to {output_path_genes_clusters}")

    return cluster_labels, total_genes_per_cluster, pred_counts

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

def plot_interactions_with_pcgs_spine(data, output_path):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).

    Args:
        data (pd.DataFrame): A DataFrame containing 'Cluster' and 'Interactions'.
        output_path (str): Path to save the plot.
    """
    import seaborn as sns

    plt.figure(figsize=(8, 5))

    # Set seaborn style without grid
    sns.set_style("white")

    # Use global CLUSTER_COLORS
    unique_clusters = sorted(data['Cluster'].unique())
    cluster_color_map = {cluster_id: CLUSTER_COLORS[cluster_id] for cluster_id in unique_clusters}

    # Create the box plot with custom colors for each cluster
    ax = sns.boxplot(x='Cluster', y='Interactions', data=data, 
                     hue='Cluster', palette=cluster_color_map, showfliers=False)

    # Overlay scatter plot of individual data points (jittering to reduce overlap)
    sns.stripplot(x='Cluster', y='Interactions', data=data, 
                  color='black', alpha=0.2, jitter=True, size=1.5)

    # Remove the legend manually
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Labels and title
    plt.ylabel("Number of interactions with PCGs", fontsize=14)
    plt.xlabel("")
    
    plt.xticks(rotation=0, ha="right")
    plt.ylim(0, 50)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.close()

    print(f"Plot saved to {output_path}")

def plot_interactions_with_kcgs_spine_rectangle(data, output_path):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).

    Args:
        data (pd.DataFrame): A DataFrame containing 'Cluster' and 'Interactions'.
        output_path (str): Path to save the plot.
    """
    import seaborn as sns

    plt.figure(figsize=(8, 5))

    # Set seaborn style without grid
    sns.set_style("white")

    # Use global CLUSTER_COLORS
    unique_clusters = sorted(data['Cluster'].unique())
    cluster_color_map = {cluster_id: CLUSTER_COLORS[cluster_id] for cluster_id in unique_clusters}

    # Create the box plot with custom colors for each cluster
    ax = sns.boxplot(x='Cluster', y='Interactions', data=data, 
                     hue='Cluster', palette=cluster_color_map, showfliers=False)

    # Overlay scatter plot of individual data points (jittering to reduce overlap)
    sns.stripplot(x='Cluster', y='Interactions', data=data, 
                  color='black', alpha=0.2, jitter=True, size=1.5)

    # Remove the legend manually
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Labels and title
    plt.ylabel("Number of interactions with KCGs", fontsize=14)
    plt.xlabel("")
    
    plt.xticks(rotation=0, ha="right")
    plt.ylim(0, 50)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.close()

    print(f"Plot saved to {output_path}")

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

    plt.ylabel("Number of interactions with PCGs", fontsize=16)
    plt.xlabel("")
    plt.xticks(rotation=0, ha="right")
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

    plt.ylabel("Number of interactions with KCGs", fontsize=16)
    plt.xlabel("")
    plt.xticks(rotation=0, ha="right")
    plt.ylim(0, 50)

    sns.despine()  # 🔻 Remove top/right spines

    plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.close()

    print(f"✅ Plot saved to {output_path}")

def plot_kcg_cancer_genes_narrow_bar(clusters, kcg_count, total_genes_per_cluster, node_names, cluster_labels, output_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Prepare data
    cluster_ids = sorted(clusters)
    total = [total_genes_per_cluster[c] for c in cluster_ids]
    kcgs = [kcg_count.get(c, 0) for c in cluster_ids]
    proportions = [k / t if t > 0 else 0 for k, t in zip(kcgs, total)]

    # Start plot
    plt.figure(figsize=(5, 2))
    sns.set_style("white")  # no grid, clean look

    bars = plt.bar(
        x=cluster_ids,
        height=proportions,
        color=[CLUSTER_COLORS.get(c, '#333333') for c in cluster_ids],
        width=0.4
    )

    # Annotate with raw count
    for bar, cluster_id in zip(bars, cluster_ids):
        height = bar.get_height()
        count = kcg_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count),
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Formatting
    plt.ylabel("KCG %", fontsize=12)
    plt.xlabel("")  # no x-axis label
    plt.xticks(cluster_ids, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, max(proportions) + 0.1)

    sns.despine()  # 🔻 remove top/right spines
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved KCG barplot to {output_path}")

def plot_pcg_cancer_genes(
    clusters,
    predicted_cancer_genes_count,
    total_genes_per_cluster,
    node_names,
    cluster_labels,
    output_path
):
    """
    Plots the percentage of predicted cancer genes (PCGs) per cluster.
    
    Args:
        clusters (iterable): Cluster identifiers.
        predicted_cancer_genes_count (dict): Cluster ID → number of PCGs.
        total_genes_per_cluster (dict): Cluster ID → total number of genes.
        node_names (list): Names of genes corresponding to the predictions.
        cluster_labels (dict): Cluster ID → label (for x-axis).
        output_path (str): Path to save the plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Prepare dataframe
    data = []
    for cluster in clusters:
        pcg_count = predicted_cancer_genes_count.get(cluster, 0)
        total = total_genes_per_cluster.get(cluster, 1)  # prevent division by zero
        percent = 100.0 * pcg_count / total
        label = cluster_labels.get(cluster, f"Cluster {cluster}")
        data.append((label, percent))

    df = pd.DataFrame(data, columns=["Cluster", "Percentage"])

    # Sort by cluster label if needed
    df = df.sort_values("Cluster")

    # Plot
    plt.figure(figsize=(5, 2))
    sns.barplot(
        x="Cluster",
        y="Percentage",
        data=df,
        palette=[CLUSTER_COLORS[int(c.replace("Cluster ", ""))] for c in df["Cluster"]],
        width=0.4
    )

    plt.ylabel("PCG %", fontsize=12)
    plt.xlabel("")  # ⛔️ no x-axis label
    plt.xticks(rotation=0, ha="center", fontsize=12)
    plt.yticks(fontsize=12)
    
    sns.despine()  # 🔻 Remove top/right spines

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved PCG barplot to {output_path}")

def plot_pcg_cancer_genes_with_spine(clusters, predicted_cancer_genes_count, total_genes_per_cluster, node_names, cluster_labels, output_path):
    """
    Plots the percentage of predicted cancer genes per cluster using a fixed color scheme.
    """
    # Convert to NumPy arrays for safe division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_genes_per_cluster = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes per cluster
    percent_predicted = np.divide(predicted_counts, total_genes_per_cluster, where=total_genes_per_cluster > 0)  

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, percent_predicted, 
                   color=[CLUSTER_COLORS.get(c, '#000000') for c in clusters],  # Default to black if cluster not in dictionary
                   edgecolor='black')

    # Add value labels on top of each bar (number of predicted cancer genes per cluster)
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        count = predicted_cancer_genes_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and formatting
    plt.ylabel("Percent of PCGs", fontsize=16)
    plt.xticks(clusters)  
    plt.ylim(0, max(percent_predicted) + 0.1)  

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_kcg_cancer_genes(clusters, kcg_count, total_genes_per_cluster, node_names, cluster_labels, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

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
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Formatting
    ##plt.xlabel("Cluster ID", fontsize=16)
    plt.ylabel("Percent of KCGs", fontsize=16)
    plt.xticks(cluster_ids)
    plt.ylim(0, max(proportions) + 0.1)

    sns.despine()  # 🔻 Remove top/right spines

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def apply_spectral_biclustering_ori(graph, embeddings, node_names, predicted_cancer_genes, n_clusters, save_path, output_path_genes_clusters):
    print(f"Running Spectral Biclustering with {n_clusters} row clusters...")

    node_features = embeddings.cpu().numpy()

    # Run Spectral Biclustering
    bicluster = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state=42)
    bicluster.fit(node_features)

    # Assign cluster labels to graph
    cluster_labels = bicluster.row_labels_
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long)

    print("Spectral Biclustering complete. Cluster labels assigned to graph.")
    print(f"Saving graph to {save_path}")
    torch.save({
        'edges': graph.edges(),
        'features': graph.ndata['feat'],
        'labels': graph.ndata.get('label', None),
        'cluster': graph.ndata['cluster']
    }, save_path)

    # Count total genes per cluster
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(n_clusters)}

    # Count predicted cancer genes per cluster
    pred_counts = {i: 0 for i in range(n_clusters)}
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    for idx in predicted_indices:
        if 0 <= idx < len(cluster_labels):
            cluster_id = cluster_labels[idx]
            pred_counts[cluster_id] += 1
        else:
            print(f"Skipping invalid predicted gene index: {idx}")

    # Visualize with t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(node_features) - 1), random_state=42)
    reduced_embeddings = tsne.fit_transform(node_features)

    plt.figure(figsize=(12, 10))
    for cluster_id in range(n_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1],
                    color=CLUSTER_COLORS.get(cluster_id, "#777777"),
                    edgecolor='k', s=100, alpha=0.8)

    for idx in predicted_indices:
        x, y = reduced_embeddings[idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=50, linewidths=2)

    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.title("Spectral Biclustering of Genes with Predicted Cancer Markers")
    plt.savefig(output_path_genes_clusters, bbox_inches="tight")
    plt.close()
    print(f"Cluster visualization saved to {output_path_genes_clusters}")

    return graph, cluster_labels, total_genes_per_cluster, pred_counts

def apply_spectral_biclustering(graph, embeddings, node_names, predicted_cancer_genes, n_clusters, save_path, save_cluster_labels_path, save_total_genes_per_cluster_path, save_predicted_counts_path, output_path_genes_clusters):
    print(f"Running Spectral Biclustering with {n_clusters} row clusters...")

    node_features = embeddings.cpu().numpy()

    # Run Spectral Biclustering
    bicluster = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state=42)
    bicluster.fit(node_features)

    # Assign cluster labels to graph
    cluster_labels = bicluster.row_labels_
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long)

    # Original graph-saving method (kept unchanged)
    print("Spectral Biclustering complete. Cluster labels assigned to graph.")
    print(f"Saving graph to {save_path}")
    torch.save({
        'edges': graph.edges(),
        'features': graph.ndata['feat'],
        'labels': graph.ndata.get('label', None),
        'cluster': graph.ndata['cluster']
    }, save_path)

    # Save cluster labels separately
    save_cluster_labels(cluster_labels, save_cluster_labels_path)

    # Count total genes per cluster
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(n_clusters)}
    save_total_genes_per_cluster(total_genes_per_cluster, save_total_genes_per_cluster_path)

    # Count predicted cancer genes per cluster
    pred_counts = {i: 0 for i in range(n_clusters)}
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    for idx in predicted_indices:
        if 0 <= idx < len(cluster_labels):
            cluster_id = cluster_labels[idx]
            pred_counts[cluster_id] += 1
        else:
            print(f"Skipping invalid predicted gene index: {idx}")

    # Save predicted counts separately
    save_predicted_counts(pred_counts, save_predicted_counts_path)

    # Visualize with t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(node_features) - 1), random_state=42)
    reduced_embeddings = tsne.fit_transform(node_features)

    plt.figure(figsize=(12, 10))
    for cluster_id in range(n_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1],
                    color=CLUSTER_COLORS.get(cluster_id, "#777777"),
                    edgecolor='k', s=100, alpha=0.8)

    for idx in predicted_indices:
        x, y = reduced_embeddings[idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=50, linewidths=2)

    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.title("Spectral Biclustering of Genes with Predicted Cancer Markers")
    plt.savefig(output_path_genes_clusters, bbox_inches="tight")
    plt.close()
    print(f"Cluster visualization saved to {output_path_genes_clusters}")

    return graph, cluster_labels, total_genes_per_cluster, pred_counts

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

def extract_summary_features_np_(features_np):
    """
    Same as extract_summary_features but works with NumPy arrays instead of PyTorch tensors.
    Args:
        features_np (np.ndarray): shape [num_nodes, 2048]
    Returns:
        np.ndarray: shape [num_nodes, 64]
    """
    num_nodes = features_np.shape[0]
    summary_features = []

    for o_idx in range(4):  # omics
        for c_idx in range(16):  # cancer
            base = o_idx * 16 * 16 + c_idx * 16
            group = features_np[:, base:base + 16]  # [num_nodes, 16]
            max_vals = group.max(axis=1, keepdims=True)  # [num_nodes, 1]
            summary_features.append(max_vals)

    return np.concatenate(summary_features, axis=1)  # [num_nodes, 64]

def save_reduced_feature_relevance_scores_(
    relevance_scores,      # Original 2048D relevance matrix (num_genes x 2048)
    gene_names,            # List of gene names (num_genes)
    output_csv_path        # Output path for CSV file
):
    """
    Reduce 2048D → 64D relevance scores using max-over-16 summarization,
    then save to CSV with structured column names: MF:Breast, CNA:Kidney, etc.
    """
    
    # Use user's own reduction logic
    reduced_scores = extract_summary_features_np(relevance_scores)

    # Column labels: omics × cancer
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck',
        'LungAd', 'Thyroid', 'Bladder', 'Esophagus', 'Liver', 'Uterus',
        'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']
    column_labels = [f"{omics.upper()}:{cancer}" for omics in omics_order for cancer in cancer_names]

    # Create DataFrame and save
    df = pd.DataFrame(reduced_scores, index=gene_names, columns=column_labels)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path)
    print(f"✅ Saved reduced feature relevance matrix to {output_csv_path}")

def plot_spectral_biclustering_heatmap_custom_color(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None
):


    # 🔹 Reduce dimensionality
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Default omics colors
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        }

    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    # 🔹 Custom grayscale colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_grey",
        [(0.0, "#FFFFFF"),
         (0.2, "#DDDDDD"),
         (0.4, "#BBBBBB"),
         (0.6, "#888888"),
         (0.8, "#444444"),
         (1.0, "#000000")]
    )

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(nrows=14, ncols=50)
    ax = fig.add_subplot(gs[0:12, 2:45])
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[4:8, 49])
    ax_legend = fig.add_subplot(gs[13, 2:45])

    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    sns.heatmap(
        sorted_scores,
        cmap=custom_cmap,
        vmin=0,
        vmax=0.1,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax,
        linewidths=2,
        linecolor='black'
    )
    ax_cbar.yaxis.label.set_size(14)

    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=12)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    ax_legend.axis("off")
    omics_patches = [Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()]
    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,
        handlelength=3
    )

    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='darkgray',
        alpha=1,
        linewidth=3
    )
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

    plot_clusterwise_feature_contributions(
        relevance_scores=relevance_scores,
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )

    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def plot_spectral_biclustering_heatmap_too_small_colorbar_number(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os
    import pandas as pd

    # 🔹 Reduce dimensionality
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default omics colors
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        }

    # 🔹 Cancer types and omics order
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Build pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # 🔹 Sort by cluster
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

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
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(nrows=14, ncols=50)
    ax = fig.add_subplot(gs[0:12, 2:45])
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[4:8, 49])
    ax_legend = fig.add_subplot(gs[13, 2:45])

    # 🔹 Compute cluster boundaries
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Use dynamic vmax for contrast
    vmax = np.percentile(sorted_scores, 99)

    # 🔹 Plot heatmap
    sns.heatmap(
        sorted_scores,
        cmap='binary',
        vmin=0,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )
    ax_cbar.yaxis.label.set_size(14)

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
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # 🔹 Color x-tick labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=16)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Omics legend
    ax_legend.axis("off")
    omics_patches = [Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()]
    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,
        handlelength=3
    )

    # 🔹 LRP curve (enhanced)
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='darkgray',
        alpha=1,
        linewidth=3
    )
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
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
    plot_clusterwise_feature_contributions(
        relevance_scores=relevance_scores,
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )

    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def plot_spectral_biclustering_heatmap_oriii_percentile(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os
    import pandas as pd

    # 🔹 Reduce dimensionality
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default omics colors
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        }

    # 🔹 Cancer types and omics order
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Build pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # 🔹 Sort by cluster
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

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
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(nrows=14, ncols=50)
    ax = fig.add_subplot(gs[0:12, 2:45])
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[4:8, 49])
    ax_legend = fig.add_subplot(gs[13, 2:45])

    # 🔹 Compute cluster boundaries
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Use dynamic vmax for contrast
    vmax = np.percentile(sorted_scores, 99)

    # 🔹 Plot heatmap
    sns.heatmap(
        sorted_scores,
        cmap='binary',
        vmin=0,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )
    ax_cbar.yaxis.label.set_size(14)

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
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # 🔹 Color x-tick labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=14)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Omics legend
    ax_legend.axis("off")
    omics_patches = [Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()]
    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,
        handlelength=3
    )

    # 🔹 LRP curve
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(y, 0, lrp_sums, color='gray', alpha=0.8, linewidth=0)
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
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
    plot_clusterwise_feature_contributions(
        relevance_scores=relevance_scores,
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )

    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def plot_spectral_biclustering_heatmap_(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None
):
    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os
    import pandas as pd

    # 🔹 Reduce dimensionality (e.g., 2048D → 64D) using summary logic
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default colors for omics types if not provided
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        }

    # 🔹 List of cancer types used for x-axis labels
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]

    # 🔹 Omics order
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Construct pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # 🔹 Sort samples by cluster
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # 🔹 Feature colors based on omics type
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    # 🔹 Create the figure
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(nrows=14, ncols=50)

    ax = fig.add_subplot(gs[0:12, 2:45])
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[4:8, 49])
    ax_legend = fig.add_subplot(gs[13, 2:45])

    # 🔹 Cluster boundaries
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Plot heatmap with 'viridis' colormap
    sns.heatmap(
        sorted_scores,
        cmap='viridis',
        vmin=0,
        vmax=np.max(sorted_scores),
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax,
        linewidths=2,
        linecolor='black'
    )
    ax_cbar.yaxis.label.set_size(14)

    # 🔹 Cluster stripes
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # 🔹 Cluster sample counts
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # 🔹 Colored x-tick labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=14)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Legend
    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,
        handlelength=3
    )

    # 🔹 LRP curve
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='#DAF7A6',
        alpha=1,
        linewidth=3
    )

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
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

    # 🔹 Optional feature contribution plot
    plot_clusterwise_feature_contributions(
        relevance_scores=relevance_scores,
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )

    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def plot_spectral_biclustering_heatmap_dark_thicker(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None  # <-- Add this
):
    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os

    # 🔹 Reduce dimensionality (e.g., 2048D → 64D) using summary logic
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default colors for omics types if not provided
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',     # Red for mutation features
            'cna': '#1F77B4',    # Blue for copy number alterations
            'meth': '#2CA02C',   # Green for methylation
            'ge': '#9467BD'      # Purple for gene expression
        }

    # 🔹 List of cancer types used for x-axis labels
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    
    # 🔹 Order in which omics types should appear
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Repeat cancer names in order of omics to create pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # 🔹 Sort samples by cluster for better heatmap visualization
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]     # Reorder relevance scores
    sorted_clusters = cluster_labels[sorted_indices]     # Reorder cluster labels

    # 🔹 Assign feature colors based on omics type for x-axis labels
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])  # Match feature to omics
                break
        else:
            feature_colors.append("#AAAAAA")  # Default gray if not matched

    # 🔹 Create the figure with custom size
    fig = plt.figure(figsize=(18, 16))

    # 🔹 Define grid layout with custom widths and heights
    gs = fig.add_gridspec(
        nrows=14,
        ncols=50
    )

    # 🔹 Create subplots for heatmap, LRP curve, colorbar, and legend
    ax = fig.add_subplot(gs[0:12, 2:45])          # Heatmap (wide)
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)  # Filled curve (medium)
    ax_cbar = fig.add_subplot(gs[4:8, 49])    # Color bar (narrow)
    ax_legend = fig.add_subplot(gs[13, 2:45])     # Legend below heatmap

    # 🔹 Compute cluster boundaries and center points
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Plot heatmap of sorted relevance scores with no opacity and strong contrast
    sns.heatmap(
        sorted_scores,
        cmap='gray_r',  # Use darker and more intense gray
        xticklabels=False,                # Hide x-tick labels for now
        yticklabels=False,                # Hide y-tick labels
        cbar_ax=ax_cbar,                  # Attach to custom colorbar axis
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"  # ← Move label & colorbar to left side
        },
        ax=ax,
        linewidths=2,  # Thicker lines
        linecolor='black'  # Black lines for high contrast
    )
    ax_cbar.yaxis.label.set_size(14)      # Label font size
    
    # 🔹 Add colored cluster stripes on the left side of each sample row with no transparency
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,                            # Position rectangle
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),  # Get cluster color
            clip_on=False
        ))

    # 🔹 Add cluster size labels next to each cluster group
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",                   # Show cluster sample count
            va='center', ha='right', fontsize=12, fontweight='bold'
        )


    # 🔹 Set x-tick positions and apply colored labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=14)  # Larger font size
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)  # Apply omics-based color

    # 🔹 Remove axis titles (optional)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Draw omics legend (mutation, CNA, GE, methylation)
    ax_legend.axis("off")  # Turn off axis box
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]

    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,  # Prevent label overlap
        handlelength=3
    )

    # 🔹 Compute per-sample LRP sums (and normalize to [0, 1])
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())

    # 🔹 Plot filled LRP curve next to heatmap with dark color
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='darkgray',  # Darker curve to match the heatmap
        alpha=1,  # No transparency
        linewidth=3  # Thicker line for emphasis
    )

    # 🔹 Extend range for better curve display
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # 🔹 Set y-range and remove visual clutter from axes
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Adjust layout and spacing
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    # 🔹 Ensure output directory exists and save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save with tight bounding box
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")
    
    # At the end of your main heatmap function:
    plot_clusterwise_feature_contributions(
        relevance_scores=relevance_scores,       # Already reduced to 64D
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )
    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def plot_spectral_biclustering_heatmap_dark_percentile(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os
    import pandas as pd

    # 🔹 Reduce dimensionality (e.g., 2048D → 64D)
    relevance_scores = extract_summary_features_np(relevance_scores)

    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        }

    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]

    omics_order = ['mf', 'cna', 'ge', 'meth']
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(nrows=14, ncols=50)

    ax = fig.add_subplot(gs[0:12, 2:45])
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[4:8, 49])
    ax_legend = fig.add_subplot(gs[13, 2:45])

    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Dynamic vmax for better contrast
    vmax_val = np.percentile(sorted_scores, 99.5)

    sns.heatmap(
        sorted_scores,
        cmap='gray_r',
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax,
        linewidths=2,
        linecolor='black',
        vmin=0,
        vmax=vmax_val  # Apply dynamic contrast
    )
    ax_cbar.yaxis.label.set_size(14)

    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=14)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,
        handlelength=3
    )

    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())

    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='darkgray',
        alpha=1,
        linewidth=3
    )

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

    plot_clusterwise_feature_contributions(
        relevance_scores=relevance_scores,
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )

    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def plot_spectral_biclustering_heatmap_dark(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None  # <-- Add this
):
    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os

    # 🔹 Reduce dimensionality (e.g., 2048D → 64D) using summary logic
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default colors for omics types if not provided
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',     # Red for mutation features
            'cna': '#1F77B4',    # Blue for copy number alterations
            'meth': '#2CA02C',   # Green for methylation
            'ge': '#9467BD'      # Purple for gene expression
        }

    # 🔹 List of cancer types used for x-axis labels
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    
    # 🔹 Order in which omics types should appear
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Repeat cancer names in order of omics to create pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # 🔹 Sort samples by cluster for better heatmap visualization
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]     # Reorder relevance scores
    sorted_clusters = cluster_labels[sorted_indices]     # Reorder cluster labels

    # 🔹 Assign feature colors based on omics type for x-axis labels
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])  # Match feature to omics
                break
        else:
            feature_colors.append("#AAAAAA")  # Default gray if not matched

    # 🔹 Create the figure with custom size
    fig = plt.figure(figsize=(18, 16))

    # 🔹 Define grid layout with custom widths and heights
    gs = fig.add_gridspec(
        nrows=14,
        ncols=50
    )

    # 🔹 Create subplots for heatmap, LRP curve, colorbar, and legend
    ax = fig.add_subplot(gs[0:12, 2:45])          # Heatmap (wide)
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)  # Filled curve (medium)
    ax_cbar = fig.add_subplot(gs[4:8, 49])    # Color bar (narrow)
    ax_legend = fig.add_subplot(gs[13, 2:45])     # Legend below heatmap

    # 🔹 Compute cluster boundaries and center points
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Plot heatmap of sorted relevance scores with no opacity and strong contrast
    sns.heatmap(
        sorted_scores,
        cmap='gray_r',  # Use darker and more intense gray
        xticklabels=False,                # Hide x-tick labels for now
        yticklabels=False,                # Hide y-tick labels
        cbar_ax=ax_cbar,                  # Attach to custom colorbar axis
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"  # ← Move label & colorbar to left side
        },
        ax=ax,
        linewidths=2,  # Thicker lines
        linecolor='black'  # Black lines for high contrast
    )
    ax_cbar.yaxis.label.set_size(14)      # Label font size
    
    # 🔹 Add colored cluster stripes on the left side of each sample row with no transparency
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,                            # Position rectangle
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),  # Get cluster color
            clip_on=False
        ))

    # 🔹 Add cluster size labels next to each cluster group
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",                   # Show cluster sample count
            va='center', ha='right', fontsize=12, fontweight='bold'
        )


    # 🔹 Set x-tick positions and apply colored labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=14)  # Larger font size
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)  # Apply omics-based color

    # 🔹 Remove axis titles (optional)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Draw omics legend (mutation, CNA, GE, methylation)
    ax_legend.axis("off")  # Turn off axis box
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]

    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,  # Prevent label overlap
        handlelength=3
    )

    # 🔹 Compute per-sample LRP sums (and normalize to [0, 1])
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())

    # 🔹 Plot filled LRP curve next to heatmap with dark color
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='darkgray',  # Darker curve to match the heatmap
        alpha=1,  # No transparency
        linewidth=3  # Thicker line for emphasis
    )

    # 🔹 Extend range for better curve display
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # 🔹 Set y-range and remove visual clutter from axes
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Adjust layout and spacing
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    # 🔹 Ensure output directory exists and save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save with tight bounding box
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")
    
    # At the end of your main heatmap function:
    plot_clusterwise_feature_contributions(
        relevance_scores=relevance_scores,       # Already reduced to 64D
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )
    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def plot_spectral_biclustering_heatmap_ori(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None
):
    relevance_scores = extract_summary_features_np(relevance_scores)

    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        }

    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck',
        'LungAd', 'Thyroid', 'Bladder', 'Esophagus', 'Liver', 'Uterus',
        'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # Create pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend(cancer_names)

    # Sort rows by cluster
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # Assign colors by omics type for each feature
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(nrows=18, ncols=18, wspace=0)


    ax = fig.add_subplot(gs[1:16, :16])         # Main heatmap
    ax_curve = fig.add_subplot(gs[1:16, 16:17]) # Sum curve
    ax_legend = fig.add_subplot(gs[-1, :])      # Legend at bottom

    # Cluster info for boundaries and counts
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # Normalize rows
    '''row_min = sorted_scores.min(axis=1, keepdims=True)
    row_max = sorted_scores.max(axis=1, keepdims=True)
    sorted_scores = (sorted_scores - row_min) / (row_max - row_min + 1e-8)'''

    # ✅ Main heatmap
    sns.heatmap(
        sorted_scores,
        ##cmap='Greys',
        cmap='binary',
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )
    ax_curve.set_ylim(ax.get_ylim())  # 🔥 Sync y-axis limits


    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # ✅ Cluster counts on left
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # ✅ Colored x-axis labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=10)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # ✅ Right-hand sum curve
    row_contributions = sorted_scores.sum(axis=1)
    ##row_contributions = row_contributions / np.max(row_contributions) ##skip normalization to max
    ## to keep the natural scale differences.
    offset = 0.15
    '''ax_curve.fill_betweenx(
        np.arange(len(row_contributions)),
        0,
        row_contributions * offset,
        color='gray',
        alpha=0.6
    )
    ax_curve.plot(
        row_contributions * offset,
        np.arange(len(row_contributions)),
        color='gray',
        linewidth=0.6
    )'''

    ax_curve.fill_betweenx(
        np.arange(len(row_contributions)),
        0,
        row_contributions,
        color='gray',
        alpha=0.6
    )
    ax_curve.plot(
        row_contributions,
        np.arange(len(row_contributions)),
        color='gray',
        linewidth=0.01
    )


    ##ax_curve.set_xlim(0, offset)
    ax_curve.set_xlim(0, row_contributions.max() * 1.05) ## dynamically expands the axis to fit the real (unscaled) curve

    ax_curve.set_xticks([])
    ax_curve.set_yticks([])
    ax_curve.set_facecolor("white")
    ax_curve.spines[:].set_visible(False)

    # ✅ Legend (omics types)
    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(handles=omics_patches, loc="center", ncol=len(omics_patches), frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_almost(
    relevance_scores,       # LRP relevance scores matrix (samples x features)
    cluster_labels,         # Cluster assignment for each sample
    feature_names,          # Names of features (e.g., gene symbols)
    omics_splits,           # Dict mapping omics types to feature index ranges
    output_path,            # File path to save the output figure
    omics_colors=None       # Optional: custom colors for omics types
):
    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os

    # 🔹 Reduce dimensionality (e.g., 2048D → 64D) using summary logic
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default colors for omics types if not provided
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',     # Red for mutation features
            'cna': '#1F77B4',    # Blue for copy number alterations
            'meth': '#2CA02C',   # Green for methylation
            'ge': '#9467BD'      # Purple for gene expression
        }

    # 🔹 List of cancer types used for x-axis labels
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    
    # 🔹 Order in which omics types should appear
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Repeat cancer names in order of omics to create pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # 🔹 Sort samples by cluster for better heatmap visualization
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]     # Reorder relevance scores
    sorted_clusters = cluster_labels[sorted_indices]     # Reorder cluster labels

    # 🔹 Assign feature colors based on omics type for x-axis labels
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])  # Match feature to omics
                break
        else:
            feature_colors.append("#AAAAAA")  # Default gray if not matched

    # 🔹 Create the figure with custom size
    fig = plt.figure(figsize=(18, 16))

    # 🔹 Define grid layout with custom widths and heights
    gs = fig.add_gridspec(
        nrows=14,
        ncols=50
    )

    # 🔹 Create subplots for heatmap, LRP curve, colorbar, and legend
    ax = fig.add_subplot(gs[0:12, 2:45])          # Heatmap (wide)
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)  # Filled curve (medium)
    ax_cbar = fig.add_subplot(gs[4:8, 49])    # Color bar (narrow)
    ax_legend = fig.add_subplot(gs[13, 2:45])     # Legend below heatmap


    # 🔹 Compute cluster boundaries and center points
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Plot heatmap of sorted relevance scores
    sns.heatmap(
        sorted_scores,
        #cmap='Greys',  
        cmap='binary', 
        xticklabels=False,                # Hide x-tick labels for now
        yticklabels=False,                # Hide y-tick labels
        cbar_ax=ax_cbar,                  # Attach to custom colorbar axis
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"  # ← Move label & colorbar to left side
        },

        ax=ax
    )
    ax_cbar.yaxis.label.set_size(14)      # Label font size
    
    # 🔹 Add colored cluster stripes on the left side of each sample row
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,                            # Position rectangle
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),  # Get cluster color
            clip_on=False
        ))

    # 🔹 Add cluster size labels next to each cluster group
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",                   # Show cluster sample count
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # 🔹 Set x-tick positions and apply colored labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=12)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)  # Apply omics-based color

    # 🔹 Remove axis titles (optional)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Draw omics legend (mutation, CNA, GE, methylation)
    ax_legend.axis("off")  # Turn off axis box
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]

    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,  # Prevent label overlap
        handlelength=3
    )
    '''pos = ax_legend.get_position()  # Get current position
    new_pos = [pos.x0, pos.y0 - 0.15, pos.width, pos.height]  # Shift down by 10%
    ax_legend.set_position(new_pos)'''
    
    # 🔹 Compute per-sample LRP sums (and normalize to [0, 1])
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())

    # 🔹 Plot filled LRP curve next to heatmap
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='gray',
        alpha=0.8,
        linewidth=0
    )

    # 🔹 Extend range for better curve display
    ##ax_curve.set_xlim(0, 1.25)

    # 🔹 Show 0–1 tick labels on top of the curve
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # 🔹 Set y-range and remove visual clutter from axes
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Adjust layout and spacing
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()


    # 🔹 Ensure output directory exists and save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_pass(
    relevance_scores,       # LRP relevance scores matrix (samples x features)
    cluster_labels,         # Cluster assignment for each sample
    feature_names,          # Names of features (e.g., gene symbols)
    omics_splits,           # Dict mapping omics types to feature index ranges
    output_path,            # File path to save the output figure
    omics_colors=None       # Optional: custom colors for omics types
):
    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os

    # 🔹 Reduce dimensionality (e.g., 2048D → 64D) using summary logic
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default colors for omics types if not provided
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',     # Red for mutation features
            'cna': '#1F77B4',    # Blue for copy number alterations
            'meth': '#2CA02C',   # Green for methylation
            'ge': '#9467BD'      # Purple for gene expression
        }

    # 🔹 List of cancer types used for x-axis labels
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    
    # 🔹 Order in which omics types should appear
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Repeat cancer names in order of omics to create pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # 🔹 Sort samples by cluster for better heatmap visualization
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]     # Reorder relevance scores
    sorted_clusters = cluster_labels[sorted_indices]     # Reorder cluster labels

    # 🔹 Assign feature colors based on omics type for x-axis labels
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])  # Match feature to omics
                break
        else:
            feature_colors.append("#AAAAAA")  # Default gray if not matched

    # 🔹 Create the figure with custom size
    fig = plt.figure(figsize=(18, 16))

    # 🔹 Define grid layout with custom widths and heights
    gs = fig.add_gridspec(
        nrows=14,
        ncols=50
    )

    # 🔹 Create subplots for heatmap, LRP curve, colorbar, and legend
    ax = fig.add_subplot(gs[0:12, 2:45])          # Heatmap (wide)
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)  # Filled curve (medium)
    ax_cbar = fig.add_subplot(gs[4:8, 49])    # Color bar (narrow)
    ax_legend = fig.add_subplot(gs[13, 2:45])     # Legend below heatmap


    # 🔹 Compute cluster boundaries and center points
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Plot heatmap of sorted relevance scores
    sns.heatmap(
        sorted_scores,
        #cmap='Greys',  
        cmap='binary', 
        xticklabels=False,                # Hide x-tick labels for now
        yticklabels=False,                # Hide y-tick labels
        cbar_ax=ax_cbar,                  # Attach to custom colorbar axis
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"  # ← Move label & colorbar to left side
        },

        ax=ax
    )
    ax_cbar.yaxis.label.set_size(14)      # Label font size
    
    # 🔹 Add colored cluster stripes on the left side of each sample row
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,                            # Position rectangle
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),  # Get cluster color
            clip_on=False
        ))

    # 🔹 Add cluster size labels next to each cluster group
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",                   # Show cluster sample count
            va='center', ha='right', fontsize=12, fontweight='bold'
        )


    # 🔹 Set x-tick positions and apply colored labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=12)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)  # Apply omics-based color

    # 🔹 Remove axis titles (optional)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Draw omics legend (mutation, CNA, GE, methylation)
    ax_legend.axis("off")  # Turn off axis box
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]

    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,  # Prevent label overlap
        handlelength=3
    )
    '''pos = ax_legend.get_position()  # Get current position
    new_pos = [pos.x0, pos.y0 - 0.15, pos.width, pos.height]  # Shift down by 10%
    ax_legend.set_position(new_pos)'''
    
    # 🔹 Compute per-sample LRP sums (and normalize to [0, 1])
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())

    # 🔹 Plot filled LRP curve next to heatmap
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='gray',
        alpha=0.8,
        linewidth=0
    )

    # 🔹 Extend range for better curve display
    ##ax_curve.set_xlim(0, 1.25)

    # 🔹 Show 0–1 tick labels on top of the curve
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # 🔹 Set y-range and remove visual clutter from axes
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Adjust layout and spacing
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()


    # 🔹 Ensure output directory exists and save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save with tight bounding box
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_passs(
    relevance_scores,       # LRP relevance scores matrix (samples x features)
    cluster_labels,         # Cluster assignment for each sample
    feature_names,          # Names of features (e.g., gene symbols)
    omics_splits,           # Dict mapping omics types to feature index ranges
    output_path,            # File path to save the output figure
    omics_colors=None       # Optional: custom colors for omics types
):
    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os

    # 🔹 Reduce dimensionality (e.g., 2048D → 64D) using summary logic
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default colors for omics types if not provided
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',     # Red for mutation features
            'cna': '#1F77B4',    # Blue for copy number alterations
            'meth': '#2CA02C',   # Green for methylation
            'ge': '#9467BD'      # Purple for gene expression
        }

    # 🔹 List of cancer types used for x-axis labels
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    
    # 🔹 Order in which omics types should appear
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Repeat cancer names in order of omics to create pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # 🔹 Sort samples by cluster for better heatmap visualization
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]     # Reorder relevance scores
    sorted_clusters = cluster_labels[sorted_indices]     # Reorder cluster labels

    # 🔹 Assign feature colors based on omics type for x-axis labels
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])  # Match feature to omics
                break
        else:
            feature_colors.append("#AAAAAA")  # Default gray if not matched

    # 🔹 Create the figure with custom size
    fig = plt.figure(figsize=(18, 16))

    # 🔹 Define grid layout with custom widths and heights
    gs = fig.add_gridspec(
        nrows=14,
        ncols=50
    )

    # 🔹 Create subplots for heatmap, LRP curve, colorbar, and legend
    ax = fig.add_subplot(gs[0:12, 2:45])          # Heatmap (wide)
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)  # Filled curve (medium)
    ax_cbar = fig.add_subplot(gs[4:8, 49])    # Color bar (narrow)
    ax_legend = fig.add_subplot(gs[13, 2:45])     # Legend below heatmap


    # 🔹 Compute cluster boundaries and center points
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Plot heatmap of sorted relevance scores
    sns.heatmap(
        sorted_scores,
        #cmap='Greys',  
        cmap='binary', 
        xticklabels=False,                # Hide x-tick labels for now
        yticklabels=False,                # Hide y-tick labels
        cbar_ax=ax_cbar,                  # Attach to custom colorbar axis
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"  # ← Move label & colorbar to left side
        },

        ax=ax
    )
    ax_cbar.yaxis.label.set_size(14)      # Label font size
    
    # 🔹 Add colored cluster stripes on the left side of each sample row
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,                            # Position rectangle
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),  # Get cluster color
            clip_on=False
        ))

    # 🔹 Add cluster size labels next to each cluster group
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",                   # Show cluster sample count
            va='center', ha='right', fontsize=12, fontweight='bold'
        )


    # 🔹 Set x-tick positions and apply colored labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=12)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)  # Apply omics-based color

    # 🔹 Remove axis titles (optional)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Draw omics legend (mutation, CNA, GE, methylation)
    ax_legend.axis("off")  # Turn off axis box
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]

    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,  # Prevent label overlap
        handlelength=3
    )
    '''pos = ax_legend.get_position()  # Get current position
    new_pos = [pos.x0, pos.y0 - 0.15, pos.width, pos.height]  # Shift down by 10%
    ax_legend.set_position(new_pos)'''
    
    # 🔹 Compute per-sample LRP sums (and normalize to [0, 1])
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())

    # 🔹 Plot filled LRP curve next to heatmap
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='gray',
        alpha=0.8,
        linewidth=0
    )

    # 🔹 Extend range for better curve display
    ##ax_curve.set_xlim(0, 1.25)

    # 🔹 Show 0–1 tick labels on top of the curve
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # 🔹 Set y-range and remove visual clutter from axes
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Adjust layout and spacing
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()


    # 🔹 Ensure output directory exists and save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save with tight bounding box
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")
    
    # At the end of your main heatmap function:
    plot_clusterwise_feature_contributions(
        relevance_scores=relevance_scores,       # Already reduced to 64D
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        ##feature_names=[f"{v}" for v in cancer_names],
        output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )

def plot_spectral_biclustering_heatmap_oriii(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None  # <-- Add this
):

    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os

    # 🔹 Reduce dimensionality (e.g., 2048D → 64D) using summary logic
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default colors for omics types if not provided
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',     # Red for mutation features
            'cna': '#1F77B4',    # Blue for copy number alterations
            'meth': '#2CA02C',   # Green for methylation
            'ge': '#9467BD'      # Purple for gene expression
        }

    # 🔹 List of cancer types used for x-axis labels
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    
    # 🔹 Order in which omics types should appear
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Repeat cancer names in order of omics to create pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # 🔹 Sort samples by cluster for better heatmap visualization
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]     # Reorder relevance scores
    sorted_clusters = cluster_labels[sorted_indices]     # Reorder cluster labels

    # 🔹 Assign feature colors based on omics type for x-axis labels
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])  # Match feature to omics
                break
        else:
            feature_colors.append("#AAAAAA")  # Default gray if not matched

    # 🔹 Create the figure with custom size
    fig = plt.figure(figsize=(18, 16))

    # 🔹 Define grid layout with custom widths and heights
    gs = fig.add_gridspec(
        nrows=14,
        ncols=50
    )

    # 🔹 Create subplots for heatmap, LRP curve, colorbar, and legend
    ax = fig.add_subplot(gs[0:12, 2:45])          # Heatmap (wide)
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)  # Filled curve (medium)
    ax_cbar = fig.add_subplot(gs[4:8, 49])    # Color bar (narrow)
    ax_legend = fig.add_subplot(gs[13, 2:45])     # Legend below heatmap


    # 🔹 Compute cluster boundaries and center points
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Plot heatmap of sorted relevance scores
    sns.heatmap(
        sorted_scores,
        #cmap='Greys',  
        cmap='binary', 
        xticklabels=False,                # Hide x-tick labels for now
        yticklabels=False,                # Hide y-tick labels
        cbar_ax=ax_cbar,                  # Attach to custom colorbar axis
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"  # ← Move label & colorbar to left side
        },

        ax=ax
    )
    ax_cbar.yaxis.label.set_size(14)      # Label font size
    
    # 🔹 Add colored cluster stripes on the left side of each sample row
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,                            # Position rectangle
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),  # Get cluster color
            clip_on=False
        ))

    # 🔹 Add cluster size labels next to each cluster group
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",                   # Show cluster sample count
            va='center', ha='right', fontsize=12, fontweight='bold'
        )


    # 🔹 Set x-tick positions and apply colored labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=12)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)  # Apply omics-based color

    # 🔹 Remove axis titles (optional)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Draw omics legend (mutation, CNA, GE, methylation)
    ax_legend.axis("off")  # Turn off axis box
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]

    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,  # Prevent label overlap
        handlelength=3
    )
    '''pos = ax_legend.get_position()  # Get current position
    new_pos = [pos.x0, pos.y0 - 0.15, pos.width, pos.height]  # Shift down by 10%
    ax_legend.set_position(new_pos)'''
    
    # 🔹 Compute per-sample LRP sums (and normalize to [0, 1])
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())

    # 🔹 Plot filled LRP curve next to heatmap
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='gray',
        alpha=0.8,
        linewidth=0
    )

    # 🔹 Extend range for better curve display
    ##ax_curve.set_xlim(0, 1.25)

    # 🔹 Show 0–1 tick labels on top of the curve
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')

    # 🔹 Set y-range and remove visual clutter from axes
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    # 🔹 Adjust layout and spacing
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()


    # 🔹 Ensure output directory exists and save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save with tight bounding box
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")
    
    # At the end of your main heatmap function:
    plot_clusterwise_feature_contributions(
        relevance_scores=relevance_scores,       # Already reduced to 64D
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        ##feature_names=[f"{v}" for v in cancer_names],
        output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )
    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def save_reduced_feature_relevance_scores(
    relevance_scores,      # Original 2048D matrix, shape [N, 2048]
    gene_names,            # List of gene symbols, length N
    output_csv_path        # File to save the reduced 64D matrix
):
    """
    Reduce 2048D LRP scores to 64D using your max-over-16 logic,
    and save to CSV with columns like 'BRCA_mf', ..., 'KIRP_meth'.
    """
    import numpy as np
    import pandas as pd
    import os

    # -- Step 1: Reduce features to 64D
    reduced_scores = extract_summary_features_np_skip(relevance_scores)  # shape [N, 64]

    # -- Step 2: Define feature names in cancer x omics order
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    omics_types = ['mf', 'cna', 'ge', 'meth']
    column_labels = [f"{cancer}_{omics}" for omics in omics_types for cancer in cancer_types]

    # -- Step 3: Build DataFrame and save
    df = pd.DataFrame(reduced_scores, index=gene_names, columns=column_labels)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path)
    print(f"✅ Saved reduced feature relevance matrix to {output_csv_path}")

def plot_gene_feature_contributions_no_probability_added(gene_name, relevance_vector, feature_names, output_path=None):
    """
    Plot a heatmap of feature contributions for a single gene.

    Parameters:
        gene_name (str): The name of the gene (e.g. 'NRAS').
        relevance_vector (np.ndarray): 1D array of relevance scores for each feature (shape: 64, assuming 4 omics × 16 cancers).
        feature_names (list of str): List of all feature names like 'BRCA_mf', 'BRCA_cna', ...
        output_path (str, optional): Path to save the plot.
    """
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    # Create DataFrame
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)

    # Pivot to wide format
    heatmap_data = df.pivot(index='Omics', columns='Cancer', values='Relevance')
    heatmap_data = heatmap_data.loc[['SNV', 'CNA', 'Meth', 'Expr']] if 'SNV' in df['Omics'].values else heatmap_data

    # Plot
    plt.figure(figsize=(8, 2.8))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar=False, linewidths=0.3, linecolor='gray')
    plt.title('Feature Contributions', fontsize=12)
    plt.yticks(rotation=0)
    plt.xticks(rotation=60, ha='right')
    plt.xlabel('')
    plt.ylabel('')
    plt.suptitle(gene_name, fontsize=18, fontweight='bold', x=0.05, y=1.15, ha='left')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_omics_barplot_under_x_label_text(df, output_path=None):
    """
    Plots a bar chart showing the total relevance per omics type.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['Omics', 'Relevance']
        output_path (str): Optional path to save the figure
    """
    omics_order = ['mf', 'cna', 'ge', 'meth']
    omics_colors = {
        'mf': '#D62728',     # Red for mutation features
        'cna': '#1F77B4',    # Blue for copy number alterations
        'meth': '#2CA02C',   # Green for methylation
        'ge': '#9467BD'      # Purple for gene expression
    }

    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)

    # 🔹 Sum relevance for each omics type
    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    # 🔹 Plot
    plt.figure(figsize=(2, 2))
    sns.barplot(
        x=omics_relevance.index,
        y=omics_relevance.values,
        palette=[omics_colors[o] for o in omics_order],
        width=0.3
    )
    ##plt.title("Total Contribution per Omics Type", fontsize=12)
    ##plt.xlabel("Omics Type", fontsize=12)
    ##plt.ylabel("Summed Relevance", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    # 🔹 Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.show()

def plot_omics_barplot_ori(df, output_path=None):
    """
    Plots a bar chart showing the total relevance per omics type.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['Omics', 'Relevance']
        output_path (str): Optional path to save the figure
    """
    omics_order = ['mf', 'cna', 'ge', 'meth']
    omics_colors = {
        'mf': '#D62728',     # Red for mutation features
        'cna': '#1F77B4',    # Blue for copy number alterations
        'meth': '#2CA02C',   # Green for methylation
        'ge': '#9467BD'      # Purple for gene expression
    }

    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)

    # 🔹 Sum relevance for each omics type
    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    # 🔹 Plot
    plt.figure(figsize=(2, 2))
    sns.barplot(
        x=omics_relevance.index,
        y=omics_relevance.values,
        palette=[omics_colors[o] for o in omics_order],
        width=0.3
    )
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel('')  # ⛔️ remove 'Omics' label
    plt.ylabel('')  # optional: remove y-axis label as well
    plt.tight_layout()

    # 🔹 Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.show()

def plot_clusterwise_feature_contributions_ori(
    relevance_scores,           # 2D array (samples x features)
    cluster_labels,             # 1D array of cluster assignments
    feature_names,              # List of feature names (e.g., MF: BRCA, ...)
    output_dir,                 # Output folder to save per-cluster plots
    omics_colors                # Dict of omics type colors (e.g., 'mf': '#D62728')
):
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # Function to get color by omics type prefix
    def get_omics_color(feature_name):
        prefix = feature_name.split(":")[0].lower()
        return omics_colors.get(prefix, "#AAAAAA")

    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in sorted(unique_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_scores = relevance_scores[indices]

        # Average relevance per feature
        avg_contribution = np.mean(cluster_scores, axis=0)
        total_score = np.sum(avg_contribution)

        fig, ax = plt.subplots(figsize=(10, 2.5))
        bars = ax.bar(
            feature_names,
            avg_contribution,
            color=[get_omics_color(name) for name in feature_names]
        )
        ax.set_ylabel("Feature Contribution")
        ax.set_title(
            f"Cluster {cluster_id} (#Genes: {len(indices)}, Avg Feature Contribution: {total_score:.2f})"
        )
        ax.set_xticks(range(len(feature_names)))

        # Get clean labels (e.g., 'Breast' from 'MF: Breast')
        clean_labels = [name.split(":")[1].strip() if ":" in name else name for name in feature_names]
        ax.set_xticklabels(clean_labels, rotation=90)

        # Set each x-label color to match its bar color
        tick_labels = ax.get_xticklabels()
        for label, feature_name in zip(tick_labels, feature_names):
            color = get_omics_color(feature_name)
            label.set_color(color)


        ax.tick_params(axis='x', labelsize=9)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"cluster_{cluster_id}_feature_contributions.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved feature contribution barplot for Cluster {cluster_id} to {save_path}")

def plot_clusterwise_feature_contributions_45(
    args,
    relevance_scores,           # 2D array (samples x features)
    cluster_labels,             # 1D array of cluster assignments
    feature_names,              # List of feature names (e.g., MF: BRCA, ...)
    per_cluster_feature_contributions_output_dir,                 # Output folder to save per-cluster plots
    omics_colors                # Dict of omics type colors (e.g., 'mf': '#D62728')
):
    os.makedirs(per_cluster_feature_contributions_output_dir, exist_ok=True)

    def get_omics_color(feature_name):
        prefix = feature_name.split(":")[0].lower()
        return omics_colors.get(prefix, "#AAAAAA")

    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in sorted(unique_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_scores = relevance_scores[indices]
        avg_contribution = np.mean(cluster_scores, axis=0)
        total_score = np.sum(avg_contribution)

        fig, ax = plt.subplots(figsize=(10, 2.5))

        x = np.linspace(0, 1, len(feature_names))  # even spacing between 0 and 1
        bar_width = 1 / len(feature_names) * 0.95  # slight gap

        bars = ax.bar(
            x,
            avg_contribution,
            width=bar_width,
            color=[get_omics_color(name) for name in feature_names],
            align='center'
        )

        ax.set_ylabel("Feature contribution")
        ax.set_title(
            f"Cluster {cluster_id}Genes {len(indices)} Average Feature Contribution = {total_score:.2f})"
        )

        # X-axis labels
        clean_labels = [name.split(":")[1].strip() if ":" in name else name for name in feature_names]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels, rotation=90)

        # Color the labels
        for label, feature_name in zip(ax.get_xticklabels(), feature_names):
            label.set_color(get_omics_color(feature_name))

        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlim(-bar_width, 1 + bar_width)  # push first and last bars to the edges

        plt.tight_layout()
        save_path = os.path.join(per_cluster_feature_contributions_output_dir, f"{args.model_type}_{args.net_type}_cluster_{cluster_id}_feature_contributions_epo{args.num_epochs}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved feature contribution barplot for Cluster {cluster_id} to {save_path}")

def plot_clusterwise_feature_contributions(
    args,
    relevance_scores,           # 2D array (samples x features)
    cluster_labels,             # 1D array of cluster assignments
    feature_names,              # List of feature names (e.g., MF: BRCA, ...)
    per_cluster_feature_contributions_output_dir,                 # Output folder to save per-cluster plots
    omics_colors                # Dict of omics type colors (e.g., 'mf': '#D62728')
):
    os.makedirs(per_cluster_feature_contributions_output_dir, exist_ok=True)


    def get_omics_color(feature_name):
        prefix = feature_name.split(":")[0].lower()
        return omics_colors.get(prefix, "#AAAAAA")

    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in sorted(unique_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_scores = relevance_scores[indices]
        avg_contribution = np.mean(cluster_scores, axis=0)
        total_score = np.sum(avg_contribution)

        fig, ax = plt.subplots(figsize=(10, 2.5))

        x = np.linspace(0, 1, len(feature_names))  # even spacing between 0 and 1
        bar_width = 1 / len(feature_names) * 0.95  # slight gap

        bars = ax.bar(
            x,
            avg_contribution,
            width=bar_width,
            color=[get_omics_color(name) for name in feature_names],
            align='center'
        )

        ax.set_ylabel("Feature contribution")
        ax.set_title(
            f"Cluster {cluster_id} ({len(indices)} genes, avg = {total_score:.2f})"
        )

        # X-axis labels
        clean_labels = [name.split(":")[1].strip() if ":" in name else name for name in feature_names]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels, rotation=90)

        # Color the labels
        for label, feature_name in zip(ax.get_xticklabels(), feature_names):
            label.set_color(get_omics_color(feature_name))

        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlim(-bar_width, 1 + bar_width)  # push first and last bars to the edges

        plt.tight_layout()
        save_path = os.path.join(per_cluster_feature_contributions_output_dir, f"{args.model_type}_{args.net_type}_cluster_{cluster_id}_feature_contributions_epo{args.num_epochs}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved feature contribution barplot for Cluster {cluster_id} to {save_path}")

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
    tsne = TSNE(n_components=2, random_state=42)
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
        # cluster_scores = top_scores[mask]
        # if cluster_scores.size > 0:
        #     top_idx_in_cluster = np.argmax(cluster_scores)
        #     name = np.array(top_names)[mask][top_idx_in_cluster]
        #     x, y = coords[top_idx_in_cluster]
        #     # Highlight the top 1 with a yellow circle (half the size of the original dot)
        #     plt.scatter(x, y, s=60, edgecolors='yellow', alpha=0.6, linewidth=1, marker='o', color='red')
        #     # Change label text color to red
        #     plt.text(x, y, name, fontsize=9, fontweight='bold', ha='center', va='center', color='black')

    plt.title("t-SNE of Top 1000 Predicted Genes by Cluster", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    # Remove legend
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Top predicted gene t-SNE plot saved to:\n{output_path}")
 
def plot_top_predicted_genes_umap(graph, node_names, scores, output_path, top_k=1000):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores  # Assuming already on CPU or NumPy

    # Get top K predicted genes
    top_indices = np.argsort(scores)[-top_k:]
    top_embeddings = embeddings[top_indices]
    top_clusters = cluster_ids[top_indices]
    top_scores = scores[top_indices]
    top_names = [node_names[i] for i in top_indices]

    # UMAP projection
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(top_embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    rcParams['pdf.fonttype'] = 42  # Prevent font issues in vector graphics

    for c in np.unique(top_clusters):
        mask = top_clusters == c
        coords = umap_coords[mask]
        plt.scatter(
            coords[:, 0], coords[:, 1],
            color=CLUSTER_COLORS.get(c, "#555555"),
            edgecolors='k',
            s=60,
            alpha=0.8
        )

        # Top 1 in this cluster (within top_k)
        # cluster_scores = top_scores[mask]
        # if cluster_scores.size > 0:
        #     top_idx_in_cluster = np.argmax(cluster_scores)
        #     name = np.array(top_names)[mask][top_idx_in_cluster]
        #     x, y = coords[top_idx_in_cluster]
        #     plt.scatter(x, y, s=60, edgecolors='yellow', alpha=0.6, linewidth=1, marker='o', color='red')
        #     plt.text(x, y, name, fontsize=9, fontweight='bold', ha='center', va='center', color='black')

    plt.title("UMAP of Top 1000 Predicted Genes by Cluster", fontsize=14)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Top predicted gene UMAP plot saved to:\n{output_path}")

def plot_tsne_predicted_genes(graph, node_names, scores, output_path, args):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores##.cpu().numpy()

    predicted_mask = scores >= args.score_threshold
    predicted_indices = np.where(predicted_mask)[0]
    predicted_scores = scores[predicted_indices]
    predicted_clusters = cluster_ids[predicted_indices]
    predicted_embeddings = embeddings[predicted_indices]

    tsne = TSNE(n_components=2, random_state=42)
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

def plot_umap_predicted_genes_top5(graph, node_names, scores, output_path, args):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores

    predicted_mask = scores >= args.score_threshold
    predicted_indices = np.where(predicted_mask)[0]
    predicted_scores = scores[predicted_indices]
    predicted_clusters = cluster_ids[predicted_indices]
    predicted_embeddings = embeddings[predicted_indices]

    # UMAP projection
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(predicted_embeddings)

    # Gather top 5 genes per cluster
    top_genes = []
    for c in np.unique(predicted_clusters):
        cluster_mask = predicted_clusters == c
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue
        top_indices = cluster_indices[np.argsort(predicted_scores[cluster_indices])[-5:]]  # top 5
        for idx in top_indices:
            top_genes.append((predicted_indices[idx], umap_coords[idx], c))

    # Plot
    plt.figure(figsize=(12, 8))
    for idx, (node_idx, coord, cluster_id) in enumerate(top_genes):
        color = CLUSTER_COLORS.get(cluster_id, "#333333")
        plt.scatter(coord[0], coord[1], color=color, s=120, edgecolor='k')
        plt.text(coord[0] + 1.5, coord[1], node_names[node_idx], fontsize=9, color=color)

    # Legend
    unique_predicted_clusters = np.unique(predicted_clusters)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=f"Cluster {c}", markersize=10)
        for c, color in CLUSTER_COLORS.items()
        if c in unique_predicted_clusters
    ]
    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove frame and axes
    plt.axis('off')

    plt.title("UMAP of Top 5 Predicted Genes per Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ UMAP plot of top 5 predicted genes per cluster saved to:\n{output_path}")

def plot_umap_predicted_genes_top5_(graph, node_names, scores, output_path, args):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores  # Already on CPU or NumPy

    predicted_mask = scores >= args.score_threshold
    predicted_indices = np.where(predicted_mask)[0]
    predicted_scores = scores[predicted_indices]
    predicted_clusters = cluster_ids[predicted_indices]
    predicted_embeddings = embeddings[predicted_indices]

    # UMAP projection
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(predicted_embeddings)

    # Gather top 5 genes per cluster
    top_genes = []
    for c in np.unique(predicted_clusters):
        cluster_mask = predicted_clusters == c
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue
        top_indices = cluster_indices[np.argsort(predicted_scores[cluster_indices])[-5:]]  # top 5
        for idx in top_indices:
            top_genes.append((predicted_indices[idx], umap_coords[idx], c))

    # Plot
    plt.figure(figsize=(12, 8))
    for idx, (node_idx, coord, cluster_id) in enumerate(top_genes):
        color = CLUSTER_COLORS.get(cluster_id, "#333333")
        plt.scatter(coord[0], coord[1], color=color, s=120, edgecolor='k')
        plt.text(coord[0] + 1.5, coord[1], node_names[node_idx], fontsize=9, color=color)

    # Legend
    unique_predicted_clusters = np.unique(predicted_clusters)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=f"Cluster {c}", markersize=10)
        for c, color in CLUSTER_COLORS.items()
        if c in unique_predicted_clusters
    ]
    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("UMAP of Top 5 Predicted Genes per Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ UMAP plot of top 5 predicted genes per cluster saved to:\n{output_path}")

def plot_gene_feature_contributions_(gene_name, relevance_vector, feature_names, score, output_path=None):
    """
    Plot a heatmap of feature contributions for a single gene.

    Parameters:
        gene_name (str): The name of the gene (e.g. 'NRAS').
        relevance_vector (np.ndarray): 1D array of relevance scores for each feature (shape: 64).
        feature_names (list of str): List of all feature names like 'BRCA_mf', 'BRCA_cna', ...
        score (float): The prediction probability for this gene (e.g. from the model).
        output_path (str, optional): Path to save the plot.
    """
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    # Create DataFrame
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    ##print('df--------------------------------------------------------------------- ',df)
    barplot_path = output_path.replace(".png", "_omics_barplot.png")
    plot_omics_barplot(df, barplot_path)
    
    # Create DataFrame
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)

    # Map omics names if needed
    omics_label_map = {'SNV': 'mf', 'CNA': 'cna', 'Expr': 'ge', 'Meth': 'meth'}
    df['Omics'] = df['Omics'].map(omics_label_map).fillna(df['Omics'])

    # Pivot to wide format
    heatmap_data = df.pivot(index='Omics', columns='Cancer', values='Relevance')
    omics_order = ['mf', 'cna', 'meth', 'ge']
    heatmap_data = heatmap_data.reindex(omics_order) if set(omics_order).intersection(heatmap_data.index) else heatmap_data

    # Plot
    plt.figure(figsize=(8, 2.8))
    sns.heatmap(heatmap_data, cmap='magma', center=0, cbar=False, linewidths=0.3, linecolor='gray')
    ##sns.heatmap(heatmap_data, cmap='cubehelix', center=0, cbar=False, linewidths=0.3, linecolor='gray')
    ##sns.heatmap(heatmap_data, cmap='plasma', center=0, cbar=False, linewidths=0.3, linecolor='gray')
    ##sns.heatmap(heatmap_data, cmap='cubehelix', center=0, cbar=False, linewidths=0.3, linecolor='gray')
    ##sns.heatmap(heatmap_data, cmap='cividis', center=0, cbar=False, linewidths=0.3, linecolor='gray')

    # Titles and axes cleanup
    ##plt.title(f"Prediction Probability: {score:.3f}", fontsize=12)
    plt.title(f"{gene_name} – Prediction Score: {score:.3f}", fontsize=14, fontweight='bold')

    ##plt.suptitle(gene_name, fontsize=18, fontweight='bold', x=0.05, y=1.15, ha='left')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90, ha='right')
    plt.xlabel('')  # ⛔️ remove 'Omics'
    plt.ylabel('')  # ⛔️ remove 'Omics'

    # Save or show
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_gene_feature_contributions__(gene_name, relevance_vector, feature_names, score, output_path=None):
    """
    Plot a heatmap of feature contributions for a single gene.

    Parameters:
        gene_name (str): The name of the gene (e.g. 'NRAS').
        relevance_vector (np.ndarray): 1D array of relevance scores for each feature (shape: 64).
        feature_names (list of str): List of all feature names like 'BRCA_mf', 'BRCA_cna', ...
        score (float): The prediction probability for this gene (e.g. from the model).
        output_path (str, optional): Path to save the plot.
    """
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    # Create DataFrame
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    
    # Save omics barplot
    barplot_path = output_path.replace(".png", "_omics_barplot.png")
    plot_omics_barplot(df, barplot_path)

    # Split into Cancer and Omics columns
    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)

    # Map omics names if needed
    omics_label_map = {'SNV': 'mf', 'CNA': 'cna', 'Expr': 'ge', 'Meth': 'meth'}
    df['Omics'] = df['Omics'].map(omics_label_map).fillna(df['Omics'])

    # Pivot table for heatmap
    heatmap_data = df.pivot(index='Omics', columns='Cancer', values='Relevance')
    omics_order = ['mf', 'cna', 'meth', 'ge']
    heatmap_data = heatmap_data.reindex(omics_order) if set(omics_order).intersection(heatmap_data.index) else heatmap_data

    # Compute average contribution
    total_score = np.mean(np.abs(relevance_vector))

    # Plot
    plt.figure(figsize=(8, 2.8))
    ax = sns.heatmap(
        heatmap_data, cmap='magma', center=0, cbar=False,
        linewidths=0.3, linecolor='gray'
    )

    # Title and axis formatting
    plt.title(f"{gene_name} (prediction score = {score:.3f})", fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel('')
    plt.ylabel('')

    # Add legend box for average score
    ax.text(
        0.01, 1.05,
        f"Average Contribution = {total_score:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )

    # Save or show
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved feature contribution heatmap to {output_path}")
    else:
        plt.show()

def plot_gene_feature_contributions(gene_name, relevance_vector, feature_names, score, output_path=None):
    """
    Plot a heatmap of feature contributions for a single gene.

    Parameters:
        gene_name (str): The name of the gene (e.g. 'NRAS').
        relevance_vector (np.ndarray): 1D array of relevance scores for each feature (shape: 64).
        feature_names (list of str): List of all feature names like 'BRCA_mf', 'BRCA_cna', ...
        score (float): The prediction probability for this gene (e.g. from the model).
        output_path (str, optional): Path to save the plot.
    """
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    # Create DataFrame
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    
    # Save omics barplot
    barplot_path = output_path.replace(".png", "_omics_barplot.png")
    plot_omics_barplot(df, barplot_path)

    # Split into Cancer and Omics columns
    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)

    # Map omics names if needed
    omics_label_map = {'SNV': 'mf', 'CNA': 'cna', 'Expr': 'ge', 'Meth': 'meth'}
    df['Omics'] = df['Omics'].map(omics_label_map).fillna(df['Omics'])

    # Pivot table for heatmap
    heatmap_data = df.pivot(index='Omics', columns='Cancer', values='Relevance')
    omics_order = ['mf', 'cna', 'meth', 'ge']
    heatmap_data = heatmap_data.reindex(omics_order) if set(omics_order).intersection(heatmap_data.index) else heatmap_data

    # Plot
    plt.figure(figsize=(8, 2.8))
    ax = sns.heatmap(
        heatmap_data, cmap='RdBu_r', center=0, cbar=False,
        linewidths=0.3, linecolor='gray'
    )

    # Title and axis formatting
    plt.title(f"{gene_name} (prediction score = {score:.3f})", fontsize=14)#, fontweight='bold')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel('')
    plt.ylabel('')

    # Save or show
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved feature contribution heatmap to {output_path}")
    else:
        plt.show()

def plot_omics_barplot_(df, output_path=None):
    """
    Plots a bar chart showing the total relevance per omics type.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['Feature', 'Relevance']
        output_path (str): Optional path to save the figure
    """
    omics_order = ['mf', 'cna', 'ge', 'meth']
    omics_colors = {
        'mf': '#D62728',
        'cna': '#1F77B4',
        'meth': '#2CA02C',
        'ge': '#9467BD'
    }

    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)
    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    # Bar positions and width
    positions = np.arange(len(omics_order))
    bar_width = 0.2

    plt.figure(figsize=(2, 2))
    plt.bar(
        positions,
        omics_relevance.values,
        color=[omics_colors[o] for o in omics_order],
        width=bar_width
    )
    
    plt.xticks(positions, omics_order, rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('')
    plt.ylabel('')
    sns.despine()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.show()

def plot_feature_importance_perplexity_error(
    relevance_vector,
    feature_names=None,
    node_name=None,
    top_k=9,
    gene_names=None,
    output_path="plots"
):
    # Convert to NumPy array
    if isinstance(relevance_vector, torch.Tensor):
        relevance_vector = relevance_vector.detach().cpu().numpy()
    else:
        relevance_vector = np.array(relevance_vector)

    # 🔄 Min-Max Normalization
    min_val = relevance_vector.min()
    max_val = relevance_vector.max()
    norm_scores = (relevance_vector - min_val) / (max_val - min_val + 1e-8)

    # 🔝 Top-K selection
    top_indices = np.argsort(norm_scores)[-top_k:][::-1]
    top_scores = norm_scores[top_indices]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(relevance_vector))]

    # 🧬 Labeling
    if gene_names is not None:
        top_labels = [gene_names[i].capitalize() if i < len(gene_names) else f"Unknown {i}" for i in top_indices]
    else:
        top_labels = [feature_names[i] for i in top_indices]

    # 🧾 Construct DataFrame
    df = pd.DataFrame({
        "feature": top_labels,
        "relevance": top_scores
    })

    # 🎨 Plot with minimal style to match omics barplot
    plt.figure(figsize=(2, 2))
    sns.barplot(
        data=df,
        x="feature",
        y="relevance",
        palette="Blues_d",
        width=0.3
    )

    plt.xticks(rotation=90, ha='center', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.title(node_name if node_name else '', fontsize=12)

    sns.despine()
    plt.tight_layout()

    # 💾 Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved plot for {node_name} to {output_path}")
    else:
        plt.show()

def plot_omics_barplot_no_rotation(df, output_path=None):
    """
    Plots a bar chart showing the total relevance per omics type.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['Feature', 'Relevance']
        output_path (str): Optional path to save the figure
    """
    omics_order = ['mf', 'cna', 'ge', 'meth']
    omics_colors = {
        'mf': '#D62728',     # Red for mutation features
        'cna': '#1F77B4',    # Blue for copy number alterations
        'meth': '#2CA02C',   # Green for methylation
        'ge': '#9467BD'      # Purple for gene expression
    }

    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)

    # 🔹 Sum relevance for each omics type
    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    # 🔹 Plot
    plt.figure(figsize=(2, 2))
    sns.barplot(
        x=omics_relevance.index,
        y=omics_relevance.values,
        palette=[omics_colors[o] for o in omics_order],
        width=0.3
    )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('', fontsize=12)  # ⛔️ no x-axis label
    plt.ylabel('', fontsize=12)  # ⛔️ no y-axis label
    sns.despine()  # 🔻 remove top/right borders

    plt.tight_layout()

    # 🔹 Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.show()

def plot_omics_barplot(df, output_path=None):
    """
    Plots a bar chart showing the total relevance per omics type.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['Feature', 'Relevance']
        output_path (str): Optional path to save the figure
    """
    omics_order = ['mf', 'cna', 'ge', 'meth']
    omics_colors = {
        'mf': '#D62728',     # Red for mutation features
        'cna': '#1F77B4',    # Blue for copy number alterations
        'meth': '#2CA02C',   # Green for methylation
        'ge': '#9467BD'      # Purple for gene expression
    }

    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)

    # 🔹 Sum relevance for each omics type
    omics_relevance = df.groupby('Omics')['Relevance'].sum().reindex(omics_order)

    # 🔹 Plot
    plt.figure(figsize=(1.3, 2))
    sns.barplot(
        x=omics_relevance.index,
        y=omics_relevance.values,
        palette=[omics_colors[o] for o in omics_order],
        width=0.7  # narrower bars for more spacing
    )

    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    sns.despine()

    plt.tight_layout()

    # 🔹 Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.show()

def plot_feature_importance_(
    relevance_vector,
    feature_names=None,
    node_name=None,
    top_k=9,
    gene_names=None,
    output_path="plots"
):
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
        width=0.6  # Narrower bars
    )

    plt.ylabel("Relevance score", fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.title(f"{node_name}", fontsize=13)

    plt.xticks(rotation=90, ha='center', fontsize=11)
    plt.yticks(fontsize=11)

    sns.despine()  # Clean up spines
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.show()

def plot_feature_importance_ori(
    relevance_vector,
    feature_names=None,
    node_name=None,
    top_k=9,
    gene_names=None,
    output_path="plots"
):
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
        width=0.6  # Narrower bars
    )

    plt.ylabel("Relevance score", fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.title(f"{node_name}", fontsize=13)

    plt.xticks(rotation=90, ha='center', fontsize=11)
    plt.yticks(fontsize=11)

    sns.despine()  # Clean up spines
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.show()

def plot_neighbor_relevance_with_zero(
    neighbor_scores, 
    gene_name="Gene", 
    node_id_to_name=None, 
    output_path=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Sort by score (descending)
    sorted_scores = dict(sorted(neighbor_scores.items(), key=lambda x: -x[1]))
    scores = np.array(list(sorted_scores.values()))
    names = [node_id_to_name.get(nid, f"Node {nid}") if node_id_to_name else f"Node {nid}"
             for nid in sorted_scores.keys()]

    # Use raw scores directly, no normalization
    df = pd.DataFrame({"neighbor": names, "relevance": scores})

    # Plot
    plt.figure(figsize=(3, 2.5))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x="neighbor", y="relevance", palette="coolwarm")

    plt.title(f"Neighbor Relevance for {gene_name}", fontsize=12)
    plt.ylabel("Relevance score")
    plt.xticks(rotation=90, ha='right', fontsize=9)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved neighbor relevance plot to {output_path}")
    else:
        plt.show()

def plot_feature_importance_skinny_line(
    relevance_vector,
    feature_names=None,
    node_name=None,
    top_k=9,
    gene_names=None,
    output_path="plots"
):
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

    plt.figure(figsize=(4, 4))
    sns.barplot(
        data=df,
        x="feature",
        y="relevance",
        palette="Blues_d",
        dodge=False,
        legend=False,
        width=0.3
    )
    plt.ylabel("Relevance score", fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.title(f"{node_name}")
    plt.xticks(rotation=90, ha='center', fontsize=12)
    plt.yticks(fontsize=12)

    sns.despine()  # 🔻 Remove top and right spines (rectangle frame lines)

    plt.tight_layout()

        # 🔹 Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved barplot to {output_path}")
    else:
        plt.show()

def plot_feature_importance_rectangle_spine(
    relevance_vector,
    feature_names=None,
    node_name=None,
    top_k=12,
    gene_names=None,
    output_path="plots"
):
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

    plt.figure(figsize=(6, 5))
    sns.barplot(
        data=df,
        x="feature",
        y="relevance",
        palette="Blues_d",
        dodge=False,
        legend=False,
        width=0.3
    )
    plt.ylabel("Relevance score", fontsize=18)
    plt.xlabel("", fontsize=18)  # Label removed but font size consistent
    plt.title(f"{node_name}")
    plt.xticks(rotation=90, ha='center', fontsize=18)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot for {node_name} to {output_path}")

def plot_feature_importance_ori_(
    relevance_vector,
    feature_names=None,
    node_name=None,
    top_k=12,
    gene_names=None,
    output_path="plots"
):
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

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=df,
        x="feature",
        y="relevance",
        palette="Blues_d",
        dodge=False,
        legend=False,
        width=0.3
    )
    plt.ylabel("Relevance score")
    plt.xlabel("")  # 🚫 Remove 'feature' label from x-axis
    plt.title(f"{node_name}")
    plt.xticks(rotation=90, ha='center')  # ✅ Align text vertically under bars
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot for {node_name} to {output_path}")

def extract_summary_features_np_skip_64(features_np):
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

def get_neighbors_gene_names(graph, node_names, name_to_index, target_genes):
    neighbors_dict = {}

    for gene in target_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in graph.")
            continue

        node_id = name_to_index[gene]

        # For undirected graphs (like PPI), use .neighbors()
        neighbors = graph.successors(node_id).tolist()
        ##neighbors = get_two_hop_neighbors(graph, node_id)
        '''neighbors = list(graph.neighbors(node_id))
        neighbor_indices = set()
        for idx in top_gene_indices:
            neighbors = graph.successors(idx).tolist()  # same as predecessors() for undirected
            neighbor_indices.update(neighbors)'''



        ##neighbors = graph.successors(node_id).tolist() + graph.predecessors(node_id).tolist()


        # Remove self-loop if exists
        neighbor_gene_names = [node_names[n] for n in neighbors if n != node_id]
        neighbors_dict[gene] = neighbor_gene_names

    return neighbors_dict

def compute_integrated_gradients(
    model, graph, features, node_indices=None, baseline=None, steps=50
):
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

def save_cluster_legend_2_row(output_path_legend, cluster_colors, num_clusters=12):
    """
    Creates and saves a separate legend image for cluster colors in two rows (Cluster 1 to 12).

    Args:
        output_path_legend (str): Path to save the legend image.
        cluster_colors (list): List of colors for each cluster.
        num_clusters (int): Number of clusters.
    """
    fig, ax = plt.subplots(figsize=(10, 2))  # Taller figure for two-row legend

    # Create legend handles labeled from Cluster 1 to Cluster 12
    legend_patches = [mpatches.Patch(color=cluster_colors[i], label=f"Cluster {i}") 
                      for i in range(num_clusters)]

    # Display legend with two rows
    ax.legend(handles=legend_patches, loc='center', ncol=num_clusters // 2,
              frameon=False, fontsize=14)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Save legend image
    plt.savefig(output_path_legend, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Legend saved to {output_path_legend}")

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

def compute_saliency_only_one_node(model, g, features, target_node, target_class=None):
    """
    Compute saliency (relevance scores) for a node's prediction using gradients.
    Similar to LRP, this function can handle multiple nodes and return relevance scores for each.

    Args:
        model: Trained GNN model (e.g., ACGNN)
        g: DGL graph
        features: Input node features (requires_grad will be set)
        target_node: Node index to explain
        target_class: If specified, explains relevance for that class. If None, use predicted class.

    Returns:
        relevance_scores: A tensor of relevance scores for each node (of shape [num_nodes, num_features])
    """
    model.eval()
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    features.requires_grad_(True)

    with torch.enable_grad():
        # Forward pass
        logits = model(g, features)

        # Determine the target class for the given node (if not provided)
        if target_class is None:
            target_class = torch.argmax(logits[target_node]).item()

        # Get the score for the target node and target class
        score = logits[target_node, target_class]
        score.backward(retain_graph=True)

        # Initialize a tensor to hold the relevance scores
        relevance_scores = torch.zeros_like(features)

        # Now, handle the backward pass for each node (similar to LRP)
        relevance_scores[target_node] = features.grad[target_node].abs().detach()

    return relevance_scores

def compute_saliency_return1(model, g, features, target_node, target_class=None):
    """
    Compute saliency (relevance scores) for a node's prediction using gradients.

    Args:
        model: Trained GNN model (e.g., ACGNN)
        g: DGL graph
        features: Input node features (requires_grad will be set)
        target_node: Node index to explain
        target_class: If specified, explains relevance for that class. If None, use predicted class.

    Returns:
        A tensor of relevance scores (one per input feature)
    """
    model.eval()
    features = features.clone().detach().requires_grad_(True)

    # Forward pass
    logits = model(g, features)

    if target_class is None:
        target_class = torch.argmax(logits[target_node]).item()

    # Get the score for the node and target class
    score = logits[target_node, target_class]
    score.backward()

    # Relevance = abs(gradient) of input features
    relevance_scores = features.grad[target_node].abs().detach()

    return relevance_scores
    
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

def compute_relevance_scores_4_groups(model, graph, features, node_indices=None, use_abs=True):
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

def plot_neighbor_relevance_spine(
    neighbor_scores, 
    gene_name="Gene", 
    node_id_to_name=None, 
    output_path=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Filter out zero scores
    filtered_scores = {k: v for k, v in neighbor_scores.items() if v != 0}

    # Sort by score (descending)
    sorted_scores = dict(sorted(filtered_scores.items(), key=lambda x: -x[1]))
    scores = np.array(list(sorted_scores.values()))
    names = [node_id_to_name.get(nid, f"Node {nid}") if node_id_to_name else f"Node {nid}"
             for nid in sorted_scores.keys()]

    df = pd.DataFrame({"neighbor": names, "relevance": scores})

    # Plot
    plt.figure(figsize=(3, 2.5))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x="neighbor", y="relevance", palette="coolwarm")

    plt.title(f"Neighbor Relevance for {gene_name}", fontsize=12)
    plt.ylabel("Relevance score")
    plt.xticks(rotation=90, ha='right', fontsize=9)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved neighbor relevance plot to {output_path}")
    else:
        plt.show()

def plot_neighbor_relevance_include_score_eq_0(
    neighbor_scores, 
    gene_name="Gene", 
    node_id_to_name=None, 
    output_path=None
):

    # Filter out zero scores
    filtered_scores = {k: v for k, v in neighbor_scores.items() if v != 0}

    # Sort by score (descending)
    sorted_scores = dict(sorted(filtered_scores.items(), key=lambda x: -x[1]))
    '''scores = np.array(list(sorted_scores.values()))
    names = [node_id_to_name.get(nid, f"Node {nid}") if node_id_to_name else f"Node {nid}"
             for nid in sorted_scores.keys()]
    df = pd.DataFrame({"neighbor": names, "relevance": scores})'''
    ##sorted_scores = dict(sorted(neighbor_scores.items(), key=lambda x: -x[1]))
    scores = np.array(list(sorted_scores.values()))
    names = [node_id_to_name.get(nid, f"Node {nid}") if node_id_to_name else f"Node {nid}" 
             for nid in sorted_scores.keys()]
    ##norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    if scores.size > 0:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    else:
        norm_scores = scores  # or np.zeros_like(scores), depending on your use case

    df = pd.DataFrame({"neighbor": names, "relevance": norm_scores})
    # Plot
    plt.figure(figsize=(3, 2.5))
    sns.set_style("white")

    ax = sns.barplot(data=df, x="neighbor", y="relevance", palette="coolwarm")

    # Only gene name as title
    ax.set_title(f"{gene_name}", fontsize=12)

    # Label formatting
    ax.set_ylabel("Relevance score")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha='right', fontsize=9)

    # Remove all spines and grid
    sns.despine(left=True, bottom=True)
    ax.grid(False)

    # Hide axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved neighbor relevance plot to {output_path}")
    else:
        plt.show()

def plot_neighbor_relevance_with_normalization(neighbor_scores, gene_name="Gene", node_id_to_name=None, output_path=None):
    # Sort and normalize
    sorted_scores = dict(sorted(neighbor_scores.items(), key=lambda x: -x[1]))
    scores = np.array(list(sorted_scores.values()))
    names = [node_id_to_name.get(nid, f"Node {nid}") if node_id_to_name else f"Node {nid}" 
             for nid in sorted_scores.keys()]
    
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    df = pd.DataFrame({"neighbor": names, "relevance": norm_scores})

    # Plot
    plt.figure(figsize=(3, 2.5))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x="neighbor", y="relevance", palette="coolwarm")

    plt.title(f"Neighbor Relevance for {gene_name}", fontsize=12)
    plt.ylabel("Relevance score")
    plt.xticks(rotation=90, ha='right', fontsize=9)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved neighbor relevance plot to {output_path}")
    else:
        plt.show()

def compute_relevance_scores(
    model,
    graph,
    features,
    node_indices,
    normalize=True,
    feature_groups=None  # e.g., {"bio": (0, 1024), "topo": (1024, 2048)}
):
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

def plot_feature_importance_1024(
    relevance_vector,
    feature_names,
    node_name=None,
    top_k=12,
    output_path="plots"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Ensure proper match in lengths
    if len(relevance_vector) != len(feature_names):
        raise ValueError(
            f"Length mismatch: relevance_vector has {len(relevance_vector)} elements, "
            f"but feature_names has {len(feature_names)}"
        )

    # Top-k filtering
    if top_k is not None and top_k < len(relevance_vector):
        top_indices = np.argsort(relevance_vector)[-top_k:][::-1]
        relevance_vector = relevance_vector[top_indices]
        feature_names = [feature_names[i] for i in top_indices]

    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })

    plt.figure(figsize=(max(4, 0.35 * len(feature_names)), 2.5))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, x="feature", y="relevance", palette="coolwarm")

    ax.set_title(f"{node_name}", fontsize=12)
    ax.set_ylabel("Relevance")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha='right', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved feature importance plot to {output_path}")

def plot_feature_importance_no_4_groups_separate(
    relevance_vector,
    feature_names,
    node_name=None,
    top_k=12,
    output_path="plots"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Ensure proper match in lengths
    if len(relevance_vector) != len(feature_names):
        raise ValueError(
            f"Length mismatch: relevance_vector has {len(relevance_vector)} elements, "
            f"but feature_names has {len(feature_names)}"
        )

    # Top-k filtering (optional)
    if top_k is not None and top_k < len(relevance_vector):
        top_indices = np.argsort(relevance_vector)[-top_k:][::-1]
        relevance_vector = relevance_vector[top_indices]
        feature_names = [feature_names[i] for i in top_indices]

    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })

    plt.figure(figsize=(max(4, 0.35 * len(feature_names)), 2.5))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, x="feature", y="relevance", palette="coolwarm")

    # Add horizontal mean line
    mean_relevance = np.mean(relevance_vector)
    ax.axhline(mean_relevance, color="black", linestyle="--", linewidth=1, label=f"Mean: {mean_relevance:.4f}")
    ax.legend(loc="upper right", fontsize=8)

    # Clean up spines
    sns.despine(top=True, right=True)

    ax.set_title(f"{node_name}", fontsize=12)
    ax.set_ylabel("Relevance")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha='right', fontsize=9)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved feature importance plot to {output_path}")

def plot_feature_importance_blue_red(
    relevance_vector,
    feature_names,
    node_name=None,
    top_k=64,
    output_path="plots"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    if len(relevance_vector) != len(feature_names):
        raise ValueError(
            f"Length mismatch: relevance_vector has {len(relevance_vector)} elements, "
            f"but feature_names has {len(feature_names)}"
        )

    if top_k is not None and top_k < len(relevance_vector):
        top_indices = np.argsort(relevance_vector)[-top_k:][::-1]
        relevance_vector = relevance_vector[top_indices]
        feature_names = [feature_names[i] for i in top_indices]

    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })

    # Omics label coloring
    def get_color(label):
        if label.startswith("mf"): return "blue"
        elif label.startswith("cna"): return "orange"
        elif label.startswith("mut"): return "green"
        elif label.startswith("rna"): return "red"
        return "black"

    label_colors = [get_color(f) for f in df["feature"]]

    plt.figure(figsize=(max(4, 0.35 * len(feature_names)), 2.5))
    sns.set_style("white")  # no grid
    ax = sns.barplot(data=df, x="feature", y="relevance", palette="coolwarm")

    # Horizontal mean line
    mean_relevance = np.mean(relevance_vector)
    ax.axhline(mean_relevance, color="black", linestyle="--", linewidth=1, label=f"Mean: {mean_relevance:.4f}")
    ax.legend(loc="upper right", fontsize=8)

    # Clean spines
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Color each x-tick label
    for tick_label, color in zip(ax.get_xticklabels(), label_colors):
        tick_label.set_color(color)

    ax.set_title(f"{node_name}", fontsize=12)
    ax.set_ylabel("Relevance")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha='right', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved feature importance plot to {output_path}")

def plot_feature_importance_same_color(
    relevance_vector,
    feature_names,
    node_name=None,
    output_path="plots"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Ensure proper match
    if len(relevance_vector) != len(feature_names):
        raise ValueError(
            f"Length mismatch: relevance_vector has {len(relevance_vector)} elements, "
            f"but feature_names has {len(feature_names)}"
        )

    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })

    # Assign colors by omics prefix
    def get_color(label):
        if label.startswith("mf"): return "#1f77b4"     # blue
        elif label.startswith("cna"): return "#ff7f0e"   # orange
        elif label.startswith("mut"): return "#2ca02c"   # green
        elif label.startswith("rna"): return "#d62728"   # red
        return "gray"

    df["bar_color"] = df["feature"].apply(get_color)

    plt.figure(figsize=(max(4, 0.35 * len(df)), 2.5))
    sns.set_style("white")  # No grid
    ax = sns.barplot(x=df["feature"], y=df["relevance"], palette=list(df["bar_color"]))

    # Add mean line
    mean_relevance = np.mean(df["relevance"])
    ax.axhline(mean_relevance, color="black", linestyle="--", linewidth=1, label=f"Mean: {mean_relevance:.4f}")
    ax.legend(loc="upper right", fontsize=8)

    # Clean up spines
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Color x-tick labels
    for tick_label, color in zip(ax.get_xticklabels(), df["bar_color"]):
        tick_label.set_color(color)

    ax.set_title(f"{node_name}", fontsize=12)
    ax.set_ylabel("Relevance")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha='right', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved feature importance plot to {output_path}")

def plot_feature_importance_no_order(
    relevance_vector,
    feature_names,
    node_name=None,
    output_path="plots"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Ensure correct match
    if len(relevance_vector) != len(feature_names):
        raise ValueError(
            f"Length mismatch: relevance_vector has {len(relevance_vector)} elements, "
            f"but feature_names has {len(feature_names)}"
        )

    # Create DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })

    # Extract cancer type from each feature
    df["cancer_type"] = df["feature"].apply(lambda x: x.split("_")[0])

    # Get 16 distinct cancer colors
    cancer_types = sorted(df["cancer_type"].unique())
    if len(cancer_types) != 16:
        raise ValueError(f"Expected 16 cancer types, but got {len(cancer_types)}.")

    palette = sns.color_palette("tab20", 20)[:16]
    cancer_to_color = dict(zip(cancer_types, palette))
    df["bar_color"] = df["cancer_type"].map(cancer_to_color)

    # Plot
    plt.figure(figsize=(20, 3))  # wide format for 64 features
    sns.set_style("white")
    ax = sns.barplot(x=df["feature"], y=df["relevance"], palette=list(df["bar_color"]))

    # Mean relevance line
    mean_relevance = np.mean(df["relevance"])
    ax.axhline(mean_relevance, color="black", linestyle="--", linewidth=1, label=f"Mean: {mean_relevance:.4f}")
    ax.legend(loc="upper right", fontsize=8)

    # Clean style
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Color x-tick labels by cancer type
    for tick_label, cancer in zip(ax.get_xticklabels(), df["cancer_type"]):
        tick_label.set_color(cancer_to_color[cancer])

    ax.set_title(node_name, fontsize=12)
    ax.set_ylabel("Relevance")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha='center', fontsize=8)

    # Save
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved feature importance plot to {output_path}")

def plot_feature_importance(
    relevance_vector,
    feature_names,
    node_name=None,
    output_path="plots"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Check lengths
    if len(relevance_vector) != len(feature_names):
        raise ValueError(
            f"Mismatch: relevance_vector has {len(relevance_vector)} elements, "
            f"but feature_names has {len(feature_names)}"
        )

    # DataFrame setup
    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })
    df["cancer_type"] = df["feature"].apply(lambda x: x.split("_")[0])
    df["omics_type"] = df["feature"].apply(lambda x: x.split("_")[1])

    # Define color map for 16 cancer types
    cancer_types = sorted(df["cancer_type"].unique())
    palette = sns.color_palette("tab20", 20)[:16]
    cancer_to_color = dict(zip(cancer_types, palette))
    df["bar_color"] = df["cancer_type"].map(cancer_to_color)

    # Sort by cancer type then by relevance descending within cancer
    df["sort_key"] = df["cancer_type"] + "_" + df["relevance"].rank(ascending=False).astype(int).astype(str)
    df = df.sort_values(by=["cancer_type", "relevance"], ascending=[True, False])
    df.reset_index(drop=True, inplace=True)

    # Plotting
    plt.figure(figsize=(22, 4))
    sns.set_style("white")
    ##ax = sns.barplot(x="feature", y="relevance", data=df, palette=df["bar_color"])

    # Use the palette as a list
    palette = df["bar_color"].tolist()

    # Plot
    ax = sns.barplot(x="feature", y="relevance", data=df, palette=palette)

    # Mean relevance line
    mean_relevance = np.mean(df["relevance"])
    ax.axhline(mean_relevance, color="black", linestyle="--", linewidth=1, label=f"Mean: {mean_relevance:.4f}")
    ax.legend(loc="upper right", fontsize=10)

    # Style and fonts
    sns.despine(top=True, right=True)
    ax.set_title(node_name, fontsize=16)
    ax.set_ylabel("Relevance", fontsize=14)
    ax.set_xlabel("Feature (Cancer_Omics)", fontsize=14)

    # X-tick label coloring and formatting
    for tick, cancer in zip(ax.get_xticklabels(), df["cancer_type"]):
        tick.set_color(cancer_to_color[cancer])
        tick.set_fontsize(10)

    plt.xticks(rotation=90, ha="center")
    plt.yticks(fontsize=10)

    # Save
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved ordered feature importance plot to {output_path}")

def plot_feature_importance_x(relevance_vector, feature_names, node_name, output_path):
    # Create DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })

    # Sort by relevance
    df = df.sort_values("relevance", ascending=False)

    # Plot
    plt.figure(figsize=(16, 6))
    ax = sns.barplot(x="feature", y="relevance", data=df, color="#1f77b4")  # use a single blue color

    # Format
    ax.set_title(f"Feature Importance for {node_name}", fontsize=18)
    ax.set_xlabel("Feature", fontsize=16)
    ax.set_ylabel("Relevance", fontsize=16)
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(False)  # no horizontal lines

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_saliency_heatmap_all(
    relevance_scores: dict,
    gene_list: list,
    name_to_index: dict,
    extract_fn,
    feature_names: list,
    part: str,
    output_dir: str,
    model_type: str,
    net_type: str
):
    """
    Plot a heatmap of saliency values (bio/topo) across confirmed genes.

    Args:
        relevance_scores (dict): {node_idx: {"bio": tensor, "topo": tensor}}
        gene_list (list): List of confirmed gene names.
        name_to_index (dict): Mapping from gene name to node index.
        extract_fn (function): Function to reduce 1024-dim vector to 64-dim.
        feature_names (list): List of 64 feature names.
        part (str): Either "bio" or "topo".
        output_dir (str): Directory to save the heatmap.
        model_type (str): For filename.
        net_type (str): For filename.
    """
    matrix = []
    valid_genes = []

    for gene in gene_list:
        if gene not in name_to_index:
            continue
        node_idx = name_to_index[gene]
        if node_idx not in relevance_scores:
            continue
        vector_1024 = relevance_scores[node_idx][part].cpu().numpy().reshape(1, -1)
        vector_64 = extract_fn(vector_1024).squeeze()
        matrix.append(vector_64)
        valid_genes.append(gene)

    if not matrix:
        print(f"⚠️ No valid relevance vectors found for part: {part}")
        return

    matrix = np.array(matrix)

    # Topo fallback feature names
    if part == "topo":
        feature_names = [f"Topo_{i}" for i in range(matrix.shape[1])]

    df_heatmap = pd.DataFrame(matrix, index=valid_genes, columns=feature_names)

    # Plot
    plt.figure(figsize=(20, max(6, len(valid_genes) * 0.4)))
    ax = sns.heatmap(
        df_heatmap,
        cmap="viridis",
        cbar_kws={"label": "Relevance"},
        linewidths=0.1,
        linecolor='lightgray'
    )

    ax.set_xlabel(f"{part.capitalize()} Features", fontsize=16)
    ax.set_ylabel("Confirmed Genes", fontsize=16)
    ax.tick_params(axis='x', rotation=90, labelsize=10)
    ax.tick_params(axis='y', labelsize=12)
    plt.title(f"Saliency Heatmap ({part.capitalize()} Features)", fontsize=18)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{model_type}_{net_type}_{part}_saliency_heatmap.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved {part} saliency heatmap to: {save_path}")

def plot_saliency_heatmap_top100(
    relevance_scores: dict,
    gene_list: list,
    name_to_index: dict,
    extract_fn,
    feature_names: list,
    part: str,
    output_dir: str,
    model_type: str,
    net_type: str
):
    """
    Plot a heatmap of top-100 saliency values (bio/topo) across confirmed genes.

    Args:
        relevance_scores (dict): {node_idx: {"bio": tensor, "topo": tensor}}
        gene_list (list): List of confirmed gene names.
        name_to_index (dict): Mapping from gene name to node index.
        extract_fn (function): Function to reduce 1024-dim vector to 64-dim.
        feature_names (list): List of 64 feature names.
        part (str): Either "bio" or "topo".
        output_dir (str): Directory to save the heatmap.
        model_type (str): For filename.
        net_type (str): For filename.
    """
    gene_vectors = []

    for gene in gene_list:
        if gene not in name_to_index:
            continue
        node_idx = name_to_index[gene]
        if node_idx not in relevance_scores:
            continue

        vector_1024 = relevance_scores[node_idx][part].cpu().numpy().reshape(1, -1)
        vector_64 = extract_fn(vector_1024).squeeze()
        gene_vectors.append((gene, vector_64))

    if not gene_vectors:
        print(f"⚠️ No valid relevance vectors found for part: {part}")
        return

    # Sort genes by total relevance (L1 norm or mean)
    gene_vectors.sort(key=lambda x: np.mean(np.abs(x[1])), reverse=True)

    # Keep only top 100
    top_genes = gene_vectors[:100]

    valid_genes = [g for g, _ in top_genes]
    matrix = np.stack([vec for _, vec in top_genes])

    # Use generic topo feature names if needed
    if part == "topo":
        feature_names = [f"Topo_{i}" for i in range(matrix.shape[1])]

    df_heatmap = pd.DataFrame(matrix, index=valid_genes, columns=feature_names)

    # Plot
    plt.figure(figsize=(20, max(6, len(valid_genes) * 0.4)))
    ax = sns.heatmap(
        df_heatmap,
        cmap="viridis",
        cbar_kws={"label": "Relevance"},
        linewidths=0.1,
        linecolor='lightgray'
    )

    ax.set_xlabel(f"{part.capitalize()} Features", fontsize=16)
    ax.set_ylabel("Top 100 Genes", fontsize=16)
    ax.tick_params(axis='x', rotation=90, labelsize=10)
    ax.tick_params(axis='y', labelsize=12)
    plt.title(f"Top 100 Saliency Heatmap ({part.capitalize()} Features)", fontsize=18)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{model_type}_{net_type}_{part}_saliency_top100_heatmap.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved {part} saliency heatmap to: {save_path}")

def plot_saliency_biclustering_heatmap(
    relevance_scores: dict,
    gene_list: list,
    name_to_index: dict,
    extract_fn,
    feature_names: list,
    part: str,
    output_dir: str,
    model_type: str,
    net_type: str,
    n_gene_clusters: int = 6,
    n_feature_clusters: int = 4
):
    """
    Bicluster and plot saliency heatmap from top 1000 genes.

    Args:
        relevance_scores (dict): {node_idx: {"bio": tensor, "topo": tensor}}
        gene_list (list): List of confirmed gene names.
        name_to_index (dict): Mapping from gene name to node index.
        extract_fn (function): Function to reduce 1024-dim vector to 64-dim.
        feature_names (list): List of 64 feature names.
        part (str): Either "bio" or "topo".
        output_dir (str): Directory to save the heatmap.
        model_type (str): For filename.
        net_type (str): For filename.
        n_gene_clusters (int): Number of gene (row) clusters.
        n_feature_clusters (int): Number of feature (column) clusters.
    """
    gene_vectors = []

    for gene in gene_list:
        if gene not in name_to_index:
            continue
        node_idx = name_to_index[gene]
        if node_idx not in relevance_scores:
            continue

        vector_1024 = relevance_scores[node_idx][part].cpu().numpy().reshape(1, -1)
        vector_64 = extract_fn(vector_1024).squeeze()
        gene_vectors.append((gene, vector_64))

    if not gene_vectors:
        print(f"⚠️ No valid relevance vectors found for part: {part}")
        return

    # Sort and take top 1000 by relevance score
    gene_vectors.sort(key=lambda x: np.mean(np.abs(x[1])), reverse=True)
    top_genes = gene_vectors[:1000]
    valid_genes = [g for g, _ in top_genes]
    matrix = np.stack([vec for _, vec in top_genes])

    if part == "topo":
        feature_names = [f"Topo_{i}" for i in range(matrix.shape[1])]

    # Apply biclustering
    model = SpectralBiclustering(n_clusters=(n_gene_clusters, n_feature_clusters), method="log", random_state=0)
    model.fit(matrix)

    # Reorder matrix according to biclustering
    row_idx = np.argsort(model.row_labels_)
    col_idx = np.argsort(model.column_labels_)
    reordered_matrix = matrix[row_idx][:, col_idx]
    reordered_genes = [valid_genes[i] for i in row_idx]
    reordered_features = [feature_names[i] for i in col_idx]

    # Create DataFrame for heatmap
    df_heatmap = pd.DataFrame(reordered_matrix, index=reordered_genes, columns=reordered_features)

    # Plot
    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(
        df_heatmap,
        cmap="viridis",
        cbar_kws={"label": "Relevance"},
        xticklabels=True,
        yticklabels=False,
        linewidths=0.05,
        linecolor='lightgray'
    )

    ax.set_xlabel(f"{part.capitalize()} Features (Clustered)", fontsize=16)
    ax.set_ylabel("Top 1000 Genes (Clustered)", fontsize=16)
    ax.tick_params(axis='x', rotation=90, labelsize=10)
    plt.title(f"Top 1000 {part.capitalize()} Gene Saliency (Biclustering)", fontsize=18)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{model_type}_{net_type}_{part}_saliency_bicluster_heatmap.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved {part} biclustered heatmap to: {save_path}")

def plot_feature_importance_bio_(relevance_vector, feature_names, node_name=None, output_path="plots"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Validate
    if len(relevance_vector) != len(feature_names):
        raise ValueError(f"Mismatch: {len(relevance_vector)} values vs {len(feature_names)} names")

    # DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })
    df["omics_type"] = df["feature"].apply(lambda x: x.split("_")[1])
    df["cancer_type"] = df["feature"].apply(lambda x: x.split("_")[0])

    # Color by omics type (4 colors)
    omics_palette = sns.color_palette("Set2", 4)
    omics_to_color = dict(zip(['mf', 'cna', 'ge', 'meth'], omics_palette))
    df["bar_color"] = df["omics_type"].map(omics_to_color)

    # Sort for nice grouping
    df = df.sort_values(by=["omics_type", "cancer_type", "feature"], ascending=True).reset_index(drop=True)

    # Plot
    plt.figure(figsize=(24, 5))
    ax = sns.barplot(x="feature", y="relevance", data=df, palette=df["bar_color"].tolist())
    ax.set_title(f"{node_name} (Biological)", fontsize=16)
    ax.set_ylabel("Relevance", fontsize=14)
    ax.set_xlabel("Feature (Cancer_Omics_Index)", fontsize=14)

    for tick, omics in zip(ax.get_xticklabels(), df["omics_type"]):
        tick.set_color(omics_to_color[omics])
        tick.set_fontsize(8)

    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved BIO plot to {output_path}")

def plot_feature_importance_topo_(relevance_vector, feature_names, node_name=None, output_path="plots"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    if len(relevance_vector) != len(feature_names):
        raise ValueError(
            f"Mismatch: relevance_vector has {len(relevance_vector)} elements, "
            f"but feature_names has {len(feature_names)}"
        )

    # Rename features to emb_0, emb_1, ...
    df = pd.DataFrame({
        "feature": [f"emb_{i}" for i in range(len(feature_names))],
        "relevance": relevance_vector
    })

    # Plotting
    plt.figure(figsize=(22, 4))
    sns.set_style("white")
    ax = sns.barplot(x="feature", y="relevance", data=df, palette="Blues_d")

    # Optional: Mean relevance line
    mean_relevance = np.mean(df["relevance"])
    ax.axhline(mean_relevance, color="black", linestyle="--", linewidth=1, label=f"Mean: {mean_relevance:.4f}")
    ax.legend(loc="upper right", fontsize=10)

    # Titles and labels
    ax.set_title(node_name, fontsize=16)
    ax.set_ylabel("Relevance", fontsize=14)
    ax.set_xlabel("Topological Embedding Feature", fontsize=14)

    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=10)

    sns.despine()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved TOPO feature importance plot to {output_path}")

def plot_feature_importance_topo_mean_outsid(
    relevance_vector,
    feature_names,
    node_name=None,
    output_path="plots"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    if len(relevance_vector) != len(feature_names):
        raise ValueError("Length mismatch between relevance vector and feature names.")

    pretty_labels = [f"emb_{i}" for i in range(len(relevance_vector))]

    df = pd.DataFrame({
        "feature": pretty_labels,
        "relevance": relevance_vector
    })

    plt.figure(figsize=(24, 5))
    sns.set_style("white")
    bars = plt.bar(df["feature"], df["relevance"], color="#607D8B")  # bluish-gray

    # Mean line without label box
    mean_val = np.mean(df["relevance"])
    plt.axhline(mean_val, color="black", linestyle="--", linewidth=1)
    plt.text(
        len(df) - 1, mean_val + 0.005,
        f"Mean: {mean_val:.4f}",
        ha="right", va="bottom", fontsize=12, color="black", bbox=None  # ← no box
    )

    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Topology Embedding Dimension", fontsize=14)
    plt.ylabel("Relevance", fontsize=14)
    plt.title(node_name, fontsize=16)
    plt.margins(x=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved TOPO feature importance plot to {output_path}")

def plot_spectral_biclustering_heatmap_64(
    args,
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None
):
    # 🔹 Reduce dimensionality
    relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default omics colors
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD',
            'topo': '#FF7F0E'   # orange, for topo
        }

    # 🔹 Cancer types and omics order
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Build pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # 🔹 Sort by cluster
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

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
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(nrows=14, ncols=50)
    ax = fig.add_subplot(gs[0:12, 2:45])
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[4:8, 49])
    ax_legend = fig.add_subplot(gs[13, 2:45])

    # 🔹 Compute cluster boundaries
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Use dynamic vmax for contrast
    vmax = np.percentile(sorted_scores, 99)

    from matplotlib.colors import LinearSegmentedColormap

    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient",
        ["#F0F3F4", "#85929e"]  # light gray → bluish gray
    )

    # 🔹 Plot heatmap
    sns.heatmap(
        sorted_scores,
        #cmap='binary',
        cmap=bluish_gray_gradient,
        vmin=0,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )
    ax_cbar.yaxis.label.set_color("#85929e")   # Match color
    ax_cbar.tick_params(colors="#85929e", labelsize=14)
    ax_cbar.yaxis.label.set_size(14)

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
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # 🔹 Color x-tick labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=14)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Omics legend
    ax_legend.axis("off")
    omics_patches = [Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()]
    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,
        handlelength=3
    )

    # 🔹 LRP curve
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='#7b241c',
        alpha=0.8,           # no transparency
        linewidth=3        # thicker line
    )

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
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
    plot_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        per_cluster_feature_contributions_output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )

    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def plot_feature_importance_bio_16_color(relevance_vector, feature_names, node_name=None, output_path="plots"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Validate
    if len(relevance_vector) != len(feature_names):
        raise ValueError(f"Mismatch: {len(relevance_vector)} values vs {len(feature_names)} names")

    # DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })
    df["omics_type"] = df["feature"].apply(lambda x: x.split("_")[1])
    df["cancer_type"] = df["feature"].apply(lambda x: x.split("_")[0])

    # Color by omics type (4 colors)
    omics_palette = sns.color_palette("Set2", 4)
    omics_to_color = dict(zip(['mf', 'cna', 'ge', 'meth'], omics_palette))
    df["bar_color"] = df["omics_type"].map(omics_to_color)

    # Sort for nice grouping
    ##df = df.sort_values(by=["omics_type", "cancer_type", "feature"], ascending=True).reset_index(drop=True)

    # Plot
    plt.figure(figsize=(24, 5))
    ax = sns.barplot(x="feature", y="relevance", data=df, palette=df["bar_color"].tolist())
    ax.set_title(f"{node_name} (Biological)", fontsize=16)
    ax.set_ylabel("Relevance", fontsize=14)
    ax.set_xlabel("Feature (Cancer_Omics_Index)", fontsize=14)

    for tick, omics in zip(ax.get_xticklabels(), df["omics_type"]):
        tick.set_color(omics_to_color[omics])
        tick.set_fontsize(8)

    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved BIO plot to {output_path}")

def plot_feature_importance_bio_4_color(relevance_vector, feature_names, node_name=None, output_path="plots"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Validate
    if len(relevance_vector) != len(feature_names):
        raise ValueError(f"Mismatch: {len(relevance_vector)} values vs {len(feature_names)} names")

    # DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })
    df["omics_type"] = df["feature"].apply(lambda x: x.split("_")[1])  # ← changed
    df["cancer_type"] = df["feature"].apply(lambda x: x.split("_")[0])  # ← changed

    # Color by omics type (4 colors)
    omics_color = {
        'mf': '#D62728',
        'cna': '#1F77B4',
        'meth': '#2CA02C',
        'ge': '#9467BD'
    }
    df["bar_color"] = df["omics_type"].map(omics_color)

    # Plot
    plt.figure(figsize=(24, 5))
    ax = sns.barplot(x="feature", y="relevance", data=df, palette=df["bar_color"].tolist())
    ##ax = sns.barplot(x="feature", y="relevance", data=df)  # No palette

    ax.set_title(f"{node_name} (Biological)", fontsize=16)
    ax.set_ylabel("Relevance", fontsize=14)
    ax.set_xlabel("Feature (Omics_Cancer)", fontsize=14)

    for tick, omics in zip(ax.get_xticklabels(), df["omics_type"]):
        tick.set_color(omics_color.get(omics, "black"))
        tick.set_fontsize(8)

    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved BIO plot to {output_path}")

def plot_feature_importance_bio_mean_outside(relevance_vector, feature_names, node_name=None, output_path="plots"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Validate input
    if len(relevance_vector) != len(feature_names):
        raise ValueError(f"Mismatch: {len(relevance_vector)} values vs {len(feature_names)} names")

    # DataFrame setup
    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })
    df["omics_type"] = df["feature"].apply(lambda x: x.split("_")[1])  # Omics type
    df["cancer_type"] = df["feature"].apply(lambda x: x.split("_")[0])  # Cancer type

    # Color mapping
    omics_color = {
        'mf': '#D62728',
        'cna': '#1F77B4',
        'meth': '#2CA02C',
        'ge': '#9467BD'
    }
    df["bar_color"] = df["omics_type"].map(omics_color)

    # Plot
    plt.figure(figsize=(24, 5))
    ax = sns.barplot(x="feature", y="relevance", data=df, palette=df["bar_color"].tolist())

    # Title & labels
    ax.set_title(f"{node_name} (Biological)", fontsize=16)
    ax.set_ylabel("Relevance", fontsize=14)
    ax.set_xlabel("Feature (Cancer_Omics_Index)", fontsize=14)

    # Tick formatting
    for tick, omics in zip(ax.get_xticklabels(), df["omics_type"]):
        tick.set_color(omics_color.get(omics, "black"))
        tick.set_fontsize(8)

    # Add mean relevance line
    mean_val = np.mean(df["relevance"])
    ax.axhline(mean_val, color="black", linestyle="--", linewidth=1)
    ax.text(
        len(df) - 1, mean_val + 0.005,
        f"Mean: {mean_val:.4f}",
        ha="right", va="bottom", fontsize=12, color="black"
    )

    # Layout and save
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved BIO plot to {output_path}")

def plot_feature_importance_topo_with_mean(
    relevance_vector,
    feature_names,
    node_name=None,
    output_path="plots"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    if len(relevance_vector) != len(feature_names):
        raise ValueError("Length mismatch between relevance vector and feature names.")

    pretty_labels = [f"emb_{i}" for i in range(len(relevance_vector))]

    df = pd.DataFrame({
        "feature": pretty_labels,
        "relevance": relevance_vector
    })

    # Calculate mean relevance
    mean_val = np.mean(df["relevance"])

    plt.figure(figsize=(24, 5))
    sns.set_style("white")
    bars = plt.bar(df["feature"], df["relevance"], color="#607D8B")  # bluish-gray

    # Mean line
    plt.axhline(mean_val, color="black", linestyle="--", linewidth=1)

    # Title with mean in it
    title_str = f"{node_name} (topo: mean={mean_val:.4f})" if node_name else f"(topo: mean={mean_val:.4f})"
    plt.title(title_str, fontsize=16)

    # Axis labels and formatting
    plt.xlabel("Topology Embedding Dimension", fontsize=14)
    plt.ylabel("Relevance", fontsize=14)
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=12)
    plt.margins(x=0)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved TOPO feature importance plot to {output_path}")

def plot_feature_importance_bio_with_mean(relevance_vector, feature_names, node_name=None, output_path="plots"):

    # Validate input
    if len(relevance_vector) != len(feature_names):
        raise ValueError(f"Mismatch: {len(relevance_vector)} values vs {len(feature_names)} names")

    # DataFrame setup
    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })
    df["omics_type"] = df["feature"].apply(lambda x: x.split("_")[1])
    df["cancer_type"] = df["feature"].apply(lambda x: x.split("_")[0])

    # Color mapping
    omics_color = {
        'mf': '#D62728',
        'cna': '#1F77B4',
        'meth': '#2CA02C',
        'ge': '#9467BD'
    }
    df["bar_color"] = df["omics_type"].map(omics_color)

    # Mean relevance
    mean_val = np.mean(df["relevance"])

    # Plot
    plt.figure(figsize=(24, 5))
    ax = sns.barplot(x="feature", y="relevance", data=df, palette=df["bar_color"].tolist())

    # Title and labels
    title_str = f"{node_name} (bio: mean={mean_val:.4f})" if node_name else f"(bio: mean={mean_val:.4f})"
    ax.set_title(title_str, fontsize=16)
    ax.set_ylabel("Relevance", fontsize=14)
    ax.set_xlabel("Feature (Cancer_Omics_Index)", fontsize=14)

    # Tick formatting
    for tick, omics in zip(ax.get_xticklabels(), df["omics_type"]):
        tick.set_color(omics_color.get(omics, "black"))
        tick.set_fontsize(8)

    # Mean line
    ax.axhline(mean_val, color="black", linestyle="--", linewidth=1)

    # Layout and save
    plt.xticks(rotation=90, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved BIO plot to {output_path}")

def plot_feature_importance_topo_with_mean_(
    relevance_vector,
    feature_names,
    node_name=None,
    output_path="plots"
):
    """
    Plot topological feature importance without mean line or mean value in title.

    Parameters:
        relevance_vector (array-like): Relevance scores.
        feature_names (list of str): Raw feature names (not used in labels).
        node_name (str, optional): Name of the node (used for title).
        output_path (str): Path to save the plot.
    """
    
    if len(relevance_vector) != len(feature_names):
        raise ValueError("Length mismatch between relevance vector and feature names.")

    # Create clean x-axis labels
    pretty_labels = [f"emb_{i}" for i in range(len(relevance_vector))]

    df = pd.DataFrame({
        "feature": pretty_labels,
        "relevance": relevance_vector
    })

    plt.figure(figsize=(24, 5))
    sns.set_style("white")
    bars = plt.bar(df["feature"], df["relevance"], color="#607D8B")  # bluish-gray

    # Title without mean
    title_str = f"{node_name}" if node_name else ""
    plt.title(title_str, fontsize=16)

    # Axis labels and formatting
    plt.xlabel("Topology Embedding Dimension", fontsize=14)
    plt.ylabel("Relevance", fontsize=14)
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=12)
    plt.margins(x=0)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved TOPO feature importance plot to {output_path}")

def plot_feature_importance_topo(
    relevance_vector,
    feature_names,
    node_name=None,
    output_path="plots"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    if len(relevance_vector) != len(feature_names):
        raise ValueError("Length mismatch between relevance vector and feature names.")

    pretty_labels = [f"emb_{i}" for i in range(len(relevance_vector))]

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

def plot_feature_importance_bio(relevance_vector, feature_names, node_name=None, output_path="plots"):
    """
    Plot biological feature importance without mean annotations.

    Parameters:
        relevance_vector (array-like): Relevance scores.
        feature_names (list of str): Feature names in the format Cancer_Omics.
        node_name (str, optional): Name of the node (used for title).
        output_path (str): Path to save the plot.
    """
    
    # Validate input
    if len(relevance_vector) != len(feature_names):
        raise ValueError(f"Mismatch: {len(relevance_vector)} values vs {len(feature_names)} names")

    # DataFrame setup
    df = pd.DataFrame({
        "feature": feature_names,
        "relevance": relevance_vector
    })
    df["omics_type"] = df["feature"].apply(lambda x: x.split("_")[1])
    df["cancer_type"] = df["feature"].apply(lambda x: x.split("_")[0])

    # Color mapping
    omics_color = {
        'mf': '#D62728',
        'cna': '#1F77B4',
        'meth': '#2CA02C',
        'ge': '#9467BD'
    }
    df["bar_color"] = df["omics_type"].map(omics_color)

    # Plot
    plt.figure(figsize=(24, 5))
    ax = sns.barplot(x="feature", y="relevance", data=df, palette=df["bar_color"].tolist())
    # Add tiny space on left and right side of bars
    num_bars = len(df)
    ax.set_xlim(-0.5, num_bars - 0.5)

    # Title and labels
    title_str = f"{node_name}" if node_name else ""
    ax.set_title(title_str, fontsize=16)
    ax.set_ylabel("Relevance", fontsize=14)
    ax.set_xlabel("Feature (Cancer_Omics_Index)", fontsize=14)

    # Tick formatting
    for tick, omics in zip(ax.get_xticklabels(), df["omics_type"]):
        tick.set_color(omics_color.get(omics, "black"))
        ##tick.set_fontsize(8)

    # Final formatting
    plt.ylabel("Relevance", fontsize=14)
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.margins(x=0)
    plt.tight_layout()
    # Add tiny space on left and right side of bars
    num_bars = len(df)
    margin = 0.75
    ax = plt.gca()
    ax.set_xlim(-margin, num_bars - 1 + margin)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved BIO plot to {output_path}")

def plot_bio_topo_saliency_ori(bio_scores, topo_scores, title="", save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(bio_scores, label="Bio Features (0-1023)", color="royalblue", linewidth=2)
    plt.plot(topo_scores, label="Topo Features (1024-2047)", color="darkorange", linewidth=2)
    ##plt.axvline(x=1024, color='gray', linestyle='--', alpha=0.7)
    plt.title(title)
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Relevance Score")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_importance_bio_(
    bio_64,
    feature_names,
    gene_name,
    save_dir,
    fontsize=12,
    figsize=(14, 6),
    save=True
):
    """
    Plot BIO feature importance for a single gene.

    Args:
        bio_64 (np.ndarray): Array of shape [64,] representing summarized BIO importance.
        feature_names (List[str]): List of 64 feature names (e.g., ["BRCA_mf", "KIRC_ge", ...]).
        gene_name (str): Name of the gene being analyzed.
        save_dir (str): Directory to save the figure.
        fontsize (int): Font size for labels and ticks.
        figsize (tuple): Figure size in inches.
        save (bool): Whether to save the plot as a PNG.
    """
    assert bio_64.shape[0] == 64, f"Expected 64 features, got {bio_64.shape[0]}"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(bio_64)), bio_64, color='skyblue', edgecolor='black')

    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel("Relevance Score", fontsize=fontsize + 2)
    plt.xlabel("BIO Features (Cancer × Omics)", fontsize=fontsize + 2)
    plt.title(f"BIO Feature Importance for {gene_name}", fontsize=fontsize + 4)

    # Optionally annotate top 5 features
    topk = 5
    topk_indices = np.argsort(bio_64)[-topk:]
    for idx in topk_indices:
        plt.text(idx, bio_64[idx] + 0.005, f"{bio_64[idx]:.2f}", ha='center', va='bottom', fontsize=fontsize - 1)

    plt.tight_layout()

    if save:
        filename = os.path.join(save_dir, f"{gene_name}_bio_importance.png")
        plt.savefig(filename, dpi=300)
        print(f"✅ Saved BIO feature importance plot for {gene_name} to: {filename}")

    plt.close()

def plot_bio_topo_saliency_not_filled_curve(bio_scores, topo_scores, title="", save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(bio_scores, label="Bio Features (0-1023)", color="royalblue", linewidth=2)
    plt.plot(topo_scores, label="Topo Features (1024-2047)", color="darkorange", linewidth=2)
    plt.axvline(x=1024, color='gray', linestyle='--', alpha=0.7)

    # Start x-axis from 0
    plt.xlim(0, len(bio_scores))  # or use len(topo_scores) if both are same length
    sns.despine(top=True, right=True)

    plt.title(title)
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Relevance Score")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_bio_topo_saliency_(bio_scores, topo_scores, title="", save_path=None):
    import numpy as np

    # Normalize to [0, 1]
    bio_scores = (bio_scores - np.min(bio_scores)) / (np.max(bio_scores) - np.min(bio_scores) + 1e-8)
    topo_scores = (topo_scores - np.min(topo_scores)) / (np.max(topo_scores) - np.min(topo_scores) + 1e-8)

    x_bio = range(len(bio_scores))
    x_topo = range(len(bio_scores), len(bio_scores) + len(topo_scores))

    plt.figure(figsize=(16, 5))
    sns.set_style("white")

    # Fill curves
    plt.fill_between(
        x_bio, 0, bio_scores,
        color="royalblue",
        alpha=0.85,
        linewidth=2,
        label="Biological Features"
    )
    plt.fill_between(
        x_topo, 0, topo_scores,
        color="darkorange",
        alpha=0.85,
        linewidth=2,
        label="Topological Features"
    )

    # Midline separating bio/topo
    plt.axvline(x=len(bio_scores), color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # Mean value text (no frame)
    mean_bio = np.mean(bio_scores)
    mean_topo = np.mean(topo_scores)
    plt.text(len(bio_scores)*0.5, mean_bio + 0.05, f"Mean: {mean_bio:.3f}", ha='center', va='bottom', fontsize=13, color="royalblue")
    plt.text(len(bio_scores) + len(topo_scores)*0.5, mean_topo + 0.05, f"Mean: {mean_topo:.3f}", ha='center', va='bottom', fontsize=13, color="darkorange")

    # Axis styling
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Feature Index", fontsize=14)
    plt.ylabel("Normalized Relevance", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="upper right", fontsize=12, frameon=False)
    sns.despine()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved saliency curve plot to {save_path}")

def plot_bio_topo_saliency_(bio_scores, topo_scores, title="", save_path=None):
    sns.set_style("white")
    plt.figure(figsize=(16, 5))

    # Normalize scores
    bio_scores = np.array(bio_scores)
    topo_scores = np.array(topo_scores)
    bio_scores_norm = (bio_scores - bio_scores.min()) / (bio_scores.max() - bio_scores.min() + 1e-8)
    topo_scores_norm = (topo_scores - topo_scores.min()) / (topo_scores.max() - topo_scores.min() + 1e-8)

    x_bio = np.arange(len(bio_scores))
    x_topo = np.arange(len(bio_scores), len(bio_scores) + len(topo_scores))

    # Filled curves
    plt.fill_between(x_bio, 0, bio_scores_norm, color="royalblue", alpha=0.85, linewidth=2, label="Biological Features")
    plt.fill_between(x_topo, 0, topo_scores_norm, color="darkorange", alpha=0.85, linewidth=2, label="Topological Features")

    # Vertical separator
    plt.axvline(x=len(bio_scores), color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # Mean lines
    mean_bio = np.mean(bio_scores_norm)
    mean_topo = np.mean(topo_scores_norm)
    plt.axhline(mean_bio, xmin=0, xmax=len(bio_scores)/ (len(bio_scores) + len(topo_scores)),
                color="royalblue", linestyle="--", linewidth=1.5, alpha=0.8)
    plt.axhline(mean_topo, xmin=len(bio_scores)/ (len(bio_scores) + len(topo_scores)), xmax=1.0,
                color="darkorange", linestyle="--", linewidth=1.5, alpha=0.8)

    # Text annotations for means (no frame)
    plt.text(len(bio_scores) * 0.5, mean_bio + 0.03, f"Mean: {mean_bio:.3f}",
             ha='center', va='bottom', fontsize=13, color="royalblue")
    plt.text(len(bio_scores) + len(topo_scores) * 0.5, mean_topo + 0.03, f"Mean: {mean_topo:.3f}",
             ha='center', va='bottom', fontsize=13, color="darkorange")

    # Aesthetics
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Feature Index", fontsize=14)
    plt.ylabel("Normalized Relevance", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="upper right", fontsize=12, frameon=False)
    sns.despine()

    # Save
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved saliency curve plot to {save_path}")

def plot_bio_topo_saliency_(bio_scores, topo_scores, title="", save_path=None):

    # 🔹 Normalize both
    bio_scores_norm = (bio_scores - bio_scores.min()) / (bio_scores.max() - bio_scores.min() + 1e-8)
    topo_scores_norm = (topo_scores - topo_scores.min()) / (topo_scores.max() - topo_scores.min() + 1e-8)

    x = np.arange(0, 1024)  # Force same range for both

    # 🔹 Mean values (normalized)
    mean_bio = bio_scores_norm.mean()
    mean_topo = topo_scores_norm.mean()

    # 🔹 Plotting
    plt.figure(figsize=(12, 6))
    plt.fill_between(x, bio_scores_norm, alpha=0.4, color="royalblue", label="Bio Features (0–1023)")
    plt.fill_between(x, topo_scores_norm, alpha=0.4, color="darkorange", label="Topo Features (1024–2047)")

    # 🔹 Mean lines
    plt.axhline(mean_bio, color="blue", linestyle="--", linewidth=1.5, label=f"Mean Bio: {mean_bio:.2f}")
    plt.axhline(mean_topo, color="darkorange", linestyle="--", linewidth=1.5, label=f"Mean Topo: {mean_topo:.2f}")

    # 🔹 Aesthetics
    plt.xlim(0, 1023)
    plt.ylim(0, 1)
    plt.xlabel("Feature Index (0–1023)", fontsize=12)
    plt.ylabel("Normalized Relevance Score", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bio_topo_saliency__(bio_scores, topo_scores, title="", save_path=None):

    # 🔹 Normalize both
    bio_scores_norm = (bio_scores - bio_scores.min()) / (bio_scores.max() - bio_scores.min() + 1e-8)
    topo_scores_norm = (topo_scores - topo_scores.min()) / (topo_scores.max() - topo_scores.min() + 1e-8)

    x = np.arange(0, 1024)  # Shared x-axis

    # 🔹 Mean values
    mean_bio = bio_scores_norm.mean()
    mean_topo = topo_scores_norm.mean()

    # 🔹 Plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(x, bio_scores_norm, alpha=0.4, color="royalblue", label="Bio Features (0–1023)")
    plt.fill_between(x, topo_scores_norm, alpha=0.4, color="darkorange", label="Topo Features (1024–2047)")

    # 🔹 Mean lines
    plt.axhline(mean_bio, color="royalblue", linestyle="--", linewidth=1.5)
    plt.axhline(mean_topo, color="darkorange", linestyle="--", linewidth=1.5)

    # 🔹 Mean text annotations (colored accordingly)
    plt.text(
        1025, mean_bio, f"Mean Bio: {mean_bio:.2f}",
        color="royalblue", fontsize=11, va='center', fontweight='bold'
    )
    plt.text(
        1025, mean_topo, f"Mean Topo: {mean_topo:.2f}",
        color="darkorange", fontsize=11, va='center', fontweight='bold'
    )

    # 🔹 Formatting
    plt.xlim(0, 1023)
    plt.ylim(0, 1)
    plt.xlabel("Feature Index (0–1023)", fontsize=12)
    plt.ylabel("Normalized Relevance Score", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right', fontsize=10, frameon=False)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bio_topo_saliency___(bio_scores, topo_scores, title="", save_path=None):
    """
    Plot saliency relevance for bio and topo features with mean values displayed in the legend.

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
    plt.fill_between(x, bio_scores_norm, alpha=0.4, color="royalblue", label="Bio Features")
    plt.fill_between(x, topo_scores_norm, alpha=0.4, color="darkorange", label="Topo Features")

    # 🔹 Mean lines
    plt.axhline(mean_bio, color="royalblue", linestyle="--", linewidth=1.5)
    plt.axhline(mean_topo, color="darkorange", linestyle="--", linewidth=1.5)

    # 🔹 Legend box with mean values
    legend_text = (
        f"Bio Mean:  {mean_bio:.2f}\n"
        f"Topo Mean: {mean_topo:.2f}"
    )
    plt.text(
        0.01, 0.99,
        legend_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
    )

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

def save_and_plot_confirmed_genes(
    args,
    node_names_topk,
    node_scores_topk,
    summary_feature_relevance,
    feature_names,
    output_dir,
    confirmed_genes_save_path,
    confirmed_gene_path="data/ncg_8886.txt"
):
    """
    Finds all confirmed cancer genes from the predictions, saves them, and plots their feature contributions.

    Args:
        node_names_topk (list): List of top-k predicted gene names, sorted by score.
        summary_feature_relevance (np.ndarray): Relevance scores, shape [topk, 64].
        feature_names (list): List of 64 feature names (e.g., from extract_summary_features_np).
        output_dir (str): Where to save the plots and confirmed list.
        num_epochs (int): For naming output files.
        confirmed_gene_path (str): Path to file with known cancer genes.
    """

    # Load known cancer genes
    with open(confirmed_gene_path) as f:
        known_cancer_genes = set(line.strip() for line in f if line.strip())

    # ✅ Get all confirmed genes from the top-k predictions
    confirmed_genes = [g for g in node_names_topk if g in known_cancer_genes]

    # Save the confirmed gene list
    ###confirmed_file = os.path.join(output_dir, f"confirmed_genes_epo{num_epochs}.txt")
    with open(confirmed_genes_save_path, "w") as f:
        for gene in confirmed_genes:
            f.write(f"{gene}\n")

    # Plot and save feature contribution plots for each confirmed gene
    '''for gene_name in confirmed_genes:
        idx = node_names_topk.index(gene_name)
        relevance_vector = summary_feature_relevance[idx]
        plot_path = os.path.join("results/gene_prediction/confirmed_feature_contributions/", f"{args.model_type}_{args.net_type}_{gene_name}_confirmed_feature_contributions_epo{args.num_epochs}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plot_gene_feature_contributions(
            gene_name=gene_name,
            relevance_vector=relevance_vector,
            feature_names=feature_names,
            output_path=plot_path,
            score=node_scores_topk[idx]
        )'''

def plot_neighbor_relevance(
    neighbor_scores, 
    gene_name="Gene", 
    node_id_to_name=None, 
    output_path=None
):
    """
    Plot relevance scores of neighbors for a given gene.

    Args:
        neighbor_scores (dict): {node_id: relevance_score}.
        gene_name (str): Name of the target gene.
        node_id_to_name (dict): Optional mapping from node_id to readable names.
        output_path (str): If given, saves the plot to this path.
    """

    # Filter out scores <= 0
    filtered_scores = {k: v for k, v in neighbor_scores.items() if v > 0}

    if not filtered_scores:
        print(f"⚠️ No positive relevance scores for {gene_name}. Skipping plot.")
        return

    # Sort by descending relevance
    sorted_scores = dict(sorted(filtered_scores.items(), key=lambda x: -x[1]))

    scores = np.array(list(sorted_scores.values()))
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    names = [node_id_to_name.get(nid, f"Node {nid}") if node_id_to_name else f"Node {nid}"
             for nid in sorted_scores.keys()]

    df = pd.DataFrame({"neighbor": names, "relevance": norm_scores})

    # Plot
    plt.figure(figsize=(3, 2.5))
    sns.set_style("white")

    ax = sns.barplot(data=df, x="neighbor", y="relevance", palette="coolwarm")
    ax.set_title(f"{gene_name}", fontsize=12)
    ax.set_ylabel("Relevance score", fontsize=10)
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha='right', fontsize=9)

    # Clean plot aesthetics
    sns.despine(left=True, bottom=True)
    ax.grid(False)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved neighbor relevance plot to {output_path}")
    else:
        plt.show()

def apply_spectral_biclustering_topo(graph, embeddings, n_clusters, save_path):
    print(f"Running Spectral Biclustering with {n_clusters} row clusters for topo embeddings...")

    node_features = embeddings.cpu().numpy()

    # Run Spectral Biclustering
    bicluster = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state=42)
    bicluster.fit(node_features)

    # Assign cluster labels to graph
    cluster_labels = bicluster.row_labels_
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long)

    # Save graph with cluster labels
    torch.save({
        'edges': graph.edges(),
        'features': graph.ndata['feat'],
        'labels': graph.ndata.get('label', None),
        'cluster': graph.ndata['cluster']
    }, save_path)

    print(f"✅ Spectral Biclustering complete. Graph with cluster labels saved to {save_path}")
    return graph, cluster_labels

def plot_spectral_biclustering_heatmap_topo(
    args,
    relevance_scores,
    cluster_labels,
    gene_names,
    omics_splits,
    output_path,
    feature_names=None  # ← NEW
):

    print(f"Plotting spectral biclustering heatmap for topo embeddings...")

    # Sort relevance scores and cluster labels
    '''sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]'''
    # Assuming cluster_labels is already aligned with relevance_scores
    num_genes = relevance_scores.shape[0]
    cluster_labels = cluster_labels[:num_genes]
    gene_names = gene_names[:num_genes]

    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_labels = cluster_labels[sorted_indices]
    sorted_gene_names = [gene_names[i] for i in sorted_indices]


    # Create color map
    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient",
        ["#F0F3F4", "#85929e"]
    )

    # Compute cluster boundaries
    unique_clusters, counts = np.unique(sorted_labels, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # Plot heatmap
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(nrows=14, ncols=50)
    ax = fig.add_subplot(gs[0:12, 2:48])
    ax_cbar = fig.add_subplot(gs[4:8, 49])
    ax_legend = fig.add_subplot(gs[13, 2:45])

    vmax = np.percentile(sorted_scores, 99)

    xtick_labels = feature_names if feature_names is not None else False

    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=0,
        vmax=vmax,
        xticklabels=xtick_labels,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )

    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=14)
    ax_cbar.yaxis.label.set_size(14)

    # Add cluster color bars
    for i, cluster in enumerate(sorted_labels):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # Legend
    ax_legend.axis("off")
    patches = [Patch(color=CLUSTER_COLORS[i], label=f"Cluster {i}") for i in unique_clusters]
    ax_legend.legend(handles=patches, loc="center", ncol=5, frameon=False, fontsize=12)

    # Save plot
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Topo embedding heatmap saved to {output_path}")

    return pd.DataFrame(sorted_scores, index=[gene_names[i] if gene_names else i for i in sorted_indices])

def plot_spectral_biclustering_heatmap_bio(
    args,
    relevance_scores,
    cluster_labels,
    gene_names,
    output_path,
    feature_names=None
):
    print(f"📊 Plotting spectral biclustering heatmap for BIO (raw)...")

    if not isinstance(cluster_labels, np.ndarray):
        cluster_labels = np.array(cluster_labels)

    num_genes = relevance_scores.shape[0]
    cluster_labels = cluster_labels[:num_genes]
    gene_names = gene_names[:num_genes]

    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_labels = cluster_labels[sorted_indices]
    sorted_gene_names = [gene_names[i] for i in sorted_indices]

    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient", ["#F0F3F4", "#85929e"]
    )

    unique_clusters, counts = np.unique(sorted_labels, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(nrows=14, ncols=50)
    ax = fig.add_subplot(gs[0:12, 2:48])
    ax_cbar = fig.add_subplot(gs[4:8, 49])
    ax_legend = fig.add_subplot(gs[13, 2:45])

    vmax = np.percentile(sorted_scores, 99)
    xtick_labels = feature_names if feature_names is not None else False

    sns.heatmap(
        sorted_scores,
        cmap=bluish_gray_gradient,
        vmin=0,
        vmax=vmax,
        xticklabels=xtick_labels,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "Saliency (BIO raw)",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )

    ax_cbar.yaxis.label.set_color("#85929e")
    ax_cbar.tick_params(colors="#85929e", labelsize=14)
    ax_cbar.yaxis.label.set_size(14)

    for i, cluster in enumerate(sorted_labels):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(-2.0, center_y, f"{count}", va='center', ha='right', fontsize=12, fontweight='bold')

    ax_legend.axis("off")
    patches = [Patch(color=CLUSTER_COLORS[i], label=f"Cluster {i}") for i in unique_clusters]
    ax_legend.legend(handles=patches, loc="center", ncol=5, frameon=False, fontsize=12)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Bio (raw) heatmap saved to {output_path}")
    return pd.DataFrame(sorted_scores, index=sorted_gene_names), sorted_labels

def plot_spectral_biclustering_heatmap(
    args,
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None,
    gene_names=None
):
    # 🔹 Reduce dimensionality
    ##relevance_scores = extract_summary_features_np(relevance_scores)

    # 🔹 Set default omics colors
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD',
            'topo': '#FF7F0E'   # orange, for topo
        }

    # 🔹 Cancer types and omics order
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # 🔹 Build pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])  # 64 labels

    # Add topo feature names (or something readable)
    pretty_labels.extend([f"Topo {i+1}" for i in range(64)])  # total = 128

    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([f"{cancer}_{omics.upper()}" for cancer in cancer_names])  # 64 total

    # Add 64 topo feature labels
    pretty_labels.extend([f"Topo {i+1}" for i in range(64)])  # 64 + 64 = 128

    # 🔹 Sort by cluster
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

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
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(nrows=14, ncols=50)
    ax = fig.add_subplot(gs[0:12, 2:45])
    ax_curve = fig.add_subplot(gs[0:12, 45:48], sharey=ax)
    ax_cbar = fig.add_subplot(gs[4:8, 49])
    ax_legend = fig.add_subplot(gs[13, 2:45])

    # 🔹 Compute cluster boundaries
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # 🔹 Use dynamic vmax for contrast
    vmax = np.percentile(sorted_scores, 99)

    bluish_gray_gradient = LinearSegmentedColormap.from_list(
        "bluish_gray_gradient",
        ["#F0F3F4", "#85929e"]  # light gray → bluish gray
    )

    # 🔹 Plot heatmap
    sns.heatmap(
        sorted_scores,
        #cmap='binary',
        cmap=bluish_gray_gradient,
        vmin=0,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.1,
            "aspect": 12,
            "pad": 0.02,
            "orientation": "vertical",
            "location": "right"
        },
        ax=ax
    )
    ax_cbar.yaxis.label.set_color("#85929e")   # Match color
    ax_cbar.tick_params(colors="#85929e", labelsize=14)
    ax_cbar.yaxis.label.set_size(14)

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
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # 🔹 Color x-tick labels
    ax.set_xticks(np.arange(128) + 0.5)
    '''ax.set_xticklabels(pretty_labels, rotation=90, fontsize=14)

    ax.set_xticks(np.arange(len(feature_names)) + 0.5)'''
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=14)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # 🔹 Omics legend
    ax_legend.axis("off")
    omics_patches = [Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()]
    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        fontsize=12,
        handleheight=1.5,
        handlelength=3
    )

    # 🔹 LRP curve
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())
    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='#7b241c',
        alpha=0.8,           # no transparency
        linewidth=3        # thicker line
    )

    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
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
    '''plot_clusterwise_feature_contributions(
        args=args,
        relevance_scores=relevance_scores,
        cluster_labels=cluster_labels,
        feature_names=[f"{k.upper()}: {v}" for k in omics_order for v in cancer_names],
        per_cluster_feature_contributions_output_dir=os.path.join(os.path.dirname(output_path), "per_cluster_feature_contributions"),
        omics_colors=omics_colors
    )'''

    return pd.DataFrame(relevance_scores, index=gene_names, columns=feature_names)

def extract_summary_features_np_skip(features_np):
    num_nodes = features_np.shape[0]
    total_dim = features_np.shape[1]
    summary_features = []

    # Constants
    num_omics = 4
    num_cancers = 16
    features_per_pair = 16
    bio_feature_dim = num_omics * num_cancers * features_per_pair  # 1024

    # 🔹 Extract bio features (mf, cna, ge, meth)
    for o_idx in range(num_omics):
        for c_idx in range(num_cancers):
            base = o_idx * num_cancers * features_per_pair + c_idx * features_per_pair
            if base + features_per_pair > total_dim:
                continue  # skip if slice exceeds bounds
            group = features_np[:, base:base + features_per_pair]  # [num_nodes, 16]
            max_vals = group.max(axis=1, keepdims=True)           # [num_nodes, 1]
            summary_features.append(max_vals)

    # 🔹 Extract topo features (assumed to follow bio features)
    if bio_feature_dim < total_dim:
        topo_features = features_np[:, bio_feature_dim:]  # remaining features
        # Apply simple summary – max across topo dim
        topo_summary = topo_features.max(axis=1, keepdims=False)  # [num_nodes]
        topo_summary = topo_summary[:, np.newaxis]                # reshape to [num_nodes, 1]
        summary_features.append(topo_summary)

    return np.concatenate(summary_features, axis=1)

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

def run_pcg_kcg_interaction_plots(
    args, graph, cluster_labels, node_names, top_gene_indices,
    ground_truth_cancer_genes, name_to_index, suffix
):
    output_dir = 'results/gene_prediction/'

    total_genes_per_cluster = {
        i: np.sum(cluster_labels == i) for i in range(len(set(cluster_labels)))
    }

    pred_counts = {
        i: sum(cluster_labels[top_gene_indices] == i) for i in range(len(set(cluster_labels)))
    }

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_{suffix}_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(),
        predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_predicted
    )

    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)
    kcg_counts = {
        i: sum((np.array(cluster_labels) == i) & np.isin(range(len(cluster_labels)), list(gt_indices)))
        for i in total_genes_per_cluster
    }

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_{suffix}_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(),
        kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_kcg
    )

    # Interaction plots
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    cluster_labels_np = np.array(cluster_labels)

    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_data = pd.DataFrame({
        "Cluster": cluster_labels_np[kcg_nodes],
        "Interactions": degrees_np[kcg_nodes]
    })

    pcg_data = pd.DataFrame({
        "Cluster": cluster_labels_np[top_gene_indices],
        "Interactions": degrees_np[top_gene_indices]
    })

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_{suffix}_epo{args.num_epochs}.png')
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_{suffix}_epo{args.num_epochs}.png')

    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

def plot_neighbor_relevance_by_mode(
    gene,
    relevance_scores,
    mode,
    scores,
    neighbors_dict,
    name_to_index,
    node_id_to_name,
    args,
    save_dir="results/gene_prediction/neighbor_feature_contributions/"
):
    os.makedirs(save_dir, exist_ok=True)

    if gene not in name_to_index:
        print(f"⚠️ Gene {gene} not found in the graph.")
        return

    node_idx = name_to_index[gene]
    gene_score = scores[node_idx]
    print(f"[{mode}] {gene} → Node {node_idx} | Predicted score: {gene_score:.4f}")

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
        gene_name=gene,
        node_id_to_name=node_id_to_name,
        output_path=output_path
    )


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    epoch_times, cpu_usages, gpu_usages = [], [], []

    # data_path = os.path.join('../gene_pathway/data/processed/json_graphs', f'{args.net_type}_omics_ppi_embeddings_graph_2048.json')
    data_path = os.path.join('data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_graph_2048.json')
    
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    graph = dgl.graph(edges, num_nodes=len(nodes))

    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

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

    print("\U0001F50D features.shape:", features.shape)
    print("\U0001F9E0 graph.num_nodes():", graph.num_nodes())
    print(f"Starting training for {args.num_epochs} epochs...")

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

    # Get neighbors of top predicted genes
    neighbor_indices = set()
    for idx in top_gene_indices:
        neighbors = graph.successors(idx).tolist()
        neighbor_indices.update(neighbors)

    # Combine top genes + their neighbors for IG analysis
    lrp_node_indices = top_gene_indices#list(set(top_gene_indices) | neighbor_indices)
    lrp_node_indices = torch.tensor(lrp_node_indices, dtype=torch.long, device=device)


    print("Generating feature importance plots...")

    top_node_features = embeddings[top_gene_indices].cpu().numpy()  # shape: [num_top_genes, feature_dim]

    # Call the function to find optimal k
    plot_silhouette_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")
    best_k = 10#find_optimal_k(top_node_features, k_range=(5, 20), plot_path=plot_silhouette_path, save_best_k_path=best_k_path)
    
    '''print('best_k=================================== ', best_k)
    if best_k > 12:
        best_k = 12
    '''
    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_cluster_labels.npy")
    total_genes_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_total_genes_per_cluster.npy")
    pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_pred_counts.npy")
    graph, cluster_labels, total_genes_per_cluster, pred_counts = apply_spectral_biclustering(
        graph=graph,
        embeddings=graph.ndata['feat'],
        node_names=node_names,
        predicted_cancer_genes=node_names_topk,  
        n_clusters=best_k,
        save_path=spectral_biclustering_graph_path,
        save_cluster_labels_path=cluster_labels_path,
        save_total_genes_per_cluster_path=total_genes_per_cluster_path,
        save_predicted_counts_path=pred_counts_path,
        output_path_genes_clusters=output_path_genes_clusters
    )

    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )
    print(f"t-SNE plot saved to {tsne_plot_path}")

    umap_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_umap_top5_predicted_genes_epo{args.num_epochs}.png')
    plot_umap_predicted_genes_top5(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        umap_plot_path,
        args
    )
    print(f"umap plot saved to {tsne_plot_path}")
     
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )

    umap_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_umap_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_umap(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        umap_plot_path
    )
    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}


    # Define where to save the clustered gene CSV

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    output_path_legend = os.path.join(output_dir, f'legend_omics.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS_OMICS, num_clusters=4)
    
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---
    
    


    # plot_degree_distributions(sorted_degree_counts_above_value, sorted_degree_counts_below_value, args, output_dir)

    # generate_kde_and_curves(logits, node_names, degree_counts_above, degree_counts_below, labels, train_mask, args)

    # plot_model_performance(args)

    # plot_venn_diagram()
        
    ##############################################################################################################################

    
    print("Running spectral_biclustering on node features...")

    # ✅ Validate node name length
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"

    # ✅ Convert features if necessary
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32, requires_grad=True)
    else:
        features = features.clone().detach().requires_grad_(True)

    # ✅ Compute relevance scores
    relevance_scores = compute_relevance_scores_4_groups(model, graph, features)

    topk_node_indices_tensor = torch.tensor(top_gene_indices, dtype=torch.int64).view(-1)

    # ✅ Subset
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]
    node_names_topk = [node_names[i] for i in topk_node_indices_tensor.tolist()]

    # ✅ Extract and summarize
    bio_scores = relevance_scores_topk[:, 0:1024].cpu().numpy()
    topo_scores = relevance_scores_topk[:, 1024:2048].cpu().numpy()

    bio_summary = extract_bio_summary_features_np(bio_scores)
    topo_summary = extract_topo_summary_features_np(topo_scores)
    combined_scores = np.concatenate([bio_summary, topo_summary], axis=1)  # (N, 128)

    relevance_scores_bio = relevance_scores[:, :1024]
    relevance_scores_topo = relevance_scores[:, 1024:]

    saliency_bio = relevance_scores_bio.sum(dim=1).cpu().numpy()
    saliency_topo = relevance_scores_topo.sum(dim=1).cpu().numpy()

    top_bio_indices = relevance_scores_bio.sum(dim=1).topk(100).indices.cpu().numpy()
    top_topo_indices = relevance_scores_topo.sum(dim=1).topk(1000).indices.cpu().numpy()

    run_pcg_kcg_interaction_plots(args, graph, cluster_labels, node_names, top_bio_indices, ground_truth_cancer_genes, name_to_index, 'bio')
    run_pcg_kcg_interaction_plots(args, graph, cluster_labels, node_names, top_topo_indices, ground_truth_cancer_genes, name_to_index, 'topo')

    # ✅ Feature naming

    cancer_types = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]

    omics_types = ['mf', 'cna', 'ge', 'meth']

    # First 64 columns: omics features
    bio_feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Next 64 columns: topo features
    topo_feature_names = [f"topo_{i+1}" for i in range(64)]

    feature_names = bio_feature_names + topo_feature_names
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
        'topo': (64, 127)
    }

    bio_feature_names = [f"{cancer}_{omics}" for omics in omics_types for cancer in cancer_types]
    topo_feature_names = [f"topo_{i+1}" for i in range(64)]
    feature_names = bio_feature_names + topo_feature_names

    # ✅ Spectral biclustering heatmap
    '''spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")
    relevance_df = plot_spectral_biclustering_heatmap(
        args=args,
        relevance_scores=combined_scores,  # instead of `relevance_scores_topk`
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        gene_names=node_names_topk,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )'''



    print("Plotting spectral biclustering heatmap for TOPO (summarized 64-dim)...")

    # Extract only topo region from 1024:2048, then summarize to 64-dim
    topo_scores = relevance_scores_topk[:, 1024:2048].cpu().numpy()
    topo_summary = extract_summary_features_np_skip(topo_scores)  # shape: [num_top_genes, 64]

    topo_feature_names = [f"Topo_{i}" for i in range(64)]
    topo_splits = {
        "topo": (0, 63),
    }

    spectral_biclustering_topo_path = os.path.join(
        output_dir,
        f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_topo_epo{args.num_epochs}.png"
    )
    plot_spectral_biclustering_heatmap(
        args=args,
        relevance_scores=topo_summary,
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=topo_feature_names,
        gene_names=node_names_topk,
        omics_splits=topo_splits,
        output_path=spectral_biclustering_topo_path
    )

    # summary_feature_relevance: shape [top_k, 64]
    reduced_scores = extract_summary_features_np_skip(relevance_scores_topk.cpu().numpy())
       
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")
    
    save_and_plot_confirmed_genes(
        args=args,
        node_names_topk=node_names_topk,
        node_scores_topk=scores[top_gene_indices],
        summary_feature_relevance=reduced_scores ,  # output from extract_summary_features_np(relevance_scores)
        feature_names=feature_names, 
        output_dir=output_dir,                 
        confirmed_genes_save_path=confirmed_file, 
        confirmed_gene_path="data/ncg_8886.txt"
    )

    #####################################################################################
    # Load confirmed driver genes
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    save_dir = "results/gene_prediction/gnnexplainer/"
    os.makedirs(save_dir, exist_ok=True)

    # Convert list to dict
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]
        print(f"{gene} → Node {node_idx} | Predicted score: {gene_score:.4f}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

        # Only proceed if there are exactly 12 neighbors
        '''if len(neighbor_indices) != 12:
            print(f"⏭️ Skipping {gene}: has {len(neighbor_indices)} neighbors (requires exactly 12).")
            continue'''

        neighbor_relevance_scores = [relevance_scores[i].sum().item() for i in neighbor_indices]
        neighbor_scores_dict = dict(zip(neighbor_indices, neighbor_relevance_scores))

        plot_path_neighbor = os.path.join(
            "results/gene_prediction/neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_neighbor_relevance_epo{args.num_epochs}.png"
        )
        os.makedirs(os.path.dirname(plot_path_neighbor), exist_ok=True)

        plot_neighbor_relevance(
            neighbor_scores=neighbor_scores_dict,
            gene_name=gene,
            node_id_to_name=node_id_to_name,
            output_path=plot_path_neighbor
        )
        
    '''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    graph = graph.to(device)
    features = features.to(device)

    # Load confirmed driver genes
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    save_dir = "results/gene_prediction/gnnexplainer/"
    os.makedirs(save_dir, exist_ok=True)

    # Convert list to dict
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)'''

    #######################################################################
    # ✅ Gradient-based saliency for bio/topo feature relevance (confirmed genes)

    print("\n🔬 Computing gradient-based bio/topo saliency for confirmed genes...")

    # Define feature groups
    feature_groups = {
        "bio": (0, 1024),
        "topo": (1024, 2048)
    }

    # Filter for confirmed genes present in the graph
    gene_indices = [name_to_index[gene] for gene in confirmed_genes if gene in name_to_index]

    # Compute relevance scores using saliency (Integrated Gradients or raw gradients)
    relevance_scores = compute_relevance_scores(
        model=model,
        graph=graph,
        features=features,
        node_indices=gene_indices,
        normalize=True,
        feature_groups=feature_groups
    )

    # Make output directory
    save_dir = os.path.join(output_dir, "gnnexplainer/")
    os.makedirs(save_dir, exist_ok=True)


    # For bio (4 omics × 16 cancers = 64 features)
    bio_feat_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # For topo (just topology_0 to topology_63)
    topo_feat_names = [f"topology_{i}" for i in range(64)]

    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]
        print(f"{gene} → Node {node_idx} | Predicted score = {gene_score:.4f}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

        # Only proceed if there are exactly 12 neighbors
        if len(neighbor_indices) != 12:
            print(f"⏭️ Skipping {gene}: has {len(neighbor_indices)} neighbors (requires exactly 12).")
            continue


        #node_idx = name_to_index[gene]

        if node_idx not in relevance_scores:
            print(f"⚠️ No relevance found for {gene} (node {node_idx})")
            continue

        # Get relevance vectors and reduce
        bio_1024 = relevance_scores[node_idx]["bio"].cpu().numpy().reshape(1, -1)
        topo_1024 = relevance_scores[node_idx]["topo"].cpu().numpy().reshape(1, -1)

        bio_64 = extract_summary_features_np_skip(bio_1024).squeeze()
        topo_64 = extract_summary_features_np_skip(topo_1024).squeeze()

        # Plot BIO saliency
        plot_feature_importance_bio(
            relevance_vector=bio_64,
            feature_names=bio_feat_names,
            node_name=f"{gene}",
            output_path=os.path.join(
                save_dir,
                f"{args.model_type}_{args.net_type}_{gene}_bio_feature_importance_epo{args.num_epochs}.png"
            )
        )

        # Plot TOPO saliency
        plot_feature_importance_topo(
            relevance_vector=topo_64,
            feature_names=topo_feat_names,
            node_name=f"{gene}",
            output_path=os.path.join(
                save_dir,
                f"{args.model_type}_{args.net_type}_{gene}_topo_feature_importance_epo{args.num_epochs}.png"
            )
        )


        from collections import defaultdict

        # This is the correct dictionary for group-wise aggregation
        groupwise_saliency = defaultdict(list)

        for node_id, group_saliency in relevance_scores.items():
            for group_name, saliency_tensor in group_saliency.items():
                groupwise_saliency[group_name].append(saliency_tensor.cpu().numpy())

        # Optional: convert lists to np.array
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
                save_path=os.path.join(save_dir, f"{args.model_type}_{args.net_type}_bio_topo_comparison_lineplot_{args.num_epochs}.png")
            )

    bio_matrix = []  # rows: genes, cols: features
    valid_genes = []  # confirmed genes that exist in graph

    for gene in confirmed_genes:
        if gene not in name_to_index or name_to_index[gene] not in relevance_scores:
            continue

        node_idx = name_to_index[gene]
        bio_1024 = relevance_scores[node_idx]["bio"].cpu().numpy().reshape(1, -1)
        bio_64 = extract_summary_features_np_skip(bio_1024).squeeze()

        bio_matrix.append(bio_64)
        valid_genes.append(gene)

    
    part = "bio"  # or "topo"

    plot_saliency_biclustering_heatmap(
        relevance_scores=relevance_scores,
        gene_list=confirmed_genes,
        name_to_index=name_to_index,
        extract_fn=extract_summary_features_np_skip,
        feature_names=feature_names if part == "bio" else None,
        part=part,
        output_dir=output_dir,
        model_type=args.model_type,
        net_type=args.net_type
    )


    # --- Spectral Biclustering with TOPO embeddings ---
    print("Running spectral biclustering for TOPO relevance...")

    topo_64_list = []
    kept_indices = []  # track which node indices were valid
    kept_gene_names = []
    kept_cluster_labels = []

    for idx, node_idx in enumerate(topk_node_indices_tensor):
        node_idx = int(node_idx)

        if node_idx not in relevance_scores:
            print(f"⚠️ Node {node_idx} not in relevance_scores, skipping.")
            continue

        topo_tensor = relevance_scores[node_idx].get("topo", None)
        if topo_tensor is None:
            print(f"⚠️ No 'topo' key for node {node_idx}, skipping.")
            continue

        topo_np = topo_tensor.cpu().numpy().reshape(1, -1)
        topo_64 = extract_summary_features_np_skip(topo_np).squeeze()
        topo_64_list.append(topo_64)

        kept_indices.append(node_idx)
        kept_gene_names.append(node_names_topk[idx])
        kept_cluster_labels.append(cluster_labels_topk[idx])

    # Only proceed if we have data
    if topo_64_list:
        relevance_scores_topo_64 = np.stack(topo_64_list)
    else:
        print("❌ No valid topo saliency scores found.")


    # Path for saving the topo heatmap
    spectral_biclustering_topo_heatmap_path = os.path.join(
        output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_topo_epo{args.num_epochs}.png"
    )

    # Call the topo heatmap plotting function
    plot_spectral_biclustering_heatmap_topo(
        args=args,
        relevance_scores=relevance_scores_topo_64,
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        gene_names=node_names_topk,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_topo_heatmap_path,
        feature_names=topo_feat_names
    )

    print("Running spectral biclustering for BIO relevance...")

    bio_64_list = []
    bio_kept_indices = []
    bio_kept_gene_names = []
    bio_kept_cluster_labels = []

    for idx, node_idx in enumerate(topk_node_indices_tensor):
        node_idx = int(node_idx)

        if node_idx not in relevance_scores:
            print(f"⚠️ Node {node_idx} not in relevance_scores, skipping.")
            continue

        bio_tensor = relevance_scores[node_idx].get("bio", None)
        if bio_tensor is None:
            print(f"⚠️ No 'bio' key for node {node_idx}, skipping.")
            continue

        bio_np = bio_tensor.cpu().numpy().reshape(1, -1)
        bio_64 = extract_summary_features_np_skip(bio_np).squeeze()
        bio_64_list.append(bio_64)

        bio_kept_indices.append(node_idx)
        bio_kept_gene_names.append(node_names_topk[idx])
        bio_kept_cluster_labels.append(cluster_labels_topk[idx])

    if bio_64_list:
        bio_64_array = np.stack(bio_64_list)

        # Optional: feature names for 64 summary features
        bio_feature_names = [f"bio_{i}" for i in range(bio_64_array.shape[1])]

        plot_spectral_biclustering_heatmap_bio(
            args=args,
            relevance_scores=bio_64_array,
            cluster_labels=bio_kept_cluster_labels,
            gene_names=bio_kept_gene_names,
            ##omics_splits=None,  # Not used, keep None
            output_path="results/heatmaps/bio_summary_heatmap.png",
            feature_names=bio_feature_names
        )
    else:
        print("❌ No valid BIO summary scores found.")

   
def train_x(args):
    # Initialize metrics
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_graph_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)
    ###data_path = os.path.join('data/multiomics_meth/', f'extracted_embeddings.json')
    ###data_path = '../___KG-PE/embedding/data/neo4j_graph_data.json'
    # data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1  # Mask for labeled nodes
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)  # Mask for testing on all nodes
    graph = dgl.add_self_loop(graph)
    in_feats = embeddings.shape[1]
    hidden_feats = args.hidden_feats
    out_feats = 1  # Binary classification (driver gene or not)
    
    
    # Compute node degrees
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)  # Reshape to (N, 1)

    # Concatenate degree feature with node embeddings
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    # Update input feature dimension
    in_feats = features.shape[1]  # Update to reflect added degree feature

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move features to device
    
    
    # Define output directory
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    # File path for saving model architecture
    model_csv_path = os.path.join(output_dir, f'{args.model_type}_model_structure.csv')

    # Choose the model
    model = choose_model(args.model_type, in_feats, hidden_feats, out_feats)
    model = model.to(args.device)

    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss(alpha=0.25, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Move data to device
    graph = graph.to(device)
    ##features = graph.ndata['feat'].to(device)
    features = features.to(device)
    labels = graph.ndata['label'].to(device).float()  # BCE loss expects float labels
    train_mask = graph.ndata['train_mask'].to(device)
    test_mask = graph.ndata['test_mask'].to(device)
        
    # Training loop with progress bar
    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()

        # Measure CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_usages.append(cpu_usage)

        # Measure GPU usage (if CUDA available)
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2  # Convert bytes to MB
        else:
            gpu_usage = 0.0  # No GPU available
        gpu_usages.append(gpu_usage)
        
        model.train()
        logits = model(graph, features).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time per epoch
        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)

        # Optionally display current metrics
        tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}, CPU: {cpu_usage}%, GPU: {gpu_usage:.2f} MB")

    # Testing and predictions
    model.eval()
    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()
        print("Predicted Scores:", scores[test_mask.cpu().numpy()])

                
    # Convert epoch_times to floats during summation
    total_time = sum(epoch_times)  # Total time in seconds
    average_time_per_epoch = total_time / args.num_epochs  # Average time per epoch
    # Calculate average memory usage for CPU and GPU
    average_cpu_usage = sum(cpu_usages) / args.num_epochs  # CPU usage in MB
    average_gpu_usage = sum(gpu_usages) / args.num_epochs  # GPU usage in MB
    # Perform clustering and visualization on predicted cancer driver genes
    save_model_details(model, args, model_csv_path, in_feats, hidden_feats, out_feats)
    label_scores = save_predicted_scores(scores, labels, nodes, args)
    save_average_scores(label_scores, args)
    plot_average_scores(label_scores, args)
    plot_score_distributions(label_scores, args)
    save_performance_metrics(epoch_times, cpu_usages, gpu_usages, args)
    # After calculating total_time, average_time_per_epoch, etc.
    save_overall_metrics(total_time, average_time_per_epoch, average_cpu_usage, average_gpu_usage, args, output_dir)


    ##################################################################
    # Rank nodes by scores for only non-labeled nodes
    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]  # Indices of non-labeled nodes
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)


    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)
    
    predicted_cancer_genes = [i for i in non_labeled_nodes if scores[i] >= args.score_threshold]
    # Run clustering and get resultsoutput_path_all_predicted_cancer_genes_per_cluster = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_all_predicted_cancer_genes_per_cluster_epo{args.num_epochs}_2048.csv')
    output_path_genes_clusters = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_all_predicted_cancer_genes_per_cluster_epo{args.num_epochs}_2048.png')
    output_path_predicted_cancer_genes = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_predicted_cancer_genes_epo{args.num_epochs}_2048.csv')
    cluster_labels, predicted_cancer_genes_count, total_genes_per_cluster = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, output_path_predicted_cancer_genes)#, num_clusters=5)

    '''result = cluster_and_visualize_predicted_genes(graph, non_labeled_nodes, predicted_cancer_genes, node_names, output_dir)
    print(len(result))  # Should print 3'''

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')

    # Call the plot function
    output_path_percent = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_cluster_percent_ground_truth_epo{args.num_epochs}_2048.png')

    plot_pcg_cancer_genes_ground_truth(
        clusters=total_genes_per_cluster.keys(),  # Cluster IDs
        predicted_cancer_genes_count=predicted_cancer_genes_count,
        total_genes_per_cluster=total_genes_per_cluster,
        ground_truth_cancer_genes=ground_truth_cancer_genes,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_percent
    )

# Call the plot function
    output_path_percent = os.path.join(
        'results/gene_prediction/', 
        f'{args.model_type}_{args.net_type}_cluster_percent_predicted_cancer_genes_epo{args.num_epochs}_2048.png'
    )

    plot_pcg_cancer_genes_ori(
        clusters=total_genes_per_cluster.keys(),  # Cluster IDs
        predicted_cancer_genes_count=predicted_cancer_genes_count,
        total_genes_per_cluster=total_genes_per_cluster,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_percent  # Removed ground_truth_cancer_genes
    )

            
    ##plot_pcg_cancer_genes(cluster_labels, predicted_cancer_genes_count, total_genes_per_cluster, output_path_percent)

    '''# Perform clustering and store cluster labels
    cluster_labels, predicted_counts, total_genes_per_cluster = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_dir="results"
    )'''

    # Now 'graph.ndata['cluster']' exists, so we can safely use it
    '''predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    
    # Get node degrees (number of edges per node)
    graph.ndata['degree'] = graph.in_degrees()  # or graph.out_degrees(), depending on the graph direction
    '''
    
    # Compute interactions (degree of connection with predicted cancer genes)
    '''
    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    graph.ndata['interactions'] = torch.zeros(graph.num_nodes(), dtype=torch.float, device=graph.device)
    graph.ndata['interactions'][predicted_nodes] = graph.ndata['degree'][predicted_nodes]'''

    graph.ndata['degree'] = graph.in_degrees().float()  # Ensure it's a float tensor

    graph.ndata['interactions'] = torch.zeros(graph.num_nodes(), dtype=torch.float, device=graph.device)

    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]

    graph.ndata['interactions'][predicted_nodes] = graph.ndata['degree'][predicted_nodes].squeeze()

    # Get degrees of the predicted nodes
    if predicted_nodes:
        predicted_degrees = graph.ndata['degree'][predicted_nodes]
        print(f"Average degree of predicted nodes: {predicted_degrees.mean().item():.2f}")
    else:
        print("No nodes predicted above the threshold.")
        '''
    if predicted_nodes:
        cluster_ids = graph.ndata['cluster'][predicted_nodes].cpu().numpy()
        interactions = graph.ndata['interactions'][predicted_nodes].cpu().numpy()

        # Create DataFrame and plot interactions
        data = pd.DataFrame({"Cluster": cluster_ids, "Interactions": interactions})
        plot_interactions_with_kcgs(data, output_path="results/kcgs_interactions.png")

    # Select nodes where the prediction score is above threshold
    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]'''



    # ---- Call the Plotting Function ----
    # Assuming `graph.ndata['cluster']` contains cluster IDs
    if predicted_nodes:
        cluster_ids = graph.ndata['cluster'][predicted_nodes].cpu().numpy()
        interactions = graph.ndata['interactions'][predicted_nodes].cpu().numpy()  # Interaction counts

        # Create DataFrame for visualization
        data = pd.DataFrame({"Cluster": cluster_ids, "Interactions": interactions})

        # Call the plotting function
        
        output_path = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}_2048.png')
        plot_interactions_with_kcgs(data, output_path)
        
    #cluster_and_visualize_predicted_genes(graph, non_labeled_nodes, predicted_cancer_genes, node_names, 'results/gene_prediction/', num_clusters=10)

    ##cluster_and_visualize_predicted_genes_ori(graph, non_labeled_nodes, node_names, 'results/gene_prediction/', num_clusters=5)
    # Load driver and reference gene sets


    # Calculate statistics
    non_labeled_nodes_count = len(non_labeled_nodes)
    ground_truth_driver_nodes = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_non_driver_nodes = [i for i, label in enumerate(labels) if label == 0]
    

    # Save both above and below threshold scores, sorted by scores in descending order
    predicted_driver_nodes_above_threshold = sorted(
        [(node_names[i], scores[i]) for i in non_labeled_nodes if scores[i] >= args.score_threshold],
        key=lambda x: x[1],
        reverse=True
    )
    predicted_driver_nodes_below_threshold = sorted(
        [(node_names[i], scores[i]) for i in non_labeled_nodes if scores[i] < args.score_threshold],
        key=lambda x: x[1],
        reverse=True
    )

    from collections import defaultdict
    import statistics

    # Get the ground truth driver gene indices and names
    ground_truth_driver_indices = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_driver_names = {node_names[i] for i in ground_truth_driver_indices}

    # Save predictions (above and below threshold) to CSV
    output_file_above = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_genes_above_epo{args.num_epochs}_2048.csv'
    )
    output_file_below = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_genes_below_epo{args.num_epochs}_2048.csv'
    )

    with open(output_file_above, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Score'])  # Header row
        csvwriter.writerows(predicted_driver_nodes_above_threshold)

    with open(output_file_below, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Score'])  # Header row
        csvwriter.writerows(predicted_driver_nodes_below_threshold)

    print(f"Predicted driver genes (above threshold) saved to {output_file_above}")
    print(f"Predicted driver genes (below threshold) saved to {output_file_below}")

    # Calculate degrees for nodes above and below the threshold (connecting only to label 1 nodes)
    degree_counts_above = defaultdict(int)
    degree_counts_below = defaultdict(int)

    for src, dst in edges:
        src_name = node_names[src]
        dst_name = node_names[dst]

        # Count only connections to ground truth driver genes (label 1 nodes)
        if dst_name in ground_truth_driver_names:
            if src_name in [gene for gene, _ in predicted_driver_nodes_above_threshold]:
                degree_counts_above[src_name] += 1
            elif src_name in [gene for gene, _ in predicted_driver_nodes_below_threshold]:
                degree_counts_below[src_name] += 1

    # Sort degrees by degree count in descending order
    sorted_degree_counts_above = sorted(degree_counts_above.items(), key=lambda x: x[1], reverse=True)
    sorted_degree_counts_below = sorted(degree_counts_below.items(), key=lambda x: x[1], reverse=True)

    # Save degrees of predicted driver genes connecting to ground truth driver genes (above threshold)
    degree_output_file_above = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_gene_degrees_above_epo{args.num_epochs}_2048.csv'
    )
    with open(degree_output_file_above, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Predicted Driver Gene', 'Degree'])  # Header row
        csvwriter.writerows(sorted_degree_counts_above)
        ##csvwriter.writerow(['Average Degree', average_degree_above])  # Save average degree

    print(f"Degrees of predicted driver genes (above threshold) saved to {degree_output_file_above}")



    # Prepare DataFrame for nodes with degrees
    nodes_with_degrees = []

    # Above threshold
    for gene, degree in degree_counts_above.items():
        nodes_with_degrees.append({'Gene_Set': 'Above Threshold', 'Degree': degree})

    # Below threshold
    for gene, degree in degree_counts_below.items():
        nodes_with_degrees.append({'Gene_Set': 'Below Threshold', 'Degree': degree})

    # Convert to DataFrame
    nodes_with_degrees = pd.DataFrame(nodes_with_degrees)

    ##print(f"sorted_degree_counts_below: {sorted_degree_counts_below}")
    sorted_degree_counts_above_value = [value for _, value in sorted_degree_counts_above if value <= 20]
    sorted_degree_counts_below_value = [value for _, value in sorted_degree_counts_below if value <= 20]
    
    output_above_file = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_output_above_file_epo{args.num_epochs}_2048.csv')
    output_below_file = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_output_below_file_epo{args.num_epochs}_2048.csv')

    calculate_and_save_prediction_stats(non_labeled_nodes, labels, node_names, scores, args)

    plot_degree_distributions(sorted_degree_counts_above_value, sorted_degree_counts_below_value, args, output_dir)

    generate_kde_and_curves(logits, node_names, degree_counts_above, degree_counts_below, labels, train_mask, args)

    plot_model_performance(args)

    plot_venn_diagram()

def train_all_passs(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    epoch_times, cpu_usages, gpu_usages = [], [], []

    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    graph = dgl.graph(edges, num_nodes=len(nodes))

    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

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

    print("\U0001F50D features.shape:", features.shape)
    print("\U0001F9E0 graph.num_nodes():", graph.num_nodes())
    print(f"Starting training for {args.num_epochs} epochs...")

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

    # Get neighbors of top predicted genes
    neighbor_indices = set()
    for idx in top_gene_indices:
        neighbors = graph.successors(idx).tolist()
        neighbor_indices.update(neighbors)

    # Combine top genes + their neighbors for IG analysis
    lrp_node_indices = top_gene_indices#list(set(top_gene_indices) | neighbor_indices)
    lrp_node_indices = torch.tensor(lrp_node_indices, dtype=torch.long, device=device)


    print("Generating feature importance plots...")

    top_node_features = embeddings[top_gene_indices].cpu().numpy()  # shape: [num_top_genes, feature_dim]

    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")
    best_k = 10#find_optimal_k(top_node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)
    
    print('best_k=================================== ', best_k)
    
    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_cluster_labels.npy")
    total_genes_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_total_genes_per_cluster.npy")
    pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_pred_counts.npy")
    graph, cluster_labels, total_genes_per_cluster, pred_counts = apply_spectral_biclustering(
        graph=graph,
        embeddings=graph.ndata['feat'],
        node_names=node_names,
        predicted_cancer_genes=node_names_topk,  
        n_clusters=best_k,
        save_path=spectral_biclustering_graph_path,
        save_cluster_labels_path=cluster_labels_path,
        save_total_genes_per_cluster_path=total_genes_per_cluster_path,
        save_predicted_counts_path=pred_counts_path,
        output_path_genes_clusters=output_path_genes_clusters
    )

    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )
    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )

    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}


    # Define where to save the clustered gene CSV

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    output_path_legend = os.path.join(output_dir, f'legend_omics.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS_OMICS, num_clusters=4)
    
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---
    ##############################################################################################################################

    print("Running spectral_biclustering on node features...")
    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"


    # If your features are in NumPy format:
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32, requires_grad=True)
    else:
        features = features.clone().detach().requires_grad_(True)

    ##relevance_scores = compute_saliency(model, graph, features, target_node=42)
    '''relevance_scores = compute_relevance_scores(model, graph, features)

    print("Relevance scores for node 42:=============================================", relevance_scores)

    topk_node_indices_tensor = torch.tensor(top_gene_indices)'''
    
    ##relevance_scores = compute_saliency(model, graph, features, target_node=42)
    relevance_scores = compute_relevance_scores_4_groups(model, graph, features)
    
    topk_node_indices_tensor = torch.tensor(top_gene_indices)
    
    # Check if the indices tensor is of the correct type
    if topk_node_indices_tensor.dtype != torch.int64:
        topk_node_indices_tensor = topk_node_indices_tensor.to(torch.int64)

    # Ensure the indices tensor is 1D
    if topk_node_indices_tensor.dim() != 1:
        topk_node_indices_tensor = topk_node_indices_tensor.view(-1)  # Flatten it if necessary

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]
    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    graph = graph.to(device)
    features = features.to(device)
       
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")     
    relevance_df = plot_spectral_biclustering_heatmap(
        args=args,
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        gene_names=node_names_topk,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )

    # summary_feature_relevance: shape [top_k, 64]
    reduced_scores = extract_summary_features_np_skip(relevance_scores_topk.cpu().numpy())
    
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")
    
    save_and_plot_confirmed_genes(
        args=args,
        node_names_topk=node_names_topk,
        node_scores_topk=scores[top_gene_indices],
        summary_feature_relevance=reduced_scores ,  # output from extract_summary_features_np(relevance_scores)
        feature_names=feature_names, 
        output_dir=output_dir,                 
        confirmed_genes_save_path=confirmed_file, 
        confirmed_gene_path="data/ncg_8886.txt"
    )

    #####################################################################################
    # Load confirmed driver genes
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    save_dir = "results/gene_prediction/gnnexplainer/"
    os.makedirs(save_dir, exist_ok=True)

    # Convert list to dict
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]
        print(f"{gene} → Node {node_idx} | Predicted score: {gene_score:.4f}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

        # Only proceed if there are exactly 12 neighbors
        '''if len(neighbor_indices) != 12:
            print(f"⏭️ Skipping {gene}: has {len(neighbor_indices)} neighbors (requires exactly 12).")
            continue'''

        neighbor_relevance_scores = [relevance_scores[i].sum().item() for i in neighbor_indices]
        neighbor_scores_dict = dict(zip(neighbor_indices, neighbor_relevance_scores))

        plot_path = os.path.join(
            "results/gene_prediction/neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_neighbor_relevance_epo{args.num_epochs}.png"
        )

        plot_neighbor_relevance(
            neighbor_scores=neighbor_scores_dict,
            gene_name=gene,
            node_id_to_name=node_id_to_name,
            output_path=plot_path
        )
        
    '''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    graph = graph.to(device)
    features = features.to(device)

    # Load confirmed driver genes
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    save_dir = "results/gene_prediction/gnnexplainer/"
    os.makedirs(save_dir, exist_ok=True)

    # Convert list to dict
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)'''

    #######################################################################
    # ✅ Gradient-based saliency for bio/topo feature relevance (confirmed genes)

    print("\n🔬 Computing gradient-based bio/topo saliency for confirmed genes...")

    # Define feature groups
    feature_groups = {
        "bio": (0, 1024),
        "topo": (1024, 2048)
    }

    # Filter for confirmed genes present in the graph
    gene_indices = [name_to_index[gene] for gene in confirmed_genes if gene in name_to_index]

    # Compute relevance scores using saliency (Integrated Gradients or raw gradients)
    relevance_scores = compute_relevance_scores(
        model=model,
        graph=graph,
        features=features,
        node_indices=gene_indices,
        normalize=True,
        feature_groups=feature_groups
    )

    # Make output directory
    save_dir = os.path.join(output_dir, "gnnexplainer/")
    os.makedirs(save_dir, exist_ok=True)


    # For bio (4 omics × 16 cancers = 64 features)
    bio_feat_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # For topo (just topology_0 to topology_63)
    topo_feat_names = [f"topology_{i}" for i in range(64)]

    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]
        print(f"{gene} → Node {node_idx} | Predicted score = {gene_score:.4f}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

        # Only proceed if there are exactly 12 neighbors
        if len(neighbor_indices) != 12:
            print(f"⏭️ Skipping {gene}: has {len(neighbor_indices)} neighbors (requires exactly 12).")
            continue

        #neighbor_relevance_scores = [relevance_scores[i].sum().item() for i in neighbor_indices]
        '''neighbor_relevance_scores = [
            relevance_scores[i].sum().item()
            for i in neighbor_indices
            if i in relevance_scores
        ]  '''
        neighbor_relevance_scores = [
            sum(t.sum().item() for t in relevance_scores[i].values())
            for i in neighbor_indices if i in relevance_scores
        ]
      
        neighbor_scores_dict = dict(zip(neighbor_indices, neighbor_relevance_scores))

        plot_path = os.path.join(
            "results/gene_prediction/neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_neighbor_relevance_epo{args.num_epochs}.png"
        )

        plot_neighbor_relevance(
            neighbor_scores=neighbor_scores_dict,
            gene_name=gene,
            node_id_to_name=node_id_to_name,
            output_path=plot_path
        )

        #node_idx = name_to_index[gene]

        if node_idx not in relevance_scores:
            print(f"⚠️ No relevance found for {gene} (node {node_idx})")
            continue

        # Get relevance vectors and reduce
        bio_1024 = relevance_scores[node_idx]["bio"].cpu().numpy().reshape(1, -1)
        topo_1024 = relevance_scores[node_idx]["topo"].cpu().numpy().reshape(1, -1)

        bio_64 = extract_summary_features_np_skip(bio_1024).squeeze()
        topo_64 = extract_summary_features_np_skip(topo_1024).squeeze()

        # Plot BIO saliency
        plot_feature_importance_bio(
            relevance_vector=bio_64,
            feature_names=bio_feat_names,
            node_name=f"{gene}",
            output_path=os.path.join(
                save_dir,
                f"{args.model_type}_{args.net_type}_{gene}_bio_feature_importance.png"
            )
        )

        # Plot TOPO saliency
        plot_feature_importance_topo(
            relevance_vector=topo_64,
            feature_names=topo_feat_names,
            node_name=f"{gene}",
            output_path=os.path.join(
                save_dir,
                f"{args.model_type}_{args.net_type}_{gene}_topo_feature_importance.png"
            )
        )

        from collections import defaultdict

        # This is the correct dictionary for group-wise aggregation
        groupwise_saliency = defaultdict(list)

        for node_id, group_saliency in relevance_scores.items():
            for group_name, saliency_tensor in group_saliency.items():
                groupwise_saliency[group_name].append(saliency_tensor.cpu().numpy())

        # Optional: convert lists to np.array
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
                save_path=os.path.join(save_dir, f"{args.model_type}_{args.net_type}_bio_topo_comparison_lineplot_{args.num_epochs}.png")
            )

    bio_matrix = []  # rows: genes, cols: features
    valid_genes = []  # confirmed genes that exist in graph

    for gene in confirmed_genes:
        if gene not in name_to_index or name_to_index[gene] not in relevance_scores:
            continue

        node_idx = name_to_index[gene]
        bio_1024 = relevance_scores[node_idx]["bio"].cpu().numpy().reshape(1, -1)
        bio_64 = extract_summary_features_np_skip(bio_1024).squeeze()

        bio_matrix.append(bio_64)
        valid_genes.append(gene)

    '''plot_saliency_heatmap(
        relevance_scores=relevance_scores,
        gene_list=confirmed_genes,
        name_to_index=name_to_index,
        extract_fn=extract_summary_features_np_skip,
        feature_names=feature_names,
        part="bio",
        output_dir=output_dir,
        model_type=args.model_type,
        net_type=args.net_type
    )

    plot_saliency_heatmap(
        relevance_scores=relevance_scores,
        gene_list=confirmed_genes,
        name_to_index=name_to_index,
        extract_fn=extract_summary_features_np_skip,
        feature_names=[],  # will be generated inside
        part="topo",
        output_dir=output_dir,
        model_type=args.model_type,
        net_type=args.net_type
    )
    '''
    
    part = "bio"  # or "topo"

    plot_saliency_biclustering_heatmap(
        relevance_scores=relevance_scores,
        gene_list=confirmed_genes,
        name_to_index=name_to_index,
        extract_fn=extract_summary_features_np_skip,
        feature_names=feature_names if part == "bio" else None,
        part=part,
        output_dir=output_dir,
        model_type=args.model_type,
        net_type=args.net_type
    )

def train_neighor_passs(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    epoch_times, cpu_usages, gpu_usages = [], [], []

    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    graph = dgl.graph(edges, num_nodes=len(nodes))

    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

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

    # Get neighbors of top predicted genes
    neighbor_indices = set()
    for idx in top_gene_indices:
        neighbors = graph.successors(idx).tolist()
        neighbor_indices.update(neighbors)

    # Combine top genes + their neighbors for IG analysis
    lrp_node_indices = top_gene_indices#list(set(top_gene_indices) | neighbor_indices)
    lrp_node_indices = torch.tensor(lrp_node_indices, dtype=torch.long, device=device)


    print("Generating feature importance plots...")

    top_node_features = embeddings[top_gene_indices].cpu().numpy()  # shape: [num_top_genes, feature_dim]

    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")
    best_k = 10#find_optimal_k(top_node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)

    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_cluster_labels.npy")
    total_genes_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_total_genes_per_cluster.npy")
    pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_pred_counts.npy")
    graph, cluster_labels, total_genes_per_cluster, pred_counts = apply_spectral_biclustering(
        graph=graph,
        embeddings=graph.ndata['feat'],
        node_names=node_names,
        predicted_cancer_genes=node_names_topk,  
        n_clusters=best_k,
        save_path=spectral_biclustering_graph_path,
        save_cluster_labels_path=cluster_labels_path,
        save_total_genes_per_cluster_path=total_genes_per_cluster_path,
        save_predicted_counts_path=pred_counts_path,
        output_path_genes_clusters=output_path_genes_clusters
    )

    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}

    graph.ndata['cluster'] = torch.tensor(cluster_labels)
    
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    ##############################################################################################################################

    print("Running spectral_biclustering on node features...")
    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"

    # If your features are in NumPy format:
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32, requires_grad=True)
    else:
        features = features.clone().detach().requires_grad_(True)

    ##relevance_scores = compute_saliency(model, graph, features, target_node=42)
    relevance_scores = compute_relevance_scores_4_groups(model, graph, features)
    topk_node_indices_tensor = torch.tensor(top_gene_indices)
    
    # Check if the indices tensor is of the correct type
    if topk_node_indices_tensor.dtype != torch.int64:
        topk_node_indices_tensor = topk_node_indices_tensor.to(torch.int64)

    # Ensure the indices tensor is 1D
    if topk_node_indices_tensor.dim() != 1:
        topk_node_indices_tensor = topk_node_indices_tensor.view(-1)  # Flatten it if necessary

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]
    print('cluster_labels_topk------------------- ', cluster_labels_topk)
    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")     
    relevance_df = plot_spectral_biclustering_heatmap(
        args=args,
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        gene_names=node_names_topk,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )

    # summary_feature_relevance: shape [top_k, 64]
    reduced_scores = extract_summary_features_np_skip(relevance_scores_topk.cpu().numpy())
    
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")
    
    save_and_plot_confirmed_genes(
        args=args,
        node_names_topk=node_names_topk,
        node_scores_topk=scores[top_gene_indices],
        summary_feature_relevance=reduced_scores ,  # output from extract_summary_features_np(relevance_scores)
        feature_names=feature_names, 
        output_dir=output_dir,                 
        confirmed_genes_save_path=confirmed_file, 
        confirmed_gene_path="data/ncg_8886.txt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    graph = graph.to(device)
    features = features.to(device)

    # Load confirmed driver genes
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    save_dir = "results/gene_prediction/gnnexplainer/"
    os.makedirs(save_dir, exist_ok=True)

    # Convert list to dict
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]
        print(f"{gene} → Node {node_idx} | Predicted score = {gene_score:.4f}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

        # Only proceed if there are exactly 12 neighbors
        '''if len(neighbor_indices) != 12:
            print(f"⏭️ Skipping {gene}: has {len(neighbor_indices)} neighbors (requires exactly 12).")
            continue'''

        neighbor_relevance_scores = [relevance_scores[i].sum().item() for i in neighbor_indices]
        neighbor_scores_dict = dict(zip(neighbor_indices, neighbor_relevance_scores))

        plot_path = os.path.join(
            "results/gene_prediction/neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_neighbor_relevance_epo{args.num_epochs}.png"
        )

        plot_neighbor_relevance(
            neighbor_scores=neighbor_scores_dict,
            gene_name=gene,
            node_id_to_name=node_id_to_name,
            output_path=plot_path
        )

def train_neighbor_pass(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    epoch_times, cpu_usages, gpu_usages = [], [], []

    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    graph = dgl.graph(edges, num_nodes=len(nodes))

    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

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

    # Get neighbors of top predicted genes
    neighbor_indices = set()
    for idx in top_gene_indices:
        neighbors = graph.successors(idx).tolist()
        neighbor_indices.update(neighbors)

    # Combine top genes + their neighbors for IG analysis
    lrp_node_indices = top_gene_indices#list(set(top_gene_indices) | neighbor_indices)
    lrp_node_indices = torch.tensor(lrp_node_indices, dtype=torch.long, device=device)


    print("Generating feature importance plots...")

    top_node_features = embeddings[top_gene_indices].cpu().numpy()  # shape: [num_top_genes, feature_dim]

    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")
    best_k = 10#find_optimal_k(top_node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)

    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_cluster_labels.npy")
    total_genes_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_total_genes_per_cluster.npy")
    pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_pred_counts.npy")
    graph, cluster_labels, total_genes_per_cluster, pred_counts = apply_spectral_biclustering(
        graph=graph,
        embeddings=graph.ndata['feat'],
        node_names=node_names,
        predicted_cancer_genes=node_names_topk,  
        n_clusters=best_k,
        save_path=spectral_biclustering_graph_path,
        save_cluster_labels_path=cluster_labels_path,
        save_total_genes_per_cluster_path=total_genes_per_cluster_path,
        save_predicted_counts_path=pred_counts_path,
        output_path_genes_clusters=output_path_genes_clusters
    )

    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}

    graph.ndata['cluster'] = torch.tensor(cluster_labels)
    
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    ##############################################################################################################################

    print("Running spectral_biclustering on node features...")
    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"

    # If your features are in NumPy format:
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32, requires_grad=True)
    else:
        features = features.clone().detach().requires_grad_(True)

    ##relevance_scores = compute_saliency(model, graph, features, target_node=42)
    relevance_scores = compute_relevance_scores(model, graph, features)
    topk_node_indices_tensor = torch.tensor(top_gene_indices)
    
    # Check if the indices tensor is of the correct type
    if topk_node_indices_tensor.dtype != torch.int64:
        topk_node_indices_tensor = topk_node_indices_tensor.to(torch.int64)

    # Ensure the indices tensor is 1D
    if topk_node_indices_tensor.dim() != 1:
        topk_node_indices_tensor = topk_node_indices_tensor.view(-1)  # Flatten it if necessary

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]
    print('cluster_labels_topk------------------- ', cluster_labels_topk)
    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")     
    relevance_df = plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        gene_names=node_names_topk,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )

    # summary_feature_relevance: shape [top_k, 64]
    reduced_scores = extract_summary_features_np_skip(relevance_scores_topk.cpu().numpy())
    
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")
    
    save_and_plot_confirmed_genes(
        args=args,
        node_names_topk=node_names_topk,
        node_scores_topk=scores[top_gene_indices],
        summary_feature_relevance=reduced_scores ,  # output from extract_summary_features_np(relevance_scores)
        feature_names=feature_names, 
        output_dir=output_dir,                 
        confirmed_genes_save_path=confirmed_file, 
        confirmed_gene_path="data/ncg_8886.txt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    graph = graph.to(device)
    features = features.to(device)

    # Load confirmed driver genes
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    save_dir = "results/gene_prediction/gnnexplainer/"
    os.makedirs(save_dir, exist_ok=True)

    # Convert list to dict
    node_id_to_name = {i: name for i, name in enumerate(node_names)}

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)

    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]
        print(f"{gene} → Node {node_idx} | Predicted score: {gene_score:.4f}")

        neighbors = neighbors_dict.get(gene, [])
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]

        # Only proceed if there are exactly 12 neighbors
        if len(neighbor_indices) != 12:
            print(f"⏭️ Skipping {gene}: has {len(neighbor_indices)} neighbors (requires exactly 12).")
            continue

        neighbor_relevance_scores = [relevance_scores[i].sum().item() for i in neighbor_indices]
        neighbor_scores_dict = dict(zip(neighbor_indices, neighbor_relevance_scores))

        plot_path = os.path.join(
            "results/gene_prediction/neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_neighbor_relevance_epo{args.num_epochs}.png"
        )

        plot_neighbor_relevance(
            neighbor_scores=neighbor_scores_dict,
            gene_name=gene,
            node_id_to_name=node_id_to_name,
            output_path=plot_path
        )

def train_graph_with_edge_node_lrp_pass(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)
    
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)
    
    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    
    ##graph = dgl.graph(edges)
    graph = dgl.graph(edges, num_nodes=len(nodes))

    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)
    
    in_feats = embeddings.shape[1]

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = embeddings.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()

        logits = model(graph, embeddings).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_times.append(time.time() - epoch_start)
        cpu_usages.append(cpu_usage)
        gpu_usages.append(gpu_usage)

        tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}, CPU: {cpu_usage}%, GPU: {gpu_usage:.2f} MB")

    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    name_to_index = {name: idx for idx, name in enumerate(node_names)}

    model.eval()
    with torch.no_grad():
        logits = model(graph, embeddings).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()

    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)

    predicted_cancer_genes = [i for i, _ in ranking[:1000]]   
    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top predicted nodes: {avg_degree:.2f}")
    else:
        print("No top nodes predicted above the threshold.")
 
    neighbor_indices = set()
    for idx in top_gene_indices:
        neighbors = graph.successors(idx).tolist()## + graph.predecessors(idx).tolist()
        neighbor_indices.update(neighbors)

    
    lrp_node_indices = list(set(top_gene_indices) | neighbor_indices)
    lrp_node_indices = torch.tensor(lrp_node_indices, dtype=torch.long)
    
    lrp_node_features = embeddings[lrp_node_indices].cpu().numpy()
        
    print("Computing LRP relevance scores for the predicted_cancer_genes and their neighbors...")

    full_lrp_features = torch.zeros(graph.num_nodes(), lrp_node_features.shape[1])  # [1980, 2048]
    full_lrp_features[lrp_node_indices] = torch.tensor(lrp_node_features, dtype=torch.float32)

    # Now call with full-sized feature tensor
    ##relevance_scores = compute_lrp_scores(model, graph, full_lrp_features)

    print("Generating feature importance plots...")

    top_node_features = embeddings[top_gene_indices].cpu().numpy()  # shape: [num_top_genes, feature_dim]

    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")
    best_k = 10#find_optimal_k(top_node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)
    
    print('best_k=================================== ', best_k)
    
    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_cluster_labels.npy")
    total_genes_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_total_genes_per_cluster.npy")
    pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_pred_counts.npy")
    graph, cluster_labels, total_genes_per_cluster, pred_counts = apply_spectral_biclustering(
        graph=graph,
        embeddings=graph.ndata['feat'],
        node_names=node_names,
        predicted_cancer_genes=node_names_topk,  
        n_clusters=best_k,
        save_path=spectral_biclustering_graph_path,
        save_cluster_labels_path=cluster_labels_path,
        save_total_genes_per_cluster_path=total_genes_per_cluster_path,
        save_predicted_counts_path=pred_counts_path,
        output_path_genes_clusters=output_path_genes_clusters
    )

    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )
    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )

    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}


    # Define where to save the clustered gene CSV

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    output_path_legend = os.path.join(output_dir, f'legend_omics.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS_OMICS, num_clusters=4)
    
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---
    ##############################################################################################################################

    print("Running spectral_biclustering on node features...")
    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"

    topk_node_indices_tensor = torch.tensor(top_gene_indices)

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")     
    relevance_df = plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        gene_names=node_names_topk,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )

    # summary_feature_relevance: shape [top_k, 64]
    reduced_scores = extract_summary_features_np_skip(relevance_scores_topk.cpu().numpy())
    
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")
    
    save_and_plot_confirmed_genes(
        args=args,
        node_names_topk=node_names_topk,
        node_scores_topk=scores[top_gene_indices],
        summary_feature_relevance=reduced_scores ,  # output from extract_summary_features_np(relevance_scores)
        feature_names=feature_names, 
        output_dir=output_dir,                 
        confirmed_genes_save_path=confirmed_file, 
        confirmed_gene_path="data/ncg_8886.txt"
    )

    ###############################################################################################################

    # Read confirmed genes from file
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)
    ##for gene in confirmed_genes_to_plot: 
    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]

        print(f"{gene} → Node {node_idx} | Predicted score: {gene_score:.4f}")
        
        neighbors = neighbors_dict.get(gene, [])
        
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]
        neighbor_relevance_scores = [relevance_scores[i].sum().item() for i in neighbor_indices]

        neighbor_gene_names = [node_names[i] for i in neighbor_indices]

        # Sort by total relevance score
        sorted_pairs = sorted(zip(neighbor_relevance_scores, neighbor_gene_names), reverse=True)
        top_pairs = sorted_pairs[:10]  # Top 10

        if not top_pairs:
            print(f"⚠️ No valid neighbors found for {gene}.")
            continue

        top_scores, top_names = zip(*top_pairs)
        plot_path = os.path.join(
            "results/gene_prediction/neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_feature_importance_epo{args.num_epochs}.png"
        )

        plot_feature_importance(
            relevance_vector=top_scores,
            gene_names=top_names,
            node_name=gene,
            output_path=plot_path
        )

def train_cp(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)
    
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)
    
    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    
    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)
    
    in_feats = embeddings.shape[1]

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = embeddings.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()

        logits = model(graph, embeddings).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_times.append(time.time() - epoch_start)
        cpu_usages.append(cpu_usage)
        gpu_usages.append(gpu_usage)

        tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}, CPU: {cpu_usage}%, GPU: {gpu_usage:.2f} MB")

    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    name_to_index = {name: idx for idx, name in enumerate(node_names)}

    model.eval()
    with torch.no_grad():
        logits = model(graph, embeddings).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()

    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)

    predicted_cancer_genes = [i for i, _ in ranking[:1000]]   
    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top predicted nodes: {avg_degree:.2f}")
    else:
        print("No top nodes predicted above the threshold.")
 
    neighbor_indices = set()
    for idx in top_gene_indices:
        neighbors = graph.successors(idx).tolist()## + graph.predecessors(idx).tolist()
        neighbor_indices.update(neighbors)

    
    lrp_node_indices = list(set(top_gene_indices) | neighbor_indices)
    lrp_node_indices = torch.tensor(lrp_node_indices, dtype=torch.long)
    
    lrp_node_features = embeddings[lrp_node_indices].cpu().numpy()
        
    print("Computing LRP relevance scores for the predicted_cancer_genes and their neighbors...")

    full_lrp_features = torch.zeros(graph.num_nodes(), lrp_node_features.shape[1])  # [1980, 2048]
    full_lrp_features[lrp_node_indices] = torch.tensor(lrp_node_features, dtype=torch.float32)

    # Now call with full-sized feature tensor
    relevance_scores = compute_lrp_scores(model, graph, full_lrp_features)

    print("Generating feature importance plots...")

    top_node_features = embeddings[top_gene_indices].cpu().numpy()  # shape: [num_top_genes, feature_dim]

    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")
    best_k = find_optimal_k(top_node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)
    
    print('best_k=================================== ', best_k)
    
    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_cluster_labels.npy")
    total_genes_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_total_genes_per_cluster.npy")
    pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_pred_counts.npy")
    graph, cluster_labels, total_genes_per_cluster, pred_counts = apply_spectral_biclustering(
        graph=graph,
        embeddings=graph.ndata['feat'],
        node_names=node_names,
        predicted_cancer_genes=node_names_topk,  
        n_clusters=best_k,
        save_path=spectral_biclustering_graph_path,
        save_cluster_labels_path=cluster_labels_path,
        save_total_genes_per_cluster_path=total_genes_per_cluster_path,
        save_predicted_counts_path=pred_counts_path,
        output_path_genes_clusters=output_path_genes_clusters
    )

    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )
    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )

    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}


    # Define where to save the clustered gene CSV

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    output_path_legend = os.path.join(output_dir, f'legend_omics.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS_OMICS, num_clusters=4)
    
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---
    ##############################################################################################################################

    print("Running spectral_biclustering on node features...")
    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"

    topk_node_indices_tensor = torch.tensor(top_gene_indices)

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")     
    relevance_df = plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        gene_names=node_names_topk,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )

    # summary_feature_relevance: shape [top_k, 64]
    reduced_scores = extract_summary_features_np_skip(relevance_scores_topk.cpu().numpy())
    
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")
    
    save_and_plot_confirmed_genes(
        args=args,
        node_names_topk=node_names_topk,
        node_scores_topk=scores[top_gene_indices],
        summary_feature_relevance=reduced_scores ,  # output from extract_summary_features_np(relevance_scores)
        feature_names=feature_names, 
        output_dir=output_dir,                 
        confirmed_genes_save_path=confirmed_file, 
        confirmed_gene_path="data/ncg_8886.txt"
    )

    ###############################################################################################################
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")

    # Read confirmed genes from file
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)
    ##for gene in confirmed_genes_to_plot: 
    for gene in confirmed_genes:

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]

        print(f"{gene} → Node {node_idx} | Predicted score: {gene_score:.4f}")
        
        neighbors = neighbors_dict.get(gene, [])
        
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]
        neighbor_relevance_scores = [relevance_scores[i].sum().item() for i in neighbor_indices]

        neighbor_gene_names = [node_names[i] for i in neighbor_indices]

        # Sort by total relevance score
        sorted_pairs = sorted(zip(neighbor_relevance_scores, neighbor_gene_names), reverse=True)
        top_pairs = sorted_pairs[:10]  # Top 10

        if not top_pairs:
            print(f"⚠️ No valid neighbors found for {gene}.")
            continue

        top_scores, top_names = zip(*top_pairs)
        plot_path = os.path.join(
            "results/gene_prediction/neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_feature_importance_epo{args.num_epochs}.png"
        )

        plot_feature_importance(
            relevance_vector=top_scores,
            gene_names=top_names,
            node_name=gene,
            output_path=plot_path
        )

def train_lrp_clean_pass(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)
    
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)
    
    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    
    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)
    
    in_feats = embeddings.shape[1]

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = embeddings.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()

        logits = model(graph, embeddings).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_times.append(time.time() - epoch_start)
        cpu_usages.append(cpu_usage)
        gpu_usages.append(gpu_usage)

        tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}, CPU: {cpu_usage}%, GPU: {gpu_usage:.2f} MB")


    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes the first 128 are embedding features

    '''omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 'THCA', 
                    'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    bio_feat_names = []
    for cancer in cancer_types:
        for omics in omics_types:
            for i in range(16):
                bio_feat_names.append(f"{cancer}_{omics}_{i}")

    topo_feat_names = [f"topology_{i}" for i in range(1024)]
    feature_names = bio_feat_names + topo_feat_names + ["degree"] + gene_feature_names'''

    target_genes = ["SRC", "SMAD3", "ESR1", "HDAC1", "ATM"]
    name_to_index = {name: idx for idx, name in enumerate(node_names)}

    

    model.eval()
    with torch.no_grad():
        logits = model(graph, embeddings).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()

    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)

    predicted_cancer_genes = [i for i, _ in ranking[:1000]]
    
    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top predicted nodes: {avg_degree:.2f}")
    else:
        print("No top nodes predicted above the threshold.")
 
    neighbor_indices = set()
    for idx in top_gene_indices:
        neighbors = graph.successors(idx).tolist()## + graph.predecessors(idx).tolist()
        neighbor_indices.update(neighbors)

    
    lrp_node_indices = list(set(top_gene_indices) | neighbor_indices)
    lrp_node_indices = torch.tensor(lrp_node_indices, dtype=torch.long)
    
    lrp_node_features = embeddings[lrp_node_indices].cpu().numpy()
        
    print("Computing LRP relevance scores for the predicted_cancer_genes and their neighbors...")

    full_lrp_features = torch.zeros(graph.num_nodes(), lrp_node_features.shape[1])  # [1980, 2048]
    full_lrp_features[lrp_node_indices] = torch.tensor(lrp_node_features, dtype=torch.float32)

    # Now call with full-sized feature tensor
    relevance_scores = compute_lrp_scores(model, graph, full_lrp_features)

    print("Generating feature importance plots...")

    top_node_features = embeddings[top_gene_indices].cpu().numpy()  # shape: [num_top_genes, feature_dim]

    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")
    best_k = find_optimal_k(top_node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)
    
    print('best_k=================================== ', best_k)
    
    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_cluster_labels.npy")
    total_genes_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_total_genes_per_cluster.npy")
    pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_pred_counts.npy")
    graph, cluster_labels, total_genes_per_cluster, pred_counts = apply_spectral_biclustering(
        graph=graph,
        embeddings=graph.ndata['feat'],
        node_names=node_names,
        predicted_cancer_genes=node_names_topk,  
        n_clusters=best_k,
        save_path=spectral_biclustering_graph_path,
        save_cluster_labels_path=cluster_labels_path,
        save_total_genes_per_cluster_path=total_genes_per_cluster_path,
        save_predicted_counts_path=pred_counts_path,
        output_path_genes_clusters=output_path_genes_clusters
    )

    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )
    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )

    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}


    # Define where to save the clustered gene CSV
    output_csv_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_clustered_genes.csv")
    predicted_cancer_gene_indices = [
        name_to_index[name] 
        for name in predicted_cancer_genes 
        if name in name_to_index
    ]

    # Save cluster gene info
    save_bioclustered_gene_info_csv(
        cluster_labels=cluster_labels,
        total_genes_per_cluster=total_genes_per_cluster,
        pred_counts=pred_counts,
        node_names=node_names,
        predicted_gene_indices=predicted_cancer_gene_indices,
        output_csv_path=output_csv_path
    )

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    output_path_legend = os.path.join(output_dir, f'legend_omics.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS_OMICS, num_clusters=4)
    
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---
    ##############################################################################################################################

    print("Running spectral_biclustering on node features...")
    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"

    topk_node_indices_tensor = torch.tensor(top_gene_indices)

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")     
    relevance_df = plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        gene_names=node_names_topk,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )

    # summary_feature_relevance: shape [top_k, 64]
    reduced_scores = extract_summary_features_np_skip(relevance_scores_topk.cpu().numpy())
    
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")
    
    save_and_plot_confirmed_genes(
        args=args,
        node_names_topk=node_names_topk,
        node_scores_topk=scores[top_gene_indices],
        summary_feature_relevance=reduced_scores ,  # output from extract_summary_features_np(relevance_scores)
        feature_names=feature_names, 
        output_dir=output_dir,                 
        confirmed_genes_save_path=confirmed_file, 
        confirmed_gene_path="data/ncg_8886.txt"
    )

    ###############################################################################################################
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")

    # Read confirmed genes from file
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)
    ##for gene in confirmed_genes_to_plot: 
    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]

        print(f"{gene} → Node {node_idx} | Predicted score: {gene_score:.4f}")
        
        neighbors = neighbors_dict.get(gene, [])
        
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]
        neighbor_relevance_scores = [relevance_scores[i].sum().item() for i in neighbor_indices]

        neighbor_gene_names = [node_names[i] for i in neighbor_indices]

        # Sort by total relevance score
        sorted_pairs = sorted(zip(neighbor_relevance_scores, neighbor_gene_names), reverse=True)
        top_pairs = sorted_pairs[:10]  # Top 10

        if not top_pairs:
            print(f"⚠️ No valid neighbors found for {gene}.")
            continue

        top_scores, top_names = zip(*top_pairs)
        plot_path = os.path.join(
            "results/gene_prediction/neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_feature_importance_epo{args.num_epochs}.png"
        )

        plot_feature_importance(
            relevance_vector=top_scores,
            gene_names=top_names,
            node_name=gene,
            output_path=plot_path
        )

def train_lrp_pass(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)
    
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)
    
    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    
    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)
    
    in_feats = embeddings.shape[1]

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = embeddings.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()

        logits = model(graph, embeddings).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_times.append(time.time() - epoch_start)
        cpu_usages.append(cpu_usage)
        gpu_usages.append(gpu_usage)

        tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}, CPU: {cpu_usage}%, GPU: {gpu_usage:.2f} MB")


    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes the first 128 are embedding features

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 'THCA', 
                    'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    bio_feat_names = []
    for cancer in cancer_types:
        for omics in omics_types:
            for i in range(16):
                bio_feat_names.append(f"{cancer}_{omics}_{i}")

    topo_feat_names = [f"topology_{i}" for i in range(1024)]
    feature_names = bio_feat_names + topo_feat_names + ["degree"] + gene_feature_names

    target_genes = ["SRC", "SMAD3", "ESR1", "HDAC1", "ATM"]
    name_to_index = {name: idx for idx, name in enumerate(node_names)}

    

    model.eval()
    with torch.no_grad():
        logits = model(graph, embeddings).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()

    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)

    predicted_cancer_genes = [i for i, _ in ranking[:1000]]
    
    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top predicted nodes: {avg_degree:.2f}")
    else:
        print("No top nodes predicted above the threshold.")
 
    neighbor_indices = set()
    for idx in top_gene_indices:
        neighbors = graph.successors(idx).tolist()## + graph.predecessors(idx).tolist()
        neighbor_indices.update(neighbors)

    
    lrp_node_indices = list(set(top_gene_indices) | neighbor_indices)
    lrp_node_indices = torch.tensor(lrp_node_indices, dtype=torch.long)
    
    lrp_node_features = embeddings[lrp_node_indices].cpu().numpy()
        
    print("Computing LRP relevance scores for the predicted_cancer_genes and their neighbors...")

    full_lrp_features = torch.zeros(graph.num_nodes(), lrp_node_features.shape[1])  # [1980, 2048]
    full_lrp_features[lrp_node_indices] = torch.tensor(lrp_node_features, dtype=torch.float32)

    # Now call with full-sized feature tensor
    relevance_scores = compute_lrp_scores(model, graph, full_lrp_features)

    print("Generating feature importance plots...")

    top_node_features = embeddings[top_gene_indices].cpu().numpy()  # shape: [num_top_genes, feature_dim]

    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")
    best_k = find_optimal_k(top_node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)
    
    print('best_k=================================== ', best_k)
    
    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_cluster_labels.npy")
    total_genes_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_total_genes_per_cluster.npy")
    pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_pred_counts.npy")
    graph, cluster_labels, total_genes_per_cluster, pred_counts = apply_spectral_biclustering(
        graph=graph,
        embeddings=graph.ndata['feat'],
        node_names=node_names,
        predicted_cancer_genes=node_names_topk,  
        n_clusters=best_k,
        save_path=spectral_biclustering_graph_path,
        save_cluster_labels_path=cluster_labels_path,
        save_total_genes_per_cluster_path=total_genes_per_cluster_path,
        save_predicted_counts_path=pred_counts_path,
        output_path_genes_clusters=output_path_genes_clusters
    )

    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )
    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )

    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}


    # Define where to save the clustered gene CSV
    output_csv_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_clustered_genes.csv")
    predicted_cancer_gene_indices = [
        name_to_index[name] 
        for name in predicted_cancer_genes 
        if name in name_to_index
    ]

    # Save cluster gene info
    save_bioclustered_gene_info_csv(
        cluster_labels=cluster_labels,
        total_genes_per_cluster=total_genes_per_cluster,
        pred_counts=pred_counts,
        node_names=node_names,
        predicted_gene_indices=predicted_cancer_gene_indices,
        output_csv_path=output_csv_path
    )

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )


    
    output_path_legend = os.path.join(output_dir, f'legend_omics.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS_OMICS, num_clusters=4)
    
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---
    ##############################################################################################################################

    print("Running spectral_biclustering on node features...")
    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes first 128 are used in embedding
    feature_names = gene_feature_names + ["degree"]

    cluster_tensor = graph.ndata['cluster']
    
    '''###################################################################################################
    #
    # Load saved bioclusted graph
    #
    ###################################################################################################

    graph = load_bioclustered_graph(spectral_biclustering_graph_path)
    cluster_labels = reload_cluster_labels(cluster_labels_path)
    total_genes_per_cluster = reload_total_genes_per_cluster(total_genes_per_cluster_path)
    pred_counts = reload_predicted_counts(pred_counts_path)'''

    ##print("Computing LRP relevance scores for the predicted_cancer_genes...")
    ##relevance_scores = compute_lrp_scores(model, graph, embeddings)

    # Convert to tensor for indexing
    topk_node_indices_tensor = torch.tensor(top_gene_indices)

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")     
    relevance_df = plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        gene_names=node_names_topk,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )

    # summary_feature_relevance: shape [top_k, 64]
    reduced_scores = extract_summary_features_np_skip(relevance_scores_topk.cpu().numpy())
    
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")
    
    save_and_plot_confirmed_genes(
        args=args,
        node_names_topk=node_names_topk,
        node_scores_topk=scores[top_gene_indices],
        summary_feature_relevance=reduced_scores ,  # output from extract_summary_features_np(relevance_scores)
        feature_names=feature_names, 
        output_dir=output_dir,                 
        confirmed_genes_save_path=confirmed_file, 
        confirmed_gene_path="data/ncg_8886.txt"
    )

    ###############################################################################################################
    confirmed_file = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_confirmed_genes_epo{args.num_epochs}.txt")

    # Read confirmed genes from file
    with open(confirmed_file, "r") as f:
        confirmed_genes = [line.strip() for line in f if line.strip()]

    neighbors_dict = get_neighbors_gene_names(graph, node_names, name_to_index, confirmed_genes)
    ##for gene in confirmed_genes_to_plot: 
    for gene in confirmed_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in the graph.")
            continue

        node_idx = name_to_index[gene]
        gene_score = scores[node_idx]

        print(f"{gene} → Node {node_idx} | Predicted score: {gene_score:.4f}")
        
        neighbors = neighbors_dict.get(gene, [])
        
        neighbor_indices = [name_to_index[n] for n in neighbors if n in name_to_index]
        neighbor_relevance_scores = [relevance_scores[i].sum().item() for i in neighbor_indices]

        neighbor_gene_names = [node_names[i] for i in neighbor_indices]

        # Sort by total relevance score
        sorted_pairs = sorted(zip(neighbor_relevance_scores, neighbor_gene_names), reverse=True)
        top_pairs = sorted_pairs[:10]  # Top 10

        if not top_pairs:
            print(f"⚠️ No valid neighbors found for {gene}.")
            continue

        top_scores, top_names = zip(*top_pairs)
        plot_path = os.path.join(
            "results/gene_prediction/neighbor_feature_contributions/",
            f"{args.model_type}_{args.net_type}_{gene}_feature_importance_epo{args.num_epochs}.png"
        )

        plot_feature_importance(
            relevance_vector=top_scores,
            gene_names=top_names,
            node_name=gene,
            output_path=plot_path
        )

def train_feature_relevance_pass(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)
    
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)
    
    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    
    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    # --- 🔹 SILHOUETTE SCORE BASED CLUSTER SELECTION ---
    node_features = embeddings.cpu().numpy()  # Convert embeddings to numpy
    
    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")

    # Find optimal k
    best_k = 12##find_optimal_k(node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)

    ##graph = load_bioclustered_graph(spectral_biclustering_graph_path)
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    
    in_feats = features.shape[1]

    graph.ndata['feat'].shape
    graph.ndata['degree'].shape
    features.shape

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()
        print("Final feature shape before model input:", features.shape)

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

        '''for c in range(12):
            node_indices = torch.nonzero(cluster_tensor == c).squeeze()
            if node_indices.numel() == 0:
                print(f"⚠️ No nodes in cluster {c}")
                continue

            cluster_scores = scores[node_indices]
            top_node_idx = node_indices[torch.argmax(cluster_scores)].item()
            gene_name = node_names[top_node_idx]

            print(f"Cluster {c} → Node {top_node_idx} ({gene_name})")

            plot_feature_importance(
                relevance_scores[top_node_idx],
                feature_names=feature_names,
                node_name=gene_name
            )
            '''


    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)


    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)
    
    ##predicted_cancer_genes = [i for i, _ in ranking[:5000]]
    ##predicted_cancer_genes = [i for i, _ in ranking[:2000]]
    predicted_cancer_genes = [i for i, _ in ranking[:1000]]
    
    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top predicted nodes: {avg_degree:.2f}")
    else:
        print("No top nodes predicted above the threshold.")


    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    '''cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, best_k
    )'''

    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    cluster_labels_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_cluster_labels.npy")
    total_genes_per_cluster_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_total_genes_per_cluster.npy")
    pred_counts_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_pred_counts.npy")
    ##graph = apply_spectral_biclustering(graph, embeddings, best_k, spectral_biclustering_graph_path)
    graph, cluster_labels, total_genes_per_cluster, pred_counts = apply_spectral_biclustering(
        graph=graph,
        embeddings=graph.ndata['feat'],
        node_names=node_names,
        predicted_cancer_genes=node_names_topk,  
        n_clusters=best_k,
        save_path=spectral_biclustering_graph_path,
        save_cluster_labels_path=cluster_labels_path,
        save_total_genes_per_cluster_path=total_genes_per_cluster_path,
        save_predicted_counts_path=pred_counts_path,
        output_path_genes_clusters=output_path_genes_clusters
    )

   
    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    ##plot_tsne_predicted_genes(graph, node_names, torch.sigmoid(logits), tsne_plot_path, args)
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )

    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )

    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}


    # Define where to save the clustered gene CSV
    output_csv_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_clustered_genes.csv")
    predicted_cancer_gene_indices = [
        name_to_index[name] 
        for name in predicted_cancer_genes 
        if name in name_to_index
    ]

    # Save cluster gene info
    save_bioclustered_gene_info_csv(
        cluster_labels=cluster_labels,
        total_genes_per_cluster=total_genes_per_cluster,
        pred_counts=pred_counts,
        node_names=node_names,
        predicted_gene_indices=predicted_cancer_gene_indices,
        output_csv_path=output_csv_path
    )

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    ##pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    
    '''topk = args.topk if hasattr(args, 'topk') else 1000  # or set a fixed value
    scores_tensor = torch.tensor(scores)

    # If `scores` is a list of floats
    top_gene_indices = torch.topk(scores_tensor, topk).indices'''
    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---
    ##############################################################################################################################
    
   
    # ----- 🔹 CLUSTERING STEP -----
    print("Running spectral_biclustering on node features...")
    '''kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)'''
    ##graph = apply_spectral_biclustering(graph, embeddings, best_k)


    '''
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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
        
    

    lrp_heatmap_top1000_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_slrp_heatmap_top1000_epo{args.num_epochs}.png")

    print("Generating feature importance plot...")

    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)
    '''
    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes first 128 are used in embedding
    feature_names = gene_feature_names + ["degree"]

    cluster_tensor = graph.ndata['cluster']
    
    '''###################################################################################################
    #
    # Load saved bioclusted graph
    #
    ###################################################################################################

    graph = load_bioclustered_graph(spectral_biclustering_graph_path)
    cluster_labels = reload_cluster_labels(cluster_labels_path)
    total_genes_per_cluster = reload_total_genes_per_cluster(total_genes_per_cluster_path)
    pred_counts = reload_predicted_counts(pred_counts_path)'''

    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)

    # ✅ Get top 1000 predicted nodes

    '''topk = 1000 if graph.num_nodes() >= 1000 else graph.num_nodes()

    # Extract just the numerical scores for top-k selection
    score_values = torch.tensor([s for _, s in non_labeled_scores])

    topk_indices = torch.topk(score_values, topk).indices

    # Retrieve the top-k (name, score) pairs
    topk_named_scores = [non_labeled_scores[i] for i in top_gene_indices]'''


    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[top_gene_indices]
    cluster_labels_topk = graph.ndata['cluster'][top_gene_indices]
    
    # Convert to tensor for indexing
    topk_node_indices_tensor = torch.tensor(top_gene_indices)

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")
    '''plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores.detach().cpu().numpy(),
        cluster_labels=graph.ndata['cluster'].cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )'''

    '''plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )'''
        
    relevance_df = plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        gene_names=node_names_topk,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )

    # Make sure the output directory exists
    gene_contributions_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_{gene}_contributions_epo{args.num_epochs}.png")
 

    # summary_feature_relevance: shape [top_k, 64]
    summary_feature_relevance = extract_summary_features_np_skip(relevance_scores_topk.cpu().numpy())

    for i, gene_name in enumerate(node_names_topk[:10]):
        relevance_vector = summary_feature_relevance[i]  # shape: (64,)
        output_path = os.path.join(output_dir, f"{gene_name}_feature_contributions_epo{args.num_epochs}.png")
        plot_gene_feature_contributions(
            gene_name=gene_name,
            relevance_vector=relevance_vector,
            feature_names=feature_names,
            output_path=output_path
        )
    # Save top 10 gene names
    top_genes_path = os.path.join(output_dir, f"top10_genes_epo{args.num_epochs}.txt")
    with open(top_genes_path, "w") as f:
        for gene_name in node_names_topk[:10]:
            f.write(f"{gene_name}\n")


    '''
    # Get the top 5 genes based on total relevance across all features
    top5_genes = relevance_df.sum(axis=1).sort_values(ascending=False).head(5).index.tolist()

    # Loop through and plot for each
    for gene in top5_genes:
        gene_contributions_path = f"output/per_gene_heatmaps/{gene}_contributions.png"
        
        plot_gene_feature_contributions(
            gene_name=gene,
            relevance_scores=relevance_df.values,
            gene_names=relevance_df.index.tolist(),
            feature_names=relevance_df.columns.tolist(),
            output_path=gene_contributions_path
        )
    '''
    feature_relevance_matrix_64D_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_feature_relevance_matrix_64D_epo{args.num_epochs}.png")
    save_reduced_feature_relevance_scores(
        relevance_scores=relevance_df.values,
        gene_names=relevance_df.index.tolist(),
        output_csv_path=feature_relevance_matrix_64D_path
    )



    embedding_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_predicted_gene_clusters_tsne_epo{args.num_epochs}.png")
    ##plot_predicted_gene_embeddings_by_cluster(graph, node_names, torch.sigmoid(logits), embedding_plot_path, score_threshold=args.score_threshold)

    '''output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top 2000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 2000 nodes predicted above the threshold.")

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())
    
    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)


    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')


    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)
    
    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        # Ensure idx is an integer
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])

        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])'''

    # === Load predicted genes with cluster info ===
    cluster_genes_df = pd.read_csv(output_path_cluster_genes)

    # === Load CPDB PPI file ===
    ppi_path = 'data/CPDB_ppi_0.99.csv'
    ppi_df = pd.read_csv(ppi_path)

    # === Match predicted genes with CPDB interactions ===
    match_records_partner1 = []
    match_records_partner2 = []

    for _, row in cluster_genes_df.iterrows():
        cluster_id = row['Cluster ID']
        gene = row['Gene Name']

        # Match when gene is partner1
        matches_p1 = ppi_df[ppi_df['partner1'] == gene]
        for _, match_row in matches_p1.iterrows():
            match_records_partner1.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

        # Match when gene is partner2
        matches_p2 = ppi_df[ppi_df['partner2'] == gene]
        for _, match_row in matches_p2.iterrows():
            match_records_partner2.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

    # === Save both partner1 and partner2 matched results ===
    output_dir = 'results/gene_prediction'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_p1 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner1_epo{args.num_epochs}.csv')
    output_path_p2 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner2_epo{args.num_epochs}.csv')

    # Save partner1 matches
    if match_records_partner1:
        with open(output_path_p1, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner1[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner1)
        print(f"✅ Saved partner1 matches to {output_path_p1}")
    else:
        print("⚠️ No partner1 matches found.")

    # Save partner2 matches
    if match_records_partner2:
        with open(output_path_p2, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner2[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner2)
        print(f"✅ Saved partner2 matches to {output_path_p2}")
    else:
        print("⚠️ No partner2 matches found.")


    # File paths
    match_output_path_p1 = output_path_p1
    match_output_path_p2 = output_path_p2

    # Load the match files
    df_p1 = pd.read_csv(match_output_path_p1)
    df_p2 = pd.read_csv(match_output_path_p2)

    # Create a helper column to form (partner1, partner2) tuple
    df_p1['pair'] = list(zip(df_p1['Matched Partner1'], df_p1['Matched Partner2']))
    df_p2['pair'] = list(zip(df_p2['Matched Partner1'], df_p2['Matched Partner2']))

    # Find common partner1–partner2 pairs
    common_pairs = set(df_p1['pair']) & set(df_p2['pair'])

    # Filter only those rows where the pair is common
    df_p1_common = df_p1[df_p1['pair'].isin(common_pairs)]
    df_p2_common = df_p2[df_p2['pair'].isin(common_pairs)]

    # Merge on partner pair
    merged = pd.merge(
        df_p1_common,
        df_p2_common,
        on=['Matched Partner1', 'Matched Partner2'],
        suffixes=('_p1', '_p2')
    )

    # Filter where matched genes from both partner1 and partner2 are from the same cluster
    same_cluster = merged[merged['Cluster ID_p1'] == merged['Cluster ID_p2']]

    # Optional: clean up or reorder columns
    output_columns = [
        'Cluster ID_p1', 'Gene Name_p1', 'Gene Name_p2',
        'Matched Partner1', 'Matched Partner2',
        'confidence_p1'
    ]
    same_cluster = same_cluster[output_columns]
    same_cluster = same_cluster.rename(columns={
        'Cluster ID_p1': 'Cluster ID',
        'Gene Name_p1': 'Matched Gene from Partner1',
        'Gene Name_p2': 'Matched Gene from Partner2',
        'confidence_p1': 'confidence'

    })

    # Save final filtered results
    output_path_common_same_cluster = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_common_same_cluster_epo{args.num_epochs}.csv')
    same_cluster.to_csv(output_path_common_same_cluster, index=False)

    print(f"✅ Saved common same-cluster matched gene pairs to {output_path_common_same_cluster}")

def train_biclustering_pass(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)
    
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)
    
    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    
    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    # --- 🔹 SILHOUETTE SCORE BASED CLUSTER SELECTION ---
    node_features = embeddings.cpu().numpy()  # Convert embeddings to numpy
    
    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")

    # Find optimal k
    best_k = 12##find_optimal_k(node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)

    ##graph = load_bioclustered_graph(spectral_biclustering_graph_path)
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    
    in_feats = features.shape[1]

    graph.ndata['feat'].shape
    graph.ndata['degree'].shape
    features.shape

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()
        print("Final feature shape before model input:", features.shape)

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

        '''for c in range(12):
            node_indices = torch.nonzero(cluster_tensor == c).squeeze()
            if node_indices.numel() == 0:
                print(f"⚠️ No nodes in cluster {c}")
                continue

            cluster_scores = scores[node_indices]
            top_node_idx = node_indices[torch.argmax(cluster_scores)].item()
            gene_name = node_names[top_node_idx]

            print(f"Cluster {c} → Node {top_node_idx} ({gene_name})")

            plot_feature_importance(
                relevance_scores[top_node_idx],
                feature_names=feature_names,
                node_name=gene_name
            )
            '''


    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    ##predicted_cancer_genes = [i for i, _ in ranking[:5000]]
    ##predicted_cancer_genes = [i for i, _ in ranking[:2000]]
    predicted_cancer_genes = [i for i, _ in ranking[:1000]]
    
    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top predicted nodes: {avg_degree:.2f}")
    else:
        print("No top nodes predicted above the threshold.")


    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    '''cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, best_k
    )'''

    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    ##graph = apply_spectral_biclustering(graph, embeddings, best_k, spectral_biclustering_graph_path)
    graph, cluster_labels, total_genes_per_cluster, pred_counts = apply_spectral_biclustering(
        graph=graph,
        embeddings=graph.ndata['feat'],
        node_names=node_names,
        predicted_cancer_genes=node_names_topk,  
        n_clusters=best_k,
        save_path=spectral_biclustering_graph_path,
        output_path_genes_clusters=output_path_genes_clusters
    )

   
    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    ##plot_tsne_predicted_genes(graph, node_names, torch.sigmoid(logits), tsne_plot_path, args)
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )

    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )

    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}


    # Define where to save the clustered gene CSV
    output_csv_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_clustered_genes.csv")
    predicted_cancer_gene_indices = [
        name_to_index[name] 
        for name in predicted_cancer_genes 
        if name in name_to_index
    ]

    # Save cluster gene info
    save_bioclustered_gene_info_csv(
        cluster_labels=cluster_labels,
        total_genes_per_cluster=total_genes_per_cluster,
        pred_counts=pred_counts,
        node_names=node_names,
        predicted_gene_indices=predicted_cancer_gene_indices,
        output_csv_path=output_csv_path
    )

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    ##pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    
    '''topk = args.topk if hasattr(args, 'topk') else 1000  # or set a fixed value
    scores_tensor = torch.tensor(scores)

    # If `scores` is a list of floats
    top_gene_indices = torch.topk(scores_tensor, topk).indices'''
    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---
    ##############################################################################################################################
    
   
    # ----- 🔹 CLUSTERING STEP -----
    print("Running spectral_biclustering on node features...")
    '''kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)'''
    ##graph = apply_spectral_biclustering(graph, embeddings, best_k)


    '''
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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
        
    

    lrp_heatmap_top1000_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_slrp_heatmap_top1000_epo{args.num_epochs}.png")

    print("Generating feature importance plot...")

    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)
    '''
    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes first 128 are used in embedding
    feature_names = gene_feature_names + ["degree"]

    cluster_tensor = graph.ndata['cluster']
    
    ###################################################################################################
    #
    # Load saved bioclusted graph
    #
    ###################################################################################################

    ##graph = load_bioclustered_graph(spectral_biclustering_graph_path)

    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)

    # ✅ Get top 1000 predicted nodes

    '''topk = 1000 if graph.num_nodes() >= 1000 else graph.num_nodes()

    # Extract just the numerical scores for top-k selection
    score_values = torch.tensor([s for _, s in non_labeled_scores])

    topk_indices = torch.topk(score_values, topk).indices

    # Retrieve the top-k (name, score) pairs
    topk_named_scores = [non_labeled_scores[i] for i in top_gene_indices]'''


    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[top_gene_indices]
    cluster_labels_topk = graph.ndata['cluster'][top_gene_indices]
    
    # Convert to tensor for indexing
    topk_node_indices_tensor = torch.tensor(top_gene_indices)

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")
    '''plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores.detach().cpu().numpy(),
        cluster_labels=graph.ndata['cluster'].cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )'''

    plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )
        
    embedding_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_predicted_gene_clusters_tsne_epo{args.num_epochs}.png")
    ##plot_predicted_gene_embeddings_by_cluster(graph, node_names, torch.sigmoid(logits), embedding_plot_path, score_threshold=args.score_threshold)

    '''output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top 2000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 2000 nodes predicted above the threshold.")

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())
    
    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)


    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')


    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)
    
    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        # Ensure idx is an integer
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])

        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])'''

    # === Load predicted genes with cluster info ===
    cluster_genes_df = pd.read_csv(output_path_cluster_genes)

    # === Load CPDB PPI file ===
    ppi_path = 'data/CPDB_ppi_0.99.csv'
    ppi_df = pd.read_csv(ppi_path)

    # === Match predicted genes with CPDB interactions ===
    match_records_partner1 = []
    match_records_partner2 = []

    for _, row in cluster_genes_df.iterrows():
        cluster_id = row['Cluster ID']
        gene = row['Gene Name']

        # Match when gene is partner1
        matches_p1 = ppi_df[ppi_df['partner1'] == gene]
        for _, match_row in matches_p1.iterrows():
            match_records_partner1.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

        # Match when gene is partner2
        matches_p2 = ppi_df[ppi_df['partner2'] == gene]
        for _, match_row in matches_p2.iterrows():
            match_records_partner2.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

    # === Save both partner1 and partner2 matched results ===
    output_dir = 'results/gene_prediction'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_p1 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner1_epo{args.num_epochs}.csv')
    output_path_p2 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner2_epo{args.num_epochs}.csv')

    # Save partner1 matches
    if match_records_partner1:
        with open(output_path_p1, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner1[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner1)
        print(f"✅ Saved partner1 matches to {output_path_p1}")
    else:
        print("⚠️ No partner1 matches found.")

    # Save partner2 matches
    if match_records_partner2:
        with open(output_path_p2, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner2[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner2)
        print(f"✅ Saved partner2 matches to {output_path_p2}")
    else:
        print("⚠️ No partner2 matches found.")


    # File paths
    match_output_path_p1 = output_path_p1
    match_output_path_p2 = output_path_p2

    # Load the match files
    df_p1 = pd.read_csv(match_output_path_p1)
    df_p2 = pd.read_csv(match_output_path_p2)

    # Create a helper column to form (partner1, partner2) tuple
    df_p1['pair'] = list(zip(df_p1['Matched Partner1'], df_p1['Matched Partner2']))
    df_p2['pair'] = list(zip(df_p2['Matched Partner1'], df_p2['Matched Partner2']))

    # Find common partner1–partner2 pairs
    common_pairs = set(df_p1['pair']) & set(df_p2['pair'])

    # Filter only those rows where the pair is common
    df_p1_common = df_p1[df_p1['pair'].isin(common_pairs)]
    df_p2_common = df_p2[df_p2['pair'].isin(common_pairs)]

    # Merge on partner pair
    merged = pd.merge(
        df_p1_common,
        df_p2_common,
        on=['Matched Partner1', 'Matched Partner2'],
        suffixes=('_p1', '_p2')
    )

    # Filter where matched genes from both partner1 and partner2 are from the same cluster
    same_cluster = merged[merged['Cluster ID_p1'] == merged['Cluster ID_p2']]

    # Optional: clean up or reorder columns
    output_columns = [
        'Cluster ID_p1', 'Gene Name_p1', 'Gene Name_p2',
        'Matched Partner1', 'Matched Partner2',
        'confidence_p1'
    ]
    same_cluster = same_cluster[output_columns]
    same_cluster = same_cluster.rename(columns={
        'Cluster ID_p1': 'Cluster ID',
        'Gene Name_p1': 'Matched Gene from Partner1',
        'Gene Name_p2': 'Matched Gene from Partner2',
        'confidence_p1': 'confidence'

    })

    # Save final filtered results
    output_path_common_same_cluster = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_common_same_cluster_epo{args.num_epochs}.csv')
    same_cluster.to_csv(output_path_common_same_cluster, index=False)

    print(f"✅ Saved common same-cluster matched gene pairs to {output_path_common_same_cluster}")

def train_cluster_gene_number_do_not_match_pas(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)
    
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)
    
    node_names = list(nodes.keys())
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    
    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    # --- 🔹 SILHOUETTE SCORE BASED CLUSTER SELECTION ---
    node_features = embeddings.cpu().numpy()  # Convert embeddings to numpy
    
    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")

    # Find optimal k
    best_k = 12##find_optimal_k(node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)
    spectral_biclustering_graph_path = os.path.join('data/', f"{args.model_type}_{args.net_type}_saved_clustered_graph.pth")
    graph = apply_spectral_biclustering(graph, embeddings, best_k, spectral_biclustering_graph_path)
    
    ##graph = load_bioclustered_graph(spectral_biclustering_graph_path)
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    
    in_feats = features.shape[1]

    graph.ndata['feat'].shape
    graph.ndata['degree'].shape
    features.shape

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()
        print("Final feature shape before model input:", features.shape)

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

        '''for c in range(12):
            node_indices = torch.nonzero(cluster_tensor == c).squeeze()
            if node_indices.numel() == 0:
                print(f"⚠️ No nodes in cluster {c}")
                continue

            cluster_scores = scores[node_indices]
            top_node_idx = node_indices[torch.argmax(cluster_scores)].item()
            gene_name = node_names[top_node_idx]

            print(f"Cluster {c} → Node {top_node_idx} ({gene_name})")

            plot_feature_importance(
                relevance_scores[top_node_idx],
                feature_names=feature_names,
                node_name=gene_name
            )
            '''


    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    ##predicted_cancer_genes = [i for i, _ in ranking[:5000]]
    ##predicted_cancer_genes = [i for i, _ in ranking[:2000]]
    predicted_cancer_genes = [i for i, _ in ranking[:1000]]

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, best_k
    )


    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}


    # Define where to save the clustered gene CSV
    output_csv_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_clustered_genes.csv")
    predicted_cancer_gene_indices = [
        name_to_index[name] 
        for name in predicted_cancer_genes 
        if name in name_to_index
    ]

    # Save cluster gene info
    save_bioclustered_gene_info_csv(
        cluster_labels=cluster_labels,
        total_genes_per_cluster=total_genes_per_cluster,
        pred_counts=pred_counts,
        node_names=node_names,
        predicted_gene_indices=predicted_cancer_gene_indices,
        output_csv_path=output_csv_path
    )

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top predicted nodes: {avg_degree:.2f}")
    else:
        print("No top nodes predicted above the threshold.")


    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices] 
    
    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    ##plot_tsne_predicted_genes(graph, node_names, torch.sigmoid(logits), tsne_plot_path, args)
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )

    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    ##pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    
    '''topk = args.topk if hasattr(args, 'topk') else 1000  # or set a fixed value
    scores_tensor = torch.tensor(scores)

    # If `scores` is a list of floats
    top_gene_indices = torch.topk(scores_tensor, topk).indices'''
    pcg_nodes = top_gene_indices##.tolist()

    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---
    ##############################################################################################################################
    
   
    # ----- 🔹 CLUSTERING STEP -----
    print("Running spectral_biclustering on node features...")
    '''kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)'''
    ##graph = apply_spectral_biclustering(graph, embeddings, best_k)


    '''
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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
        
    

    lrp_heatmap_top1000_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_slrp_heatmap_top1000_epo{args.num_epochs}.png")

    print("Generating feature importance plot...")

    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)
    '''
    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes first 128 are used in embedding
    feature_names = gene_feature_names + ["degree"]

    cluster_tensor = graph.ndata['cluster']
    
    ###################################################################################################
    #
    # Load saved bioclusted graph
    #
    ###################################################################################################

    graph = load_bioclustered_graph(spectral_biclustering_graph_path)

    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)

    # ✅ Get top 1000 predicted nodes

    '''topk = 1000 if graph.num_nodes() >= 1000 else graph.num_nodes()

    # Extract just the numerical scores for top-k selection
    score_values = torch.tensor([s for _, s in non_labeled_scores])

    topk_indices = torch.topk(score_values, topk).indices

    # Retrieve the top-k (name, score) pairs
    topk_named_scores = [non_labeled_scores[i] for i in top_gene_indices]'''


    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[top_gene_indices]
    cluster_labels_topk = graph.ndata['cluster'][top_gene_indices]
    
    # Convert to tensor for indexing
    topk_node_indices_tensor = torch.tensor(top_gene_indices)

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_node_indices_tensor]
    cluster_labels_topk = graph.ndata['cluster'][topk_node_indices_tensor]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")
    '''plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores.detach().cpu().numpy(),
        cluster_labels=graph.ndata['cluster'].cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )'''

    plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )
        
    embedding_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_predicted_gene_clusters_tsne_epo{args.num_epochs}.png")
    ##plot_predicted_gene_embeddings_by_cluster(graph, node_names, torch.sigmoid(logits), embedding_plot_path, score_threshold=args.score_threshold)

    '''output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top 2000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 2000 nodes predicted above the threshold.")

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())
    
    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)


    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')


    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)
    
    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        # Ensure idx is an integer
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])

        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])'''

    # === Load predicted genes with cluster info ===
    cluster_genes_df = pd.read_csv(output_path_cluster_genes)

    # === Load CPDB PPI file ===
    ppi_path = 'data/CPDB_ppi_0.99.csv'
    ppi_df = pd.read_csv(ppi_path)

    # === Match predicted genes with CPDB interactions ===
    match_records_partner1 = []
    match_records_partner2 = []

    for _, row in cluster_genes_df.iterrows():
        cluster_id = row['Cluster ID']
        gene = row['Gene Name']

        # Match when gene is partner1
        matches_p1 = ppi_df[ppi_df['partner1'] == gene]
        for _, match_row in matches_p1.iterrows():
            match_records_partner1.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

        # Match when gene is partner2
        matches_p2 = ppi_df[ppi_df['partner2'] == gene]
        for _, match_row in matches_p2.iterrows():
            match_records_partner2.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

    # === Save both partner1 and partner2 matched results ===
    output_dir = 'results/gene_prediction'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_p1 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner1_epo{args.num_epochs}.csv')
    output_path_p2 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner2_epo{args.num_epochs}.csv')

    # Save partner1 matches
    if match_records_partner1:
        with open(output_path_p1, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner1[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner1)
        print(f"✅ Saved partner1 matches to {output_path_p1}")
    else:
        print("⚠️ No partner1 matches found.")

    # Save partner2 matches
    if match_records_partner2:
        with open(output_path_p2, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner2[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner2)
        print(f"✅ Saved partner2 matches to {output_path_p2}")
    else:
        print("⚠️ No partner2 matches found.")


    # File paths
    match_output_path_p1 = output_path_p1
    match_output_path_p2 = output_path_p2

    # Load the match files
    df_p1 = pd.read_csv(match_output_path_p1)
    df_p2 = pd.read_csv(match_output_path_p2)

    # Create a helper column to form (partner1, partner2) tuple
    df_p1['pair'] = list(zip(df_p1['Matched Partner1'], df_p1['Matched Partner2']))
    df_p2['pair'] = list(zip(df_p2['Matched Partner1'], df_p2['Matched Partner2']))

    # Find common partner1–partner2 pairs
    common_pairs = set(df_p1['pair']) & set(df_p2['pair'])

    # Filter only those rows where the pair is common
    df_p1_common = df_p1[df_p1['pair'].isin(common_pairs)]
    df_p2_common = df_p2[df_p2['pair'].isin(common_pairs)]

    # Merge on partner pair
    merged = pd.merge(
        df_p1_common,
        df_p2_common,
        on=['Matched Partner1', 'Matched Partner2'],
        suffixes=('_p1', '_p2')
    )

    # Filter where matched genes from both partner1 and partner2 are from the same cluster
    same_cluster = merged[merged['Cluster ID_p1'] == merged['Cluster ID_p2']]

    # Optional: clean up or reorder columns
    output_columns = [
        'Cluster ID_p1', 'Gene Name_p1', 'Gene Name_p2',
        'Matched Partner1', 'Matched Partner2',
        'confidence_p1'
    ]
    same_cluster = same_cluster[output_columns]
    same_cluster = same_cluster.rename(columns={
        'Cluster ID_p1': 'Cluster ID',
        'Gene Name_p1': 'Matched Gene from Partner1',
        'Gene Name_p2': 'Matched Gene from Partner2',
        'confidence_p1': 'confidence'

    })

    # Save final filtered results
    output_path_common_same_cluster = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_common_same_cluster_epo{args.num_epochs}.csv')
    same_cluster.to_csv(output_path_common_same_cluster, index=False)

    print(f"✅ Saved common same-cluster matched gene pairs to {output_path_common_same_cluster}")

def train_(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    # --- 🔹 SILHOUETTE SCORE BASED CLUSTER SELECTION ---
    node_features = embeddings.cpu().numpy()  # Convert embeddings to numpy
    
    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")

    # Find optimal k
    best_k = 12##find_optimal_k(node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    
    in_feats = features.shape[1]

    graph.ndata['feat'].shape
    graph.ndata['degree'].shape
    features.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()
        print("Final feature shape before model input:", features.shape)

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    ##predicted_cancer_genes = [i for i, _ in ranking[:5000]]
    ##predicted_cancer_genes = [i for i, _ in ranking[:2000]]
    predicted_cancer_genes = [i for i, _ in ranking[:1000]]

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, best_k
    )

    ##############################################################################################################################
      
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(best_k)}


    # Define where to save the clustered gene CSV
    output_csv_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_clustered_genes.csv")
    predicted_cancer_gene_indices = [
        name_to_index[name] 
        for name in predicted_cancer_genes 
        if name in name_to_index
    ]

    # Save cluster gene info
    save_bioclustered_gene_info_csv(
        cluster_labels=cluster_labels,
        total_genes_per_cluster=total_genes_per_cluster,
        pred_counts=pred_counts,
        node_names=node_names,
        predicted_gene_indices=predicted_cancer_gene_indices,
        output_csv_path=output_csv_path
    )

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top 2000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 2000 nodes predicted above the threshold.")

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---
    ##############################################################################################################################
    
   
    # ----- 🔹 CLUSTERING STEP -----
    print("Running spectral_biclustering on node features...")
    '''kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)'''
    graph = apply_spectral_biclustering(graph, embeddings, best_k)


    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    lrp_heatmap_top1000_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_slrp_heatmap_top1000_epo{args.num_epochs}.png")

    print("Generating feature importance plot...")

    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)

        '''for c in range(12):
            node_indices = torch.nonzero(cluster_tensor == c).squeeze()
            if node_indices.numel() == 0:
                print(f"⚠️ No nodes in cluster {c}")
                continue

            cluster_scores = scores[node_indices]
            top_node_idx = node_indices[torch.argmax(cluster_scores)].item()
            gene_name = node_names[top_node_idx]

            print(f"Cluster {c} → Node {top_node_idx} ({gene_name})")

            plot_feature_importance(
                relevance_scores[top_node_idx],
                feature_names=feature_names,
                node_name=gene_name
            )
            '''


    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes first 128 are used in embedding
    feature_names = gene_feature_names + ["degree"]

    cluster_tensor = graph.ndata['cluster']

    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)

    # ✅ Get top 1000 predicted nodes
    scores = torch.sigmoid(logits)
    topk = 1000 if graph.num_nodes() >= 1000 else graph.num_nodes()
    top_gene_indices = torch.topk(scores, topk).indices

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[top_gene_indices]
    cluster_labels_topk = graph.ndata['cluster'][top_gene_indices]

    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices.cpu().numpy()]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")
    '''plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores.detach().cpu().numpy(),
        cluster_labels=graph.ndata['cluster'].cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )'''

    plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )
        
    embedding_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_predicted_gene_clusters_tsne_epo{args.num_epochs}.png")
    ##plot_predicted_gene_embeddings_by_cluster(graph, node_names, torch.sigmoid(logits), embedding_plot_path, score_threshold=args.score_threshold)

    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    ##plot_tsne_predicted_genes(graph, node_names, torch.sigmoid(logits), tsne_plot_path, args)
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )

    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )





    '''output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top 2000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 2000 nodes predicted above the threshold.")

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())
    
    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)


    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')


    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)
    
    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        # Ensure idx is an integer
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])

        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])'''

    # === Load predicted genes with cluster info ===
    cluster_genes_df = pd.read_csv(output_path_cluster_genes)

    # === Load CPDB PPI file ===
    ppi_path = 'data/CPDB_ppi_0.99.csv'
    ppi_df = pd.read_csv(ppi_path)

    # === Match predicted genes with CPDB interactions ===
    match_records_partner1 = []
    match_records_partner2 = []

    for _, row in cluster_genes_df.iterrows():
        cluster_id = row['Cluster ID']
        gene = row['Gene Name']

        # Match when gene is partner1
        matches_p1 = ppi_df[ppi_df['partner1'] == gene]
        for _, match_row in matches_p1.iterrows():
            match_records_partner1.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

        # Match when gene is partner2
        matches_p2 = ppi_df[ppi_df['partner2'] == gene]
        for _, match_row in matches_p2.iterrows():
            match_records_partner2.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

    # === Save both partner1 and partner2 matched results ===
    output_dir = 'results/gene_prediction'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_p1 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner1_epo{args.num_epochs}.csv')
    output_path_p2 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner2_epo{args.num_epochs}.csv')

    # Save partner1 matches
    if match_records_partner1:
        with open(output_path_p1, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner1[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner1)
        print(f"✅ Saved partner1 matches to {output_path_p1}")
    else:
        print("⚠️ No partner1 matches found.")

    # Save partner2 matches
    if match_records_partner2:
        with open(output_path_p2, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner2[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner2)
        print(f"✅ Saved partner2 matches to {output_path_p2}")
    else:
        print("⚠️ No partner2 matches found.")


    # File paths
    match_output_path_p1 = output_path_p1
    match_output_path_p2 = output_path_p2

    # Load the match files
    df_p1 = pd.read_csv(match_output_path_p1)
    df_p2 = pd.read_csv(match_output_path_p2)

    # Create a helper column to form (partner1, partner2) tuple
    df_p1['pair'] = list(zip(df_p1['Matched Partner1'], df_p1['Matched Partner2']))
    df_p2['pair'] = list(zip(df_p2['Matched Partner1'], df_p2['Matched Partner2']))

    # Find common partner1–partner2 pairs
    common_pairs = set(df_p1['pair']) & set(df_p2['pair'])

    # Filter only those rows where the pair is common
    df_p1_common = df_p1[df_p1['pair'].isin(common_pairs)]
    df_p2_common = df_p2[df_p2['pair'].isin(common_pairs)]

    # Merge on partner pair
    merged = pd.merge(
        df_p1_common,
        df_p2_common,
        on=['Matched Partner1', 'Matched Partner2'],
        suffixes=('_p1', '_p2')
    )

    # Filter where matched genes from both partner1 and partner2 are from the same cluster
    same_cluster = merged[merged['Cluster ID_p1'] == merged['Cluster ID_p2']]

    # Optional: clean up or reorder columns
    output_columns = [
        'Cluster ID_p1', 'Gene Name_p1', 'Gene Name_p2',
        'Matched Partner1', 'Matched Partner2',
        'confidence_p1'
    ]
    same_cluster = same_cluster[output_columns]
    same_cluster = same_cluster.rename(columns={
        'Cluster ID_p1': 'Cluster ID',
        'Gene Name_p1': 'Matched Gene from Partner1',
        'Gene Name_p2': 'Matched Gene from Partner2',
        'confidence_p1': 'confidence'

    })

    # Save final filtered results
    output_path_common_same_cluster = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_common_same_cluster_epo{args.num_epochs}.csv')
    same_cluster.to_csv(output_path_common_same_cluster, index=False)

    print(f"✅ Saved common same-cluster matched gene pairs to {output_path_common_same_cluster}")

def train_lrp_pass(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)
    
    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    # --- 🔹 SILHOUETTE SCORE BASED CLUSTER SELECTION ---
    node_features = embeddings.cpu().numpy()  # Convert embeddings to numpy
    
    # Call the function to find optimal k
    plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_silhouette_score_plot_epo{args.num_epochs}.png")
    best_k_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_best_k_epo{args.num_epochs}.txt")

    # Find optimal k
    best_k = find_optimal_k(node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)

    print(f"Optimal k saved in {best_k_path} and plot saved at {plot_path}")
    
    # ----- 🔹 CLUSTERING STEP -----
    print("Running spectral_biclustering on node features...")
    '''kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)'''
    graph = apply_spectral_biclustering(graph, embeddings, best_k)


    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    lrp_heatmap_top1000_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_slrp_heatmap_top1000_epo{args.num_epochs}.png")

    print("Generating feature importance plot...")

    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)

        '''for c in range(12):
            node_indices = torch.nonzero(cluster_tensor == c).squeeze()
            if node_indices.numel() == 0:
                print(f"⚠️ No nodes in cluster {c}")
                continue

            cluster_scores = scores[node_indices]
            top_node_idx = node_indices[torch.argmax(cluster_scores)].item()
            gene_name = node_names[top_node_idx]

            print(f"Cluster {c} → Node {top_node_idx} ({gene_name})")

            plot_feature_importance(
                relevance_scores[top_node_idx],
                feature_names=feature_names,
                node_name=gene_name
            )
            '''


    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes first 128 are used in embedding
    feature_names = gene_feature_names + ["degree"]

    cluster_tensor = graph.ndata['cluster']

    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)

    # ✅ Get top 1000 predicted nodes
    scores = torch.sigmoid(logits)
    topk = 1000 if graph.num_nodes() >= 1000 else graph.num_nodes()
    top_gene_indices = torch.topk(scores, topk).indices

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[top_gene_indices]
    cluster_labels_topk = graph.ndata['cluster'][top_gene_indices]

    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in top_gene_indices.cpu().numpy()]

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")
    '''plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores.detach().cpu().numpy(),
        cluster_labels=graph.ndata['cluster'].cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )'''

    plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores_topk.detach().cpu().numpy(),
        cluster_labels=cluster_labels_topk.cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )
        
    embedding_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_predicted_gene_clusters_tsne_epo{args.num_epochs}.png")
    ##plot_predicted_gene_embeddings_by_cluster(graph, node_names, torch.sigmoid(logits), embedding_plot_path, score_threshold=args.score_threshold)

    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    ##plot_tsne_predicted_genes(graph, node_names, torch.sigmoid(logits), tsne_plot_path, args)
    plot_tsne_predicted_genes(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path,
        args
    )

    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[top_gene_indices], 
        tsne_plot_path
    )



def find_optimal_k(node_features, k_range=(2, 16), plot_path='silhouette_score_plot.png', save_best_k_path='best_k.txt'):
    """
    Find the optimal number of clusters k using silhouette score.
    
    Parameters:
        node_features (numpy.ndarray): The features of the nodes (embeddings).
        k_range (tuple): A range (min, max) for the values of k to evaluate.
        plot_path (str): Path to save the silhouette score plot.
        save_best_k_path (str): Path to save the best k value.
        
    Returns:
        int: The optimal number of clusters based on the highest silhouette score.
    """
    silhouette_scores = []
    K_range = range(k_range[0], k_range[1] + 1)

    # Compute silhouette score for each k in the range
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(node_features)
        score = silhouette_score(node_features, cluster_labels)
        silhouette_scores.append(score)
        print(f"Silhouette score for k={k}: {score:.4f}")
    
    # Find the best k (highest silhouette score)
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters (k) based on Silhouette Score: {best_k}")

    # Plot silhouette scores
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different k')

    # Save the plot to the specified path
    plt.savefig(plot_path)
    plt.close()  # Close the plot after saving

    # Save the best k value to a text file
    with open(save_best_k_path, 'w') as f:
        f.write(f"Best k: {best_k}\n")

    return best_k

def plot_lrp_cluster_heatmap_ori(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    filename_prefix: str = "lrp_heatmap"
):
    """
    Plot a heatmap of LRP scores averaged over nodes in each cluster.

    Args:
        all_cluster_scores: [num_nodes, num_features] tensor of LRP scores.
        feature_names: List of feature names.
        cluster_tensor: Tensor assigning each node to a cluster.
        output_dir: Path to directory to save the heatmap.
        model_type: Model type string (e.g., 'ACGNN').
        net_type: Graph type string.
        score_threshold: Score threshold for filtering (used in filename).
        num_epochs: Number of training epochs (used in filename).
        filename_prefix: Prefix for the plot file name.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 🔹 Convert to numpy and pandas
    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()

    # 🔹 Fix mismatch between feature names and actual tensor shape
    scores_np = scores_np[:, :len(feature_names)]
    feature_names_trimmed = feature_names[:scores_np.shape[1]]

    df = pd.DataFrame(scores_np, columns=feature_names_trimmed)
    df['cluster'] = cluster_ids

    # 🔹 Average LRP scores per cluster
    cluster_means = df.groupby('cluster').mean(numeric_only=True)

    # 🔹 Plot heatmap
    plt.figure(figsize=(22, 10))
    sns.heatmap(
        cluster_means,
        cmap='Greys',
        cbar_kws={"label": "Total feature contribution"},
        yticklabels=True,
        xticklabels=True,
        square=False
    )
    plt.title("Average LRP Feature Contribution per Cluster", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Cluster", fontsize=14)
    plt.xticks(rotation=90)

    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🔥 Saved LRP heatmap to: {plot_path}")

def train_before_clean(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    # ----- 🔹 CLUSTERING STEP -----
    print("Running spectral_biclustering on node features...")
    '''kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)'''
    graph = apply_spectral_biclustering(graph, embeddings, n_clusters=12)


    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes first 128 are used in embedding
    feature_names = gene_feature_names + ["degree"]

    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)
    cluster_tensor = graph.ndata['cluster']

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }


    lrp_heatmap_top1000_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_slrp_heatmap_top1000_epo{args.num_epochs}.png")
    '''plot_spectral_biclustering_lrp_heatmap_top_genes(
        relevance_scores=relevance_scores,  # shape: (num_nodes, num_features)
        feature_names=feature_names,
        top_k=1000,
        save_path=lrp_heatmap_top1000_path
    )'''


    '''
    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")
    plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores.detach().cpu().numpy(),
        cluster_labels=graph.ndata['cluster'].cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )

    top_node_feature_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_feature_clustered_omics_colored_heatmap_epo{args.num_epochs}.png")
    plot_feature_cluster_heatmap_colored_top_nodes(
        relevance_scores=relevance_scores,
        feature_names=feature_names,
        save_path=top_node_feature_heatmap_path,
        omics_splits=omics_splits,
        omics_colors={
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        },
        n_top_nodes=1000,
        n_feature_clusters=6
    )


    colored_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_feature_clustered_omics_colored_heatmap_epo{args.num_epochs}.png")
    plot_feature_cluster_heatmap_colored(
        relevance_scores=relevance_scores,
        feature_names=feature_names,
        save_path=colored_heatmap_path,
        omics_splits=omics_splits,
        omics_colors={
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        },
        n_feature_clusters=6  # or adjust
    )
    '''

    print("Generating feature importance plot...")

    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)

        '''for c in range(12):
            node_indices = torch.nonzero(cluster_tensor == c).squeeze()
            if node_indices.numel() == 0:
                print(f"⚠️ No nodes in cluster {c}")
                continue

            cluster_scores = scores[node_indices]
            top_node_idx = node_indices[torch.argmax(cluster_scores)].item()
            gene_name = node_names[top_node_idx]

            print(f"Cluster {c} → Node {top_node_idx} ({gene_name})")

            plot_feature_importance(
                relevance_scores[top_node_idx],
                feature_names=feature_names,
                node_name=gene_name
            )
            '''

    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")
    plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores.detach().cpu().numpy(),
        cluster_labels=graph.ndata['cluster'].cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )
    '''
    # 🔹 Get top 1000 predicted nodes (based on score)
    topk = 1000
    _, top_indices = torch.topk(scores, k=topk)

    top_relevance_scores = relevance_scores[top_indices]
    top_cluster_tensor = cluster_tensor[top_indices]


    multiomics_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_multiomics_heatmap_epo{args.num_epochs}.png")
    
    plot_sorted_multiomics_heatmap(
        output_path=multiomics_heatmap_path,
        relevance_scores=relevance_scores,
        cluster_tensor=graph.ndata['cluster'],
        scores=scores,
        omics_splits=['mf', 'cna', 'meth', 'ge'],
        omics_colors=['#D62728', '#1F77B4', '#2CA02C', '#9467BD']  # optional for later color legend
    )


    heatmap_save_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_real_multiomics_heatmap_epo{args.num_epochs}.png")

    plot_real_clustered_multiomics_heatmap(
        relevance_scores=top_relevance_scores,
        cluster_tensor=top_cluster_tensor,
        feature_names=feature_names,
        omics_splits=omics_splits,
        save_path=heatmap_save_path
    )


    final_output_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_cluster_lrp_final_heatmap_epo{args.num_epochs}.png")


    generate_cluster_lrp_omics_heatmap(
        relevance_scores=relevance_scores,
        cluster_tensor=graph.ndata['cluster'],
        output_path=final_output_path
    )

    output_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_lrp_cluster_heatmap_epo{args.num_epochs}.png")
    plot_cluster_lrp_heatmap(
        relevance_scores=relevance_scores,
        cluster_tensor=graph.ndata['cluster'],
        feature_names=feature_names,
        output_path=output_path,
        n_clusters=12
    )
    # Simulated data
    n_clusters = 12
    n_features = 160
    omics_types = ['mf', 'cna', 'meth', 'ge']
    np.random.seed(42)

    lrp_matrix = np.random.rand(n_clusters, n_features)
    cluster_sizes = np.random.randint(low=5, high=50, size=n_clusters)

    output_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_cluster_omics_heatmap_numpy_style_epo{args.num_epochs}.png")

    plot_cluster_omics_heatmap_numpy(
        lrp_matrix,
        cluster_sizes,
        omics_types,
        output_path
    )

    '''
    embedding_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_predicted_gene_clusters_tsne_epo{args.num_epochs}.png")
    ##plot_predicted_gene_embeddings_by_cluster(graph, node_names, torch.sigmoid(logits), embedding_plot_path, score_threshold=args.score_threshold)

    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    plot_tsne_predicted_genes(graph, node_names, torch.sigmoid(logits), tsne_plot_path, args)
    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    omics_types = ['mf', 'cna', 'meth', 'ge']  # List of omics types
    cluster_omics_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_cluster_omics_heatmap_epo{args.num_epochs}.png")
    plot_cluster_omics_heatmap(relevance_scores, cluster_tensor, feature_names, omics_types, cluster_omics_heatmap_path, n_clusters=12)    
    
    relevance_scores_np = compute_lrp_scores(model, graph, features).detach().cpu().numpy()
    cluster_tensor_np = graph.ndata['cluster'].cpu().numpy()
    output_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_heatmap_np_epo{args.num_epochs}.png')
    plot_cluster_omics_heatmap_np(
        relevance_scores=relevance_scores_np,
        cluster_labels=cluster_tensor_np,
        feature_names=feature_names,
        omics_types=['mf', 'cna', 'meth', 'ge'],
        output_path='results/plot.png',
        n_clusters=12
    )

    
    plot_cluster_lrp_heatmap(
        relevance_scores=relevance_scores,
        cluster_tensor=graph.ndata['cluster'],
        feature_names=feature_names,
        output_path=os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_lrp_heatmap_colored_epo{args.num_epochs}.png'),
        n_clusters=12
    )

    print("Generating cluster-wise LRP heatmap from model outputs...")

    n_clusters = 12
    feature_dim = relevance_scores.shape[1]
    lrp_matrix = np.zeros((n_clusters, feature_dim))
    cluster_sizes = np.zeros(n_clusters, dtype=int)

    for c in range(n_clusters):
        node_indices = torch.nonzero(cluster_tensor == c).squeeze()
        if node_indices.numel() == 0:
            print(f"⚠️ No nodes in cluster {c}")
            continue

        if node_indices.ndim == 0:
            node_indices = node_indices.unsqueeze(0)

        cluster_scores = relevance_scores[node_indices]
        lrp_matrix[c] = cluster_scores.mean(dim=0).cpu().numpy()
        cluster_sizes[c] = node_indices.numel()

    output_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_cluster_omics_heatmap_numpy_style.png")

    omics_types = ['mf', 'cna', 'meth', 'ge']  # adjust if needed based on actual omics ordering
    plot_cluster_omics_heatmap_numpy(
        lrp_matrix=lrp_matrix,
        cluster_sizes=cluster_sizes,
        omics_types=omics_types,
        output_path=output_path
    )

def train_(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    ##data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    data_path = os.path.join('../gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    # ----- 🔹 CLUSTERING STEP -----
    print("Running spectral_biclustering on node features...")
    '''kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)'''
    graph = apply_spectral_biclustering(graph, embeddings, n_clusters=12)


    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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


    print("Generating feature importance plot...")

    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)

        '''for c in range(12):
            node_indices = torch.nonzero(cluster_tensor == c).squeeze()
            if node_indices.numel() == 0:
                print(f"⚠️ No nodes in cluster {c}")
                continue

            cluster_scores = scores[node_indices]
            top_node_idx = node_indices[torch.argmax(cluster_scores)].item()
            gene_name = node_names[top_node_idx]

            print(f"Cluster {c} → Node {top_node_idx} ({gene_name})")

            plot_feature_importance(
                relevance_scores[top_node_idx],
                feature_names=feature_names,
                node_name=gene_name
            )
            '''

    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes first 128 are used in embedding
    feature_names = gene_feature_names + ["degree"]

    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)
    cluster_tensor = graph.ndata['cluster']

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    # 64 new feature names (e.g., 'BRCA_mf', 'KIRC_mf', ..., 'KIRP_meth')
    feature_names = [
        f"{cancer}_{omics}"
        for omics in omics_types
        for cancer in cancer_types
    ]

    # Update omics_splits for the new layout
    omics_splits = {
        'mf': (0, 15),
        'cna': (16, 31),
        'ge': (32, 47),
        'meth': (48, 63),
    }


    lrp_heatmap_top1000_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_slrp_heatmap_top1000_epo{args.num_epochs}.png")
    '''plot_spectral_biclustering_lrp_heatmap_top_genes(
        relevance_scores=relevance_scores,  # shape: (num_nodes, num_features)
        feature_names=feature_names,
        top_k=1000,
        save_path=lrp_heatmap_top1000_path
    )'''


    spectral_biclustering_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_spectral_biclustering_heatmap_epo{args.num_epochs}.png")
    plot_spectral_biclustering_heatmap(
        relevance_scores=relevance_scores.detach().cpu().numpy(),
        cluster_labels=graph.ndata['cluster'].cpu().numpy(),
        feature_names=feature_names,
        omics_splits=omics_splits,
        output_path=spectral_biclustering_heatmap_path
    )

    top_node_feature_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_feature_clustered_omics_colored_heatmap_epo{args.num_epochs}.png")
    plot_feature_cluster_heatmap_colored_top_nodes(
        relevance_scores=relevance_scores,
        feature_names=feature_names,
        save_path=top_node_feature_heatmap_path,
        omics_splits=omics_splits,
        omics_colors={
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        },
        n_top_nodes=1000,
        n_feature_clusters=6
    )


    colored_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_feature_clustered_omics_colored_heatmap_epo{args.num_epochs}.png")
    plot_feature_cluster_heatmap_colored(
        relevance_scores=relevance_scores,
        feature_names=feature_names,
        save_path=colored_heatmap_path,
        omics_splits=omics_splits,
        omics_colors={
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        },
        n_feature_clusters=6  # or adjust
    )


    # 🔹 Get top 1000 predicted nodes (based on score)
    topk = 1000
    _, top_indices = torch.topk(scores, k=topk)

    top_relevance_scores = relevance_scores[top_indices]
    top_cluster_tensor = cluster_tensor[top_indices]


    multiomics_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_multiomics_heatmap_epo{args.num_epochs}.png")
    
    plot_sorted_multiomics_heatmap(
        output_path=multiomics_heatmap_path,
        relevance_scores=relevance_scores,
        cluster_tensor=graph.ndata['cluster'],
        scores=scores,
        omics_splits=['mf', 'cna', 'meth', 'ge'],
        omics_colors=['#D62728', '#1F77B4', '#2CA02C', '#9467BD']  # optional for later color legend
    )


    heatmap_save_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_real_multiomics_heatmap_epo{args.num_epochs}.png")

    plot_real_clustered_multiomics_heatmap(
        relevance_scores=top_relevance_scores,
        cluster_tensor=top_cluster_tensor,
        feature_names=feature_names,
        omics_splits=omics_splits,
        save_path=heatmap_save_path
    )


    final_output_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_cluster_lrp_final_heatmap_epo{args.num_epochs}.png")


    generate_cluster_lrp_omics_heatmap(
        relevance_scores=relevance_scores,
        cluster_tensor=graph.ndata['cluster'],
        output_path=final_output_path
    )

    output_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_lrp_cluster_heatmap_epo{args.num_epochs}.png")
    '''plot_cluster_lrp_heatmap(
        relevance_scores=relevance_scores,
        cluster_tensor=graph.ndata['cluster'],
        feature_names=feature_names,
        output_path=output_path,
        n_clusters=12
    )'''
    # Simulated data
    n_clusters = 12
    n_features = 160
    omics_types = ['mf', 'cna', 'meth', 'ge']
    np.random.seed(42)

    lrp_matrix = np.random.rand(n_clusters, n_features)
    cluster_sizes = np.random.randint(low=5, high=50, size=n_clusters)

    output_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_cluster_omics_heatmap_numpy_style_epo{args.num_epochs}.png")

    plot_cluster_omics_heatmap_numpy(
        lrp_matrix,
        cluster_sizes,
        omics_types,
        output_path
    )


    embedding_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_predicted_gene_clusters_tsne_epo{args.num_epochs}.png")
    ##plot_predicted_gene_embeddings_by_cluster(graph, node_names, torch.sigmoid(logits), embedding_plot_path, score_threshold=args.score_threshold)

    tsne_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_tsne_top2_predicted_genes_epo{args.num_epochs}.png')
    plot_tsne_predicted_genes(graph, node_names, torch.sigmoid(logits), tsne_plot_path, args)
    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    omics_types = ['mf', 'cna', 'meth', 'ge']  # List of omics types
    cluster_omics_heatmap_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_cluster_omics_heatmap_epo{args.num_epochs}.png")
    plot_cluster_omics_heatmap(relevance_scores, cluster_tensor, feature_names, omics_types, cluster_omics_heatmap_path, n_clusters=12)    
    
    relevance_scores_np = compute_lrp_scores(model, graph, features).detach().cpu().numpy()
    cluster_tensor_np = graph.ndata['cluster'].cpu().numpy()
    output_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_heatmap_np_epo{args.num_epochs}.png')
    plot_cluster_omics_heatmap_np(
        relevance_scores=relevance_scores_np,
        cluster_labels=cluster_tensor_np,
        feature_names=feature_names,
        omics_types=['mf', 'cna', 'meth', 'ge'],
        output_path='results/plot.png',
        n_clusters=12
    )

    
    plot_cluster_lrp_heatmap(
        relevance_scores=relevance_scores,
        cluster_tensor=graph.ndata['cluster'],
        feature_names=feature_names,
        output_path=os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_lrp_heatmap_colored_epo{args.num_epochs}.png'),
        n_clusters=12
    )

    print("Generating cluster-wise LRP heatmap from model outputs...")

    n_clusters = 12
    feature_dim = relevance_scores.shape[1]
    lrp_matrix = np.zeros((n_clusters, feature_dim))
    cluster_sizes = np.zeros(n_clusters, dtype=int)

    for c in range(n_clusters):
        node_indices = torch.nonzero(cluster_tensor == c).squeeze()
        if node_indices.numel() == 0:
            print(f"⚠️ No nodes in cluster {c}")
            continue

        if node_indices.ndim == 0:
            node_indices = node_indices.unsqueeze(0)

        cluster_scores = relevance_scores[node_indices]
        lrp_matrix[c] = cluster_scores.mean(dim=0).cpu().numpy()
        cluster_sizes[c] = node_indices.numel()

    output_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_cluster_omics_heatmap_numpy_style.png")

    omics_types = ['mf', 'cna', 'meth', 'ge']  # adjust if needed based on actual omics ordering
    plot_cluster_omics_heatmap_numpy(
        lrp_matrix=lrp_matrix,
        cluster_sizes=cluster_sizes,
        omics_types=omics_types,
        output_path=output_path
    )

def plot_spectral_biclustering_heatmap_ori_(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None
):
    relevance_scores = extract_summary_features_np(relevance_scores)

    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        }

    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck',
        'LungAd', 'Thyroid', 'Bladder', 'Esophagus', 'Liver', 'Uterus',
        'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # Create pretty x-axis labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend(cancer_names)

    # Sort rows by cluster
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # Assign colors by omics type for each feature
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(nrows=18, ncols=18, wspace=0)


    ax = fig.add_subplot(gs[1:16, :16])         # Main heatmap
    ax_curve = fig.add_subplot(gs[1:16, 16:17]) # Sum curve
    ax_legend = fig.add_subplot(gs[-1, :])      # Legend at bottom

    # Cluster info for boundaries and counts
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # Normalize rows
    '''row_min = sorted_scores.min(axis=1, keepdims=True)
    row_max = sorted_scores.max(axis=1, keepdims=True)
    sorted_scores = (sorted_scores - row_min) / (row_max - row_min + 1e-8)'''

    # ✅ Main heatmap
    sns.heatmap(
        sorted_scores,
        ##cmap='Greys',
        cmap='binary',
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )
    ax_curve.set_ylim(ax.get_ylim())  # 🔥 Sync y-axis limits


    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # ✅ Cluster counts on left
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # ✅ Colored x-axis labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=10)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # ✅ Right-hand sum curve
    row_contributions = sorted_scores.sum(axis=1)
    ##row_contributions = row_contributions / np.max(row_contributions) ##skip normalization to max
    ## to keep the natural scale differences.
    offset = 0.15
    '''ax_curve.fill_betweenx(
        np.arange(len(row_contributions)),
        0,
        row_contributions * offset,
        color='gray',
        alpha=0.6
    )
    ax_curve.plot(
        row_contributions * offset,
        np.arange(len(row_contributions)),
        color='gray',
        linewidth=0.6
    )'''

    ax_curve.fill_betweenx(
        np.arange(len(row_contributions)),
        0,
        row_contributions,
        color='gray',
        alpha=0.6
    )
    ax_curve.plot(
        row_contributions,
        np.arange(len(row_contributions)),
        color='gray',
        linewidth=0.01
    )


    ##ax_curve.set_xlim(0, offset)
    ax_curve.set_xlim(0, row_contributions.max() * 1.05) ## dynamically expands the axis to fit the real (unscaled) curve

    ax_curve.set_xticks([])
    ax_curve.set_yticks([])
    ax_curve.set_facecolor("white")
    ax_curve.spines[:].set_visible(False)

    # ✅ Legend (omics types)
    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(handles=omics_patches, loc="center", ncol=len(omics_patches), frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")
  
def plot_spectral_biclustering_heatmap_x(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None
):
    relevance_scores = extract_summary_features_np(relevance_scores)

    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        }

    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend(cancer_names)

    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(nrows=11, ncols=20)  # Increased rows to move legend on top

    ax_legend = fig.add_subplot(gs[0, :])         # 🔺 Legend goes on top row
    ax = fig.add_subplot(gs[1:, :15])             # Main heatmap
    ax_bar = fig.add_subplot(gs[1:, 15:16])       # Contribution bar (thin)
    ax_curve = fig.add_subplot(gs[1:, 17:])       # Sum curve


    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    row_min = sorted_scores.min(axis=1, keepdims=True)
    row_max = sorted_scores.max(axis=1, keepdims=True)
    sorted_scores = (sorted_scores - row_min) / (row_max - row_min + 1e-8)

    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.2,
            "aspect": 12,
            "pad": 0.01
        },
        ax=ax
    )

    
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1, i), 1, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=12)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # Spacer (invisible)
    ##ax_spacer.axis("off")

    # ✅ Compute sum of contributions per row
    row_contributions = sorted_scores.sum(axis=1)

    # ✅ Scale contributions to be 1/4 of the heatmap's width
    row_contributions_scaled = row_contributions / row_contributions.max() * (ax.get_position().width / 4)

    # ✅ Plot on the right, filling between the curve and the right edge
    ax_curve.fill_betweenx(
        np.arange(len(row_contributions)),
        1 - row_contributions_scaled,  # 1 means the far-right side of the plot
        1,  # Start from the right edge
        color='gray',
        alpha=0.6
    )



    # ✅ Omics legend
    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(handles=omics_patches, loc="center", ncol=len(omics_patches), frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_sum_contribution_plt(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os

    # --- Reduce 2048D to 64D using max-over-16 logic ---
    relevance_scores = extract_summary_features_np(relevance_scores)

    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',     # red
            'cna': '#1F77B4',    # blue
            'meth': '#2CA02C',   # green
            'ge': '#9467BD'      # purple
        }

    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # ---- 🔹 Sort nodes by cluster ----
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # ---- 🔹 Feature colors ----
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    fig = plt.figure(figsize=(18, 18))
    '''gs = fig.add_gridspec(nrows=10, ncols=1)
    ax = fig.add_subplot(gs[:-1, 0])       # heatmap
    ax_legend = fig.add_subplot(gs[-1, 0]) # legend

    fig = plt.figure(figsize=(18, 16))  # make taller and wider'''
    gs = fig.add_gridspec(nrows=10, ncols=20)  # more columns

    ax = fig.add_subplot(gs[:-1, :18])        # heatmap
    ax_bar = fig.add_subplot(gs[:-1, 18:])    # ✅ contribution bar
    ax_legend = fig.add_subplot(gs[-1, :])    # legend

    # ✅ Compute cluster boundaries and counts
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # ---- 🔹 Heatmap ----
    # ---- 🔹 Normalize scores per row ----
    row_min = sorted_scores.min(axis=1, keepdims=True)
    row_max = sorted_scores.max(axis=1, keepdims=True)
    sorted_scores = (sorted_scores - row_min) / (row_max - row_min + 1e-8)

    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={
            "label": "LRP Contribution",
            "shrink": 0.2,     # reduce height
            "aspect": 6,      # reduce width
            "pad": 0.01        # reduce space between heatmap and colorbar
        },
        ax=ax
    )

    # ✅ Compute sum of contributions per row
    row_contributions = sorted_scores.sum(axis=1)

    # ✅ Plot on the right
    ax_bar.barh(
        np.arange(len(row_contributions)),
        row_contributions,
        height=1.0,
        color='black'
    )
    ax_bar.invert_yaxis()  # align with heatmap
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_xlim(left=0)  # ensure bar is right-directed
    ax_bar.set_xlabel("")  # optional: label if desired
    ax_bar.spines[['top', 'right', 'bottom']].set_visible(False)

    # ---- 🔹 Cluster bar ----
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1, i), 1, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # ✅ Add cluster labels with counts on y-axis, near axis
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",  # just number of genes
            va='center', ha='right', fontsize=9, fontweight='bold'
        )

    # ---- 🔹 Colored x-axis labels ----
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=10)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    # ❌ REMOVE: x-label, y-label, title
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # ---- 🔹 Omics legend ----
    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(handles=omics_patches, loc="center", ncol=len(omics_patches), frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_number_of_genes_added(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os
    
    # --- Reduce 2048D to 64D using max-over-16 logic ---
    relevance_scores = extract_summary_features_np(relevance_scores)


    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',     # red
            'cna': '#1F77B4',    # blue
            'meth': '#2CA02C',   # green
            'ge': '#9467BD'      # purple
        }

    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # Construct pretty labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # ---- 🔹 Sort nodes by cluster ----
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # ---- 🔹 Feature colors ----
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(nrows=10, ncols=1)
    ax = fig.add_subplot(gs[:-1, 0])       # heatmap
    ax_legend = fig.add_subplot(gs[-1, 0]) # legend

    # ---- 🔹 Sort nodes by cluster ----
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # ✅ Compute cluster boundaries and counts
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # ---- 🔹 Heatmap ----
    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "LRP Contribution"},
        ax=ax
    )

    # ---- 🔹 Cluster bar ----
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1, i), 1, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # ✅ Add cluster labels with counts on y-axis
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -3.5, center_y, f"Cluster {cluster_id} ({count})",
            va='center', ha='right', fontsize=9, fontweight='bold'
        )

    # ---- 🔹 Colored x-axis labels ----
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=6)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("Cancer Type (Colored by Omics Type)")
    ax.set_ylabel("Nodes (Grouped by Cluster)")
    ax.set_title("Spectral Biclustering of LRP Relevance scores")

    # ---- 🔹 Omics legend ----
    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(handles=omics_patches, loc="center", ncol=len(omics_patches), frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_on_top_bar_x(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os
    
    # --- Reduce 2048D to 64D using max-over-16 logic ---
    relevance_scores = extract_summary_features_np(relevance_scores)


    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',     # red
            'cna': '#1F77B4',    # blue
            'meth': '#2CA02C',   # green
            'ge': '#9467BD'      # purple
        }

    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    # Construct pretty labels
    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # ---- 🔹 Sort nodes by cluster ----
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # ---- 🔹 Feature colors ----
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(nrows=10, ncols=1)
    ax = fig.add_subplot(gs[:-1, 0])       # heatmap
    ax_legend = fig.add_subplot(gs[-1, 0]) # legend

    # 🔹 Heatmap with horizontal colorbar on top
    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        ax=ax,
        cbar_kws={
            "orientation": "horizontal",
            "label": "LRP Contribution",
            "pad": 0.02,       # small gap between heatmap and colorbar
            "shrink": 0.4,     # shrink width of colorbar
            "aspect": 20       # makes it thinner
        }
    )

    # 🔹 Move colorbar label and ticks to top
    cbar = ax.collections[0].colorbar
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.tick_top()



    # ---- 🔹 Cluster bar ----
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1, i), 1, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # ---- 🔹 Colored x-axis labels ----
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=6)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("Cancer Type (Colored by Omics Type)")
    ax.set_ylabel("Nodes (Grouped by Cluster)")
    ax.set_title("Spectral Biclustering of LRP Relevance scores")

    # ---- 🔹 Omics legend ----
    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(handles=omics_patches, loc="center", ncol=len(omics_patches), frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_x(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None
):
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',   # red
            'cna': '#1F77B4',  # blue
            'meth': '#2CA02C', # green
            'ge': '#9467BD'    # purple
        }

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
    
    # Sort rows by cluster labels
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # Determine x-axis colors by omics
    feature_colors = []
    omics_labels = []
    cancer_labels = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                omics_labels.append(omics)
                break
        else:
            feature_colors.append("#AAAAAA")
            omics_labels.append("")

        # Infer cancer type by modulo
        cancer_idx = (i % len(cancer_types))
        cancer_labels.append(cancer_types[cancer_idx])

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(nrows=12, ncols=1)
    ax = fig.add_subplot(gs[:-2, 0])         # heatmap
    ax_legend = fig.add_subplot(gs[-2, 0])   # omics legend
    ax_labels = fig.add_subplot(gs[-1, 0])   # omics label row

    # Heatmap
    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "LRP Contribution"},
        ax=ax
    )

    # Cluster bar on Y-axis
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1, i), 1, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # Omics type colored labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("Features (Colored by Omics Type)")
    ax.set_ylabel("Nodes (Grouped by Cluster)")
    ax.set_title("Spectral Biclustering of LRP Relevance scores")

    # Omics type mini-bars under each feature
    # Add bottom color label bar for omics
    ax_labels = fig.add_subplot(gs[-2, 0])
    label_bar = np.array([to_rgb(c) for c in feature_colors]).reshape(1, -1, 3)
    ax_labels.imshow(label_bar, aspect='auto')
    ax_labels.axis("off")


    '''ax_labels.imshow(
        np.array([feature_colors]),
        aspect='auto',
        extent=[0, len(feature_colors), 0, 1]
    )
    ax_labels.axis('off')'''

    # Legend
    ax_legend.axis("off")
    omics_patches = [Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()]
    ax_legend.legend(handles=omics_patches, loc="center", ncol=len(omics_patches), frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_bottom_legend_pas(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None
):
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        }

    # Sort nodes by cluster
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # Determine x-axis label colors
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(nrows=10, ncols=1)
    ax = fig.add_subplot(gs[:-1, 0])  # heatmap area
    ax_legend = fig.add_subplot(gs[-1, 0])  # legend bar below

    # Plot heatmap
    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "LRP Contribution"},
        ax=ax
    )

    # Add cluster color bar on Y-axis
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1, i), 1, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # Add x-axis labels with colored omics type
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("Features (Colored by Omics Type)")
    ax.set_ylabel("Nodes (Grouped by Cluster)")
    ax.set_title("Spectral Biclustering of LRP Relevance scores")

    # Omics legend bar
    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(handles=omics_patches, loc="center", ncol=len(omics_patches), frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_x_label_only_red_text_pas(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None
):
    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',
            'cna': '#1F77B4',
            'meth': '#2CA02C',
            'ge': '#9467BD'
        }

    # Sort by cluster
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # Prepare feature label colors
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")  # fallback gray

    fig, ax = plt.subplots(figsize=(16, 10))

    # Main heatmap
    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "LRP Contribution"},
        ax=ax
    )

    # Cluster bar on the left Y-axis
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1, i), 1, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # Colored feature names on X-axis
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("Features (Colored by Omics Type)")
    ax.set_ylabel("Nodes (Grouped by Cluster)")
    ax.set_title("Spectral Biclustering of LRP Relevance scores")

    # Legends
    cluster_patches = [Patch(color=color, label=f"Cluster {i}") for i, color in CLUSTER_COLORS.items()]
    omics_patches = [Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()]
    ax.legend(handles=cluster_patches + omics_patches,
              title="Legend", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_lrp_heatmap_top_genes(
    relevance_scores: torch.Tensor,
    feature_names: list,
    top_k: int,
    save_path: str,
    n_row_clusters: int = 12,
    n_col_clusters: int = 4
):
    omics_types = ['mf', 'cna', 'meth', 'ge']
    omics_colors = {
        'mf': '#b22222',     # dark red
        'cna': '#9370DB',    # purple
        'meth': '#00008B',   # dark blue
        'ge': '#228B22'      # dark green
    }

    # ---- 🔹 Remove "degree" feature ----
    if "degree" in feature_names:
        degree_idx = feature_names.index("degree")
        feature_names = feature_names[:degree_idx] + feature_names[degree_idx + 1:]
        relevance_scores = relevance_scores[:, :degree_idx].clone()

    # ---- 🔹 Select top-K ----
    top_k = min(top_k, relevance_scores.shape[0])
    relevance_scores = relevance_scores[:top_k]

    # ---- 🔹 Spectral Biclustering ----
    model = SpectralBiclustering(n_clusters=(n_row_clusters, n_col_clusters), random_state=0)
    model.fit(relevance_scores)
    row_order = np.argsort(model.row_labels_)
    col_order = np.argsort(model.column_labels_)
    reordered_scores = relevance_scores[row_order][:, col_order].cpu().numpy()
    reordered_feature_names = [feature_names[i] for i in col_order]
    reordered_cluster_labels = model.row_labels_[row_order]

    # ---- 🔹 Omics color mapping ----
    num_features = len(reordered_feature_names)
    features_per_omic = num_features // len(omics_types)
    feature_omics_map = []
    for omic in omics_types:
        feature_omics_map.extend([omic] * features_per_omic)
    feature_colors = [omics_colors[o] for o in feature_omics_map]

    # ---- 🔹 Plot ----
    fig, ax = plt.subplots(figsize=(max(12, num_features * 0.25), 10))
    sns.heatmap(
        reordered_scores,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar_kws={'label': 'LRP Relevance'},
        linewidths=0.1
    )
    ax.set_title(f"Spectral Biclustering on Top {top_k} Genes", fontsize=16)

    # ---- 🔹 Add omics-colored x labels ----
    for idx, label in enumerate(reordered_feature_names):
        color = feature_colors[idx]
        ax.text(idx + 0.5, -top_k * 0.01, label, rotation=90,
                ha='center', va='top', fontsize=8, color=color, clip_on=False)

    # ---- 🔹 Add colored y-axis cluster labels ----
    for i, cluster_id in enumerate(reordered_cluster_labels):
        cluster_color = to_rgba(CLUSTER_COLORS.get(cluster_id, "#555555"))
        ax.text(-1.2, i + 0.5, f"C{cluster_id}", va='center', ha='right',
                fontsize=8, color=cluster_color)

    # ---- 🔹 Red bounding box ----
    rect = Rectangle((0, 0), reordered_scores.shape[1], reordered_scores.shape[0],
                     linewidth=1.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # ---- 🔹 Omics legend ----
    legend_elements = [
        Patch(facecolor=omics_colors['mf'], label='Mutation (mf)'),
        Patch(facecolor=omics_colors['cna'], label='CNA (cna)'),
        Patch(facecolor=omics_colors['meth'], label='Methylation (meth)'),
        Patch(facecolor=omics_colors['ge'], label='Expression (ge)')
    ]
    plt.subplots_adjust(bottom=0.3)
    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.35),
        ncol=4,
        frameon=False,
        fontsize=10
    )

    # ---- 🔹 Save ----
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Spectral biclustering LRP heatmap saved to:\n{save_path}")

def plot_spectral_biclustering_lrp_heatmap_top_genes(
    relevance_scores: torch.Tensor,
    feature_names: list,
    top_k: int,
    save_path: str,
    n_row_clusters: int = 12,
    n_col_clusters: int = 4
):
    # ---- 🔹 Omics Setup ----
    omics_types = ['mf', 'cna', 'meth', 'ge']
    omics_colors = {
        'mf': '#b22222',     # dark red
        'cna': '#9370DB',    # purple
        'meth': '#00008B',   # dark blue
        'ge': '#228B22'      # dark green
    }

    # ---- 🔹 Remove "degree" feature ----
    if "degree" in feature_names:
        degree_idx = feature_names.index("degree")
        feature_names = feature_names[:degree_idx] + feature_names[degree_idx + 1:]
        relevance_scores = relevance_scores[:, :degree_idx].clone()

    # ---- 🔹 Filter top-K genes ----
    top_k = min(top_k, relevance_scores.shape[0])
    relevance_scores = relevance_scores[:top_k]

    # ---- 🔹 Spectral Biclustering ----
    model = SpectralBiclustering(n_clusters=(n_row_clusters, n_col_clusters), random_state=0)
    model.fit(relevance_scores)
    row_order = np.argsort(model.row_labels_)
    col_order = np.argsort(model.column_labels_)
    reordered_scores = relevance_scores[row_order][:, col_order].cpu().numpy()
    reordered_feature_names = [feature_names[i] for i in col_order]

    # ---- 🔹 Assign omics colors ----
    num_features = len(reordered_feature_names)
    features_per_omic = num_features // len(omics_types)
    feature_omics_map = []
    for omic in omics_types:
        feature_omics_map.extend([omic] * features_per_omic)
    feature_colors = [omics_colors[o] for o in feature_omics_map]

    # ---- 🔹 Plot ----
    fig, ax = plt.subplots(figsize=(max(12, num_features * 0.25), 10))
    sns.heatmap(
        reordered_scores,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar_kws={'label': 'LRP Relevance'},
        linewidths=0.1
    )

    ax.set_title(f"Spectral Biclustering on Top {top_k} Genes", fontsize=16)

    # ---- 🔹 Omics-colored x labels ----
    for idx, label in enumerate(reordered_feature_names):
        color = feature_colors[idx]
        ax.text(idx + 0.5, -top_k * 0.01, label, rotation=90,
                ha='center', va='top', fontsize=8, color=color, clip_on=False)

    # ---- 🔹 Highlight region (optional) ----
    # Draw red rectangle around entire heatmap
    rect = Rectangle((0, 0), reordered_scores.shape[1], reordered_scores.shape[0],
                     linewidth=1.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # ---- 🔹 Omics Legend ----
    legend_elements = [
        Patch(facecolor=omics_colors['mf'], label='Mutation (mf)'),
        Patch(facecolor=omics_colors['cna'], label='CNA (cna)'),
        Patch(facecolor=omics_colors['meth'], label='Methylation (meth)'),
        Patch(facecolor=omics_colors['ge'], label='Expression (ge)')
    ]
    plt.subplots_adjust(bottom=0.3)
    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.35),
        ncol=4,
        frameon=False,
        fontsize=10
    )

    # ---- 🔹 Save ----
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Spectral biclustering LRP heatmap saved to:\n{save_path}")

def plot_spectral_biclustering_lrp_heatmap_top_genes(
    relevance_scores: torch.Tensor,
    feature_names: list,
    top_k: int,
    save_path: str,
    n_row_clusters: int = 12,
    n_col_clusters: int = 4
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch, Rectangle
    import os
    import numpy as np
    from sklearn.cluster import SpectralBiclustering

    # ---- 🔹 Omics Setup ----
    omics_types = ['mf', 'cna', 'ge', 'meth']
    omics_colors = {
        'mf': '#b22222',     # dark red
        'cna': '#9370DB',    # purple
        'meth': '#00008B',   # dark blue
        'ge': '#228B22'      # dark green
    }
    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]

    # ---- 🔹 Remove "degree" feature ----
    if "degree" in feature_names:
        degree_idx = feature_names.index("degree")
        feature_names = feature_names[:degree_idx] + feature_names[degree_idx + 1:]
        relevance_scores = relevance_scores[:, :degree_idx].clone()

    # ---- 🔹 Filter top-K genes ----
    top_k = min(top_k, relevance_scores.shape[0])
    relevance_scores = relevance_scores[:top_k]

    # ---- 🔹 Spectral Biclustering ----
    model = SpectralBiclustering(n_clusters=(n_row_clusters, n_col_clusters), random_state=0)
    model.fit(relevance_scores)
    row_order = np.argsort(model.row_labels_)
    col_order = np.argsort(model.column_labels_)
    reordered_scores = relevance_scores[row_order][:, col_order].cpu().numpy()

    # ---- 🔹 Construct new feature names by omics + cancer label ----
    feature_labels = []
    for omics in omics_types:
        for cancer in cancer_names:
            feature_labels.append(cancer)

    reordered_feature_labels = [feature_labels[i] for i in col_order]
    feature_omics_map = [omics for omics in omics_types for _ in range(len(cancer_names))]
    reordered_feature_omics = [feature_omics_map[i] for i in col_order]
    feature_colors = [omics_colors[omic] for omic in reordered_feature_omics]

    # ---- 🔹 Plot ----
    fig, ax = plt.subplots(figsize=(max(12, len(reordered_feature_labels) * 0.25), 10))
    sns.heatmap(
        reordered_scores,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar_kws={'label': 'LRP Relevance'},
        linewidths=0.1
    )

    ax.set_title(f"Spectral Biclustering on Top {top_k} Genes", fontsize=16)

    # ---- 🔹 Omics-colored x labels ----
    for idx, label in enumerate(reordered_feature_labels):
        color = feature_colors[idx]
        ax.text(idx + 0.5, -top_k * 0.01, label, rotation=90,
                ha='center', va='top', fontsize=8, color=color, clip_on=False)

    # ---- 🔹 Highlight region (optional) ----
    rect = Rectangle((0, 0), reordered_scores.shape[1], reordered_scores.shape[0],
                     linewidth=1.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # ---- 🔹 Omics Legend ----
    legend_elements = [
        Patch(facecolor=omics_colors['mf'], label='Mutation (mf)'),
        Patch(facecolor=omics_colors['cna'], label='CNA (cna)'),
        Patch(facecolor=omics_colors['meth'], label='Methylation (meth)'),
        Patch(facecolor=omics_colors['ge'], label='Expression (ge)')
    ]
    plt.subplots_adjust(bottom=0.3)
    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.35),
        ncol=4,
        frameon=False,
        fontsize=10
    )

    # ---- 🔹 Save ----
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Spectral biclustering LRP heatmap saved to:\n{save_path}")

def plot_spectral_biclustering_heatmap_pas_right_track(
    relevance_scores, cluster_labels, feature_names, omics_splits, output_path
):
    import numpy as np

    # Sort by cluster
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # Omics bar color setup
    omics_colors = {
        'mf': '#D62728',
        'cna': '#1F77B4',
        'meth': '#2CA02C',
        'ge': '#9467BD'
    }

    omics_bar = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                omics_bar.append(omics_colors[omics])
                break
        else:
            omics_bar.append('#CCCCCC')  # fallback

    # Plot
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)

    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        yticklabels=False,
        xticklabels=False,
        cbar_kws={"label": "LRP Contribution"},
        ax=ax
    )

    # Cluster label colors
    for i, idx in enumerate(sorted_indices):
        cluster = sorted_clusters[i]
        ax.add_patch(plt.Rectangle(
            (0, i), len(feature_names), 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF'), alpha=0.3)
        ))

    # Omics x-axis label bar
    for i, color in enumerate(omics_bar):
        ax.add_patch(plt.Rectangle(
            (i, 0), 1, sorted_scores.shape[0],
            linewidth=0,
            facecolor=to_rgba(color, alpha=0.1)
        ))

    ax.set_xlabel("Features (Omics)")
    ax.set_ylabel("Nodes (Grouped by Spectral Clusters)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_cluster_heatmap_colored_top_nodes(
    relevance_scores,
    feature_names,
    save_path,
    omics_splits,
    omics_colors,
    n_top_nodes=1000,
    n_feature_clusters=6,
    n_labeled_features=50
):
    """
    Clusters and visualizes features based on LRP scores of top nodes.
    Only labels top N most relevant features, colored by omics type.
    """
    if isinstance(relevance_scores, torch.Tensor):
        relevance_scores = relevance_scores.detach().cpu().numpy()

    # Get top N nodes
    node_relevance_sum = relevance_scores.sum(axis=1)
    top_indices = np.argsort(node_relevance_sum)[-n_top_nodes:]
    top_node_scores = relevance_scores[top_indices]  # [n_top_nodes, n_features]

    # Transpose: features x top_nodes
    feature_data = top_node_scores.T

    # Normalize features
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)

    # Cluster features
    model = SpectralBiclustering(n_clusters=n_feature_clusters, method='log', random_state=42)
    model.fit(feature_data_scaled)

    # Sort features by cluster
    sorted_indices = np.argsort(model.row_labels_)
    sorted_data = feature_data[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    # Omics coloring
    feature_omics_colors = []
    for idx in sorted_indices:
        assigned = False
        for omics_type, (start, end) in omics_splits.items():
            if start <= idx <= end:
                feature_omics_colors.append(omics_colors[omics_type])
                assigned = True
                break
        if not assigned:
            feature_omics_colors.append("lightgray")

    # Compute average relevance per feature
    avg_feature_relevance = top_node_scores.mean(axis=0)
    top_feature_indices = np.argsort(avg_feature_relevance)[-n_labeled_features:]
    top_feature_set = set(top_feature_indices)

    # Label only top N most relevant features
    yticklabels = []
    for original_idx in sorted_indices:
        if original_idx in top_feature_set:
            yticklabels.append(feature_names[original_idx])
        else:
            yticklabels.append("")

    # Plot
    plt.figure(figsize=(20, 8))
    sns.heatmap(
        sorted_data,
        cmap="coolwarm",
        yticklabels=yticklabels,
        xticklabels=False,
        cbar_kws={"label": "LRP relevance"},
        linewidths=0.05,
        linecolor='lightgray'
    )

    for ticklabel, color in zip(plt.gca().get_yticklabels(), feature_omics_colors):
        ticklabel.set_color(color)

    # Legend
    handles = [Patch(color=color, label=omics) for omics, color in omics_colors.items()]
    plt.legend(handles=handles, title="Omics Types", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.xlabel("Top 1000 Nodes")
    plt.ylabel("Features (Clustered & Colored)")
    plt.title("Feature Clustering from Top 1000 Nodes (LRP)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved top-node-based feature clustering heatmap to {save_path}")

def plot_feature_cluster_heatmap(relevance_scores, feature_names, save_path, n_feature_clusters=6):
    """
    Clusters the columns of the LRP relevance score matrix and plots feature-wise clustering heatmap.

    Args:
        relevance_scores (torch.Tensor or np.ndarray): LRP relevance scores of shape (n_nodes, n_features).
        feature_names (list): Names of each feature column.
        save_path (str): Output path for the heatmap.
        n_feature_clusters (int): Number of feature clusters.
    """
    if isinstance(relevance_scores, torch.Tensor):
        relevance_scores = relevance_scores.detach().cpu().numpy()

    # Transpose: [features x nodes]
    relevance_transposed = relevance_scores.T

    # Standardize across nodes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(relevance_transposed)

    # Apply spectral biclustering to columns (features)
    model = SpectralBiclustering(n_clusters=n_feature_clusters, method='log', random_state=42)
    model.fit(X_scaled)

    # Reorder features by assigned cluster
    sorted_indices = np.argsort(model.row_labels_)
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_scores = relevance_transposed[sorted_indices]

    # Plot
    plt.figure(figsize=(20, 8))
    sns.heatmap(sorted_scores, cmap="vlag", center=0, yticklabels=sorted_feature_names)
    plt.title("Spectral Biclustering of LRP Feature Contributions", fontsize=16)
    plt.xlabel("Nodes")
    plt.ylabel("Features (clustered)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved feature clustering heatmap to {save_path}")

def plot_feature_cluster_heatmap_colored(
    relevance_scores,
    feature_names,
    save_path,
    omics_splits,
    omics_colors,
    n_feature_clusters=6
):
    """
    Spectrally clusters features (columns) of the LRP score matrix, and colors rows by omics group.

    Args:
        relevance_scores: [n_nodes, n_features] Tensor or array of LRP scores.
        feature_names: List of feature names.
        save_path: Output path to save heatmap.
        omics_splits: Dict of {omics_type: (start_idx, end_idx)}.
        omics_colors: Dict of {omics_type: color}.
        n_feature_clusters: Number of feature clusters.
    """
    if isinstance(relevance_scores, torch.Tensor):
        relevance_scores = relevance_scores.detach().cpu().numpy()

    # Transpose so features are rows
    feature_data = relevance_scores.T

    # Normalize
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)

    # Cluster features
    model = SpectralBiclustering(n_clusters=n_feature_clusters, method='log', random_state=42)
    model.fit(feature_data_scaled)

    # Sort features by cluster
    sorted_indices = np.argsort(model.row_labels_)
    sorted_data = feature_data[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    # Generate color annotation for omics groups
    feature_omics = []
    for idx in sorted_indices:
        assigned = False
        for omics_type, (start, end) in omics_splits.items():
            if start <= idx <= end:
                feature_omics.append(omics_colors[omics_type])
                assigned = True
                break
        if not assigned:
            feature_omics.append("lightgray")

    # Plot heatmap
    plt.figure(figsize=(20, 8))
    sns.heatmap(
        sorted_data,
        cmap="coolwarm",
        yticklabels=sorted_feature_names,
        xticklabels=False,
        cbar_kws={"label": "LRP relevance"},
        linewidths=0.05,
        linecolor='lightgray'
    )

    for ticklabel, color in zip(plt.gca().get_yticklabels(), feature_omics):
        ticklabel.set_color(color)

    # Legend
    handles = [Patch(color=color, label=omics) for omics, color in omics_colors.items()]
    plt.legend(handles=handles, title="Omics Types", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.xlabel("Nodes")
    plt.ylabel("Features (Clustered & Colored)")
    plt.title("Column-Level Spectral Clustering of Features (LRP)")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🎨 Feature-clustered heatmap with omics colors saved to {save_path}")

def plot_lrp_cluster_heatmap_color_ori(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    feature_groups: dict = None,
    filename_prefix: str = "lrp_heatmap"
):
    """
    Plot a heatmap of LRP scores averaged over nodes in each cluster, styled similar to reference image.

    Args:
        all_cluster_scores: [num_nodes, num_features] tensor of LRP scores.
        feature_names: List of feature names.
        cluster_tensor: Tensor assigning each node to a cluster.
        output_dir: Path to directory to save the heatmap.
        model_type: Model type string (e.g., 'ACGNN').
        net_type: Graph type string.
        score_threshold: Score threshold for filtering (used in filename).
        num_epochs: Number of training epochs (used in filename).
        feature_groups: Optional dict of {feature_name: group_name}.
        filename_prefix: Prefix for the plot file name.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy for plotting
    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()

    # Assuming scores_np has shape (7528, 129)
    # Feature names should match this number (129), not 2177
    feature_names = ['feature_{}'.format(i) for i in range(129)]  # This should match the columns of scores_np

    df = pd.DataFrame(scores_np, columns=feature_names)

    df['cluster'] = cluster_ids

    # Group by cluster and compute mean scores
    cluster_means = df.groupby('cluster').mean()
    cluster_means = cluster_means.sort_index()

    # Feature group coloring
    if feature_groups:
        group_colors = {
            'Mutation': '#a50026',
            'CNA': '#b35806',
            'DNA methylation': '#313695',
            'Gene expression': '#1b7837'
        }
        col_colors = [
            group_colors.get(feature_groups.get(f, ''), 'lightgray') for f in cluster_means.columns
        ]
    else:
        col_colors = None

    # Plotting setup
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(1, 10, width_ratios=[0.3] + [9]*9, wspace=0.05)

    # Cluster side color bar
    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(
        np.expand_dims(cluster_means.index, 1),
        cmap='tab20',
        cbar=False,
        ax=ax0,
        yticklabels=False,
        xticklabels=False
    )
    ax0.set_ylabel('Cluster')
    ax0.set_xticks([])

    # Main heatmap
    ax1 = fig.add_subplot(gs[1:])
    sns.heatmap(
        cluster_means,
        cmap='Greys',
        cbar_kws={"label": "Total feature contribution"},
        yticklabels=True,
        xticklabels=True,
        square=False,
        ax=ax1,
        linewidths=0.1,
        linecolor='lightgrey',
        ##col_colors=col_colors  # Add color by feature groups
    )
    ax1.set_xlabel("Features", fontsize=14)
    ax1.set_ylabel("Cluster", fontsize=14)
    ax1.set_title("Average LRP Feature Contribution per Cluster", fontsize=16)
    ax1.tick_params(axis='x', labelrotation=90)

    # Optional legend for feature group coloring
    if feature_groups:
        handles = [Patch(color=c, label=label) for label, c in group_colors.items()]
        ax1.legend(handles=handles, title="Feature groups", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Save plot
    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🔥 Saved LRP heatmap to: {plot_path}")

def plot_sorted_multiomics_heatmap(output_path, relevance_scores, cluster_tensor, scores, omics_splits, omics_colors):
    """
    Plot multi-omics heatmap, sorted by model scores within each cluster.

    Parameters:
    - output_path: where to save the figure.
    - relevance_scores: [N, F] numpy array of relevance scores (e.g., from LRP).
    - cluster_tensor: torch tensor of cluster IDs per node (shape [N]).
    - scores: torch tensor of predicted scores per node (shape [N]).
    - omics_splits: list of omics names, e.g., ['mf', 'cna', 'meth', 'ge'].
    - omics_colors: list of RGB tuples or color hex codes for each omics type.
    """
    scores = scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()
    num_clusters = cluster_ids.max() + 1
    n_omics = len(omics_splits)

    # Assume each omics feature chunk is equally split
    omics_dim = relevance_scores.shape[1] // n_omics

    cluster_gene_rows = []

    for cluster_id in range(num_clusters):
        cluster_mask = cluster_ids == cluster_id
        cluster_scores = scores[cluster_mask]
        cluster_relevance = relevance_scores[cluster_mask]

        if cluster_relevance.shape[0] == 0:
            continue

        # Sort indices by descending prediction score
        sorted_indices = np.argsort(-cluster_scores)
        cluster_relevance = cluster_relevance[sorted_indices]

        # Split each row into omics segments
        for row in cluster_relevance:
            row_segments = [row[i * omics_dim:(i + 1) * omics_dim] for i in range(n_omics)]
            cluster_gene_rows.append(row_segments)

    # Normalize all patches for color mapping
    all_values = np.concatenate([np.concatenate(patch) for patch in cluster_gene_rows])
    vmin, vmax = all_values.min(), all_values.max()

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, row_segments in enumerate(cluster_gene_rows):
        for j, segment in enumerate(row_segments):
            segment = segment.reshape(1, -1)
            ax.imshow(segment, extent=[j, j + 1, i, i + 1], aspect='auto', cmap='YlGnBu', vmin=vmin, vmax=vmax)

    # Add vertical omics boundaries
    for j in range(1, n_omics):
        ax.axvline(x=j, color='white', linewidth=1)

    # Omics labels
    ax.set_xticks([i + 0.5 for i in range(n_omics)])
    ax.set_xticklabels(omics_splits, rotation=90, ha='right', fontsize=12)
    ax.set_yticks([])
    ax.set_title("Multi-omics Feature Importance Sorted by Prediction Score", fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_real_clustered_multiomics_heatmap(
    relevance_scores: torch.Tensor,
    cluster_tensor: torch.Tensor,
    feature_names: list,
    omics_splits: dict,
    save_path: str,
    cmap='Greys',
    figsize=(14, 10)
):
    """
    Plots a clustered heatmap with cluster-wise average LRP scores per omics type.

    Parameters:
    - relevance_scores (torch.Tensor): [N, F] tensor of LRP scores per node.
    - cluster_tensor (torch.Tensor): [N] cluster ID for each node.
    - feature_names (list): List of all feature names (e.g., ["f1", ..., "degree"]).
    - omics_splits (dict): Dictionary mapping omics type to index ranges, e.g., {"mf": (0, 64), "cna": (64, 128), ...}
    - save_path (str): Path to save the plot.
    - cmap (str): Color map for the heatmap.
    - figsize (tuple): Size of the figure.
    """

    num_clusters = cluster_tensor.max().item() + 1
    num_features = relevance_scores.shape[1]

    # Create matrix [num_clusters x num_features]
    lrp_matrix = np.zeros((num_clusters, num_features))
    cluster_sizes = []

    for c in range(num_clusters):
        indices = torch.nonzero(cluster_tensor == c, as_tuple=True)[0]
        cluster_sizes.append(len(indices))
        if len(indices) == 0:
            continue
        cluster_relevance = relevance_scores[indices]
        avg_relevance = cluster_relevance.mean(dim=0).cpu().numpy()
        lrp_matrix[c] = avg_relevance

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        lrp_matrix,
        cmap=cmap,
        xticklabels=False,
        yticklabels=[f"C{c} ({sz})" for c, sz in enumerate(cluster_sizes)],
        cbar_kws={"label": "Avg. feature contribution"},
        ax=ax
    )

    # Draw vertical lines to split omics types
    x_start = 0
    for omics_type, (start, end) in omics_splits.items():
        center = (start + end) / 2
        ax.text(center, -0.5, omics_type, ha='center', va='bottom', fontsize=10, rotation=90)
        ax.axvline(end, color='red', linestyle='--', linewidth=0.5)

    ax.set_ylabel("Clusters")
    ax.set_xlabel("Omics Features")
    ax.set_title("Cluster-wise Multi-Omics LRP Heatmap")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved heatmap to: {save_path}")
    plt.close()

def generate_cluster_lrp_omics_heatmap(
    relevance_scores: torch.Tensor,
    cluster_tensor: torch.Tensor,
    output_path: str,
    omics_types: list = ['mf', 'cna', 'meth', 'ge'],
    feature_block_size: int = 256,
    n_clusters: int = 12
):
    """
    Generates and saves an omics-type heatmap using averaged LRP scores per cluster.

    Args:
        relevance_scores (torch.Tensor): Tensor of shape [num_nodes, num_features]
        cluster_tensor (torch.Tensor): Tensor of shape [num_nodes], with cluster indices
        output_path (str): Path to save the heatmap image
        omics_types (list): List of omics types in order
        feature_block_size (int): Number of features per omics type
        n_clusters (int): Number of clusters
    """
    print("🧠 Generating LRP-based cluster omics heatmap...")
    
    feature_dim = relevance_scores.shape[1]
    lrp_matrix = np.zeros((n_clusters, feature_dim))
    cluster_sizes = np.zeros(n_clusters, dtype=int)

    for c in range(n_clusters):
        node_indices = torch.nonzero(cluster_tensor == c).squeeze()
        if node_indices.numel() == 0:
            print(f"⚠️ No nodes in cluster {c}")
            continue
        if node_indices.ndim == 0:
            node_indices = node_indices.unsqueeze(0)

        cluster_scores = relevance_scores[node_indices]
        lrp_matrix[c] = cluster_scores.mean(dim=0).cpu().numpy()
        cluster_sizes[c] = node_indices.numel()

    omics_splits = [feature_block_size] * len(omics_types)

    # Now call your plotting function
    plot_cluster_omics_heatmap_numpy_omics_splits(
        lrp_matrix=lrp_matrix,
        cluster_sizes=cluster_sizes,
        omics_types=omics_types,
        output_path=output_path,
        omics_splits=omics_splits
    )

    print(f"✅ Saved LRP cluster omics heatmap to: {output_path}")

def plot_cluster_omics_heatmap_numpy_omics_splits(
    lrp_matrix,
    cluster_sizes,
    omics_types,
    output_path,
    omics_splits=None
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 8))
    sns.heatmap(lrp_matrix, cmap='viridis', cbar_kws={'label': 'LRP Score'})

    if omics_splits:
        split_positions = np.cumsum(omics_splits)[:-1]
        for x in split_positions:
            plt.axvline(x=x, color='white', linestyle='--', linewidth=1)

    plt.yticks(ticks=np.arange(len(cluster_sizes)) + 0.5, labels=[f'C{i}' for i in range(len(cluster_sizes))])
    plt.xticks([])
    plt.title("Cluster-wise Omics LRP Heatmap")
    plt.xlabel("Feature Index")
    plt.ylabel("Cluster")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_cluster_omics_heatmap_numpy(
    lrp_matrix,
    cluster_sizes,
    omics_types,
    output_path,
    cluster_colors=None
):
    n_clusters, n_features = lrp_matrix.shape

    # Define omics color mapping
    omics_colors = {
        'mf': '#b22222',     # dark red
        'cna': '#9370DB',    # purple
        'meth': '#00008B',   # dark blue
        'ge': '#228B22'      # dark green
    }

    # Assign feature → omics type
    features_per_omic = n_features // len(omics_types)
    feature_omics_map = []
    for omic in omics_types:
        feature_omics_map.extend([omic] * features_per_omic)
    feature_colors = [omics_colors[o] for o in feature_omics_map]

    # Start plotting
    fig, ax = plt.subplots(figsize=(16, 8))

    # Heatmap
    im = ax.imshow(lrp_matrix, cmap='Greys', aspect='auto')

    # Overlay cluster-colored rectangles
    for i, size in enumerate(cluster_sizes):
        cluster_color = cluster_colors[i] if cluster_colors else f"C{i % 10}"
        ax.add_patch(plt.Rectangle((0, i - 0.5), 1, 1, color=cluster_color, alpha=0.8))

    # Omics-colored x-axis labels
    for idx, omic in enumerate(feature_omics_map):
        ax.text(idx, n_clusters + 0.3, omic, rotation=90, fontsize=7,
                ha='center', va='bottom', color=omics_colors[omic], clip_on=False)

    ax.set_title("Cluster-wise Omics Heatmap", fontsize=16)
    ax.set_xlabel("Omics Features", fontsize=12)
    ax.set_ylabel("Clusters", fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([])

    legend_elements = [
        Patch(facecolor=omics_colors['mf'], label='Mutation (mf)'),
        Patch(facecolor=omics_colors['cna'], label='CNA (cna)'),
        Patch(facecolor=omics_colors['meth'], label='Methylation (meth)'),
        Patch(facecolor=omics_colors['ge'], label='Expression (ge)')
    ]
    plt.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved NumPy-only cluster-omics heatmap to: {output_path}")

def plot_cluster_omics_heatmap_np(relevance_scores, cluster_labels, feature_names, omics_types, output_path, n_clusters=12):
    omics_colors = {
        'mf': '#b22222',     # Mutation - dark red
        'cna': '#9370DB',    # CNA - purple
        'meth': '#00008B',   # Methylation - dark blue
        'ge': '#228B22'      # Gene expression - dark green
    }

    # 🔹 Remove "degree" feature if present
    if "degree" in feature_names:
        degree_idx = feature_names.index("degree")
        feature_names = feature_names[:degree_idx] + feature_names[degree_idx + 1:]
        relevance_scores = np.delete(relevance_scores, degree_idx, axis=1)

    # 🔹 Assign omics colors
    num_features = len(feature_names)
    features_per_omic = num_features // len(omics_types)
    feature_omics_map = []
    for omic in omics_types:
        feature_omics_map.extend([omic] * features_per_omic)
    feature_colors = [omics_colors[o] for o in feature_omics_map]

    # 🔹 Compute average relevance scores per cluster
    clusterwise_lrp = []
    cluster_sizes = []
    for c in range(n_clusters):
        indices = np.where(cluster_labels == c)[0]
        cluster_sizes.append(len(indices))
        if len(indices) == 0:
            clusterwise_lrp.append(np.zeros(num_features))
        else:
            cluster_scores = relevance_scores[indices]
            cluster_mean = cluster_scores.mean(axis=0)
            clusterwise_lrp.append(cluster_mean)

    lrp_matrix = np.stack(clusterwise_lrp)

    # 🔹 Plot heatmap
    fig, ax = plt.subplots(figsize=(max(12, num_features * 0.3), 8))
    sns.heatmap(
        lrp_matrix,
        cmap="Greys",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'LRP Relevance'},
        linewidths=0.1,
        ax=ax
    )

    ax.set_title("Cluster-wise Omics Heatmap", fontsize=16)
    ax.set_ylabel("Clusters", fontsize=12, labelpad=20)

    # 🔹 Omics-colored x-axis feature labels
    for idx, label in enumerate(feature_names):
        color = feature_colors[idx]
        ax.text(idx + 0.5, n_clusters + 0.2, label, rotation=90,
                ha='center', va='bottom', fontsize=8, color=color, clip_on=False)

    # 🔹 Omics legend
    legend_elements = [
        Patch(facecolor=omics_colors['mf'], label='Mutation (mf)'),
        Patch(facecolor=omics_colors['cna'], label='CNA (cna)'),
        Patch(facecolor=omics_colors['meth'], label='Methylation (meth)'),
        Patch(facecolor=omics_colors['ge'], label='Expression (ge)')
    ]
    plt.subplots_adjust(bottom=0.3)
    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=4,
        frameon=False,
        fontsize=10
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Cluster-wise Omics heatmap saved to:\n{output_path}")

def plot_cluster_omics_heatmap(relevance_scores, cluster_tensor, feature_names, omics_types, output_path, n_clusters=12):
    omics_colors = {
        'mf': '#b22222',     # dark red
        'cna': '#9370DB',    # purple
        'meth': '#00008B',   # dark blue
        'ge': '#228B22'      # dark green
    }

    # ---- 🔹 Remove "degree" feature ----
    if "degree" in feature_names:
        degree_idx = feature_names.index("degree")
        feature_names = feature_names[:degree_idx] + feature_names[degree_idx + 1:]
        relevance_scores = relevance_scores[:, :degree_idx].clone()
    
    # ---- 🔹 Assign omics colors ----
    num_features = len(feature_names)
    features_per_omic = num_features // len(omics_types)
    feature_omics_map = []
    for omic in omics_types:
        feature_omics_map.extend([omic] * features_per_omic)

    feature_colors = [omics_colors[o] for o in feature_omics_map]

    # ---- 🔹 Compute cluster-wise average LRP ----
    clusterwise_lrp = []
    cluster_sizes = []  # To store the number of genes in each cluster
    for c in range(n_clusters):
        node_indices = torch.nonzero(cluster_tensor == c).squeeze()
        cluster_sizes.append(node_indices.numel())
        if node_indices.numel() == 0:
            clusterwise_lrp.append(torch.zeros(len(feature_names)))
        else:
            cluster_scores = relevance_scores[node_indices]
            cluster_mean = cluster_scores.mean(dim=0)
            clusterwise_lrp.append(cluster_mean)

    lrp_matrix = torch.stack(clusterwise_lrp).cpu().numpy()
    cluster_sizes = np.array(cluster_sizes)

    # ---- 🔹 Plot ----
    fig, ax = plt.subplots(figsize=(max(12, len(feature_names) * 0.3), 8))

    # Use sns heatmap to plot cluster-wise LRP with varying row heights
    sns.heatmap(
        lrp_matrix,
        cmap="Greys",
        xticklabels=False,
        yticklabels=False,  # No cluster names on the y-axis
        cbar_kws={'label': 'LRP Relevance'},
        linewidths=0.1,
        ax=ax
    )

    # ---- 🔹 Plot cluster-specific color bars ----
    for i, size in enumerate(cluster_sizes):
        # Adjust the color bar for each cluster based on its size
        cluster_color = CLUSTER_COLORS.get(i, "#555555")
        ax.add_patch(plt.Rectangle((0, i), 1, size, color=cluster_color, alpha=0.8))

    ax.set_title("Cluster-wise Omics Heatmap", fontsize=16)

    # ---- 🔹 Move y-axis label outside the plot ----
    ax.set_ylabel("Clusters", fontsize=12, labelpad=20)  # Increased labelpad to push the label outwards

    # ---- 🔹 Omics-colored x-axis feature labels (move them away from the x-axis) ----
    for idx, label in enumerate(feature_names):
        color = feature_colors[idx]
        ax.text(idx + 0.5, n_clusters + 0.2, label, rotation=90,  # Adjusted y position
                ha='center', va='bottom', fontsize=8, color=color, clip_on=False)

    # ---- 🔹 Omics legend bar closer to the plot ----
    legend_elements = [
        Patch(facecolor=omics_colors['mf'], label='Mutation (mf)'),
        Patch(facecolor=omics_colors['cna'], label='CNA (cna)'),
        Patch(facecolor=omics_colors['meth'], label='Methylation (meth)'),
        Patch(facecolor=omics_colors['ge'], label='Expression (ge)')
    ]

    # Adjust the position to bring the legend bar closer
    plt.subplots_adjust(bottom=0.3)  
    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=4,
        frameon=False,
        fontsize=10
    )

    # ---- 🔹 Save ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Cluster-wise Omics heatmap saved to:\n{output_path}")

def plot_cluster_lrp_heatmap(relevance_scores, cluster_tensor, feature_names, output_path, n_clusters=12):
    omics_types = ['mf', 'cna', 'meth', 'ge']
    omics_colors = {
        'mf': '#b22222',     # dark red
        'cna': '#9370DB',    # purple
        'meth': '#00008B',   # dark blue
        'ge': '#228B22'      # dark green
    }

    # ---- 🔹 Remove "degree" feature ----
    if "degree" in feature_names:
        degree_idx = feature_names.index("degree")
        feature_names = feature_names[:degree_idx] + feature_names[degree_idx + 1:]
        relevance_scores = relevance_scores[:, :degree_idx].clone()
    
    # ---- 🔹 Assign omics colors ----
    num_features = len(feature_names)
    features_per_omic = num_features // len(omics_types)
    feature_omics_map = []
    for omic in omics_types:
        feature_omics_map.extend([omic] * features_per_omic)

    feature_colors = [omics_colors[o] for o in feature_omics_map]

    # ---- 🔹 Compute cluster-wise average LRP ----
    clusterwise_lrp = []
    for c in range(n_clusters):
        node_indices = torch.nonzero(cluster_tensor == c).squeeze()
        if node_indices.numel() == 0:
            clusterwise_lrp.append(torch.zeros(len(feature_names)))
        else:
            cluster_scores = relevance_scores[node_indices]
            cluster_mean = cluster_scores.mean(dim=0)
            clusterwise_lrp.append(cluster_mean)

    lrp_matrix = torch.stack(clusterwise_lrp).cpu().numpy()

    # ---- 🔹 Plot ----
    plt.figure(figsize=(max(12, len(feature_names) * 0.3), 8))
    ax = sns.heatmap(
        lrp_matrix,
        cmap="Greys",
        xticklabels=False,
        yticklabels=[f"C{c}" for c in range(n_clusters)],  # Cluster labels
        cbar_kws={'label': 'LRP Relevance'},
        linewidths=0.1
    )

    ax.set_title("Cluster-wise LRP Feature Relevance", fontsize=16)

    # ---- 🔹 Move y-axis label outside the plot ----
    ax.set_ylabel("Clusters", fontsize=12, labelpad=20)  # Increased labelpad to push the label outwards

    # ---- 🔹 Omics-colored x-axis feature labels (move them away from the x-axis) ----
    for idx, label in enumerate(feature_names):
        color = feature_colors[idx]
        ax.text(idx + 0.5, n_clusters + 0.2, label, rotation=90,  # Adjusted y position
                ha='center', va='bottom', fontsize=8, color=color, clip_on=False)

    # ---- 🔹 Cluster-colored y-axis tick labels ----
    for tick_label in ax.get_yticklabels():
        tick_label.set_color("white")  # Set color to white since we're not using the labels

    # ---- 🔹 Add color bar to the y-axis (on the left) ----
    cluster_color_list = [CLUSTER_COLORS.get(i, "#555555") for i in range(n_clusters)]
    cluster_cmap = ListedColormap(cluster_color_list)
    
    norm = mcolors.BoundaryNorm(boundaries=np.arange(n_clusters+1), ncolors=n_clusters)
    sm = plt.cm.ScalarMappable(cmap=cluster_cmap, norm=norm)
    sm.set_array([])

    # Place the color bar on the left of the plot
    plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label="Clusters", location="left")

    # ---- 🔹 Omics legend bar closer to the plot ----
    legend_elements = [
        Patch(facecolor=omics_colors['mf'], label='Mutation (mf)'),
        Patch(facecolor=omics_colors['cna'], label='CNA (cna)'),
        Patch(facecolor=omics_colors['meth'], label='Methylation (meth)'),
        Patch(facecolor=omics_colors['ge'], label='Expression (ge)')
    ]

    # Adjust the position to bring the legend bar closer
    plt.subplots_adjust(bottom=0.3)  
    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=4,
        frameon=False,
        fontsize=10
    )

    # ---- 🔹 Save ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Cluster-wise LRP heatmap saved to:\n{output_path}")

def plot_cluster_lrp_heatmap_y_pas(relevance_scores, cluster_tensor, feature_names, output_path, n_clusters=12):
    omics_types = ['mf', 'cna', 'meth', 'ge']
    omics_colors = {
        'mf': '#b22222',     # dark red
        'cna': '#9370DB',    # purple
        'meth': '#00008B',   # dark blue
        'ge': '#228B22'      # dark green
    }

    # ---- 🔹 Remove "degree" feature ----
    if "degree" in feature_names:
        degree_idx = feature_names.index("degree")
        feature_names = feature_names[:degree_idx] + feature_names[degree_idx + 1:]
        relevance_scores = relevance_scores[:, :degree_idx].clone()
    
    # ---- 🔹 Assign omics colors ----
    num_features = len(feature_names)
    features_per_omic = num_features // len(omics_types)
    feature_omics_map = []
    for omic in omics_types:
        feature_omics_map.extend([omic] * features_per_omic)

    feature_colors = [omics_colors[o] for o in feature_omics_map]

    # ---- 🔹 Compute cluster-wise average LRP ----
    clusterwise_lrp = []
    for c in range(n_clusters):
        node_indices = torch.nonzero(cluster_tensor == c).squeeze()
        if node_indices.numel() == 0:
            clusterwise_lrp.append(torch.zeros(len(feature_names)))
        else:
            cluster_scores = relevance_scores[node_indices]
            cluster_mean = cluster_scores.mean(dim=0)
            clusterwise_lrp.append(cluster_mean)

    lrp_matrix = torch.stack(clusterwise_lrp).cpu().numpy()

    # ---- 🔹 Plot ----
    plt.figure(figsize=(max(12, len(feature_names) * 0.3), 8))
    ax = sns.heatmap(
        lrp_matrix,
        cmap="Greys",
        xticklabels=False,
        yticklabels=False,  # Removed cluster names from y-axis
        cbar_kws={'label': 'LRP Relevance'},
        linewidths=0.1
    )

    ax.set_title("Cluster-wise LRP Feature Relevance", fontsize=16)

    # ---- 🔹 Move y-axis label outside the plot ----
    ax.set_ylabel("Clusters", fontsize=12, labelpad=20)  # Increased labelpad to push the label outwards

    # ---- 🔹 Omics-colored x-axis feature labels (move them away from the x-axis) ----
    for idx, label in enumerate(feature_names):
        color = feature_colors[idx]
        ax.text(idx + 0.5, n_clusters + 0.2, label, rotation=90,  # Adjusted y position
                ha='center', va='bottom', fontsize=8, color=color, clip_on=False)

    # ---- 🔹 Cluster-colored y-axis tick labels ----
    for tick_label in ax.get_yticklabels():
        tick_label.set_color("white")  # Set color to white since we're not using the labels

    # ---- 🔹 Add color bar to the y-axis (on the left) ----
    cluster_color_list = [CLUSTER_COLORS.get(i, "#555555") for i in range(n_clusters)]
    cluster_cmap = ListedColormap(cluster_color_list)
    
    norm = mcolors.BoundaryNorm(boundaries=np.arange(n_clusters+1), ncolors=n_clusters)
    sm = plt.cm.ScalarMappable(cmap=cluster_cmap, norm=norm)
    sm.set_array([])

    # Place the color bar on the left of the plot
    plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label="Clusters", location="left")

    # ---- 🔹 Omics legend bar closer to the plot ----
    legend_elements = [
        Patch(facecolor=omics_colors['mf'], label='Mutation (mf)'),
        Patch(facecolor=omics_colors['cna'], label='CNA (cna)'),
        Patch(facecolor=omics_colors['meth'], label='Methylation (meth)'),
        Patch(facecolor=omics_colors['ge'], label='Expression (ge)')
    ]

    # Adjust the position to bring the legend bar closer
    plt.subplots_adjust(bottom=0.3)  
    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=4,
        frameon=False,
        fontsize=10
    )

    # ---- 🔹 Save ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Cluster-wise LRP heatmap saved to:\n{output_path}")

def plot_cluster_lrp_heatmap_no_cluster_embedding_color_pass(relevance_scores, cluster_tensor, feature_names, output_path, n_clusters=12):
    omics_types = ['mf', 'cna', 'meth', 'ge']
    omics_colors = {
        'mf': '#b22222',     # dark red
        'cna': '#9370DB',    # purple
        'meth': '#00008B',   # dark blue
        'ge': '#228B22'      # dark green
    }

    # ---- 🔹 Remove "degree" ----
    if "degree" in feature_names:
        degree_idx = feature_names.index("degree")
        feature_names = feature_names[:degree_idx]
        relevance_scores = relevance_scores[:, :degree_idx]

    # ---- 🔹 Assign omics colors ----
    num_features = len(feature_names)
    features_per_omic = num_features // len(omics_types)
    feature_omics_map = []
    for omic in omics_types:
        feature_omics_map.extend([omic] * features_per_omic)

    feature_colors = [omics_colors[o] for o in feature_omics_map]

    # ---- 🔹 Compute cluster-wise average LRP ----
    clusterwise_lrp = []
    for c in range(n_clusters):
        node_indices = torch.nonzero(cluster_tensor == c).squeeze()
        if node_indices.numel() == 0:
            clusterwise_lrp.append(torch.zeros(len(feature_names)))
        else:
            cluster_scores = relevance_scores[node_indices]
            cluster_mean = cluster_scores.mean(dim=0)
            clusterwise_lrp.append(cluster_mean)

    lrp_matrix = torch.stack(clusterwise_lrp).cpu().numpy()

    # ---- 🔹 Plot ----
    plt.figure(figsize=(max(12, len(feature_names) * 0.3), 8))
    ax = sns.heatmap(
        lrp_matrix,
        cmap="Greys",
        xticklabels=False,
        yticklabels=[f"C{c}" for c in range(n_clusters)],
        cbar_kws={'label': 'LRP Relevance'},
        linewidths=0.1
    )

    ax.set_title("Cluster-wise LRP Feature Relevance", fontsize=16)
    ax.set_ylabel("Clusters", fontsize=12)

    # ---- 🔹 Omics-colored x-axis feature labels ----
    for idx, label in enumerate(feature_names):
        color = feature_colors[idx]
        ax.text(idx + 0.5, n_clusters + 0.1, label, rotation=90,
                ha='center', va='bottom', fontsize=8, color=color, clip_on=False)

    # ---- 🔹 Cluster-colored y-axis tick labels ----
    for tick_label in ax.get_yticklabels():
        cluster_idx = int(tick_label.get_text().replace("C", ""))
        tick_label.set_color(CLUSTER_COLORS.get(cluster_idx, "black"))
        tick_label.set_fontweight("bold")

    # ---- 🔹 Omics legend ----
    legend_elements = [
        Patch(facecolor=omics_colors['mf'], label='Mutation (mf)'),
        Patch(facecolor=omics_colors['cna'], label='CNA (cna)'),
        Patch(facecolor=omics_colors['meth'], label='Methylation (meth)'),
        Patch(facecolor=omics_colors['ge'], label='Expression (ge)')
    ]

    plt.subplots_adjust(bottom=0.35)
    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=4,
        frameon=False,
        fontsize=10
    )

    # ---- 🔹 Save ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_predicted_genes_tsne_red_circle_pass(graph, node_names, scores, output_path, top_k=1000):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores.cpu().numpy()

    # Get top K predicted genes
    top_indices = np.argsort(scores)[-top_k:]
    top_embeddings = embeddings[top_indices]
    top_clusters = cluster_ids[top_indices]
    top_scores = scores[top_indices]
    top_names = [node_names[i] for i in top_indices]

    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=42)
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
            # Highlight the top 1 with a red circle (not a dot)
            plt.scatter(x, y, s=150, edgecolors='red', alpha=1.0, linewidth=2, marker='o')
            # Change label text color to pink
            plt.text(x, y, name, fontsize=9, fontweight='bold', ha='center', va='center', color='green')

    plt.title("t-SNE of Top 1000 Predicted Genes by Cluster", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    # Remove legend
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Top predicted gene t-SNE plot saved to:\n{output_path}")

def plot_top_predicted_genes_tsne_red_dot_pass(graph, node_names, scores, output_path, top_k=1000):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores.cpu().numpy()

    # Get top K predicted genes
    top_indices = np.argsort(scores)[-top_k:]
    top_embeddings = embeddings[top_indices]
    top_clusters = cluster_ids[top_indices]
    top_scores = scores[top_indices]
    top_names = [node_names[i] for i in top_indices]

    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=42)
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
            # Highlight the top 1 with a red circle
            plt.scatter(x, y, color='red', s=100, edgecolors='k', alpha=1.0, linewidth=2)
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

def plot_top_predicted_genes_tsne_with_legend_pass(graph, node_names, scores, output_path, top_k=1000):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores.cpu().numpy()

    # Get top K predicted genes
    top_indices = np.argsort(scores)[-top_k:]
    top_embeddings = embeddings[top_indices]
    top_clusters = cluster_ids[top_indices]
    top_scores = scores[top_indices]
    top_names = [node_names[i] for i in top_indices]

    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=42)
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
            label=f"Cluster {c}",
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
            plt.text(x, y, name, fontsize=9, fontweight='bold', ha='center', va='center')

    plt.title("t-SNE of Top 1000 Predicted Genes by Cluster", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Top predicted gene t-SNE plot saved to:\n{output_path}")

def plot_predicted_gene_embeddings_by_cluster(graph, node_names, scores, output_path, score_threshold=0.5):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores.cpu().numpy()

    predicted_mask = scores >= score_threshold
    predicted_indices = np.where(predicted_mask)[0]

    if len(predicted_indices) == 0:
        print("⚠️ No predicted genes above threshold.")
        return

    predicted_embeddings = embeddings[predicted_indices]
    predicted_clusters = cluster_ids[predicted_indices]

    tsne = TSNE(n_components=2, random_state=42)
    tsne_coords = tsne.fit_transform(predicted_embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    for c in np.unique(predicted_clusters):
        cluster_mask = predicted_clusters == c
        coords = tsne_coords[cluster_mask]
        label = f"Cluster {c}"
        plt.scatter(
            coords[:, 0], coords[:, 1],
            color=CLUSTER_COLORS.get(c, "#555555"),
            label=label,
            alpha=0.8,
            edgecolors='k',
            s=60
        )

    plt.title("t-SNE of Predicted Genes Colored by Cluster", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Clustered t-SNE plot of predicted genes saved to:\n{output_path}")

def plot_cluster_lrp_heatmap_color_label_text_pass(relevance_scores, cluster_tensor, feature_names, output_path, n_clusters=12):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    # ----- 🔹 Setup -----
    omics_types = ['mf', 'cna', 'meth', 'ge']
    omics_colors = {
        'mf': '#b22222',     # dark red
        'cna': '#9370DB',    # purple
        'meth': '#00008B',   # dark blue
        'ge': '#228B22'      # dark green
    }

    # ----- 🔹 Remove "degree" -----
    if "degree" in feature_names:
        degree_idx = feature_names.index("degree")
        feature_names = feature_names[:degree_idx]
        relevance_scores = relevance_scores[:, :degree_idx]

    # ----- 🔹 Map features to omics -----
    num_features = len(feature_names)
    features_per_omic = num_features // len(omics_types)
    feature_omics_map = []
    for omic in omics_types:
        feature_omics_map.extend([omic] * features_per_omic)

    feature_colors = [omics_colors[o] for o in feature_omics_map]

    # ----- 🔹 Compute clusterwise LRP -----
    clusterwise_lrp = []
    for c in range(n_clusters):
        node_indices = torch.nonzero(cluster_tensor == c).squeeze()
        if node_indices.numel() == 0:
            clusterwise_lrp.append(torch.zeros(len(feature_names)))
        else:
            cluster_scores = relevance_scores[node_indices]
            cluster_mean = cluster_scores.mean(dim=0)
            clusterwise_lrp.append(cluster_mean)

    lrp_matrix = torch.stack(clusterwise_lrp).cpu().numpy()

    # ----- 🔹 Plotting -----
    plt.figure(figsize=(max(12, len(feature_names) * 0.3), 8))
    ax = sns.heatmap(
        lrp_matrix,
        cmap="Greys",
        xticklabels=False,
        yticklabels=[f"C{c}" for c in range(n_clusters)],
        cbar_kws={'label': 'LRP Relevance'},
        linewidths=0.1
    )

    ax.set_title("Cluster-wise LRP Feature Relevance", fontsize=16)
    ax.set_ylabel("Clusters", fontsize=12)

    # ----- 🔹 Add text labels on x-axis with omics coloring -----
    for idx, label in enumerate(feature_names):
        color = feature_colors[idx]
        ax.text(idx + 0.5, n_clusters + 0.1, label, rotation=90,
                ha='center', va='bottom', fontsize=8, color=color, clip_on=False)

    # ----- 🔹 Add legend -----
    legend_elements = [
        Patch(facecolor=omics_colors['mf'], label='Mutation (mf)'),
        Patch(facecolor=omics_colors['cna'], label='CNA (cna)'),
        Patch(facecolor=omics_colors['meth'], label='Methylation (meth)'),
        Patch(facecolor=omics_colors['ge'], label='Expression (ge)')
    ]

    plt.subplots_adjust(bottom=0.35)
    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=4,
        frameon=False,
        fontsize=10
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cluster_lrp_heatmap_omics_legend_bar_pass(relevance_scores, cluster_tensor, feature_names, output_path, n_clusters=12):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    # Step 1: Aggregate relevance scores per cluster
    clusterwise_lrp = []
    for c in range(n_clusters):
        node_indices = torch.nonzero(cluster_tensor == c).squeeze()
        if node_indices.numel() == 0:
            clusterwise_lrp.append(torch.zeros(len(feature_names)))
        else:
            cluster_scores = relevance_scores[node_indices]
            cluster_mean = cluster_scores.mean(dim=0)
            clusterwise_lrp.append(cluster_mean)

    lrp_matrix = torch.stack(clusterwise_lrp).cpu().numpy()

    # Step 2: Define omics coloring scheme
    omics_types = ['mf', 'cna', 'meth', 'ge']  # Order matches your input
    omics_colors = {
        'mf': '#b22222',     # dark red
        'cna': '#9370DB',    # purple
        'meth': '#00008B',   # dark blue
        'ge': '#228B22'      # dark green
    }

    # Assume feature_names are grouped and evenly divided by omics
    num_per_omics = (len(feature_names) - 1) // len(omics_types)  # Subtract "degree" if at end
    feature_to_omics = []
    for omic in omics_types:
        feature_to_omics.extend([omic] * num_per_omics)
    feature_to_omics.append("degree")  # Last feature is degree

    feature_colors = [omics_colors.get(t, 'gray') for t in feature_to_omics]

    # Step 3: Plot heatmap
    plt.figure(figsize=(22, 10))
    ax = sns.heatmap(
        lrp_matrix,
        cmap="Greys",
        xticklabels=False,
        yticklabels=[f"C{c}" for c in range(n_clusters)],
        cbar_kws={'label': 'LRP Relevance'},
        linewidths=0.2
    )

    ax.set_title("Cluster-wise LRP Feature Relevance", fontsize=16)
    ax.set_ylabel("Clusters", fontsize=12)

    # Step 4: Colored strip under heatmap
    for i, color in enumerate(feature_colors):
        ax.add_patch(Rectangle((i, n_clusters), 1, 0.3, linewidth=0, facecolor=color, clip_on=False))

    plt.subplots_adjust(bottom=0.3)

    # Step 5: Add legend for omics types
    legend_elements = [
        Patch(facecolor='#b22222', label='Mutation (mf)'),
        Patch(facecolor='#9370DB', label='CNA (cna)'),
        Patch(facecolor='#00008B', label='Methylation (meth)'),
        Patch(facecolor='#228B22', label='Expression (ge)'),
        Patch(facecolor='gray', label='Degree')
    ]
    plt.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=5,
        frameon=False
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cluster_lrp_heatmap_omics_color_pa(relevance_scores, cluster_tensor, feature_names, output_path, n_clusters=12):
    # Aggregate LRP scores by cluster
    clusterwise_lrp = []
    for c in range(n_clusters):
        node_indices = torch.nonzero(cluster_tensor == c).squeeze()
        if node_indices.numel() == 0:
            clusterwise_lrp.append(torch.zeros(len(feature_names)))
        else:
            cluster_scores = relevance_scores[node_indices]
            cluster_mean = cluster_scores.mean(dim=0)
            clusterwise_lrp.append(cluster_mean)

    lrp_matrix = torch.stack(clusterwise_lrp).cpu().numpy()

    # Create a colormap for different feature types
    feature_types = ['Mutation'] * 32 + ['CNA'] * 32 + ['Methylation'] * 32 + ['Expression'] * 32 + ['Degree']
    palette = {
        'Mutation': '#b22222',
        'CNA': '#9370DB',
        'Methylation': '#00008B',
        'Expression': '#228B22',
        'Degree': 'gray'
    }
    col_colors = [palette[t] for t in feature_types]

    # Plot heatmap
    plt.figure(figsize=(20, 8))
    sns.heatmap(lrp_matrix, cmap="Greys", xticklabels=feature_names, yticklabels=[f"C{c}" for c in range(n_clusters)],
                cbar_kws={'label': 'LRP Relevance'}, linewidths=0.2)

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.title("Mean LRP Feature Relevance per Cluster", fontsize=14)

    # Add colored bars under x-axis to show feature type
    for i, color in enumerate(col_colors):
        plt.gca().add_patch(Rectangle((i, n_clusters), 1, 0.3, linewidth=0, facecolor=color, clip_on=False))

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
def train_color_pa(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    print("Running KMeans clustering on node features...")
    kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 'THCA', 
                    'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    bio_feat_names = []
    for cancer in cancer_types:
        for omics in omics_types:
            for i in range(16):
                bio_feat_names.append(f"{cancer}_{omics}_{i}")

    topo_feat_names = [f"topology_{i}" for i in range(1024)]
    gene_feature_names = node_names[:embeddings.shape[1]]
    feature_names = bio_feat_names + topo_feat_names + ["degree"] + gene_feature_names

    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)

    cluster_tensor = graph.ndata['cluster']
    cluster_relevance = torch.zeros((12, relevance_scores.shape[1]))
    for c in range(12):
        node_indices = (cluster_tensor == c).nonzero(as_tuple=True)[0]
        if node_indices.numel() > 0:
            cluster_relevance[c] = relevance_scores[node_indices].mean(dim=0)

    print("Generating cluster heatmap...")

    # Optional feature group labels
    feature_groups = {
        'TP53_mut': 'Mutation',
        'MYC_amp': 'CNA',
        'BRCA1_expr': 'Gene expression',
        'CDKN2A_meth': 'DNA methylation',
        # etc... for each feature
    }

    '''plot_lrp_cluster_heatmap(
        all_cluster_scores=relevance_scores,  # use relevance_scores here
        feature_names=feature_names,           # feature names should already be defined
        cluster_tensor=cluster_tensor,         # use cluster_tensor here
        output_dir="results/gene_prediction/",
        model_type=args.model_type,
        net_type=args.net_type,
        score_threshold=args.score_threshold,
        num_epochs=args.num_epochs,
        feature_groups=feature_groups
    )'''

    plot_lrp_cluster_heatmap(
        all_cluster_scores=relevance_scores,  # LRP scores tensor
        feature_names=feature_names,           # List of feature names
        cluster_tensor=graph.ndata['cluster'], # Tensor with cluster assignments
        output_dir="results/gene_prediction/",
        model_type=args.model_type,
        net_type=args.net_type,
        score_threshold=args.score_threshold,
        num_epochs=args.num_epochs,                      # Number of epochs
        feature_groups=feature_groups, 
        filename_prefix='lrp_heatmap'          # Prefix for the filename
    )

def train_worse(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    print("Running KMeans clustering on node features...")
    kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"

    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 'THCA', 
                    'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    bio_feat_names = []
    for cancer in cancer_types:
        for omics in omics_types:
            for i in range(16):
                bio_feat_names.append(f"{cancer}_{omics}_{i}")

    topo_feat_names = [f"topology_{i}" for i in range(1024)]
    gene_feature_names = node_names[:embeddings.shape[1]]
    feature_names = bio_feat_names + topo_feat_names + ["degree"] + gene_feature_names

    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)

    cluster_tensor = graph.ndata['cluster']
    cluster_relevance = torch.zeros((12, relevance_scores.shape[1]))
    for c in range(12):
        node_indices = (cluster_tensor == c).nonzero(as_tuple=True)[0]
        if node_indices.numel() > 0:
            cluster_relevance[c] = relevance_scores[node_indices].mean(dim=0)

    print("Generating cluster heatmap...")

    # Optional feature group labels
    feature_groups = {
        'TP53_mut': 'Mutation',
        'MYC_amp': 'CNA',
        'BRCA1_expr': 'Gene expression',
        'CDKN2A_meth': 'DNA methylation',
        # etc... for each feature
    }

    '''plot_lrp_cluster_heatmap(
        all_cluster_scores=relevance_scores,  # use relevance_scores here
        feature_names=feature_names,           # feature names should already be defined
        cluster_tensor=cluster_tensor,         # use cluster_tensor here
        output_dir="results/gene_prediction/",
        model_type=args.model_type,
        net_type=args.net_type,
        score_threshold=args.score_threshold,
        num_epochs=args.num_epochs,
        feature_groups=feature_groups
    )'''

    plot_lrp_cluster_heatmap(
        all_cluster_scores=relevance_scores,  # LRP scores tensor
        feature_names=feature_names,           # List of feature names
        cluster_tensor=graph.ndata['cluster'], # Tensor with cluster assignments
        output_dir="results/gene_prediction/",
        model_type=args.model_type,
        net_type=args.net_type,
        score_threshold=args.score_threshold,
        num_epochs=args.num_epochs,                      # Number of epochs
        feature_groups=feature_groups, 
        filename_prefix='lrp_heatmap'          # Prefix for the filename
    )

def plot_lrp_cluster_heatmap_worse(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    feature_groups: dict = None,
    filename_prefix: str = "lrp_heatmap"
):
    os.makedirs(output_dir, exist_ok=True)

    # === Prepare matrix: clusters (rows) × features (columns) ===
    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()

    if len(feature_names) != scores_np.shape[1]:
        feature_names = [f'feature_{i}' for i in range(scores_np.shape[1])]

    df = pd.DataFrame(scores_np, columns=feature_names)
    df['cluster'] = cluster_ids
    cluster_means = df.groupby('cluster').mean()
    cluster_means = cluster_means.sort_index()

    # === Transpose for visualization: features (x-axis) × clusters (y-axis) ===
    heatmap_data = cluster_means.T

    # === Plot with omics color bar on top ===
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 20], hspace=0.05)

    # Omics bar
    ax_omics = fig.add_subplot(gs[0])
    if feature_groups:
        OMICS_COLORS = {
            'Mutation': '#a50026',
            'CNA': '#b35806',
            'DNA methylation': '#313695',
            'Gene expression': '#1b7837'
        }

        omics_labels = [feature_groups.get(f, '') for f in heatmap_data.index]
        omics_colors = [OMICS_COLORS.get(label, 'lightgray') for label in omics_labels]
        rgba_matrix = np.array([[to_rgba(color) for color in omics_colors]])
        ax_omics.imshow(rgba_matrix, aspect='auto')
        ax_omics.set_xticks([])
        ax_omics.set_yticks([])

        # Draw group names on top
        label_map = {}
        for idx, label in enumerate(omics_labels):
            if label:
                label_map.setdefault(label, []).append(idx)

        for label, indices in label_map.items():
            avg_x = np.mean(indices)
            ax_omics.text(avg_x, 1.05, label, ha='center', va='bottom', fontsize=10)

    # Main heatmap
    ax_heatmap = fig.add_subplot(gs[1])
    sns.heatmap(
        heatmap_data,
        cmap='Greys',
        ax=ax_heatmap,
        cbar_kws={"label": "Total feature contribution"},
        xticklabels=True,
        yticklabels=True,
        linewidths=0.05,
        linecolor='lightgray'
    )
    ax_heatmap.set_xlabel("Features", fontsize=14)
    ax_heatmap.set_ylabel("Cluster", fontsize=14)
    ax_heatmap.set_title("Average LRP Feature Contribution per Cluster", fontsize=16)

    # Rotate feature labels to vertical
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)

    # Save
    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🔥 Saved aligned LRP heatmap to: {plot_path}")

def plot_lrp_cluster_heatmap_90(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    feature_groups: dict = None,
    filename_prefix: str = "lrp_heatmap"
):
    """
    Plot a heatmap of LRP scores averaged over nodes in each cluster,
    rotated 90 degrees clockwise to match uploaded image style.
    """
    os.makedirs(output_dir, exist_ok=True)

    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()

    if len(feature_names) != scores_np.shape[1]:
        feature_names = ['feature_{}'.format(i) for i in range(scores_np.shape[1])]

    df = pd.DataFrame(scores_np, columns=feature_names)
    df['cluster'] = cluster_ids
    cluster_means = df.groupby('cluster').mean()
    cluster_means = cluster_means.sort_index()

    # Transpose for rotation (features on y-axis, clusters on x-axis)
    heatmap_data = cluster_means.T

    # GridSpec: 2 rows (heatmap + omics bar)
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 1, height_ratios=[20, 1], hspace=0.1)

    # Main heatmap
    ax_heatmap = fig.add_subplot(gs[0])
    sns.heatmap(
        heatmap_data,
        cmap='Greys',
        cbar_kws={"label": "Total feature contribution"},
        yticklabels=True,
        xticklabels=True,
        square=False,
        ax=ax_heatmap,
        linewidths=0.1,
        linecolor='lightgrey'
    )
    ax_heatmap.set_xlabel("Cluster", fontsize=14)
    ax_heatmap.set_ylabel("Feature", fontsize=14)
    ax_heatmap.set_title("Average LRP Feature Contribution per Cluster", fontsize=16)
    ax_heatmap.tick_params(axis='x', labelrotation=0)
    ax_heatmap.tick_params(axis='y', labelrotation=0)

    if feature_groups:
        OMICS_COLORS = {
            'Mutation': '#a50026',
            'CNA': '#b35806',
            'DNA methylation': '#313695',
            'Gene expression': '#1b7837'
        }

        # Match omics group colors for features
        omics_labels = [feature_groups.get(f, '') for f in heatmap_data.index]
        omics_colors = [OMICS_COLORS.get(label, 'lightgray') for label in omics_labels]

        # Color bar: 1 row of colors
        ax_colorbar = fig.add_subplot(gs[1])
        color_matrix = np.array([omics_colors])
        rgba_matrix = np.array([[to_rgba(color) for color in color_matrix[0]]])

        ax_colorbar.imshow(rgba_matrix, aspect='auto')
        ax_colorbar.set_xticks([])
        ax_colorbar.set_yticks([])

        # Draw omics group names under patches (average x position per group)
        label_map = {}
        for idx, label in enumerate(omics_labels):
            if label:
                label_map.setdefault(label, []).append(idx)

        for label, indices in label_map.items():
            avg_x = np.mean(indices)
            ax_colorbar.text(
                avg_x, 1.1, label, ha='center', va='bottom', fontsize=10, rotation=0
            )

    # Save
    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🔥 Saved rotated LRP heatmap to: {plot_path}")

def plot_lrp_cluster_heatmap_still_not_horizontal(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    feature_groups: dict = None,
    filename_prefix: str = "lrp_heatmap"
):
    """
    Plot a heatmap of LRP scores averaged over nodes in each cluster, styled similar to reference image.
    """
    os.makedirs(output_dir, exist_ok=True)

    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()

    if len(feature_names) != scores_np.shape[1]:
        feature_names = ['feature_{}'.format(i) for i in range(scores_np.shape[1])]

    df = pd.DataFrame(scores_np, columns=feature_names)
    df['cluster'] = cluster_ids
    cluster_means = df.groupby('cluster').mean()
    cluster_means = cluster_means.sort_index()

    # Set up plot layout
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 1, height_ratios=[20, 1], hspace=0.05)

    # Main heatmap
    ax_heatmap = fig.add_subplot(gs[0])
    sns.heatmap(
        cluster_means,
        cmap='Greys',
        cbar_kws={"label": "Total feature contribution"},
        yticklabels=True,
        xticklabels=True,
        square=False,
        ax=ax_heatmap,
        linewidths=0.1,
        linecolor='lightgrey'
    )
    ax_heatmap.set_xlabel("Features", fontsize=14)
    ax_heatmap.set_ylabel("Cluster", fontsize=14)
    ax_heatmap.set_title("Average LRP Feature Contribution per Cluster", fontsize=16)
    ax_heatmap.tick_params(axis='x', labelrotation=90)

    # Horizontal omics color bar
    if feature_groups:
        OMICS_COLORS = {
            'Mutation': '#a50026',
            'CNA': '#b35806',
            'DNA methylation': '#313695',
            'Gene expression': '#1b7837'
        }
        omics_colors = [OMICS_COLORS.get(feature_groups.get(f, ''), 'lightgray') for f in cluster_means.columns]

        # Convert to RGBA
        color_matrix = np.array([omics_colors])
        rgba_matrix = np.array([[to_rgba(color) for color in row] for row in color_matrix])

        # Bottom color bar
        ax_colorbar = fig.add_subplot(gs[1])
        ax_colorbar.imshow(rgba_matrix, aspect='auto')
        ax_colorbar.set_xticks([])
        ax_colorbar.set_yticks([])
        ax_colorbar.set_ylabel("Omics", rotation=0, labelpad=40)

        # Legend
        handles = [Patch(color=color, label=group) for group, color in OMICS_COLORS.items()]
        ax_heatmap.legend(handles=handles, title="Omics Types", bbox_to_anchor=(1.01, 1), loc="upper left")

    # Save
    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🔥 Saved LRP heatmap to: {plot_path}")

def plot_lrp_cluster_heatmap_not_horizontal_patches_pass(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    feature_groups: dict = None,
    filename_prefix: str = "lrp_heatmap"
):
    """
    Plot a heatmap of LRP scores averaged over nodes in each cluster, styled similar to reference image.
    """
    os.makedirs(output_dir, exist_ok=True)

    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()

    if len(feature_names) != scores_np.shape[1]:
        feature_names = ['feature_{}'.format(i) for i in range(scores_np.shape[1])]

    df = pd.DataFrame(scores_np, columns=feature_names)
    df['cluster'] = cluster_ids
    cluster_means = df.groupby('cluster').mean()
    cluster_means = cluster_means.sort_index()

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 1, height_ratios=[20, 1], hspace=0.05)

    ax_heatmap = fig.add_subplot(gs[0])
    sns.heatmap(
        cluster_means,
        cmap='Greys',
        cbar_kws={"label": "Total feature contribution"},
        yticklabels=True,
        xticklabels=True,
        square=False,
        ax=ax_heatmap,
        linewidths=0.1,
        linecolor='lightgrey'
    )
    ax_heatmap.set_xlabel("Features", fontsize=14)
    ax_heatmap.set_ylabel("Cluster", fontsize=14)
    ax_heatmap.set_title("Average LRP Feature Contribution per Cluster", fontsize=16)
    ax_heatmap.tick_params(axis='x', labelrotation=90)

    if feature_groups:
        OMICS_COLORS = {
            'Mutation': '#a50026',
            'CNA': '#b35806',
            'DNA methylation': '#313695',
            'Gene expression': '#1b7837'
        }
        omics_colors = [OMICS_COLORS.get(feature_groups.get(f, ''), 'lightgray') for f in cluster_means.columns]

        ax_colorbar = fig.add_subplot(gs[1])
        color_matrix = np.array([omics_colors])
        

        # Convert color_matrix of hex strings to RGBA
        rgba_matrix = np.array([[to_rgba(color) for color in row] for row in color_matrix])

        ax_colorbar.imshow(rgba_matrix, aspect='auto')
        ax_colorbar.set_xticks([])
        ax_colorbar.set_yticks([])
        ax_colorbar.set_ylabel("Omics", rotation=0, labelpad=40)

        handles = [Patch(color=color, label=group) for group, color in OMICS_COLORS.items()]
        ax_heatmap.legend(handles=handles, title="Omics Types", bbox_to_anchor=(1.01, 1), loc="upper left")

    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\U0001F525 Saved LRP heatmap to: {plot_path}")

def plot_lrp_cluster_heatmap_orii(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    feature_groups: dict = None,
    filename_prefix: str = "lrp_heatmap"
):
    """
    Plot a heatmap of LRP scores averaged over nodes in each cluster, styled similar to reference image.

    Args:
        all_cluster_scores: [num_nodes, num_features] tensor of LRP scores.
        feature_names: List of feature names.
        cluster_tensor: Tensor assigning each node to a cluster.
        output_dir: Path to directory to save the heatmap.
        model_type: Model type string (e.g., 'ACGNN').
        net_type: Graph type string.
        score_threshold: Score threshold for filtering (used in filename).
        num_epochs: Number of training epochs (used in filename).
        feature_groups: Optional dict of {feature_name: group_name}.
        filename_prefix: Prefix for the plot file name.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy for plotting
    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()

    # Assuming scores_np has shape (7528, 129)
    # Feature names should match this number (129), not 2177
    feature_names = ['feature_{}'.format(i) for i in range(129)]  # This should match the columns of scores_np

    df = pd.DataFrame(scores_np, columns=feature_names)

    df['cluster'] = cluster_ids

    # Group by cluster and compute mean scores
    cluster_means = df.groupby('cluster').mean()
    cluster_means = cluster_means.sort_index()

    # Feature group coloring
    if feature_groups:
        group_colors = {
            'Mutation': '#a50026',
            'CNA': '#b35806',
            'DNA methylation': '#313695',
            'Gene expression': '#1b7837'
        }
        col_colors = [
            group_colors.get(feature_groups.get(f, ''), 'lightgray') for f in cluster_means.columns
        ]
    else:
        col_colors = None

    # Plotting setup
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(1, 10, width_ratios=[0.3] + [9]*9, wspace=0.05)

    # Cluster side color bar
    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(
        np.expand_dims(cluster_means.index, 1),
        cmap='tab20',
        cbar=False,
        ax=ax0,
        yticklabels=False,
        xticklabels=False
    )
    ax0.set_ylabel('Cluster')
    ax0.set_xticks([])

    # Main heatmap
    ax1 = fig.add_subplot(gs[1:])
    sns.heatmap(
        cluster_means,
        cmap='Greys',
        cbar_kws={"label": "Total feature contribution"},
        yticklabels=True,
        xticklabels=True,
        square=False,
        ax=ax1,
        linewidths=0.1,
        linecolor='lightgrey',
        ##col_colors=col_colors  # Add color by feature groups
    )
    ax1.set_xlabel("Features", fontsize=14)
    ax1.set_ylabel("Cluster", fontsize=14)
    ax1.set_title("Average LRP Feature Contribution per Cluster", fontsize=16)
    ax1.tick_params(axis='x', labelrotation=90)

    # Optional legend for feature group coloring
    if feature_groups:
        handles = [Patch(color=c, label=label) for label, c in group_colors.items()]
        ax1.legend(handles=handles, title="Feature groups", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Save plot
    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🔥 Saved LRP heatmap to: {plot_path}")

def plot_lrp_cluster_heatmap_2177(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    feature_groups: dict = None,
    filename_prefix: str = "lrp_heatmap"
):
    os.makedirs(output_dir, exist_ok=True)

    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()
    df = pd.DataFrame(scores_np, columns=feature_names)
    df['cluster'] = cluster_ids

    cluster_means = df.groupby('cluster').mean()
    cluster_means = cluster_means.sort_index()

    # Column colors by feature group
    col_colors = None
    group_colors = {}
    if feature_groups:
        group_colors = {
            'Mutation': '#a50026',
            'CNA': '#b35806',
            'DNA methylation': '#313695',
            'Gene expression': '#1b7837'
        }
        col_colors = pd.DataFrame([
            group_colors.get(feature_groups.get(f, ''), 'lightgray') for f in cluster_means.columns
        ]).T

    # Row colors for clusters
    row_colors = pd.DataFrame([
        CLUSTER_COLORS.get(cid, '#cccccc') for cid in cluster_means.index
    ])

    # Plot
    sns.set(style="white")
    g = sns.clustermap(
        cluster_means,
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap='Greys',
        linewidths=0.05,
        figsize=(20, 14),
        cbar_kws={"label": "Total feature contribution"}
    )

    g.ax_heatmap.set_xlabel("Features", fontsize=14)
    g.ax_heatmap.set_ylabel("Cluster", fontsize=14)
    g.ax_heatmap.set_title("Average LRP Feature Contribution per Cluster", fontsize=16)
    g.ax_heatmap.tick_params(axis='x', labelrotation=90)

    # Custom legends
    cluster_legend_handles = [
        Patch(facecolor=color, label=f"Cluster {cid}") for cid, color in CLUSTER_COLORS.items()
    ]
    group_legend_handles = [
        Patch(facecolor=color, label=label) for label, color in group_colors.items()
    ]

    # Add both legends
    plt.legend(
        handles=cluster_legend_handles + group_legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.45),
        ncol=4,
        fontsize=12,
        title="Legend",
        frameon=False
    )

    # Save
    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"🔥 Saved LRP heatmap to: {plot_path}")

def plot_lrp_cluster_heatmap_2177(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    feature_groups: dict = None,
    filename_prefix: str = "lrp_heatmap"
):
    os.makedirs(output_dir, exist_ok=True)

    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()
    df = pd.DataFrame(scores_np, columns=feature_names)
    df['cluster'] = cluster_ids

    cluster_means = df.groupby('cluster').mean().sort_index()

    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 1, height_ratios=[10, 0.4], hspace=0.05)

    ax_heatmap = fig.add_subplot(gs[0])

    sns.heatmap(
        cluster_means,
        cmap='Greys',
        cbar_kws={"label": "Total feature contribution"},
        yticklabels=True,
        xticklabels=True,
        ax=ax_heatmap,
        linewidths=0.1,
        linecolor='lightgrey'
    )
    ax_heatmap.set_xlabel("")
    ax_heatmap.set_ylabel("Cluster", fontsize=14)
    ax_heatmap.set_title("Average LRP Feature Contribution per Cluster", fontsize=16)
    ax_heatmap.tick_params(axis='x', labelrotation=90)

    if feature_groups:
        OMICS_COLORS = {
            'Mutation': '#a50026',
            'CNA': '#b35806',
            'DNA methylation': '#313695',
            'Gene expression': '#1b7837'
        }
        color_bar = []
        for f in cluster_means.columns:
            group = feature_groups.get(f, 'Mutation')
            color_bar.append(OMICS_COLORS.get(group, 'lightgray'))
        
        ax_colorbar = fig.add_subplot(gs[1])
        color_array = np.array(color_bar).reshape(1, -1)
        ax_colorbar.imshow(color_array, aspect='auto')
        ax_colorbar.set_xticks([])
        ax_colorbar.set_yticks([])
        ax_colorbar.set_ylabel("Omics", rotation=0, labelpad=40)

        # Optional legend
        handles = [Patch(color=c, label=label) for label, c in OMICS_COLORS.items()]
        ax_heatmap.legend(handles=handles, title="Omics types", bbox_to_anchor=(1.01, 1), loc="upper left")

    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🔥 Saved LRP heatmap with horizontal color bar to: {plot_path}")

def plot_lrp_cluster_heatmap_(all_cluster_scores, feature_names, cluster_labels, feature_groups=None, save_path=None):
    cluster_means = np.zeros((len(set(cluster_labels)), all_cluster_scores.shape[1]))
    for i in range(cluster_means.shape[0]):
        cluster_means[i] = all_cluster_scores[cluster_labels == i].mean(axis=0)

    # Generate feature group colors (for columns)
    if feature_groups:
        feature_group_list = [feature_groups.get(f, 'Other') for f in feature_names]
        unique_groups = sorted(set(feature_group_list))
        group_colors = {
            'Mutation': '#a50026',
            'CNA': '#b35806',
            'DNA methylation': '#313695',
            'Gene expression': '#1b7837',
            'Other': 'lightgray'
        }
        col_colors = pd.DataFrame([
            group_colors[g] for g in feature_group_list
        ], index=feature_names).T
    else:
        col_colors = None

    # Row (cluster) color bar
    row_colors = pd.DataFrame([
        CLUSTER_COLORS[i] for i in range(cluster_means.shape[0])
    ], index=[f'Cluster {i}' for i in range(cluster_means.shape[0])])

    # Plot
    sns.set(font_scale=0.9)
    g = sns.clustermap(
        cluster_means,
        cmap='Greys',
        row_cluster=False,
        col_cluster=False,
        col_colors=col_colors,
        row_colors=row_colors,
        xticklabels=feature_names,
        yticklabels=[f'Cluster {i}' for i in range(cluster_means.shape[0])],
        linewidths=0.1,
        figsize=(22, 12)
    )
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.fig.suptitle("Average LRP Feature Contribution per Cluster", fontsize=18)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_lrp_cluster_heatmap_color_ori(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    feature_groups: dict = None,
    filename_prefix: str = "lrp_heatmap"
):
    """
    Plot a heatmap of LRP scores averaged over nodes in each cluster, styled similar to reference image.

    Args:
        all_cluster_scores: [num_nodes, num_features] tensor of LRP scores.
        feature_names: List of feature names.
        cluster_tensor: Tensor assigning each node to a cluster.
        output_dir: Path to directory to save the heatmap.
        model_type: Model type string (e.g., 'ACGNN').
        net_type: Graph type string.
        score_threshold: Score threshold for filtering (used in filename).
        num_epochs: Number of training epochs (used in filename).
        feature_groups: Optional dict of {feature_name: group_name}.
        filename_prefix: Prefix for the plot file name.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy for plotting
    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()

    # Assuming scores_np has shape (7528, 129)
    # Feature names should match this number (129), not 2177
    feature_names = ['feature_{}'.format(i) for i in range(129)]  # This should match the columns of scores_np

    df = pd.DataFrame(scores_np, columns=feature_names)

    df['cluster'] = cluster_ids

    # Group by cluster and compute mean scores
    cluster_means = df.groupby('cluster').mean()
    cluster_means = cluster_means.sort_index()

    # Feature group coloring
    if feature_groups:
        group_colors = {
            'Mutation': '#a50026',
            'CNA': '#b35806',
            'DNA methylation': '#313695',
            'Gene expression': '#1b7837'
        }
        col_colors = [
            group_colors.get(feature_groups.get(f, ''), 'lightgray') for f in cluster_means.columns
        ]
    else:
        col_colors = None

    # Plotting setup
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(1, 10, width_ratios=[0.3] + [9]*9, wspace=0.05)

    # Cluster side color bar
    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(
        np.expand_dims(cluster_means.index, 1),
        cmap='tab20',
        cbar=False,
        ax=ax0,
        yticklabels=False,
        xticklabels=False
    )
    ax0.set_ylabel('Cluster')
    ax0.set_xticks([])

    # Main heatmap
    ax1 = fig.add_subplot(gs[1:])
    sns.heatmap(
        cluster_means,
        cmap='Greys',
        cbar_kws={"label": "Total feature contribution"},
        yticklabels=True,
        xticklabels=True,
        square=False,
        ax=ax1,
        linewidths=0.1,
        linecolor='lightgrey',
        ##col_colors=col_colors  # Add color by feature groups
    )
    ax1.set_xlabel("Features", fontsize=14)
    ax1.set_ylabel("Cluster", fontsize=14)
    ax1.set_title("Average LRP Feature Contribution per Cluster", fontsize=16)
    ax1.tick_params(axis='x', labelrotation=90)

    # Optional legend for feature group coloring
    if feature_groups:
        handles = [Patch(color=c, label=label) for label, c in group_colors.items()]
        ax1.legend(handles=handles, title="Feature groups", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Save plot
    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🔥 Saved LRP heatmap to: {plot_path}")

def plot_lrp_cluster_heatmap_ori(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    filename_prefix: str = "lrp_heatmap"
):
    """
    Plot a heatmap of LRP scores averaged over nodes in each cluster.

    Args:
        all_cluster_scores: [num_nodes, num_features] tensor of LRP scores.
        feature_names: List of feature names.
        cluster_tensor: Tensor assigning each node to a cluster.
        output_dir: Path to directory to save the heatmap.
        model_type: Model type string (e.g., 'ACGNN').
        net_type: Graph type string.
        score_threshold: Score threshold for filtering (used in filename).
        num_epochs: Number of training epochs (used in filename).
        filename_prefix: Prefix for the plot file name.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 🔹 Convert to numpy and pandas
    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()

    # 🔹 Fix mismatch between feature names and actual tensor shape
    scores_np = scores_np[:, :len(feature_names)]
    feature_names_trimmed = feature_names[:scores_np.shape[1]]

    df = pd.DataFrame(scores_np, columns=feature_names_trimmed)
    df['cluster'] = cluster_ids

    # 🔹 Average LRP scores per cluster
    cluster_means = df.groupby('cluster').mean(numeric_only=True)

    # 🔹 Plot heatmap
    plt.figure(figsize=(22, 10))
    sns.heatmap(
        cluster_means,
        cmap='Greys',
        cbar_kws={"label": "Total feature contribution"},
        yticklabels=True,
        xticklabels=True,
        square=False
    )
    plt.title("Average LRP Feature Contribution per Cluster", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Cluster", fontsize=14)
    plt.xticks(rotation=90)

    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🔥 Saved LRP heatmap to: {plot_path}")

def plot_lrp_cluster_heatmap_x(
    all_cluster_scores: torch.Tensor,
    feature_names: list,
    cluster_tensor: torch.Tensor,
    output_dir: str,
    model_type: str,
    net_type: str,
    score_threshold: float,
    num_epochs: int,
    filename_prefix: str = "lrp_heatmap"
):
    """
    Plot a heatmap of LRP scores averaged over nodes in each cluster.

    Args:
        all_cluster_scores: [num_nodes, num_features] tensor of LRP scores.
        feature_names: List of feature names.
        cluster_tensor: Tensor assigning each node to a cluster.
        output_dir: Path to directory to save the heatmap.
        model_type: Model type string (e.g., 'ACGNN').
        net_type: Graph type string.
        score_threshold: Score threshold for filtering (used in filename).
        num_epochs: Number of training epochs (used in filename).
        filename_prefix: Prefix for the plot file name.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy and pandas for plotting
    scores_np = all_cluster_scores.detach().cpu().numpy()
    cluster_ids = cluster_tensor.cpu().numpy()

    df = pd.DataFrame(scores_np, columns=feature_names)
    df['cluster'] = cluster_ids

    # Average LRP scores per cluster
    cluster_means = df.groupby('cluster').mean()

    # Plot
    plt.figure(figsize=(22, 10))
    sns.heatmap(
        cluster_means,
        cmap='Greys',
        cbar_kws={"label": "Total feature contribution"},
        yticklabels=True,
        xticklabels=True,
        square=False
    )
    plt.title("Average LRP Feature Contribution per Cluster", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Cluster", fontsize=14)
    plt.xticks(rotation=90)

    plot_path = os.path.join(
        output_dir,
        f"{filename_prefix}_{model_type}_{net_type}_threshold{score_threshold}_epo{num_epochs}.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🔥 Saved LRP heatmap to: {plot_path}")

def train_omics_name_pass(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    # ----- 🔹 CLUSTERING STEP -----
    print("Running KMeans clustering on node features...")
    kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes first 128 are used in embedding
    ##feature_names = gene_feature_names + ["degree"]
    
    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 'THCA', 
                    'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    bio_feat_names = []
    for cancer in cancer_types:
        for omics in omics_types:
            for i in range(16):
                bio_feat_names.append(f"{cancer}_{omics}_{i}")

    topo_feat_names = [f"topology_{i}" for i in range(1024)]

    feature_names = bio_feat_names + topo_feat_names + ["degree"] + gene_feature_names


    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)
    print("Generating feature importance plot...")

    cluster_tensor = graph.ndata['cluster']

    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)

        for c in range(12):
            node_indices = torch.nonzero(cluster_tensor == c).squeeze()
            if node_indices.numel() == 0:
                print(f"⚠️ No nodes in cluster {c}")
                continue

            cluster_scores = scores[node_indices]
            top_node_idx = node_indices[torch.argmax(cluster_scores)].item()
            gene_name = node_names[top_node_idx]

            print(f"Cluster {c} → Node {top_node_idx} ({gene_name})")

            plot_feature_importance(
                relevance_scores[top_node_idx],
                feature_names=feature_names,
                node_name=gene_name
            )

        
        ##high_confidence = torch.nonzero(scores > 0.9).squeeze()
        '''high_confidence = torch.nonzero(scores > args.score_threshold).squeeze()


        if high_confidence.numel() == 0:
            print("⚠️ No high-confidence predictions found. Try using a lower threshold.")
            return

        top_idx = high_confidence[0].item()
        relevance_vector = relevance_scores[top_idx]

        # Assuming 128 gene embeddings + 1 degree feature
        feature_names = [f"Gene_{i}" for i in range(128)] + ["degree"]
        plot_feature_importance(relevance_vector, feature_names, node_id=top_idx)'''

def train_gene_name_pass_lrp(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    # ----- 🔹 CLUSTERING STEP -----
    print("Running KMeans clustering on node features...")
    kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings.cpu().numpy()
    cluster_labels = kmeans.fit_predict(node_features)
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    # ✅ Get node and feature names
    node_names = list(nodes.keys())
    assert len(node_names) == graph.num_nodes(), "Node names length mismatch!"
    
    gene_feature_names = node_names[:embeddings.shape[1]]  # Assumes first 128 are used in embedding
    feature_names = gene_feature_names + ["degree"]

    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)
    print("Generating feature importance plot...")

    cluster_tensor = graph.ndata['cluster']

    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)

        for c in range(12):
            node_indices = torch.nonzero(cluster_tensor == c).squeeze()
            if node_indices.numel() == 0:
                print(f"⚠️ No nodes in cluster {c}")
                continue

            cluster_scores = scores[node_indices]
            top_node_idx = node_indices[torch.argmax(cluster_scores)].item()
            gene_name = node_names[top_node_idx]

            print(f"Cluster {c} → Node {top_node_idx} ({gene_name})")

            plot_feature_importance(
                relevance_scores[top_node_idx],
                feature_names=feature_names,
                node_name=gene_name
            )

        
        ##high_confidence = torch.nonzero(scores > 0.9).squeeze()
        '''high_confidence = torch.nonzero(scores > args.score_threshold).squeeze()


        if high_confidence.numel() == 0:
            print("⚠️ No high-confidence predictions found. Try using a lower threshold.")
            return

        top_idx = high_confidence[0].item()
        relevance_vector = relevance_scores[top_idx]

        # Assuming 128 gene embeddings + 1 degree feature
        feature_names = [f"Gene_{i}" for i in range(128)] + ["degree"]
        plot_feature_importance(relevance_vector, feature_names, node_id=top_idx)'''

'''
    # Add cluster labels to graph
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)
    print("Cluster labels assigned to graph.ndata['cluster']")


    print("Computing LRP relevance scores for entire graph...")
    relevance_scores = compute_lrp_scores(model, graph, features)

    # Assuming feature names
    feature_names = [f"Gene_{i}" for i in range(128)] + ["degree"]

    # Pick one node per cluster with a high prediction score
    print("Visualizing top node in each cluster...")
    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)
        cluster_tensor = graph.ndata['cluster']

        for c in range(12):
            node_indices = torch.nonzero(cluster_tensor == c).squeeze()
            if node_indices.numel() == 0:
                print(f"⚠️ No nodes in cluster {c}")
                continue

            # Select top scoring node in cluster
            cluster_scores = scores[node_indices]
            top_node_idx = node_indices[torch.argmax(cluster_scores)].item()

            print(f"Cluster {c} → Visualizing node {top_node_idx}")
            plot_feature_importance(relevance_scores[top_node_idx], feature_names, node_id=top_node_idx)'''

def train_ori_both_cluster_interaction_pass(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    
    in_feats = features.shape[1]

    graph.ndata['feat'].shape
    graph.ndata['degree'].shape
    features.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()
        print("Final feature shape before model input:", features.shape)

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    predicted_cancer_genes = [i for i, _ in ranking[:5000]]
    ##predicted_cancer_genes = [i for i, _ in ranking[:2000]]

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top 2000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 2000 nodes predicted above the threshold.")

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])
    # --- End LRP-related section ---


    '''output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top 2000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 2000 nodes predicted above the threshold.")

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())
    
    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)


    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')


    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)
    
    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        # Ensure idx is an integer
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])

        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])'''

    # === Load predicted genes with cluster info ===
    cluster_genes_df = pd.read_csv(output_path_cluster_genes)

    # === Load CPDB PPI file ===
    ppi_path = 'data/CPDB_ppi_0.99.csv'
    ppi_df = pd.read_csv(ppi_path)

    # === Match predicted genes with CPDB interactions ===
    match_records_partner1 = []
    match_records_partner2 = []

    for _, row in cluster_genes_df.iterrows():
        cluster_id = row['Cluster ID']
        gene = row['Gene Name']

        # Match when gene is partner1
        matches_p1 = ppi_df[ppi_df['partner1'] == gene]
        for _, match_row in matches_p1.iterrows():
            match_records_partner1.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

        # Match when gene is partner2
        matches_p2 = ppi_df[ppi_df['partner2'] == gene]
        for _, match_row in matches_p2.iterrows():
            match_records_partner2.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

    # === Save both partner1 and partner2 matched results ===
    output_dir = 'results/gene_prediction'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_p1 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner1_epo{args.num_epochs}.csv')
    output_path_p2 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner2_epo{args.num_epochs}.csv')

    # Save partner1 matches
    if match_records_partner1:
        with open(output_path_p1, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner1[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner1)
        print(f"✅ Saved partner1 matches to {output_path_p1}")
    else:
        print("⚠️ No partner1 matches found.")

    # Save partner2 matches
    if match_records_partner2:
        with open(output_path_p2, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner2[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner2)
        print(f"✅ Saved partner2 matches to {output_path_p2}")
    else:
        print("⚠️ No partner2 matches found.")


    # File paths
    match_output_path_p1 = output_path_p1
    match_output_path_p2 = output_path_p2

    # Load the match files
    df_p1 = pd.read_csv(match_output_path_p1)
    df_p2 = pd.read_csv(match_output_path_p2)

    # Create a helper column to form (partner1, partner2) tuple
    df_p1['pair'] = list(zip(df_p1['Matched Partner1'], df_p1['Matched Partner2']))
    df_p2['pair'] = list(zip(df_p2['Matched Partner1'], df_p2['Matched Partner2']))

    # Find common partner1–partner2 pairs
    common_pairs = set(df_p1['pair']) & set(df_p2['pair'])

    # Filter only those rows where the pair is common
    df_p1_common = df_p1[df_p1['pair'].isin(common_pairs)]
    df_p2_common = df_p2[df_p2['pair'].isin(common_pairs)]

    # Merge on partner pair
    merged = pd.merge(
        df_p1_common,
        df_p2_common,
        on=['Matched Partner1', 'Matched Partner2'],
        suffixes=('_p1', '_p2')
    )

    # Filter where matched genes from both partner1 and partner2 are from the same cluster
    same_cluster = merged[merged['Cluster ID_p1'] == merged['Cluster ID_p2']]

    # Optional: clean up or reorder columns
    output_columns = [
        'Cluster ID_p1', 'Gene Name_p1', 'Gene Name_p2',
        'Matched Partner1', 'Matched Partner2',
        'confidence_p1'
    ]
    same_cluster = same_cluster[output_columns]
    same_cluster = same_cluster.rename(columns={
        'Cluster ID_p1': 'Cluster ID',
        'Gene Name_p1': 'Matched Gene from Partner1',
        'Gene Name_p2': 'Matched Gene from Partner2',
        'confidence_p1': 'confidence'

    })

    # Save final filtered results
    output_path_common_same_cluster = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_common_same_cluster_epo{args.num_epochs}.csv')
    same_cluster.to_csv(output_path_common_same_cluster, index=False)

    print(f"✅ Saved common same-cluster matched gene pairs to {output_path_common_same_cluster}")

def train_lrp_pass(args):
    import time, os, psutil
    from tqdm import tqdm
    import torch.nn as nn
    import dgl

    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    assert graph.num_nodes() == embeddings.shape[0], f"Mismatch: {graph.num_nodes()} nodes vs {embeddings.shape[0]} embeddings"
    assert features.shape[0] == labels.shape[0], f"Mismatch: {features.shape[0]} features vs {labels.shape[0]} labels"

    print("Final feature shape before model input:", features.shape)

    in_feats = features.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    print("Training complete. Computing LRP scores...")
    relevance_scores = compute_lrp_scores(model, graph, features)

    print("Generating feature importance plot...")
    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits)
        high_confidence = torch.nonzero(scores > 0.9).squeeze()

        if high_confidence.numel() == 0:
            print("⚠️ No high-confidence predictions found. Try using a lower threshold.")
            return

        top_idx = high_confidence[0].item()
        relevance_vector = relevance_scores[top_idx]

        # Assuming 128 gene embeddings + 1 degree feature
        feature_names = [f"Gene_{i}" for i in range(128)] + ["degree"]
        plot_feature_importance(relevance_vector, feature_names, node_id=top_idx)

def train_(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()
        print("Final feature shape before model input:", features.shape)

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    predicted_cancer_genes = [i for i, _ in ranking[:5000]]
    ##predicted_cancer_genes = [i for i, _ in ranking[:2000]]

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    '''output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)'''

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top 2000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 2000 nodes predicted above the threshold.")

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())
    
    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)


    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')


    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)
    
    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        # Ensure idx is an integer
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])

        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])


    # === Load predicted genes ===
    '''
    cluster_genes_df = pd.read_csv(output_path_cluster_genes)

    # Group genes by cluster to form pairs within each cluster
    cluster_to_genes = cluster_genes_df.groupby("Cluster ID")["Gene Name"].apply(list).to_dict()

    # === Omics and cancer types ===
    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    match_records = []

    for omics in omics_types:
        for cancer in cancer_types:
            file_path = f'data/multiomics_{omics}/CPDB_PPI_{cancer}.csv'
            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path)

            # Ensure partner columns are strings for comparison
            df['partner1'] = df['partner1'].astype(str)
            df['partner2'] = df['partner2'].astype(str)

            for cluster_id, gene_list in cluster_to_genes.items():
                # Generate all unique gene pairs in the cluster
                gene_pairs = combinations(sorted(set(gene_list)), 2)

                for gene1, gene2 in gene_pairs:
                    # Match gene1-gene2 or gene2-gene1 as unordered pair
                    matches = df[((df['partner1'] == gene1) & (df['partner2'] == gene2)) |
                                ((df['partner1'] == gene2) & (df['partner2'] == gene1))]

                    for _, match_row in matches.iterrows():
                        match_records.append({
                            'Cluster ID': cluster_id,
                            'Gene1': gene1,
                            'Gene2': gene2,
                            'Matched Partner1': match_row['partner1'],
                            'Matched Partner2': match_row['partner2'],
                            'Omics Type': omics,
                            'Cancer Type': cancer,
                            'p_value': match_row.get('p_value', ''),
                            'confidence': match_row.get('confidence', ''),
                            'significance': match_row.get('significance', '')
                        })

    # === Save matched results ===
    match_output_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_omics_pair_matches_epo{args.num_epochs}.csv')
    os.makedirs(os.path.dirname(match_output_path), exist_ok=True)

    if match_records:
        with open(match_output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records[0].keys())
            writer.writeheader()
            writer.writerows(match_records)
        print(f"Saved matched gene pairs to {match_output_path}")
    else:
        print("No matching gene pairs found in omics interaction files.")
        '''


    
    '''cluster_genes_df = pd.read_csv(output_path_cluster_genes)

    # === Omics and cancer types ===
    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    match_records = []

    for omics in omics_types:
        for cancer in cancer_types:
            file_path = f'data/multiomics_{omics}/CPDB_PPI_{cancer}.csv'
            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path)

            for _, row in cluster_genes_df.iterrows():
                cluster_id = row['Cluster ID']
                gene = row['Gene Name']
                ##print('predicted gene ==================================== ', gene)

                matches = df[(df['partner1'] == gene) | (df['partner2'] == gene)]
                ##print('partner1 ==================================== ', df['partner1'])
                for _, match_row in matches.iterrows():
                    match_records.append({
                        'Cluster ID': cluster_id,
                        'Gene Name': gene,
                        'Matched Partner1': match_row['partner1'],
                        'Matched Partner2': match_row['partner2'],
                        'Omics Type': omics,
                        'Cancer Type': cancer,
                        'p_value': match_row.get('p_value', ''),
                        'confidence': match_row.get('confidence', ''),
                        'significance': match_row.get('significance', '')
                    })

    # === Save matched results ===
    match_output_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_omics_matches_epo{args.num_epochs}.csv')
    
    os.makedirs(os.path.dirname(match_output_path), exist_ok=True)

    if match_records:
        with open(match_output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records[0].keys())
            writer.writeheader()
            writer.writerows(match_records)
        print(f"Saved matched interactions to {match_output_path}")
    else:
        print("No matches found between predicted genes and omics files.")
    '''

    # === Load predicted genes with cluster info ===
    cluster_genes_df = pd.read_csv(output_path_cluster_genes)

    # === Load CPDB PPI file ===
    ppi_path = 'data/CPDB_ppi_0.99.csv'
    ppi_df = pd.read_csv(ppi_path)

    # === Match predicted genes with CPDB interactions ===
    match_records_partner1 = []
    match_records_partner2 = []

    for _, row in cluster_genes_df.iterrows():
        cluster_id = row['Cluster ID']
        gene = row['Gene Name']

        # Match when gene is partner1
        matches_p1 = ppi_df[ppi_df['partner1'] == gene]
        for _, match_row in matches_p1.iterrows():
            match_records_partner1.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

        # Match when gene is partner2
        matches_p2 = ppi_df[ppi_df['partner2'] == gene]
        for _, match_row in matches_p2.iterrows():
            match_records_partner2.append({
                'Cluster ID': cluster_id,
                'Gene Name': gene,
                'Matched Partner1': match_row['partner1'],
                'Matched Partner2': match_row['partner2'],
                'confidence': match_row.get('confidence', '')
            })

    # === Save both partner1 and partner2 matched results ===
    output_dir = 'results/gene_prediction'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_p1 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner1_epo{args.num_epochs}.csv')
    output_path_p2 = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_partner2_epo{args.num_epochs}.csv')

    # Save partner1 matches
    if match_records_partner1:
        with open(output_path_p1, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner1[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner1)
        print(f"✅ Saved partner1 matches to {output_path_p1}")
    else:
        print("⚠️ No partner1 matches found.")

    # Save partner2 matches
    if match_records_partner2:
        with open(output_path_p2, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records_partner2[0].keys())
            writer.writeheader()
            writer.writerows(match_records_partner2)
        print(f"✅ Saved partner2 matches to {output_path_p2}")
    else:
        print("⚠️ No partner2 matches found.")


    # File paths
    match_output_path_p1 = output_path_p1
    match_output_path_p2 = output_path_p2

    # Load the match files
    df_p1 = pd.read_csv(match_output_path_p1)
    df_p2 = pd.read_csv(match_output_path_p2)

    # Create a helper column to form (partner1, partner2) tuple
    df_p1['pair'] = list(zip(df_p1['Matched Partner1'], df_p1['Matched Partner2']))
    df_p2['pair'] = list(zip(df_p2['Matched Partner1'], df_p2['Matched Partner2']))

    # Find common partner1–partner2 pairs
    common_pairs = set(df_p1['pair']) & set(df_p2['pair'])

    # Filter only those rows where the pair is common
    df_p1_common = df_p1[df_p1['pair'].isin(common_pairs)]
    df_p2_common = df_p2[df_p2['pair'].isin(common_pairs)]

    # Merge on partner pair
    merged = pd.merge(
        df_p1_common,
        df_p2_common,
        on=['Matched Partner1', 'Matched Partner2'],
        suffixes=('_p1', '_p2')
    )

    # Filter where matched genes from both partner1 and partner2 are from the same cluster
    same_cluster = merged[merged['Cluster ID_p1'] == merged['Cluster ID_p2']]

    # Optional: clean up or reorder columns
    output_columns = [
        'Cluster ID_p1', 'Gene Name_p1', 'Gene Name_p2',
        'Matched Partner1', 'Matched Partner2',
        'confidence_p1'
    ]
    same_cluster = same_cluster[output_columns]
    same_cluster = same_cluster.rename(columns={
        'Cluster ID_p1': 'Cluster ID',
        'Gene Name_p1': 'Matched Gene from Partner1',
        'Gene Name_p2': 'Matched Gene from Partner2',
        'confidence_p1': 'confidence'

    })

    # Save final filtered results
    output_path_common_same_cluster = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_matches_ppi_common_same_cluster_epo{args.num_epochs}.csv')
    same_cluster.to_csv(output_path_common_same_cluster, index=False)

    print(f"✅ Saved common same-cluster matched gene pairs to {output_path_common_same_cluster}")

def train_(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    # Load graph data
    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    # Create DGL graph
    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    # Feature engineering: degree and embeddings
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    # Device and model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

        model.train()
        print("Final feature shape before model input:", features.shape)

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

    # Get ranking of unlabeled nodes
    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    predicted_cancer_genes = [i for i, _ in ranking[:5000]]

    # Visualize predicted cancer genes clustering
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    # Visualize attention scores (if model is GAT-based)
    if hasattr(model, 'attention_weights'):
        attention_weights = model.attention_weights.cpu().detach().numpy()
        visualize_attention_weights(attention_weights, node_names, output_dir, args.num_epochs)

    # Layer-wise Relevance Propagation (LRP)
    lrp_scores = compute_lrp(model, graph, features, logits)
    visualize_lrp(lrp_scores, node_names, output_dir, args.num_epochs)

    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')
    save_predicted_genes_by_cluster(predicted_cancer_genes, total_genes_per_cluster, output_path_cluster_genes)

def compute_lrp(model, graph, features, logits):
    """
    Compute Layer-wise Relevance Propagation (LRP) for the GNN model.
    Returns the relevance scores for each node.
    """
    # Perform LRP (this is just a placeholder; actual LRP implementation varies by model)
    # LRP can be computed using the backward pass and aggregating relevance scores.
    relevance = logits  # In actual LRP, this should be a layer-by-layer propagation.
    return relevance

def visualize_lrp(lrp_scores, node_names, output_dir, epoch):
    """
    Visualize the Layer-wise Relevance Propagation (LRP) scores.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(lrp_scores)), lrp_scores, align='center')
    plt.yticks(range(len(lrp_scores)), node_names)
    plt.xlabel('Relevance score')
    plt.title(f'LRP Scores for Gene Predictions (Epoch {epoch})')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'lrp_scores_epoch_{epoch}.png')
    plt.savefig(output_path)
    plt.close()

def visualize_attention_weights(attention_weights, node_names, output_dir, epoch):
    """
    Visualize attention weights (if the model is a GAT).
    """
    plt.figure(figsize=(10, 6))
    plt.hist(attention_weights.flatten(), bins=50, color='blue', alpha=0.7)
    plt.xlabel('Attention Weights')
    plt.ylabel('Frequency')
    plt.title(f'Attention Weights Distribution (Epoch {epoch})')
    output_path = os.path.join(output_dir, f'attention_weights_epoch_{epoch}.png')
    plt.savefig(output_path)
    plt.close()

def save_predicted_genes_by_cluster(predicted_cancer_genes, total_genes_per_cluster, output_path):
    """
    Save the predicted cancer genes by cluster.
    """
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for gene in predicted_cancer_genes:
        cluster_id = gene % len(total_genes_per_cluster)  # Assuming cluster_id is determined by gene's index
        cluster_to_genes[cluster_id].append(gene)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])
        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])

def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

def plot_feature_importance_id(relevance_vector, feature_names=None, node_name=None, top_k=20):
    relevance_vector = relevance_vector.detach().cpu().numpy()
    norm_scores = normalize(torch.tensor(relevance_vector)).numpy()
    
    top_indices = np.argsort(norm_scores)[-top_k:][::-1]
    top_scores = norm_scores[top_indices]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(relevance_vector))]
    top_names = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(10, 5))
    plt.barh(top_names[::-1], top_scores[::-1], color='skyblue')
    plt.xlabel("Relevance score")
    plt.title(f"Top {top_k} Feature Importances for Node {node_name}")
    plt.tight_layout()
    plt.show()

def plot_feature_importance_(relevance_vector, feature_names=None, node_name=None, top_k=20):
    relevance_vector = relevance_vector.detach().cpu().numpy()
    norm_scores = normalize(torch.tensor(relevance_vector)).numpy()

    top_indices = np.argsort(norm_scores)[-top_k:][::-1]
    top_scores = norm_scores[top_indices]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(relevance_vector))]
    top_names = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(10, 5))
    plt.barh(top_names[::-1], top_scores[::-1], color='skyblue')
    plt.xlabel("Relevance score")
    plt.title(f"Top {top_k} Feature Importances for {node_name}")  # 👈 Shows gene name in title
    plt.tight_layout()
    plt.show()

def train_lrp(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    print("graph.ndata['feat'].shape:", graph.ndata['feat'].shape)
    print("graph.ndata['degree'].shape:", graph.ndata['degree'].shape)

    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    # ⬇️ LRP Step (after evaluation)
    features.requires_grad = True
    lrp_scores = compute_lrp_scores(model, graph, features)

    # Optionally save or visualize relevance scores
    torch.save(lrp_scores.cpu(), os.path.join(output_dir, 'lrp_scores.pt'))
    print("Saved LRP relevance scores.")

def train_match(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    print("graph.ndata['feat'].shape:", graph.ndata['feat'].shape)
    print("graph.ndata['degree'].shape:", graph.ndata['degree'].shape)

    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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


    ##top_1000_genes = [i for i, _ in ranking[:1000]]

    ##################################################################
    
    # Rank nodes by scores for only non-labeled nodes
    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]  # Indices of non-labeled nodes
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)


    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)
    
    predicted_cancer_genes = [i for i in non_labeled_nodes if scores[i] >= args.score_threshold]
    # Run clustering and get resultsoutput_path_all_predicted_cancer_genes_per_cluster = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_all_predicted_cancer_genes_per_cluster_epo{args.num_epochs}_2048.csv')
    '''output_path_genes_clusters = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_all_predicted_cancer_genes_per_cluster_epo{args.num_epochs}_2048.png')
    output_path_predicted_cancer_genes = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_predicted_cancer_genes_epo{args.num_epochs}_2048.csv')
    cluster_labels, predicted_cancer_genes_count, total_genes_per_cluster = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, output_path_predicted_cancer_genes)#, num_clusters=5)
    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )
    '''
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )


    ##======================================================================================================================    
    # Clustering Step (using KMeans as an example)
    kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings  # You can also include degree or other features
    cluster_labels = kmeans.fit_predict(node_features)  # Cluster the nodes based on their features

    # Store the cluster labels in the graph
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)

    with open('data/796_drivers.txt') as f:
        ground_truth_cancer_genes = set(line.strip() for line in f)

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())    

    '''output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)'''

    # === PCG Plotting ===
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )
    
    # === KCG Plotting ===
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')#('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    # === Degree Summary ===
    if predicted_cancer_genes:
        avg_degree = graph.ndata['degree'][predicted_cancer_genes].float().mean().item()
        print(f"Average degree of predicted cancer genes above threshold: {avg_degree:.2f}")
    else:
        print("No predicted cancer genes above the threshold.")

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)


    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')

    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes:
        # Ensure idx is an integer
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])

        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])

    # === Load predicted genes by cluster ===
    cluster_genes_df = pd.read_csv(output_path_cluster_genes)

    # === Omics and cancer types ===
    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    match_records = []

    for omics in omics_types:
        for cancer in cancer_types:
            file_path = f'../../../gat/data/multiomics_{omics}/CPDB_PPI_{cancer}.csv'
            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path)

            for _, row in cluster_genes_df.iterrows():
                cluster_id = row['Cluster ID']
                gene = row['Gene Name']
                print('predicted gene ==================================== ', gene)

                matches = df[(df['partner1'] == gene) | (df['partner2'] == gene)]
                print('partner1 ==================================== ', df['partner1'])
                for _, match_row in matches.iterrows():
                    match_records.append({
                        'Cluster ID': cluster_id,
                        'Gene Name': gene,
                        'Matched Partner1': match_row['partner1'],
                        'Matched Partner2': match_row['partner2'],
                        'Omics Type': omics,
                        'Cancer Type': cancer,
                        'p_value': match_row.get('p_value', ''),
                        'confidence': match_row.get('confidence', ''),
                        'significance': match_row.get('significance', '')
                    })

    # === Save matched results ===
    match_output_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_omics_matches_epo{args.num_epochs}.png')
    
    os.makedirs(os.path.dirname(match_output_path), exist_ok=True)

    if match_records:
        with open(match_output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records[0].keys())
            writer.writeheader()
            writer.writerows(match_records)
        print(f"Saved matched interactions to {match_output_path}")
    else:
        print("No matches found between predicted genes and omics files.")

def train_kcg_pcg_x(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    ##non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]

    ##top_predicted_genes = [i for i, _ in non_labeled_scores if scores[i] >= args.score_threshold]
    # Include index, name, and score
    non_labeled_scores = [(i, node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    # Filter using index and score
    top_predicted_genes = [i for i, _, score in non_labeled_scores if score >= args.score_threshold]


    # Clustering Step
    kmeans = KMeans(n_clusters=12, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)

    cluster_labels_np = cluster_labels
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())

    # KCGs
    with open('data/796_drivers.txt') as f:
        ground_truth_cancer_genes = set(line.strip() for line in f)
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)

    # PCGs
    pcg_nodes = top_predicted_genes
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes.csv')

    cluster_labels_tensor = torch.tensor(cluster_labels, dtype=torch.long)
    cluster_labels_dict, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, top_predicted_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels_tensor, output_path=output_path_predicted
    )

    # KCG plot
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')##('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels_tensor, output_path=output_path_kcg
    )

    if top_predicted_genes:
        avg_degree = graph.ndata['degree'][top_predicted_genes].float().mean().item()
        print(f"Average degree of predicted nodes above threshold: {avg_degree:.2f}")
    else:
        print("No nodes predicted above the threshold.")

def train_0_0_cluster(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1  # Mask for labeled nodes
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)  # Mask for testing on all nodes
    graph = dgl.add_self_loop(graph)
    
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)  # Reshape to (N,1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0
        
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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)
    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)
    
    predicted_cancer_genes = [i for i in non_labeled_nodes if scores[i] >= args.score_threshold]
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_clusters_epo{args.num_epochs}.png')
    output_path_predicted_cancer_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_cancer_genes.csv')
    
    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, 
        output_path_genes_clusters
    )
    
    output_path_legend = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_legend_epo{args.num_epochs}.png')

    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)
            
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    output_path_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_ground_truth_epo{args.num_epochs}.png')
    
    plot_pcg_cancer_genes_ground_truth(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, ground_truth_cancer_genes=ground_truth_cancer_genes,
        node_names=node_names, cluster_labels=cluster_labels, output_path=output_path_percent,
        ##cluster_colors=CLUSTER_COLORS
    )

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted,
        ##cluster_colors=CLUSTER_COLORS
    )

    graph.ndata['interactions'] = torch.zeros(graph.num_nodes(), dtype=torch.float, device=graph.device)
    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    graph.ndata['interactions'][predicted_nodes] = graph.ndata['degree'][predicted_nodes].squeeze()

    cluster_colors = [
        '#0077B6', '#0000FF', '#00B4D8', '#48EAC4', '#F1C0E8', 
        '#B9FBC0', '#32CD32', '#FFD700', '#8A2BE2', '#E377C2', 
        '#8EECF5', '#A3C4F3'  
    ]
    
    if predicted_nodes:
        print(f"Average degree of predicted nodes: {graph.ndata['degree'][predicted_nodes].mean().item():.2f}")
        cluster_ids = graph.ndata['cluster'][predicted_nodes].cpu().numpy()
        interactions = graph.ndata['interactions'][predicted_nodes].cpu().numpy()
        data = pd.DataFrame({"Cluster": cluster_ids, "Interactions": interactions})
        output_path_interactions = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
        ##plot_interactions_with_kcgs(data, output_path_interactions, CLUSTER_COLORS)
        plot_interactions_with_kcgs(data, output_path_interactions)#, cluster_colors)
    else:
        print("No nodes predicted above the threshold.")

def train_(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1  # Mask for labeled nodes
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)  # Mask for testing on all nodes
    graph = dgl.add_self_loop(graph)
    
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)  # Reshape to (N,1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0
        
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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)
    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)
    
    predicted_cancer_genes = [i for i in non_labeled_nodes if scores[i] >= args.score_threshold]
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_clusters_epo{args.num_epochs}.png')
    output_path_predicted_cancer_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_cancer_genes.csv')
    
    '''cluster_labels, pred_counts, total_genes_per_cluster = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, output_path_predicted_cancer_genes,
        cluster_colors=CLUSTER_COLORS
    )'''

    '''CLUSTER_COLORS = {
        0: '#0077B6',  1: '#0000FF',  2: '#00B4D8',  3: '#48EAC4',
        4: '#F1C0E8',  5: '#B9FBC0',  6: '#32CD32',  7: '#FFD700',
        8: '#8A2BE2',  9: '#E377C2', 10: '#8EECF5', 11: '#A3C4F3'
    }
    cluster_colors = [CLUSTER_COLORS[i] for i in range(12)]'''
    
    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, 
        output_path_genes_clusters
    )
    
    output_path_legend = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_legend_epo{args.num_epochs}.png')

    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)
            
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    output_path_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_ground_truth_epo{args.num_epochs}.png')
    
    plot_pcg_cancer_genes_ground_truth(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, ground_truth_cancer_genes=ground_truth_cancer_genes,
        node_names=node_names, cluster_labels=cluster_labels, output_path=output_path_percent,
        ##cluster_colors=CLUSTER_COLORS
    )

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted,
        ##cluster_colors=CLUSTER_COLORS
    )

    graph.ndata['interactions'] = torch.zeros(graph.num_nodes(), dtype=torch.float, device=graph.device)
    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    graph.ndata['interactions'][predicted_nodes] = graph.ndata['degree'][predicted_nodes].squeeze()

    cluster_colors = [
        '#0077B6', '#0000FF', '#00B4D8', '#48EAC4', '#F1C0E8', 
        '#B9FBC0', '#32CD32', '#FFD700', '#8A2BE2', '#E377C2', 
        '#8EECF5', '#A3C4F3'  
    ]
    
    if predicted_nodes:
        print(f"Average degree of predicted nodes: {graph.ndata['degree'][predicted_nodes].mean().item():.2f}")
        cluster_ids = graph.ndata['cluster'][predicted_nodes].cpu().numpy()
        interactions = graph.ndata['interactions'][predicted_nodes].cpu().numpy()
        data = pd.DataFrame({"Cluster": cluster_ids, "Interactions": interactions})
        output_path_interactions = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
        ##plot_interactions_with_kcgs(data, output_path_interactions, CLUSTER_COLORS)
        plot_interactions_with_kcgs(data, output_path_interactions)#, cluster_colors)
    else:
        print("No nodes predicted above the threshold.")

def train_kcg_interaction(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    top_1000_genes = [i for i, _ in ranking[:1000]]

    # Clustering Step (using KMeans as an example)
    kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings  # You can also include degree or other features
    cluster_labels = kmeans.fit_predict(node_features)  # Cluster the nodes based on their features

    # Store the cluster labels in the graph
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)

    # Identify KCGs (ground truth known cancer genes)
    with open('data/796_drivers.txt') as f:
        ground_truth_cancer_genes = set(line.strip() for line in f)

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})


    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    # Plot PCG interactions
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

def train_all_predicted_cancer_genes(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    predicted_cancer_genes = [i for i, _ in ranking]

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_all_predicted_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_cancer_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_all_predicted_cluster_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    all_predicted_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if all_predicted_gene_indices:
        ##avg_degree = graph.ndata['degree'][all_predicted_gene_indices].float().mean().item()
        name_to_index = {name: idx for idx, name in enumerate(node_names)}
        all_predicted_gene_indices = [name_to_index[name] for name in all_predicted_gene_indices if name in name_to_index]

        avg_degree = graph.ndata['degree'][all_predicted_gene_indices].float().mean().item() if all_predicted_gene_indices else 0

        print(f"Average degree of all predicted nodes: {avg_degree:.2f}")
    else:
        print("No nodes predicted above the threshold.")

def train_(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    top_1000_genes = [i for i, _ in ranking[:1000]]

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, top_1000_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/intogen_23444.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_pcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_pcg
    )

    top_1000_gene_indices = [name_to_index[name] for name in top_1000_genes if name in name_to_index]

    if top_1000_gene_indices:
        avg_degree = graph.ndata['degree'][top_1000_gene_indices].float().mean().item()
        print(f"Average degree of top 1000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 1000 nodes predicted above the threshold.")


    # Clustering Step (using KMeans as an example)
    kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings  # You can also include degree or other features
    cluster_labels = kmeans.fit_predict(node_features)  # Cluster the nodes based on their features

    # Store the cluster labels in the graph
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)

    # Identify KCGs (ground truth known cancer genes)
    '''with open('data/796_drivers.txt') as f:
        ground_truth_cancer_genes = set(line.strip() for line in f)'''

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})


    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    # Plot PCG interactions
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

def train_kcg_pcg_pas(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    top_1000_genes = [i for i, _ in ranking[:1000]]

    # Clustering Step (using KMeans as an example)
    kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings  # You can also include degree or other features
    cluster_labels = kmeans.fit_predict(node_features)  # Cluster the nodes based on their features

    # Store the cluster labels in the graph
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)

# Identify KCGs (ground truth known cancer genes)
    with open('data/796_drivers.txt') as f:
        ground_truth_cancer_genes = set(line.strip() for line in f)

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())

    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})


    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    # Plot PCG interactions
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)

def train_match_pass(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    predicted_cancer_genes = [i for i, _ in ranking[:2000]]
    ##predicted_cancer_genes = [i for i, _ in ranking[:2000]]

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    '''output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)'''

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top 2000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 2000 nodes predicted above the threshold.")

    # Cluster labels and degrees (for plotting)
    cluster_labels_np = graph.ndata['cluster'].cpu().numpy()
    degrees_np = graph.ndata['degree'].squeeze().cpu().numpy()
    node_names = list(nodes.keys())
    
    # Get KCG interaction data
    kcg_nodes = [i for i, name in enumerate(node_names) if name in ground_truth_cancer_genes]
    kcg_clusters = cluster_labels_np[kcg_nodes]
    kcg_degrees = degrees_np[kcg_nodes]
    kcg_data = pd.DataFrame({"Cluster": kcg_clusters, "Interactions": kcg_degrees})

    # Get PCG interaction data
    pcg_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    pcg_clusters = cluster_labels_np[pcg_nodes]
    pcg_degrees = degrees_np[pcg_nodes]
    pcg_data = pd.DataFrame({"Cluster": pcg_clusters, "Interactions": pcg_degrees})

    output_path_interactions_kcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_kcgs(kcg_data, output_path_interactions_kcgs)
    
    output_path_interactions_pcgs = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcgs_interaction_epo{args.num_epochs}.png')
    plot_interactions_with_pcgs(pcg_data, output_path_interactions_pcgs)


    # Save predicted genes by cluster
    output_path_cluster_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_genes_by_cluster_epo{args.num_epochs}.csv')


    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_cancer_genes_indices = set(name_to_index[name] for name in predicted_cancer_genes if name in name_to_index)
    
    # Reverse map: cluster_id → list of predicted gene names in that cluster
    cluster_to_genes = {cluster_id: [] for cluster_id in total_genes_per_cluster}
    for idx in predicted_cancer_genes_indices:
        # Ensure idx is an integer
        cluster_id = cluster_labels[idx.item()] if isinstance(idx, torch.Tensor) else cluster_labels[idx]
        gene_name = node_names[idx.item()] if isinstance(idx, torch.Tensor) else node_names[idx]
        cluster_to_genes[cluster_id].append(gene_name)

    # Write to CSV: columns = Cluster ID, Gene Name
    with open(output_path_cluster_genes, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cluster ID', 'Gene Name'])

        for cluster_id, genes in cluster_to_genes.items():
            for gene in genes:
                writer.writerow([cluster_id, gene])

    # === Load predicted genes by cluster ===
    cluster_genes_df = pd.read_csv(output_path_cluster_genes)

    # === Omics and cancer types ===
    omics_types = ['mf', 'cna', 'ge', 'meth']
    cancer_types = ['BRCA', 'KIRC', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    match_records = []

    for omics in omics_types:
        for cancer in cancer_types:
            file_path = f'data/multiomics_{omics}/CPDB_PPI_{cancer}.csv'
            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path)

            for _, row in cluster_genes_df.iterrows():
                cluster_id = row['Cluster ID']
                gene = row['Gene Name']
                ##print('predicted gene ==================================== ', gene)

                matches = df[(df['partner1'] == gene) | (df['partner2'] == gene)]
                ##print('partner1 ==================================== ', df['partner1'])
                for _, match_row in matches.iterrows():
                    match_records.append({
                        'Cluster ID': cluster_id,
                        'Gene Name': gene,
                        'Matched Partner1': match_row['partner1'],
                        'Matched Partner2': match_row['partner2'],
                        'Omics Type': omics,
                        'Cancer Type': cancer,
                        'p_value': match_row.get('p_value', ''),
                        'confidence': match_row.get('confidence', ''),
                        'significance': match_row.get('significance', '')
                    })

    # === Save matched results ===
    match_output_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_omics_matches_epo{args.num_epochs}.csv')
    
    os.makedirs(os.path.dirname(match_output_path), exist_ok=True)

    if match_records:
        with open(match_output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=match_records[0].keys())
            writer.writeheader()
            writer.writerows(match_records)
        print(f"Saved matched interactions to {match_output_path}")
    else:
        print("No matches found between predicted genes and omics files.")
        
def train_pas(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    top_1000_genes = [i for i, _ in ranking[:1000]]

    # Clustering Step (using KMeans as an example)
    kmeans = KMeans(n_clusters=12, random_state=42)
    node_features = embeddings  # You can also include degree or other features
    cluster_labels = kmeans.fit_predict(node_features)  # Cluster the nodes based on their features

    # Store the cluster labels in the graph
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)

    # Add plotting interaction step
    graph.ndata['interactions'] = torch.zeros(graph.num_nodes(), dtype=torch.float, device=graph.device)
    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    graph.ndata['interactions'][predicted_nodes] = graph.ndata['degree'][predicted_nodes].squeeze()

    if predicted_nodes:
        print(f"Average degree of predicted nodes: {graph.ndata['degree'][predicted_nodes].mean().item():.2f}")
        cluster_ids = graph.ndata['cluster'][predicted_nodes].cpu().numpy()
        interactions = graph.ndata['interactions'][predicted_nodes].cpu().numpy()
        data = pd.DataFrame({"Cluster": cluster_ids, "Interactions": interactions})
        output_path_interactions = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
        plot_interactions_with_kcgs(data, output_path_interactions)
    else:
        print("No nodes predicted above the threshold.")

def train_cluster_pass(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    predicted_cancer_genes = [i for i, _ in ranking[:2000]]

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if top_gene_indices:
        avg_degree = graph.ndata['degree'][top_gene_indices].float().mean().item()
        print(f"Average degree of top 2000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 2000 nodes predicted above the threshold.")

def train_pa(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    top_1000_genes = [i for i, _ in ranking[:1000]]

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, top_1000_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    # PCG counts (from pred_counts earlier)
    pcg_counts = pred_counts  # already computed above

    # Plot percentages
    '''output_path_kcg_pcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_pcg_percent_epo{args.num_epochs}.png')
    plot_kcg_pcg_percentages(
        clusters=total_genes_per_cluster.keys(),
        kcg_counts=kcg_counts,
        pcg_counts=pcg_counts,
        total_genes_per_cluster=total_genes_per_cluster,
        output_path=output_path_kcg_pcg
    )'''


    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    '''output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )'''

    # Plot KCG percentage
    output_path_kcg_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_epo{args.num_epochs}.png')
    plot_percentage_per_cluster(
        counts=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster,
        clusters=total_genes_per_cluster.keys(),
        label="KCGs",
        color="orange",
        output_path=output_path_kcg_percent
    )

    # Plot PCG percentage
    output_path_pcg_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_epo{args.num_epochs}.png')
    plot_percentage_per_cluster(
        counts=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster,
        clusters=total_genes_per_cluster.keys(),
        label="PCGs",
        color="skyblue",
        output_path=output_path_pcg_percent
    )


    top_1000_gene_indices = [name_to_index[name] for name in top_1000_genes if name in name_to_index]

    if top_1000_gene_indices:
        avg_degree = graph.ndata['degree'][top_1000_gene_indices].float().mean().item()
        print(f"Average degree of top 1000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 1000 nodes predicted above the threshold.")

def train_cluster_pass(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0

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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)

    top_1000_genes = [i for i, _ in ranking[:1000]]

    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_genes.csv')

    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, top_1000_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_genes_per_cluster:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_1000_gene_indices = [name_to_index[name] for name in top_1000_genes if name in name_to_index]

    if top_1000_gene_indices:
        avg_degree = graph.ndata['degree'][top_1000_gene_indices].float().mean().item()
        print(f"Average degree of top 1000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 1000 nodes predicted above the threshold.")

def plot_percentage_per_cluster(counts, total_genes_per_cluster, clusters, label, color, output_path):
    """
    Plot the percentage of a specific gene type (KCG or PCG) per cluster.

    Args:
        counts (dict): Count of genes (KCG or PCG) per cluster.
        total_genes_per_cluster (dict): Total number of genes per cluster.
        clusters (iterable): Cluster IDs.
        label (str): Label for the bar (e.g., "KCGs" or "PCGs").
        color (str): Color of the bars.
        output_path (str): Where to save the plot.
    """
    clusters = list(clusters)
    percentages = [100 * counts.get(c, 0) / total_genes_per_cluster[c] for c in clusters]

    plt.figure(figsize=(12, 6))
    plt.bar(clusters, percentages, color=color)
    plt.xticks(clusters, [f"Cluster {c}" for c in clusters], rotation=90)
    plt.ylabel("Percentage (%)")
    plt.title(f"Percentage of {label} per Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"{label} percentage plot saved to {output_path}")

def plot_kcg_cancer_genes_(clusters, kcg_per_cluster, total_genes_per_cluster, node_names, cluster_labels, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    cluster_list = list(clusters)
    total_genes_per_cluster = [total_genes_per_cluster[c] for c in cluster_list]
    kcg_counts = [kcg_per_cluster.get(c, 0) for c in cluster_list]

    x = np.arange(len(cluster_list))
    width = 0.4

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, total_genes_per_cluster, width, label='Total Genes', color='lightgray')
    plt.bar(x + width / 2, kcg_counts, width, label='Known Cancer Genes', color='firebrick')

    plt.xlabel('Cluster')
    plt.ylabel('Number of Genes')
    ##plt.title('Known Cancer Genes Distribution by Cluster')
    plt.ylim(0, 0.4)
    plt.xticks(ticks=x, labels=[f'Cluster {c}' for c in cluster_list], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_kcg_cancer_genes_ori(clusters, kcg_count, total_genes_per_cluster, node_names, cluster_labels, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

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
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Formatting
    ##plt.xlabel("Cluster ID", fontsize=16)
    plt.ylabel("Percent of KCGs", fontsize=16)
    plt.xticks(cluster_ids)
    plt.ylim(0, max(proportions) + 0.1)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pcg_cancer_genes_(clusters, predicted_cancer_genes_count, total_genes_per_cluster, node_names, cluster_labels, output_path):
    """
    Plots the percentage of predicted cancer genes per cluster using a fixed color scheme.
    """
    # Convert to NumPy arrays for safe division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_genes_per_cluster = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes per cluster
    percent_predicted = np.divide(predicted_counts, total_genes_per_cluster, where=total_genes_per_cluster > 0)  

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, percent_predicted, 
                   color=[CLUSTER_COLORS.get(c, '#000000') for c in clusters],  # Default to black if cluster not in dictionary
                   edgecolor='black')

    # Add value labels on top of each bar (number of predicted cancer genes per cluster)
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        count = predicted_cancer_genes_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and formatting
    plt.ylabel("Percent of PCGs", fontsize=16)
    plt.xticks(clusters)  
    plt.ylim(0, max(percent_predicted) + 0.1)  

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def train_top_1000(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1  # Mask for labeled nodes
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)  # Mask for testing on all nodes
    graph = dgl.add_self_loop(graph)
    
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)  # Reshape to (N,1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0
        
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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)
    
    # Select top 1000 predicted cancer genes
    top_1000_genes = [i for i, _ in ranking[:1000]]
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_genes.csv')
    
    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, top_1000_genes, node_names, output_path_genes_clusters
    )
    
    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)
    
    '''output_path_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_cluster_ground_truth_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes_ground_truth(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_percent
    )'''
    
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )
    
    # Mapping gene names to their corresponding indices in the graph
    name_to_index = {name: idx for idx, name in enumerate(node_names)}

    # Convert top_1000_genes (which likely contains names) to their indices
    top_1000_gene_indices = [name_to_index[name] for name in top_1000_genes if name in name_to_index]

    if top_1000_gene_indices:
        avg_degree = graph.ndata['degree'][top_1000_gene_indices].float().mean().item()
        print(f"Average degree of top 1000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 1000 nodes predicted above the threshold.")

def train_before_top_1000(args):
    epoch_times = []
    cpu_usages = []
    gpu_usages = []

    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1  # Mask for labeled nodes
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)  # Mask for testing on all nodes
    graph = dgl.add_self_loop(graph)
    
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)  # Reshape to (N,1)
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    graph = graph.to(device)
    features, labels = features.to(device), labels.to(device).float()
    train_mask, test_mask = graph.ndata['train_mask'].to(device), graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0
        
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

    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)
    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)
    
    predicted_cancer_genes = [i for i in non_labeled_nodes if scores[i] >= args.score_threshold]
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_clusters_epo{args.num_epochs}.png')
    output_path_predicted_cancer_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_cancer_genes.csv')
    
    cluster_labels, total_genes_per_cluster, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, 
        output_path_genes_clusters
    )
    
    output_path_legend = os.path.join(output_dir, f'legend.png')

    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)
            
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    output_path_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_ground_truth_epo{args.num_epochs}.png')
    
    plot_pcg_cancer_genes_ground_truth(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, ground_truth_cancer_genes=ground_truth_cancer_genes,
        node_names=node_names, cluster_labels=cluster_labels, output_path=output_path_percent,
        ##cluster_colors=CLUSTER_COLORS
    )

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_genes_per_cluster, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted,
        ##cluster_colors=CLUSTER_COLORS
    )

    graph.ndata['interactions'] = torch.zeros(graph.num_nodes(), dtype=torch.float, device=graph.device)
    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    graph.ndata['interactions'][predicted_nodes] = graph.ndata['degree'][predicted_nodes].squeeze()

    if predicted_nodes:
        print(f"Average degree of predicted nodes: {graph.ndata['degree'][predicted_nodes].mean().item():.2f}")
        cluster_ids = graph.ndata['cluster'][predicted_nodes].cpu().numpy()
        interactions = graph.ndata['interactions'][predicted_nodes].cpu().numpy()
        data = pd.DataFrame({"Cluster": cluster_ids, "Interactions": interactions})
        output_path_interactions = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}.png')
        ##plot_interactions_with_kcgs(data, output_path_interactions, CLUSTER_COLORS)
        plot_interactions_with_kcgs(data, output_path_interactions)#, cluster_colors)
    else:
        print("No nodes predicted above the threshold.")

def plot_kcg_pcg_percentages(clusters, kcg_counts, pcg_counts, total_genes_per_cluster, output_path):
    """
    Plots a bar chart showing the percentage of KCGs and PCGs in each cluster.

    Args:
        clusters (iterable): Cluster IDs.
        kcg_counts (dict): Count of known cancer genes per cluster.
        pcg_counts (dict): Count of predicted cancer genes per cluster.
        total_genes_per_cluster (dict): Total genes per cluster.
        output_path (str): Path to save the figure.
    """
    clusters = list(clusters)
    kcg_percents = [100 * kcg_counts.get(c, 0) / total_genes_per_cluster[c] for c in clusters]
    pcg_percents = [100 * pcg_counts.get(c, 0) / total_genes_per_cluster[c] for c in clusters]

    x = np.arange(len(clusters))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, pcg_percents, width, label='Predicted Cancer Genes (PCGs)', color='skyblue')
    plt.bar(x + width/2, kcg_percents, width, label='Known Cancer Genes (KCGs)', color='orange')

    plt.xticks(x, [f"Cluster {c}" for c in clusters], rotation=90)
    plt.ylabel("Percentage (%)")
    plt.title("Percentage of PCGs and KCGs in Each Cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"KCG/PCG percentage plot saved to {output_path}")


def save_cluster_legend_0_11(output_path_legend, cluster_colors, num_clusters=12):
    """
    Creates and saves a separate legend image for cluster colors in two rows.

    Args:
        output_path_legend (str): Path to save the legend image.
        cluster_colors (list): List of colors for each cluster.
        num_clusters (int): Number of clusters.
    """
    fig, ax = plt.subplots(figsize=(10, 2))  # Taller figure for two rows

    # Create legend handles (square patches)
    legend_patches = [mpatches.Patch(color=cluster_colors[i], label=f"Cluster {i}") 
                      for i in range(num_clusters)]

    # Display legend with two rows (ncol = half of num_clusters)
    ax.legend(handles=legend_patches, loc='center', ncol=num_clusters // 2, 
              frameon=False, fontsize=14)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Save legend image
    plt.savefig(output_path_legend, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Legend saved to {output_path_legend}")

def save_cluster_legend_one_row(output_path_legend, cluster_colors, num_clusters=12):
    """
    Creates and saves a separate legend image for cluster colors.

    Args:
        output_path_legend (str): Path to save the legend image.
        cluster_colors (list): List of colors for each cluster.
        num_clusters (int): Number of clusters.
    """
    fig, ax = plt.subplots(figsize=(10, 1))  # Small figure for legend

    # Create legend handles (square patches)
    legend_patches = [mpatches.Patch(color=cluster_colors[i], label=f"Cluster {i}") 
                      for i in range(num_clusters)]

    # Create legend with fontsize=14
    ax.legend(handles=legend_patches, loc='center', ncol=num_clusters, 
              frameon=False, fontsize=14)  # ✅ Set font size

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Save legend image
    plt.savefig(output_path_legend, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Legend saved to {output_path_legend}")

def cluster_and_visualize_predicted_genes_ori(graph, predicted_cancer_genes, node_names, 
                                          output_path_genes_clusters, num_clusters=12):
    """
    Clusters gene embeddings into groups using KMeans, visualizes them with t-SNE, 
    and marks predicted cancer genes with red circles (half the size of non-cancer dots).

    Returns:
        cluster_labels (np.ndarray): Cluster assignments for each gene.
        total_genes_per_cluster (dict): Total number of genes per cluster.
        pred_counts (dict): Number of predicted cancer genes per cluster.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Store cluster labels in graph
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)

    # Calculate the total number of genes per cluster
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(num_clusters)}

    # Calculate the number of predicted cancer genes per cluster
    pred_counts = {i: 0 for i in range(num_clusters)}
    '''for gene_idx in predicted_cancer_genes:
        print(f"gene_idx type: {type(gene_idx)}, value: {gene_idx}")
        print(f"cluster_labels type: {type(cluster_labels)}, length: {len(cluster_labels)}")

        cluster_id = cluster_labels[gene_idx]
        pred_counts[cluster_id] += 1'''

    # Convert gene names to their corresponding indices
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    predicted_cancer_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    # Ensure valid indices before using them
    for gene_idx in predicted_cancer_gene_indices:
        if 0 <= gene_idx < len(cluster_labels):  
            cluster_id = cluster_labels[gene_idx]
            pred_counts[cluster_id] += 1
        else:
            print(f"Skipping invalid index: {gene_idx}")


    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))

    # Plot clusters (Non-cancer genes)
    non_cancer_dot_size = 100  # Default dot size
    red_circle_size = non_cancer_dot_size / 2  # Half the size

    for cluster_id in range(num_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=CLUSTER_COLORS[cluster_id], 
                    edgecolor='k', s=non_cancer_dot_size, alpha=0.8)

    # Mark predicted cancer genes with red circles (⚪, half the size)
    for gene_idx in predicted_cancer_gene_indices:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=red_circle_size, linewidths=2)


    # Labels and title
    ##plt.title("Gene clustering with predicted cancer genes")
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)

    # Save plot
    plt.savefig(output_path_genes_clusters, bbox_inches="tight")  # Ensure proper cropping
    plt.close()

    print(f"Cluster visualization saved to {output_path_genes_clusters}")

    return cluster_labels, total_genes_per_cluster, pred_counts

def cluster_and_visualize_predicted_genes_with_legend(graph, predicted_cancer_genes, node_names, 
                                          output_path_genes_clusters, num_clusters=12):
    """
    Clusters gene embeddings into groups using KMeans, visualizes them with t-SNE, 
    and marks predicted cancer genes with red circles (half the size of non-cancer dots).
    
    Returns:
        cluster_labels (np.ndarray): Cluster assignments for each gene.
        total_genes_per_cluster (dict): Total number of genes per cluster.
        pred_counts (dict): Number of predicted cancer genes per cluster.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()
    
    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Store cluster labels in graph
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)
    
    # Calculate the total number of genes per cluster
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(num_clusters)}
    
    # Calculate the number of predicted cancer genes per cluster
    pred_counts = {i: 0 for i in range(num_clusters)}
    for gene_idx in predicted_cancer_genes:
        cluster_id = cluster_labels[gene_idx]
        pred_counts[cluster_id] += 1
    
    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    # Plot clusters (Non-cancer genes)
    non_cancer_dot_size = 100  # Default dot size
    red_circle_size = non_cancer_dot_size / 2  # Half the size
    
    for cluster_id in range(num_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=CLUSTER_COLORS[cluster_id], 
                    edgecolor='k', s=non_cancer_dot_size, alpha=0.8, 
                    label=f"Cluster {cluster_id}")
    
    # Mark predicted cancer genes with red circles (⚪, half the size)
    for gene_idx in predicted_cancer_genes:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=red_circle_size, linewidths=2)
    
    # Legend and labels
    plt.title("Gene clustering with predicted cancer genes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # Adjust layout to fit legend outside
    plt.subplots_adjust(bottom=0.25)  # Add space for the legend

    # Legend outside the plot at the bottom
    plt.legend(
        title="Clusters", 
        loc="upper center",  
        bbox_to_anchor=(0.5, -0.2),  # Move further below the plot
        ncol=6,  
        frameon=False  
    )

    # Save plot
    plt.savefig(output_path_genes_clusters, bbox_inches="tight")  # Ensure legend is included
    plt.close()

    print(f"Cluster visualization saved to {output_path_genes_clusters}")
    
    return cluster_labels, total_genes_per_cluster, pred_counts

def cluster_and_visualize_predicted_genes_(graph, predicted_cancer_genes, node_names, 
                                          output_path_genes_clusters, output_path_predicted_cancer_genes, num_clusters=12):
    """
    Clusters gene embeddings into groups using KMeans, visualizes them with t-SNE, 
    and marks predicted cancer genes with red circles (half the size of non-cancer dots).
    Saves predicted cancer genes per cluster in CSV files.
    
    Returns:
        cluster_labels (np.ndarray): Cluster assignments for each gene.
        predicted_cancer_genes_count (dict): Count of predicted cancer genes per cluster.
        total_genes_per_cluster (dict): Total number of genes per cluster.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()
    
    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Store cluster labels in graph
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)
    
    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    
    # Store predicted cancer genes per cluster
    pcg_genes = {i: [] for i in range(num_clusters)}
    
    # Count total genes per cluster
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(num_clusters)}
    
    # Plot clusters (Non-cancer genes)
    non_cancer_dot_size = 100  # Default dot size
    red_circle_size = non_cancer_dot_size / 2  # Half the size
    
    for cluster_id in range(num_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=CLUSTER_COLORS[cluster_id], 
                    edgecolor='k', s=non_cancer_dot_size, alpha=0.8, 
                    label=f"Cluster {cluster_id}")
    
    # Mark predicted cancer genes with red circles (⚪, half the size)
    for gene_idx in predicted_cancer_genes:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=red_circle_size, linewidths=2)
        
        # Store gene in corresponding cluster
        cluster_id = cluster_labels[gene_idx]
        pcg_genes[cluster_id].append(node_names[gene_idx])
    
    # Legend and labels
    plt.title("Gene clustering with predicted cancer genes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Clusters", loc="best")
    
    # Save plot
    plt.savefig(output_path_genes_clusters)
    plt.close()
    print(f"Cluster visualization saved to {output_path_genes_clusters}")
    
    # Save predicted genes per cluster to separate CSV files
    '''predicted_cancer_genes = []
    predicted_cancer_genes_count = {}
    
    for cluster_id, genes in pcg_genes.items():
        predicted_cancer_genes_count[cluster_id] = len(genes)  # Store count of cancer genes per cluster
        
        with open(output_path_genes_clusters, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Gene Name"])  # Header
            csvwriter.writerows([[gene] for gene in genes])
        
        print(f"Cluster {cluster_id}: {len(genes)} predicted cancer genes saved to {output_path_genes_clusters}")
        
        # Store for overall CSV
        for gene in genes:
            predicted_cancer_genes.append([gene, cluster_id])
    
    # Save all predicted cancer genes in a single CSV
    with open(output_path_predicted_cancer_genes, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Gene Name", "Cluster ID"])  # Header
        csvwriter.writerows(predicted_cancer_genes)
    
    print(f"All predicted cancer genes saved to {output_path_predicted_cancer_genes}")
    '''
    return cluster_labels, total_genes_per_cluster

def plot_pcg_cancer_genes_ori(clusters, predicted_cancer_genes_count, total_genes_per_cluster, node_names, cluster_labels, output_path, cluster_colors):
    """
    Plots the percentage of predicted cancer genes per cluster.
    """
    # Convert to NumPy arrays for safe division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_genes_per_cluster = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes per cluster
    percent_predicted = np.divide(predicted_counts, total_genes_per_cluster, where=total_genes_per_cluster > 0)  

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, percent_predicted, color=[cluster_colors[c] for c in clusters], edgecolor='black')

    # Add value labels on top of each bar (number of predicted cancer genes per cluster)
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        count = predicted_cancer_genes_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and formatting
    plt.ylabel("Percent of PCGs", fontsize=14)
    ##plt.xlabel("Cluster Number")
    plt.xticks(clusters)  
    plt.ylim(0, max(percent_predicted) + 0.1)  

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_pcg_cancer_genes_ground_truth_ori(clusters, predicted_cancer_genes_count, total_genes_per_cluster, ground_truth_cancer_genes, node_names, cluster_labels, output_path, cluster_colors):
    """
    Plots the percentage of predicted cancer genes in each cluster.
    The number on top of each bar represents the count of ground truth cancer genes in that cluster.
    """
    # Convert to NumPy arrays for safer division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_genes_per_cluster = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes safely
    percent_predicted = np.divide(predicted_counts, total_genes_per_cluster, where=total_genes_per_cluster > 0)

    # Compute ground truth cancer gene count per cluster
    ground_truth_cancer_genes_count = {i: 0 for i in range(len(clusters))}
    
    for i, gene in enumerate(node_names):
        if gene in ground_truth_cancer_genes:
            cluster_id = cluster_labels[i]
            ground_truth_cancer_genes_count[cluster_id] += 1

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, percent_predicted, color=[cluster_colors[c] for c in clusters], edgecolor='black')

    # Add value labels on top of each bar (ground truth cancer genes per cluster)
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        true_count = ground_truth_cancer_genes_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(true_count), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and formatting
    plt.ylabel("Percent of Predicted Cancer Genes", fontsize=14)
    ##plt.xlabel("Cluster Number", fontsize=12)
    plt.xticks(clusters)  
    plt.ylim(0, max(percent_predicted) + 0.1)  

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_interactions_with_kcgs_random_order_color(data, output_path, cluster_colors):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).

    Args:
        data (pd.DataFrame): A DataFrame containing 'Cluster' and 'Interactions'.
        cluster_colors (list): List of colors corresponding to clusters.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))

    # Set seaborn style without grid
    sns.set_style("white")

    # Create a color map to match clusters to colors
    cluster_color_map = {cluster_id: cluster_colors[i % len(cluster_colors)] for i, cluster_id in enumerate(data['Cluster'].unique())}

    # Create the box plot with custom colors for each cluster
    ax = sns.boxplot(x='Cluster', y='Interactions', data=data, 
                     hue='Cluster', palette=cluster_color_map, showfliers=False)  # No outlier markers

    # Overlay scatter plot of individual data points (jittering to reduce overlap)
    sns.stripplot(x='Cluster', y='Interactions', data=data, 
                  color='black', alpha=0.2, jitter=True, size=1.5)  # Reduce dot size

    # Remove the legend manually
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Labels and title
    plt.ylabel("Number of interactions with PCGs", fontsize=14)
    plt.xlabel("")
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=0, ha="right")

    plt.ylim(0, 400)

    # Save and close (Legend completely removed)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_interactions_with_kcgs_lenged_inside(data, output_path, cluster_colors):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).

    Args:
        data (pd.DataFrame): A DataFrame containing 'Cluster' and 'Interactions'.
        cluster_colors (list): List of colors corresponding to clusters.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 5))

    # Set seaborn style without grid
    sns.set_style("white")

    # Create a color map to match clusters to colors
    cluster_color_map = {cluster_id: cluster_colors[i % len(cluster_colors)] for i, cluster_id in enumerate(data['Cluster'].unique())}

    # Create the box plot with custom colors for each cluster
    sns.boxplot(x='Cluster', y='Interactions', data=data, 
                hue='Cluster', palette=cluster_color_map, showfliers=False)  # No outlier markers

    # Overlay scatter plot of individual data points (jittering to reduce overlap)
    sns.stripplot(x='Cluster', y='Interactions', data=data, 
                  color='black', alpha=0.2, jitter=True, size=1.5)  # Reduce dot size

    # Labels and title
    plt.ylabel("Number of interactions with PCGs", fontsize=14)
    plt.xlabel("")
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=0, ha="right")

    plt.ylim(0, 400)

    # Adjust layout to fit the legend
    '''plt.subplots_adjust(bottom=0.3)  # Add space below for the legend

    # Create legend handles with square patches
    legend_handles = [
        Patch(facecolor=cluster_colors[i], edgecolor='black', label=f"Cluster {cluster_id}")
        for i, cluster_id in enumerate(data['Cluster'].unique())
    ]

    # Legend outside the plot at the bottom with square markers
    plt.legend(
        handles=legend_handles, 
        title="Clusters", 
        loc="upper center",  
        bbox_to_anchor=(0.5, -0.15),  # Move legend below the plot
        ncol=6,  
        frameon=False  
    )
    '''
    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # Ensure legend is included
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_interactions_with_kcgs_legend_inside(data, output_path, cluster_colors):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).

    Args:
        data (pd.DataFrame): A DataFrame containing 'Cluster' and 'Interactions'.
        cluster_colors (list): List of colors corresponding to clusters.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))

    # Set seaborn style without grid
    sns.set_style("white")

    # Create a color map to match clusters to colors
    cluster_color_map = {cluster_id: cluster_colors[i % len(cluster_colors)] for i, cluster_id in enumerate(data['Cluster'].unique())}

    # Create the box plot with custom colors for each cluster
    sns.boxplot(x='Cluster', y='Interactions', data=data, 
                hue='Cluster', palette=cluster_color_map, showfliers=False)  # No outlier markers


    # Overlay scatter plot of individual data points (jittering to reduce overlap)
    sns.stripplot(x='Cluster', y='Interactions', data=data, 
                  color='black', alpha=0.2, jitter=True, size=1.5)  # Reduce dot size

    # Labels and title
    plt.ylabel("Number of interactions with predicted cancer genes", fontsize=14)
    plt.xlabel("")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=0, ha="right")

    plt.ylim(0, 400)

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_interactions_with_kcgs_missing_key(data, output_path, cluster_colors):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).

    Args:
        data (pd.DataFrame): A DataFrame containing 'Cluster' and 'Interactions'.
        cluster_colors (list): List of colors corresponding to clusters.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))

    # Set seaborn style without grid
    sns.set_style("white")

    # Create the box plot with custom colors for each cluster
    sns.boxplot(x='Cluster', y='Interactions', data=data, 
                palette=cluster_colors, showfliers=False)  # No outlier markers

    # Overlay scatter plot of individual data points (jittering to reduce overlap)
    sns.stripplot(x='Cluster', y='Interactions', data=data, 
                  color='black', alpha=0.2, jitter=True, size=1.5)  # Reduce dot size

    # Labels and title
    plt.ylabel("Number of interactions with predicted cancer genes", fontsize=14)
    plt.xlabel("")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=0, ha="right")

    plt.ylim(0, 400)

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_pcg_cancer_genes_ground_truth(clusters, predicted_cancer_genes_count, total_genes_per_cluster, ground_truth_cancer_genes, node_names, cluster_labels, output_path):
    """
    Plots the percentage of predicted cancer genes in each cluster.
    The number on top of each bar represents the count of ground truth cancer genes in that cluster.
    """
    # Convert to NumPy arrays for safer division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_genes_per_cluster = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes safely
    percent_predicted = np.divide(predicted_counts, total_genes_per_cluster, where=total_genes_per_cluster > 0)

    # Compute ground truth cancer gene count per cluster
    ground_truth_cancer_genes_count = {i: 0 for i in clusters}
    
    for i, gene in enumerate(node_names):
        if gene in ground_truth_cancer_genes:
            cluster_id = cluster_labels[i]
            if cluster_id in ground_truth_cancer_genes_count:
                ground_truth_cancer_genes_count[cluster_id] += 1

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, percent_predicted, 
                   color=[CLUSTER_COLORS.get(c, '#000000') for c in clusters],  # Default to black if cluster not in dictionary
                   edgecolor='black')

    # Add value labels on top of each bar (ground truth cancer genes per cluster)
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        true_count = ground_truth_cancer_genes_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(true_count), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and formatting
    plt.ylabel("Percent of PCGs", fontsize=14)
    plt.xticks(clusters)  
    plt.ylim(0, max(percent_predicted) + 0.1)  

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_pcg_cancer_genes(clusters, predicted_cancer_genes_count, total_genes_per_cluster, node_names, cluster_labels, output_path):
    """
    Plots the percentage of predicted cancer genes per cluster using a fixed color scheme.
    """
    # Convert to NumPy arrays for safe division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_genes_per_cluster = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes per cluster
    percent_predicted = np.divide(predicted_counts, total_genes_per_cluster, where=total_genes_per_cluster > 0)  

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, percent_predicted, 
                   color=[CLUSTER_COLORS.get(c, '#000000') for c in clusters],  # Default to black if cluster not in dictionary
                   edgecolor='black')

    # Add value labels on top of each bar (number of predicted cancer genes per cluster)
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        count = predicted_cancer_genes_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and formatting
    plt.ylabel("Percent of PCGs", fontsize=16)
    plt.xticks(clusters)  
    plt.ylim(0, max(percent_predicted) + 0.1)  

    sns.despine()  # 🔻 Remove top/right spines
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def train_x(args):
    # Initialize metrics
    epoch_times = []
    cpu_usages = []
    gpu_usages = []
    
    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1  # Mask for labeled nodes
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)  # Mask for testing on all nodes
    graph = dgl.add_self_loop(graph)
    
    # Compute node degrees
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)  # Reshape to (N, 1)
    
    # Concatenate degree feature with node embeddings
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)
    in_feats = features.shape[1]  # Update input dimension

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    model = choose_model(args.model_type, in_feats, args.hidden_feats, 1)
    model = model.to(device)
    
    loss_fn = FocalLoss(alpha=0.25, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Move data to device
    graph = graph.to(device)
    features = features.to(device)
    labels = graph.ndata['label'].to(device).float()
    train_mask = graph.ndata['train_mask'].to(device)
    test_mask = graph.ndata['test_mask'].to(device)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()
        
        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_usages.append(cpu_usage)
        
        gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2 if torch.cuda.is_available() else 0.0
        gpu_usages.append(gpu_usage)
        
        model.train()
        logits = model(graph, features).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_times.append(time.time() - epoch_start)
        tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}, CPU: {cpu_usage}%, GPU: {gpu_usage:.2f} MB")

    # Testing
    model.eval()
    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()
    
    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)
    
    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt",
                        "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt",
                        node_names, non_labeled_nodes)
    
    predicted_cancer_genes = [i for i in non_labeled_nodes if scores[i] >= args.score_threshold]
    
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes_clusters.png')
    output_path_predicted_cancer_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_predicted_cancer_genes.csv')
    
    cluster_labels, predicted_cancer_genes_count, total_genes_per_cluster = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, output_path_predicted_cancer_genes)
    
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    
    output_path_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_percent_ground_truth.png')
    plot_pcg_cancer_genes_ground_truth(
        clusters=total_genes_per_cluster.keys(),
        predicted_cancer_genes_count=predicted_cancer_genes_count,
        total_genes_per_cluster=total_genes_per_cluster,
        ground_truth_cancer_genes=ground_truth_cancer_genes,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_percent)
    
    output_path_percent_pred = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_percent_predicted.png')
    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(),
        predicted_cancer_genes_count=predicted_cancer_genes_count,
        total_genes_per_cluster=total_genes_per_cluster,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_percent_pred)
    
    graph.ndata['degree'] = graph.in_degrees().float()
    graph.ndata['interactions'] = torch.zeros(graph.num_nodes(), dtype=torch.float, device=graph.device)
    
    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    graph.ndata['interactions'][predicted_nodes] = graph.ndata['degree'][predicted_nodes].squeeze()
    
    if predicted_nodes:
        print(f"Average degree of predicted nodes: {graph.ndata['degree'][predicted_nodes].mean().item():.2f}")
    else:
        print("No nodes predicted above the threshold.")
    
    data = pd.DataFrame({"Cluster": graph.ndata['cluster'][predicted_nodes].cpu().numpy(),
                          "Interactions": graph.ndata['interactions'][predicted_nodes].cpu().numpy()})
    
    output_path_interactions = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcgs_interaction.png')
    plot_interactions_with_kcgs(data, output_path_interactions)

def train_(args):
    # Initialize metrics
    epoch_times = []
    cpu_usages = []
    gpu_usages = []
    data_path = os.path.join('data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    ###data_path = os.path.join('data/multiomics_meth/', f'extracted_embeddings.json')
    ###data_path = '../___KG-PE/embedding/data/neo4j_graph_data.json'
    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1  # Mask for labeled nodes
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)  # Mask for testing on all nodes
    graph = dgl.add_self_loop(graph)
    in_feats = embeddings.shape[1]
    hidden_feats = args.hidden_feats
    out_feats = 1  # Binary classification (driver gene or not)
    
    
    # Compute node degrees
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)  # Reshape to (N, 1)

    # Concatenate degree feature with node embeddings
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    # Update input feature dimension
    in_feats = features.shape[1]  # Update to reflect added degree feature

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move features to device
    
    
    # Define output directory
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    # File path for saving model architecture
    model_csv_path = os.path.join(output_dir, f'{args.model_type}_model_structure.csv')

    # Choose the model
    model = choose_model(args.model_type, in_feats, hidden_feats, out_feats)
    model = model.to(args.device)

    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss(alpha=0.25, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Move data to device
    graph = graph.to(device)
    ##features = graph.ndata['feat'].to(device)
    features = features.to(device)
    labels = graph.ndata['label'].to(device).float()  # BCE loss expects float labels
    train_mask = graph.ndata['train_mask'].to(device)
    test_mask = graph.ndata['test_mask'].to(device)
        
    # Training loop with progress bar
    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()

        # Measure CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_usages.append(cpu_usage)

        # Measure GPU usage (if CUDA available)
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2  # Convert bytes to MB
        else:
            gpu_usage = 0.0  # No GPU available
        gpu_usages.append(gpu_usage)
        
        model.train()
        logits = model(graph, features).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time per epoch
        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)

        # Optionally display current metrics
        tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}, CPU: {cpu_usage}%, GPU: {gpu_usage:.2f} MB")

    # Testing and predictions
    model.eval()
    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()
        print("Predicted Scores:", scores[test_mask.cpu().numpy()])

    ##################################################################
    # Rank nodes by scores for only non-labeled nodes
    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]  # Indices of non-labeled nodes
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)


    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)
    
    predicted_cancer_genes = [i for i in non_labeled_nodes if scores[i] >= args.score_threshold]
    # Run clustering and get resultsoutput_path_all_predicted_cancer_genes_per_cluster = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_all_predicted_cancer_genes_per_cluster_epo{args.num_epochs}_2048.csv')
    output_path_genes_clusters = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_all_predicted_cancer_genes_per_cluster_epo{args.num_epochs}_2048.png')
    output_path_predicted_cancer_genes = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_predicted_cancer_genes_epo{args.num_epochs}_2048.csv')
    cluster_labels, predicted_cancer_genes_count, total_genes_per_cluster = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, output_path_predicted_cancer_genes)#, num_clusters=5)

    '''result = cluster_and_visualize_predicted_genes(graph, non_labeled_nodes, predicted_cancer_genes, node_names, output_dir)
    print(len(result))  # Should print 3'''

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')

    # Call the plot function
    output_path_percent = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_cluster_percent_ground_truth_epo{args.num_epochs}_2048.png')

    plot_pcg_cancer_genes_ground_truth(
        clusters=total_genes_per_cluster.keys(),  # Cluster IDs
        predicted_cancer_genes_count=predicted_cancer_genes_count,
        total_genes_per_cluster=total_genes_per_cluster,
        ground_truth_cancer_genes=ground_truth_cancer_genes,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_percent
    )

# Call the plot function
    output_path_percent = os.path.join(
        'results/gene_prediction/', 
        f'{args.model_type}_{args.net_type}_cluster_percent_predicted_cancer_genes_epo{args.num_epochs}_2048.png'
    )

    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(),  # Cluster IDs
        predicted_cancer_genes_count=predicted_cancer_genes_count,
        total_genes_per_cluster=total_genes_per_cluster,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_percent  # Removed ground_truth_cancer_genes
    )

    graph.ndata['degree'] = graph.in_degrees().float()  # Ensure it's a float tensor

    graph.ndata['interactions'] = torch.zeros(graph.num_nodes(), dtype=torch.float, device=graph.device)

    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]

    graph.ndata['interactions'][predicted_nodes] = graph.ndata['degree'][predicted_nodes].squeeze()

    # Get degrees of the predicted nodes
    if predicted_nodes:
        predicted_degrees = graph.ndata['degree'][predicted_nodes]
        print(f"Average degree of predicted nodes: {predicted_degrees.mean().item():.2f}")
    else:
        print("No nodes predicted above the threshold.")

        cluster_ids = graph.ndata['cluster'][predicted_nodes].cpu().numpy()
        interactions = graph.ndata['interactions'][predicted_nodes].cpu().numpy()  # Interaction counts

        # Create DataFrame for visualization
        data = pd.DataFrame({"Cluster": cluster_ids, "Interactions": interactions})

        # Call the plotting function
        
        output_path = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}_2048.png')
        plot_interactions_with_kcgs(data, output_path)
   
def train_ori(args):
    # Initialize metrics
    epoch_times = []
    cpu_usages = []
    gpu_usages = []
    data_path = os.path.join('data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    ###data_path = os.path.join('data/multiomics_meth/', f'extracted_embeddings.json')
    ###data_path = '../___KG-PE/embedding/data/neo4j_graph_data.json'
    data_path = '../___KG-PE/embedding/data/merged_gene_embeddings.json'
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1  # Mask for labeled nodes
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)  # Mask for testing on all nodes
    graph = dgl.add_self_loop(graph)
    in_feats = embeddings.shape[1]
    hidden_feats = args.hidden_feats
    out_feats = 1  # Binary classification (driver gene or not)
    
    
    # Compute node degrees
    graph.ndata['degree'] = graph.in_degrees().float().unsqueeze(1)  # Reshape to (N, 1)

    # Concatenate degree feature with node embeddings
    features = torch.cat([graph.ndata['feat'], graph.ndata['degree']], dim=1)

    # Update input feature dimension
    in_feats = features.shape[1]  # Update to reflect added degree feature

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move features to device
    
    
    # Define output directory
    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    # File path for saving model architecture
    model_csv_path = os.path.join(output_dir, f'{args.model_type}_model_structure.csv')

    # Choose the model
    model = choose_model(args.model_type, in_feats, hidden_feats, out_feats)
    model = model.to(args.device)

    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss(alpha=0.25, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Move data to device
    graph = graph.to(device)
    ##features = graph.ndata['feat'].to(device)
    features = features.to(device)
    labels = graph.ndata['label'].to(device).float()  # BCE loss expects float labels
    train_mask = graph.ndata['train_mask'].to(device)
    test_mask = graph.ndata['test_mask'].to(device)
        
    # Training loop with progress bar
    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()

        # Measure CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_usages.append(cpu_usage)

        # Measure GPU usage (if CUDA available)
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2  # Convert bytes to MB
        else:
            gpu_usage = 0.0  # No GPU available
        gpu_usages.append(gpu_usage)
        
        model.train()
        logits = model(graph, features).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time per epoch
        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)

        # Optionally display current metrics
        tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}, CPU: {cpu_usage}%, GPU: {gpu_usage:.2f} MB")

    # Testing and predictions
    model.eval()
    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()
        print("Predicted Scores:", scores[test_mask.cpu().numpy()])

                
    # Convert epoch_times to floats during summation
    total_time = sum(epoch_times)  # Total time in seconds
    average_time_per_epoch = total_time / args.num_epochs  # Average time per epoch
    # Calculate average memory usage for CPU and GPU
    average_cpu_usage = sum(cpu_usages) / args.num_epochs  # CPU usage in MB
    average_gpu_usage = sum(gpu_usages) / args.num_epochs  # GPU usage in MB
    # Perform clustering and visualization on predicted cancer driver genes
    save_model_details(model, args, model_csv_path, in_feats, hidden_feats, out_feats)
    label_scores = save_predicted_scores(scores, labels, nodes, args)
    save_average_scores(label_scores, args)
    plot_average_scores(label_scores, args)
    plot_score_distributions(label_scores, args)
    save_performance_metrics(epoch_times, cpu_usages, gpu_usages, args)
    # After calculating total_time, average_time_per_epoch, etc.
    save_overall_metrics(total_time, average_time_per_epoch, average_cpu_usage, average_gpu_usage, args, output_dir)


    ##################################################################
    # Rank nodes by scores for only non-labeled nodes
    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]  # Indices of non-labeled nodes
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)


    process_predictions(ranking, args, "data/796_drivers.txt", "data/oncokb_1172.txt", "data/ongene_803.txt", "data/ncg_8886.txt", "data/intogen_23444.txt", node_names, non_labeled_nodes)
    
    predicted_cancer_genes = [i for i in non_labeled_nodes if scores[i] >= args.score_threshold]
    # Run clustering and get resultsoutput_path_all_predicted_cancer_genes_per_cluster = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_all_predicted_cancer_genes_per_cluster_epo{args.num_epochs}_2048.csv')
    output_path_genes_clusters = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_all_predicted_cancer_genes_per_cluster_epo{args.num_epochs}_2048.png')
    output_path_predicted_cancer_genes = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_predicted_cancer_genes_epo{args.num_epochs}_2048.csv')
    cluster_labels, predicted_cancer_genes_count, total_genes_per_cluster = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, output_path_predicted_cancer_genes)#, num_clusters=5)

    '''result = cluster_and_visualize_predicted_genes(graph, non_labeled_nodes, predicted_cancer_genes, node_names, output_dir)
    print(len(result))  # Should print 3'''

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')

    # Call the plot function
    output_path_percent = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_cluster_percent_ground_truth_epo{args.num_epochs}_2048.png')

    plot_pcg_cancer_genes_ground_truth(
        clusters=total_genes_per_cluster.keys(),  # Cluster IDs
        predicted_cancer_genes_count=predicted_cancer_genes_count,
        total_genes_per_cluster=total_genes_per_cluster,
        ground_truth_cancer_genes=ground_truth_cancer_genes,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_percent
    )

# Call the plot function
    output_path_percent = os.path.join(
        'results/gene_prediction/', 
        f'{args.model_type}_{args.net_type}_cluster_percent_predicted_cancer_genes_epo{args.num_epochs}_2048.png'
    )

    plot_pcg_cancer_genes_ori(
        clusters=total_genes_per_cluster.keys(),  # Cluster IDs
        predicted_cancer_genes_count=predicted_cancer_genes_count,
        total_genes_per_cluster=total_genes_per_cluster,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_percent  # Removed ground_truth_cancer_genes
    )

            
    ##plot_pcg_cancer_genes(cluster_labels, predicted_cancer_genes_count, total_genes_per_cluster, output_path_percent)

    '''# Perform clustering and store cluster labels
    cluster_labels, predicted_counts, total_genes_per_cluster = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_dir="results"
    )'''

    # Now 'graph.ndata['cluster']' exists, so we can safely use it
    '''predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    
    # Get node degrees (number of edges per node)
    graph.ndata['degree'] = graph.in_degrees()  # or graph.out_degrees(), depending on the graph direction
    '''
    
    # Compute interactions (degree of connection with predicted cancer genes)
    '''
    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]
    graph.ndata['interactions'] = torch.zeros(graph.num_nodes(), dtype=torch.float, device=graph.device)
    graph.ndata['interactions'][predicted_nodes] = graph.ndata['degree'][predicted_nodes]'''

    graph.ndata['degree'] = graph.in_degrees().float()  # Ensure it's a float tensor

    graph.ndata['interactions'] = torch.zeros(graph.num_nodes(), dtype=torch.float, device=graph.device)

    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]

    graph.ndata['interactions'][predicted_nodes] = graph.ndata['degree'][predicted_nodes].squeeze()

    # Get degrees of the predicted nodes
    if predicted_nodes:
        predicted_degrees = graph.ndata['degree'][predicted_nodes]
        print(f"Average degree of predicted nodes: {predicted_degrees.mean().item():.2f}")
    else:
        print("No nodes predicted above the threshold.")
        '''
    if predicted_nodes:
        cluster_ids = graph.ndata['cluster'][predicted_nodes].cpu().numpy()
        interactions = graph.ndata['interactions'][predicted_nodes].cpu().numpy()

        # Create DataFrame and plot interactions
        data = pd.DataFrame({"Cluster": cluster_ids, "Interactions": interactions})
        plot_interactions_with_kcgs(data, output_path="results/kcgs_interactions.png")

    # Select nodes where the prediction score is above threshold
    predicted_nodes = [i for i in range(graph.num_nodes()) if scores[i] >= args.score_threshold]'''



    # ---- Call the Plotting Function ----
    # Assuming `graph.ndata['cluster']` contains cluster IDs
    if predicted_nodes:
        cluster_ids = graph.ndata['cluster'][predicted_nodes].cpu().numpy()
        interactions = graph.ndata['interactions'][predicted_nodes].cpu().numpy()  # Interaction counts

        # Create DataFrame for visualization
        data = pd.DataFrame({"Cluster": cluster_ids, "Interactions": interactions})

        # Call the plotting function
        
        output_path = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_kcgs_interaction_epo{args.num_epochs}_2048.png')
        plot_interactions_with_kcgs(data, output_path)
        
    #cluster_and_visualize_predicted_genes(graph, non_labeled_nodes, predicted_cancer_genes, node_names, 'results/gene_prediction/', num_clusters=10)

    ##cluster_and_visualize_predicted_genes_ori(graph, non_labeled_nodes, node_names, 'results/gene_prediction/', num_clusters=5)
    # Load driver and reference gene sets


    # Calculate statistics
    non_labeled_nodes_count = len(non_labeled_nodes)
    ground_truth_driver_nodes = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_non_driver_nodes = [i for i, label in enumerate(labels) if label == 0]
    

    # Save both above and below threshold scores, sorted by scores in descending order
    predicted_driver_nodes_above_threshold = sorted(
        [(node_names[i], scores[i]) for i in non_labeled_nodes if scores[i] >= args.score_threshold],
        key=lambda x: x[1],
        reverse=True
    )
    predicted_driver_nodes_below_threshold = sorted(
        [(node_names[i], scores[i]) for i in non_labeled_nodes if scores[i] < args.score_threshold],
        key=lambda x: x[1],
        reverse=True
    )

    from collections import defaultdict
    import statistics

    # Get the ground truth driver gene indices and names
    ground_truth_driver_indices = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_driver_names = {node_names[i] for i in ground_truth_driver_indices}

    # Save predictions (above and below threshold) to CSV
    output_file_above = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_genes_above_epo{args.num_epochs}_2048.csv'
    )
    output_file_below = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_genes_below_epo{args.num_epochs}_2048.csv'
    )

    with open(output_file_above, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Score'])  # Header row
        csvwriter.writerows(predicted_driver_nodes_above_threshold)

    with open(output_file_below, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Score'])  # Header row
        csvwriter.writerows(predicted_driver_nodes_below_threshold)

    print(f"Predicted driver genes (above threshold) saved to {output_file_above}")
    print(f"Predicted driver genes (below threshold) saved to {output_file_below}")

    # Calculate degrees for nodes above and below the threshold (connecting only to label 1 nodes)
    degree_counts_above = defaultdict(int)
    degree_counts_below = defaultdict(int)

    for src, dst in edges:
        src_name = node_names[src]
        dst_name = node_names[dst]

        # Count only connections to ground truth driver genes (label 1 nodes)
        if dst_name in ground_truth_driver_names:
            if src_name in [gene for gene, _ in predicted_driver_nodes_above_threshold]:
                degree_counts_above[src_name] += 1
            elif src_name in [gene for gene, _ in predicted_driver_nodes_below_threshold]:
                degree_counts_below[src_name] += 1

    # Sort degrees by degree count in descending order
    sorted_degree_counts_above = sorted(degree_counts_above.items(), key=lambda x: x[1], reverse=True)
    sorted_degree_counts_below = sorted(degree_counts_below.items(), key=lambda x: x[1], reverse=True)

    # Save degrees of predicted driver genes connecting to ground truth driver genes (above threshold)
    degree_output_file_above = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_gene_degrees_above_epo{args.num_epochs}_2048.csv'
    )
    with open(degree_output_file_above, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Predicted Driver Gene', 'Degree'])  # Header row
        csvwriter.writerows(sorted_degree_counts_above)
        ##csvwriter.writerow(['Average Degree', average_degree_above])  # Save average degree

    print(f"Degrees of predicted driver genes (above threshold) saved to {degree_output_file_above}")



    # Prepare DataFrame for nodes with degrees
    nodes_with_degrees = []

    # Above threshold
    for gene, degree in degree_counts_above.items():
        nodes_with_degrees.append({'Gene_Set': 'Above Threshold', 'Degree': degree})

    # Below threshold
    for gene, degree in degree_counts_below.items():
        nodes_with_degrees.append({'Gene_Set': 'Below Threshold', 'Degree': degree})

    # Convert to DataFrame
    nodes_with_degrees = pd.DataFrame(nodes_with_degrees)

    ##print(f"sorted_degree_counts_below: {sorted_degree_counts_below}")
    sorted_degree_counts_above_value = [value for _, value in sorted_degree_counts_above if value <= 20]
    sorted_degree_counts_below_value = [value for _, value in sorted_degree_counts_below if value <= 20]
    
    output_above_file = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_output_above_file_epo{args.num_epochs}_2048.csv')
    output_below_file = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_output_below_file_epo{args.num_epochs}_2048.csv')

    calculate_and_save_prediction_stats(non_labeled_nodes, labels, node_names, scores, args)

    plot_degree_distributions(sorted_degree_counts_above_value, sorted_degree_counts_below_value, args, output_dir)

    generate_kde_and_curves(logits, node_names, degree_counts_above, degree_counts_below, labels, train_mask, args)

    plot_model_performance(args)

    plot_venn_diagram()

def get_cluster_colors(cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    colors = plt.get_cmap("tab10").colors  # Use a predefined colormap
    cluster_color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
    return cluster_color_map

def plot_interactions_with_kcgs_ori(data, output_path):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).

    Args:
        data (pd.DataFrame): A DataFrame containing 'Cluster' and 'Interactions'.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))

    # Set seaborn style without grid
    sns.set_style("white")

    # Create the box plot
    sns.boxplot(x='Cluster', y='Interactions', data=data, 
                palette='tab20', showfliers=False)  # No outlier markers

    # Overlay scatter plot of individual data points (jittering to reduce overlap)
    sns.stripplot(x='Cluster', y='Interactions', data=data, 
                  color='black', alpha=0.2, jitter=True, size=1.5)  # Reduce dot size

    # Labels and title
    plt.ylabel("Number of interactions with predicted cancer genes", fontsize=14)
    plt.xlabel("")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=0, ha="right")

    plt.ylim(0, 400)

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {output_path}")
    
def plot_interactions_with_kcgs_dot_big(data, output_path):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).

    Args:
        data (pd.DataFrame): A DataFrame containing 'Cluster' and 'Interactions'.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))

    # Set seaborn style without grid
    sns.set_style("white")

    # Create the box plot
    sns.boxplot(x='Cluster', y='Interactions', data=data, 
                palette='tab20', showfliers=False)  # No outlier markers

    # Overlay scatter plot of individual data points (jittering to reduce overlap)
    sns.stripplot(x='Cluster', y='Interactions', data=data, 
                  color='black', alpha=0.5, jitter=True)

    # Labels and title
    plt.ylabel("Number of interactions with predicted cancer genes", fontsize=14)
    plt.xlabel("")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=0, ha="right")

    plt.ylim(0, 400)

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_interactions_with_kcgs_y_1750(data, output_path):
    """
    Creates a box plot with individual data points showing 
    the number of interactions with known cancer genes (KCGs).
    
    Args:
        data (pd.DataFrame): A DataFrame containing 'Cluster' and 'Interactions'.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    
    # Set seaborn style
    sns.set(style="whitegrid")

    # Create the box plot
    sns.boxplot(x='Cluster', y='Interactions', data=data, 
                palette='tab20', showfliers=False)  # No outlier markers

    # Overlay scatter plot of individual data points (jittering to reduce overlap)
    sns.stripplot(x='Cluster', y='Interactions', data=data, 
                  color='black', alpha=0.5, jitter=True)

    # Labels and title
    plt.ylabel("Number of interactions with KCGs", fontsize=14)
    plt.xlabel("")
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=0, ha="right")
    
    # Save and show
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")

def cluster_and_visualize_predicted_genes_ori(graph, predicted_cancer_genes, node_names, output_path_genes_clusters, output_path_predicted_cancer_genes, num_clusters=12):
    """
    Clusters gene embeddings into groups using KMeans, visualizes them with t-SNE, 
    and marks predicted cancer genes with red circles (half the size of non-cancer dots).
    Saves predicted cancer genes per cluster in CSV files.

    Returns:
        cluster_labels (np.ndarray): Cluster assignments for each gene.
        predicted_cancer_genes_count (dict): Count of predicted cancer genes per cluster.
        total_genes_per_cluster (dict): Total number of genes per cluster.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()
    
    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # 🛠 Store cluster labels in graph.ndata
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long, device=graph.device)
    
    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Define 10 distinct colors for clusters (excluding red)
    cluster_colors = [
        '#0077B6', '#0000FF', '#00B4D8', '#48EAC4', '#F1C0E8', 
        '#B9FBC0', '#32CD32', '#FFD700', '#8A2BE2', '#E377C2', 
        '#8EECF5', '#A3C4F3'  
    ]


    plt.figure(figsize=(12, 10))

    # Store predicted cancer genes per cluster
    pcg_genes = {i: [] for i in range(num_clusters)}
    print('pcg_genes------------------\n',pcg_genes)

    # Count total genes per cluster
    total_genes_per_cluster = {i: np.sum(cluster_labels == i) for i in range(num_clusters)}
    print('total_genes_per_cluster==================\n',total_genes_per_cluster)

    # Plot clusters (Non-cancer genes)
    non_cancer_dot_size = 100  # Default dot size
    red_circle_size = non_cancer_dot_size / 2  # Half the size

    for cluster_id in range(num_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=cluster_colors[cluster_id % len(cluster_colors)], 
                    edgecolor='k', s=non_cancer_dot_size, alpha=0.8, 
                    label=f"Cluster {cluster_id}")

    # Mark predicted cancer genes with red circles (⚪, half the size)
    for gene_idx in predicted_cancer_genes:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=red_circle_size, linewidths=2)  # Red circle

        # Store gene in corresponding cluster
        cluster_id = cluster_labels[gene_idx]
        pcg_genes[cluster_id].append(node_names[gene_idx])

    # Legend and labels
    plt.title("Gene clustering with predicted cancer genes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Clusters", loc="best")
    
    # Save plot
    #os.makedirs(output_dir, exist_ok=True)
    #plot_path = os.path.join(output_dir, "clustered_genes_clusters.png")
    plt.savefig(output_path_genes_clusters)
    plt.close()
    
    print(f"Cluster visualization saved to {output_path_genes_clusters}")

    # Save predicted genes per cluster to separate CSV files
    predicted_cancer_genes = []
    predicted_cancer_genes_count = {}

    for cluster_id, genes in pcg_genes.items():
        predicted_cancer_genes_count[cluster_id] = len(genes)  # Store count of cancer genes per cluster

        ##cluster_csv_path = os.path.join(output_dir, f"cluster_{cluster_id}_predicted_cancer_genes.csv")
        with open(output_path_genes_clusters, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Gene Name"])  # Header
            csvwriter.writerows([[gene] for gene in genes])
        
        print(f"Cluster {cluster_id}: {len(genes)} predicted cancer genes saved to {output_path_genes_clusters}")

        # Store for overall CSV
        for gene in genes:
            predicted_cancer_genes.append([gene, cluster_id])

    # Save all predicted cancer genes in a single CSV
    ##summary_csv_path = os.path.join(output_path)
    with open(output_path_predicted_cancer_genes, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Gene Name", "Cluster ID"])  # Header
        csvwriter.writerows(predicted_cancer_genes)

    print(f"All predicted cancer genes saved to {output_path_predicted_cancer_genes}")

    return cluster_labels, predicted_cancer_genes_count, total_genes_per_cluster

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

def plot_pcg_cancer_genes_ori(clusters, predicted_cancer_genes_count, total_genes_per_cluster, node_names, cluster_labels, output_path):
    """
    Plots the percentage of predicted cancer genes per cluster.
    """
    # Convert to NumPy arrays for safe division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_genes_per_cluster = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes per cluster
    percent_predicted = np.divide(predicted_counts, total_genes_per_cluster, where=total_genes_per_cluster > 0)  

    # Define distinct colors
    colors = [
        '#1F77B4', '#B9FBC0', '#2CA02C', '#D62728', '#9467BD', '#8C564B',  
        '#E377C2', '#BCBD22', '#17BECF', '#76C7C0', '#F4A261', '#A8DADC'
    ]

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, percent_predicted, color=colors[:len(clusters)], edgecolor='black')

    # Add value labels on top of each bar (number of predicted cancer genes per cluster)
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        count = predicted_cancer_genes_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and formatting
    plt.ylabel("Percent of Predicted Cancer Genes")
    plt.xlabel("Cluster Number")
    plt.xticks(clusters)  
    plt.ylim(0, max(percent_predicted) + 0.1)  

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_pcg_cancer_genes_ground_truth_ori(clusters, predicted_cancer_genes_count, total_genes_per_cluster, ground_truth_cancer_genes, node_names, cluster_labels, output_path):
    """
    Plots the percentage of predicted cancer genes in each cluster.
    The number on top of each bar represents the count of ground truth cancer genes in that cluster.
    
    Args:
        clusters (dict): Cluster numbers as keys.
        predicted_cancer_genes_count (dict): Count of predicted cancer genes per cluster.
        total_genes_per_cluster (dict): Total number of genes per cluster.
        ground_truth_cancer_genes (set): Set of ground truth cancer driver genes.
        node_names (list): List of gene names corresponding to node indices.
        cluster_labels (np.ndarray): Array of cluster labels for each gene.
        output_path (str): Path to save the plot.
    """
    # Convert to NumPy arrays for safer division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_genes_per_cluster = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes safely
    percent_predicted = np.divide(predicted_counts, total_genes_per_cluster, where=total_genes_per_cluster > 0)

    # Compute ground truth cancer gene count per cluster
    ground_truth_cancer_genes_count = {i: 0 for i in range(len(clusters))}
    
    for i, gene in enumerate(node_names):
        if gene in ground_truth_cancer_genes:
            cluster_id = cluster_labels[i]
            ground_truth_cancer_genes_count[cluster_id] += 1

    # Define distinct colors for each bar
    colors = [
        '#1F77B4', '#B9FBC0', '#2CA02C', '#D62728', '#9467BD', '#8C564B',  
        '#E377C2', '#BCBD22', '#17BECF', '#76C7C0', '#F4A261', '#A8DADC'
    ]

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, percent_predicted, color=colors, edgecolor='black')

    # Add value labels on top of each bar (ground truth cancer genes per cluster)
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        true_count = ground_truth_cancer_genes_count.get(cluster_id, 0)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(true_count), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and formatting
    plt.ylabel("Percent of Predicted Cancer Genes", fontsize=14)
    plt.xlabel("Cluster Number", fontsize=12)
    plt.xticks(clusters)  
    plt.ylim(0, max(percent_predicted) + 0.1)  

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_pcg_cancer_genes_(clusters, predicted_cancer_genes_count, total_genes_per_cluster, ground_truth_cancer_genes_count, output_path):
    """
    Plots the percentage of predicted cancer genes per cluster.
    Labels the bars with the number of ground truth cancer genes in each cluster.
    
    Args:
        clusters (list): List of cluster IDs.
        predicted_cancer_genes_count (dict): Count of predicted cancer genes per cluster.
        total_genes_per_cluster (dict): Total number of genes per cluster.
        ground_truth_cancer_genes_count (dict): Count of ground truth cancer genes per cluster.
        output_path (str): Path to save the output figure.
    """
    import numpy as np

    # Convert to NumPy arrays for safer division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_genes_per_cluster = np.array(list(total_genes_per_cluster.values()))  # Total genes in each cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Perform element-wise division safely
    percent_predicted = np.divide(predicted_counts, total_genes_per_cluster, where=total_genes_per_cluster > 0)  

    # Define distinct colors for each bar
    colors = [
        '#1F77B4', '#B9FBC0', '#2CA02C', '#D62728', '#9467BD', '#8C564B',  
        '#E377C2', '#BCBD22', '#17BECF', '#76C7C0', '#F4A261', '#A8DADC'
    ]

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, percent_predicted, color=colors, edgecolor='black')

    # Add value labels (Ground Truth Cancer Genes Count) on top of each bar
    for bar, cluster_id in zip(bars, clusters):
        height = bar.get_height()
        ground_truth_count = ground_truth_cancer_genes_count.get(cluster_id, 0)  # Get ground truth count
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(ground_truth_count), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and formatting
    plt.ylabel("Percent of Predicted Cancer Genes")
    plt.xlabel("Cluster Number")
    plt.xticks(clusters)  
    plt.ylim(0, max(percent_predicted) + 0.1)  

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def plot_pcg_cancer_genes_bar_top_number_x(clusters, predicted_cancer_genes_count, total_genes_per_cluster, output_path):
    # Calculate percentage of predicted cancer genes per cluster
    ##percent_predicted = predicted_cancer_genes_count / total_genes_per_cluster
    
    import numpy as np

    # Convert to NumPy arrays for safer division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs (0 to 9)
    total_genes_per_cluster = np.array(list(total_genes_per_cluster.values()))  # Total genes in each cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Perform element-wise division safely
    percent_predicted = np.divide(predicted_counts, total_genes_per_cluster, where=total_genes_per_cluster > 0)  


    '''percent_predicted = {
    cluster_id: (predicted_cancer_genes_count.get(cluster_id, 0) / total_genes_per_cluster[cluster_id])
        for cluster_id in total_genes_per_cluster
    }'''


    # Define distinct colors for each bar
    colors = [
        '#1F77B4', '#B9FBC0', '#2CA02C', '#D62728', '#9467BD', '#8C564B',  
        '#E377C2', '#BCBD22', '#17BECF', '#76C7C0', '#F4A261', '#A8DADC'
    ]


    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, percent_predicted, color=colors, edgecolor='black')

    # Add value labels on top of each bar
    for bar, count in zip(bars, predicted_cancer_genes_count):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(count), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Labels and formatting
    plt.ylabel("Percent of Predicted Cancer Genes")
    plt.xlabel("Cluster Number")
    plt.xticks(clusters)  
    plt.ylim(0, max(percent_predicted) + 0.1)  

    # Save the plot
    #os.makedirs(output_dir, exist_ok=True)
    #output_path = os.path.join(output_dir, "cluster_percent_predicted_cancer_genes.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")

def cluster_and_visualize_predicted_genes_(graph, non_labeled_nodes, predicted_cancer_genes, node_names, output_dir, num_clusters=10):
    """
    Clusters gene embeddings into 10 groups using KMeans, visualizes them using t-SNE, 
    and marks predicted cancer genes with red circles (half the size of non-cancer dots).
    Saves predicted cancer genes per cluster in CSV files.
    Returns the number of predicted cancer genes in each cluster.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()
    
    # Run KMeans clustering with 10 clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Define 10 distinct colors for clusters (excluding red)
    cluster_colors = [
        '#0077B6', '#0000FF', '#00B4D8', '#48EAC4', '#F1C0E8', 
        '#B9FBC0', '#32CD32', '#FFD700', '#8A2BE2', '#E377C2'
    ]

    plt.figure(figsize=(12, 10))

    # Store predicted cancer genes per cluster
    pcg_genes = {i: [] for i in range(num_clusters)}
    predicted_cancer_genes_count = {i: 0 for i in range(num_clusters)}

    # Plot clusters (Non-cancer genes)
    non_cancer_dot_size = 100  # Default dot size
    red_circle_size = non_cancer_dot_size / 2  # Half the size

    for cluster_id in range(num_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=cluster_colors[cluster_id % len(cluster_colors)], 
                    edgecolor='k', s=non_cancer_dot_size, alpha=0.8, 
                    label=f"Cluster {cluster_id}")

    # Mark predicted cancer genes with red circles (⚪, half the size)
    for gene_idx in predicted_cancer_genes:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=red_circle_size, linewidths=2)  # Red circle

        # Store gene in corresponding cluster
        cluster_id = cluster_labels[gene_idx]
        pcg_genes[cluster_id].append(node_names[gene_idx])
        predicted_cancer_genes_count[cluster_id] += 1  # Count predicted genes per cluster

    # Legend and labels
    plt.title("Gene Clustering (10 Clusters) with Predicted Cancer Genes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Clusters", loc="best")
    
    # Save plot
    plot_path = os.path.join(output_dir, "clustered_genes_10_clusters.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Cluster visualization saved to {plot_path}")

    # Save predicted genes per cluster to separate CSV files
    predicted_cancer_genes = []
    for cluster_id, genes in pcg_genes.items():
        cluster_csv_path = os.path.join(output_dir, f"cluster_{cluster_id}_predicted_cancer_genes.csv")
        with open(cluster_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Gene Name"])  # Header
            csvwriter.writerows([[gene] for gene in genes])
        
        print(f"Cluster {cluster_id}: {len(genes)} predicted cancer genes saved to {cluster_csv_path}")

        # Store for overall CSV
        for gene in genes:
            predicted_cancer_genes.append([gene, cluster_id])

    # Save all predicted cancer genes in a single CSV
    summary_csv_path = os.path.join(output_dir, "all_predicted_cancer_genes_per_cluster.csv")
    with open(summary_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Gene Name", "Cluster ID"])  # Header
        csvwriter.writerows(predicted_cancer_genes)

    print(f"All predicted cancer genes saved to {summary_csv_path}")

    return cluster_labels, predicted_cancer_genes_count

def cluster_and_visualize_predicted_genes_no_count_for_bar(graph, non_labeled_nodes, predicted_cancer_genes, node_names, output_dir, num_clusters=10):
    """
    Clusters gene embeddings into 10 groups using KMeans, visualizes them using t-SNE, 
    and marks predicted cancer genes with red circles (half the size of non-cancer dots).
    Saves predicted cancer genes per cluster in CSV files.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()
    
    # Run KMeans clustering with 10 clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Define 10 distinct colors for clusters (excluding red)
    cluster_colors = [
        '#0077B6', '#0000FF', '#00B4D8', '#48EAC4', '#F1C0E8', 
        '#B9FBC0', '#32CD32', '#FFD700', '#8A2BE2', '#E377C2'
    ]

    plt.figure(figsize=(12, 10))

    # Store predicted cancer genes per cluster
    pcg_genes = {i: [] for i in range(num_clusters)}

    # Plot clusters (Non-cancer genes)
    non_cancer_dot_size = 100  # Default dot size
    red_circle_size = non_cancer_dot_size / 2  # Half the size

    for cluster_id in range(num_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=cluster_colors[cluster_id % len(cluster_colors)], 
                    edgecolor='k', s=non_cancer_dot_size, alpha=0.8, 
                    label=f"Cluster {cluster_id}")

    # Mark predicted cancer genes with red circles (⚪, half the size)
    for gene_idx in predicted_cancer_genes:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=red_circle_size, linewidths=2)  # Red circle

        # Store gene in corresponding cluster
        cluster_id = cluster_labels[gene_idx]
        pcg_genes[cluster_id].append(node_names[gene_idx])

    # Legend and labels
    plt.title("Gene Clustering (10 Clusters) with Predicted Cancer Genes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Clusters", loc="best")
    
    # Save plot
    plot_path = os.path.join(output_dir, "clustered_genes_10_clusters.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Cluster visualization saved to {plot_path}")

    # Save predicted genes per cluster to separate CSV files
    predicted_cancer_genes = []
    for cluster_id, genes in pcg_genes.items():
        cluster_csv_path = os.path.join(output_dir, f"cluster_{cluster_id}_predicted_cancer_genes.csv")
        with open(cluster_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Gene Name"])  # Header
            csvwriter.writerows([[gene] for gene in genes])
        
        print(f"Cluster {cluster_id}: {len(genes)} predicted cancer genes saved to {cluster_csv_path}")

        # Store for overall CSV
        for gene in genes:
            predicted_cancer_genes.append([gene, cluster_id])

    # Save all predicted cancer genes in a single CSV
    summary_csv_path = os.path.join(output_dir, "all_predicted_cancer_genes_per_cluster.csv")
    with open(summary_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Gene Name", "Cluster ID"])  # Header
        csvwriter.writerows(predicted_cancer_genes)

    print(f"All predicted cancer genes saved to {summary_csv_path}")

    return cluster_labels

def cluster_and_visualize_predicted_genes_circle(graph, non_labeled_nodes, predicted_cancer_genes, node_names, output_dir, num_clusters=10):
    """
    Clusters gene embeddings into 10 groups using KMeans, visualizes them using t-SNE, 
    and marks predicted cancer genes with red circles while keeping cluster legends.
    Also saves predicted cancer genes per cluster in CSV files.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()
    
    # Run KMeans clustering with 10 clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Define 10 distinct colors for clusters (excluding red)
    cluster_colors = [
        '#0077B6', '#0000FF', '#00B4D8', '#48EAC4', '#F1C0E8', 
        '#B9FBC0', '#32CD32', '#FFD700', '#8A2BE2', '#E377C2'
    ]

    plt.figure(figsize=(12, 10))

    # Store predicted cancer genes per cluster
    pcg_genes = {i: [] for i in range(num_clusters)}

    # Plot clusters
    for cluster_id in range(num_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=cluster_colors[cluster_id % len(cluster_colors)], 
                    edgecolor='k', s=100, alpha=0.8, 
                    label=f"Cluster {cluster_id}")

    # Mark predicted cancer genes with red circles (⚪)
    for gene_idx in predicted_cancer_genes:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, facecolors='none', edgecolors='red', s=150, linewidths=2)  # Red circle

        # Store gene in corresponding cluster
        cluster_id = cluster_labels[gene_idx]
        pcg_genes[cluster_id].append(node_names[gene_idx])

    # Legend and labels
    plt.title("Gene Clustering (10 Clusters) with Predicted Cancer Genes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Clusters", loc="best")
    
    # Save plot
    plot_path = os.path.join(output_dir, "clustered_genes_10_clusters.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Cluster visualization saved to {plot_path}")

    # Save predicted genes per cluster to separate CSV files
    predicted_cancer_genes = []
    for cluster_id, genes in pcg_genes.items():
        cluster_csv_path = os.path.join(output_dir, f"cluster_{cluster_id}_predicted_cancer_genes.csv")
        with open(cluster_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Gene Name"])  # Header
            csvwriter.writerows([[gene] for gene in genes])
        
        print(f"Cluster {cluster_id}: {len(genes)} predicted cancer genes saved to {cluster_csv_path}")

        # Store for overall CSV
        for gene in genes:
            predicted_cancer_genes.append([gene, cluster_id])

    # Save all predicted cancer genes in a single CSV
    summary_csv_path = os.path.join(output_dir, "all_predicted_cancer_genes_per_cluster.csv")
    with open(summary_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Gene Name", "Cluster ID"])  # Header
        csvwriter.writerows(predicted_cancer_genes)

    print(f"All predicted cancer genes saved to {summary_csv_path}")

    return cluster_labels

def cluster_and_visualize_predicted_genes_red_dot(graph, non_labeled_nodes, predicted_cancer_genes, node_names, output_dir, num_clusters=5):
    """
    Clusters gene embeddings using KMeans, visualizes them using t-SNE, 
    and marks predicted cancer genes in red while keeping cluster legends.
    Also saves predicted cancer genes per cluster in CSV files.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()
    
    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Define distinct colors (Cluster 4 is no longer red)
    cluster_colors = ['#0077B6', '#0000FF', '#00B4D8', '#48EAC4', '#F1C0E8']  # No red

    plt.figure(figsize=(10, 8))

    # Store predicted cancer genes per cluster
    pcg_genes = {i: [] for i in range(num_clusters)}

    # Plot clusters
    for cluster_id in range(num_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=cluster_colors[cluster_id % len(cluster_colors)], 
                    edgecolor='k', s=100, alpha=0.8, 
                    label=f"Cluster {cluster_id}")

    # Mark predicted cancer genes in red
    for gene_idx in predicted_cancer_genes:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, color='red', edgecolor='k', s=100, alpha=1.0)  # Red dot

        # Store gene in corresponding cluster
        cluster_id = cluster_labels[gene_idx]
        pcg_genes[cluster_id].append(node_names[gene_idx])

    # Legend and labels
    plt.title("Gene Clustering with Predicted Cancer Genes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Clusters", loc="best")
    
    # Save plot
    plot_path = os.path.join(output_dir, "clustered_genes.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Cluster visualization saved to {plot_path}")

    # Save predicted genes per cluster to separate CSV files
    predicted_cancer_genes = []
    for cluster_id, genes in pcg_genes.items():
        cluster_csv_path = os.path.join(output_dir, f"cluster_{cluster_id}_predicted_cancer_genes.csv")
        with open(cluster_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Gene Name"])  # Header
            csvwriter.writerows([[gene] for gene in genes])
        
        print(f"Cluster {cluster_id}: {len(genes)} predicted cancer genes saved to {cluster_csv_path}")

        # Store for overall CSV
        for gene in genes:
            predicted_cancer_genes.append([gene, cluster_id])

    # Save all predicted cancer genes in a single CSV
    summary_csv_path = os.path.join(output_dir, "all_predicted_cancer_genes_per_cluster.csv")
    with open(summary_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Gene Name", "Cluster ID"])  # Header
        csvwriter.writerows(predicted_cancer_genes)

    print(f"All predicted cancer genes saved to {summary_csv_path}")

    return cluster_labels

def cluster_and_visualize_predicted_genes_x(graph, non_labeled_nodes, predicted_cancer_genes, node_names, output_dir, num_clusters=5):
    """
    Clusters gene embeddings using KMeans, visualizes them using t-SNE, 
    and marks predicted cancer genes in red while keeping cluster legends.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()
    
    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Define distinct colors (Cluster 4 is no longer red)
    cluster_colors = ['#0077B6', '#0000FF', '#00B4D8', '#48EAC4', '#F1C0E8']  # Removed red

    plt.figure(figsize=(10, 8))

    # Store predicted cancer genes per cluster
    pcg_genes = {i: [] for i in range(num_clusters)}

    # Plot clusters
    for cluster_id in range(num_clusters):
        idx = np.where(cluster_labels == cluster_id)[0]  # Get indices of this cluster
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    color=cluster_colors[cluster_id % len(cluster_colors)], 
                    edgecolor='k', s=100, alpha=0.8, 
                    label=f"Cluster {cluster_id}")

    # Mark predicted cancer genes in red
    for gene_idx in predicted_cancer_genes:
        x, y = reduced_embeddings[gene_idx]
        plt.scatter(x, y, color='red', edgecolor='k', s=100, alpha=1.0)  # Red dot

        # Store gene in corresponding cluster
        cluster_id = cluster_labels[gene_idx]
        pcg_genes[cluster_id].append(node_names[gene_idx])

    # Legend and labels
    plt.title("Gene Clustering with Predicted Cancer Genes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Clusters", loc="best")
    
    # Save plot
    plot_path = os.path.join(output_dir, "clustered_genes.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Cluster visualization saved to {plot_path}")

    # Save predicted genes per cluster to CSV
    for cluster_id, genes in pcg_genes.items():
        cluster_csv_path = os.path.join(output_dir, f"cluster_{cluster_id}_predicted_cancer_genes.csv")
        with open(cluster_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Gene Name"])  # Header
            csvwriter.writerows([[gene] for gene in genes])
        
        print(f"Cluster {cluster_id}: {len(genes)} predicted cancer genes saved to {cluster_csv_path}")

    return cluster_labels

def cluster_and_visualize_predicted_genes_star(graph, non_labeled_nodes, predicted_cancer_genes, node_names, output_dir, num_clusters=5):
    """
    Clusters gene embeddings and visualizes them using t-SNE.
    Marks predicted cancer genes and saves them per cluster.
    """
    # Extract embeddings
    embeddings = graph.ndata['feat'].cpu().numpy()
    
    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Reduce dimensions with t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Define colors for visualization
    colors = ['#0077B6', '#0000FF', '#00B4D8', '#48EAC4', '#FF0054']
    
    plt.figure(figsize=(10, 8))
    
    # Store predicted cancer genes per cluster
    pcg_genes = {i: [] for i in range(num_clusters)}

    for i, label in enumerate(cluster_labels):
        x, y = reduced_embeddings[i]
        color = colors[label % len(colors)]

        # Mark predicted cancer genes separately
        if i in predicted_cancer_genes:
            plt.scatter(x, y, color='red', edgecolor='k', s=150, marker='*', label="Predicted Cancer Gene" if i == predicted_cancer_genes[0] else "")
            pcg_genes[label].append(node_names[i])
        else:
            plt.scatter(x, y, color=color, edgecolor='k', s=100, alpha=0.8)

    plt.title("Gene Clustering with Predicted Cancer Genes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(output_dir, "clustered_genes.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Cluster visualization saved to {plot_path}")

    # Save predicted genes per cluster to CSV
    for cluster_id, genes in pcg_genes.items():
        cluster_csv_path = os.path.join(output_dir, f"cluster_{cluster_id}_predicted_cancer_genes.csv")
        with open(cluster_csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Gene Name"])  # Header
            csvwriter.writerows([[gene] for gene in genes])
        
        print(f"Cluster {cluster_id}: {len(genes)} predicted cancer genes saved to {cluster_csv_path}")

    return cluster_labels

def visualize_clusters_with_highlighted_genes_no_save_genes(embeddings, cluster_labels, predicted_gene_indices, node_names, title, file_name, output_dir, method="tsne"):
    """
    Visualizes clustered embeddings, highlighting predicted cancer genes within each cluster.

    Args:
        embeddings (numpy.ndarray): Feature embeddings of the nodes.
        cluster_labels (numpy.ndarray): Cluster labels assigned by KMeans.
        predicted_gene_indices (list): Indices of predicted cancer genes (above threshold).
        node_names (list): List of gene names.
        title (str): Title for the plot.
        file_name (str): Name of the file to save the plot.
        output_dir (str): Directory where the plot will be saved.
        method (str): Choose between 'tsne' or 'pca' for visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reduce dimensionality using t-SNE or PCA
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Define a color palette
    colors = ['#0077B6', '#0000ff', '#00B4D8', '#48EAC4', '#ff0054', '#F1C0E8', '#CAF0F8']
    unique_labels = np.unique(cluster_labels)

    plt.figure(figsize=(10, 8))

    # Plot each cluster with a different color
    for i, label in enumerate(unique_labels):
        idx = cluster_labels == label  # Nodes in this cluster
        plt.scatter(
            reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
            label=f"Cluster {label}", 
            alpha=0.8, 
            marker='o',  
            s=100,  
            edgecolor='k',
            color=colors[i % len(colors)]
        )

    # Highlight predicted cancer genes
    for i in predicted_gene_indices:
        cluster = cluster_labels[i]
        plt.scatter(
            reduced_embeddings[i, 0], reduced_embeddings[i, 1], 
            label=f"Predicted Cancer Gene (Cluster {cluster})" if f"Predicted Cancer Gene (Cluster {cluster})" not in plt.gca().get_legend_handles_labels()[1] else "", 
            alpha=1.0, 
            marker='*',  # Star marker for cancer genes
            s=200,  # Larger size for visibility
            edgecolor='k',
            color='red'  # Distinct color for emphasis
        )

    plt.legend(title="Clusters & Cancer Genes", loc="best", fontsize=10)
    plt.title(title)
    plt.xlabel(f"{method.upper()} Dim 1")
    plt.ylabel(f"{method.upper()} Dim 2")

    plot_path = os.path.join(output_dir, file_name)
    plt.savefig(plot_path)
    plt.close()

    print(f"{method.upper()} visualization with highlighted cancer genes saved to {plot_path}")

def cluster_and_visualize_predicted_genes_(graph, non_labeled_nodes, predicted_cancer_genes, node_names, output_dir, num_clusters=5):
    """
    Performs clustering and visualization on the predicted cancer driver genes, highlighting cancer genes in each cluster.

    Args:
        graph (DGLGraph): The graph containing embeddings.
        non_labeled_nodes (list): Indices of non-labeled (predicted) nodes.
        predicted_cancer_genes (list): Indices of predicted cancer genes (above threshold).
        node_names (list): List of gene names.
        output_dir (str): Directory to save results.
        num_clusters (int): Number of clusters to form.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract embeddings for predicted nodes
    embeddings = graph.ndata['feat'][non_labeled_nodes].cpu().numpy()

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Save cluster assignments
    cluster_output_file = os.path.join(output_dir, "clustered_predicted_genes.csv")
    with open(cluster_output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Cluster Label', 'Cancer Gene'])
        for idx, (gene, cluster) in enumerate(zip([node_names[i] for i in non_labeled_nodes], cluster_labels)):
            is_cancer_gene = "Yes" if idx in predicted_cancer_genes else "No"
            csvwriter.writerow([gene, cluster, is_cancer_gene])

    print(f"Cluster assignments saved to {cluster_output_file}")

    # Visualize clusters
    visualize_clusters_with_highlighted_genes(embeddings, cluster_labels, predicted_cancer_genes, node_names, "t-SNE Visualization of Predicted Gene Clusters", "tsne_clusters.png", output_dir, method="tsne")
    visualize_clusters_with_highlighted_genes(embeddings, cluster_labels, predicted_cancer_genes, node_names, "PCA Visualization of Predicted Gene Clusters", "pca_clusters.png", output_dir, method="pca")

def visualize_clusters_ori(embeddings, cluster_labels, title, file_name, output_dir, method="tsne"):
    """
    Visualizes clustered embeddings using t-SNE or PCA, matching the defined visualization style.

    Args:
        embeddings (numpy.ndarray): Feature embeddings of the nodes.
        cluster_labels (numpy.ndarray): Cluster labels assigned by KMeans.
        title (str): Title for the plot.
        file_name (str): Name of the file to save the plot.
        output_dir (str): Directory where the plot will be saved.
        method (str): Choose between 'tsne' or 'pca' for visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reduce dimensionality using t-SNE or PCA
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Define a color palette
    colors = ['#0077B6', '#0000ff', '#00B4D8', '#48EAC4', '#ff0054', '#F1C0E8', '#CAF0F8']
    unique_labels = np.unique(cluster_labels)

    plt.figure(figsize=(10, 8))

    # Plot each cluster with a different color
    for i, label in enumerate(unique_labels):
        idx = cluster_labels == label
        plt.scatter(
            reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
            label=f"Cluster {label}", 
            alpha=1.0, 
            marker='o',  
            s=100,  
            edgecolor='k',
            color=colors[i % len(colors)]
        )

    plt.legend(title="Clusters", loc="best")
    plt.title(title)
    plt.xlabel(f"{method.upper()} Dim 1")
    plt.ylabel(f"{method.upper()} Dim 2")

    plot_path = os.path.join(output_dir, file_name)
    plt.savefig(plot_path)
    plt.close()

    print(f"{method.upper()} visualization saved to {plot_path}")

def cluster_and_visualize_predicted_genes_ori(graph, non_labeled_nodes, node_names, output_dir, num_clusters=5):
    """
    Performs clustering and visualization on the predicted cancer driver genes.

    Args:
        graph (DGLGraph): The graph containing embeddings.
        non_labeled_nodes (list): Indices of non-labeled (predicted) nodes.
        node_names (list): List of gene names.
        output_dir (str): Directory to save results.
        num_clusters (int): Number of clusters to form.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract embeddings for predicted nodes
    embeddings = graph.ndata['feat'][non_labeled_nodes].cpu().numpy()

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Save cluster assignments
    cluster_output_file = os.path.join(output_dir, "clustered_predicted_genes.csv")
    with open(cluster_output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Cluster Label'])
        for gene, cluster in zip([node_names[i] for i in non_labeled_nodes], cluster_labels):
            csvwriter.writerow([gene, cluster])

    print(f"Cluster assignments saved to {cluster_output_file}")

    # Visualize clusters
    visualize_clusters(embeddings, cluster_labels, "t-SNE Visualization of Predicted Gene Clusters", "tsne_clusters.png", output_dir, method="tsne")
    visualize_clusters(embeddings, cluster_labels, "PCA Visualization of Predicted Gene Clusters", "pca_clusters.png", output_dir, method="pca")

def perform_kmeans_clustering(embeddings, num_clusters=5):
    """
    Performs KMeans clustering on given embeddings.

    Args:
        embeddings (numpy.ndarray): Feature embeddings of the nodes.
        num_clusters (int): Number of clusters to form.

    Returns:
        numpy.ndarray: Cluster labels for each embedding.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=55)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def save_cluster_assignments(node_names, cluster_labels, output_file):
    """
    Saves the cluster assignments to a CSV file.

    Args:
        node_names (list): List of gene names.
        cluster_labels (numpy.ndarray): Cluster labels assigned by KMeans.
        output_file (str): Path to save the CSV file.
    """
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Cluster Label'])
        for gene, cluster in zip(node_names, cluster_labels):
            csvwriter.writerow([gene, cluster])
    
    print(f"Cluster assignments saved to {output_file}")

def plot_pca_tsne(embeddings, cluster_labels, output_dir, method='pca'):
    """
    Plots and saves PCA or t-SNE visualization of clustered embeddings.

    Args:
        embeddings (numpy.ndarray): Feature embeddings of the nodes.
        cluster_labels (numpy.ndarray): Cluster labels assigned by KMeans.
        output_dir (str): Directory to save the visualization.
        method (str): Choose between 'pca' or 'tsne' for visualization.
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = "PCA Visualization of Predicted Gene Clusters"
        filename = os.path.join(output_dir, "pca_clusters.png")
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        title = "t-SNE Visualization of Predicted Gene Clusters"
        filename = os.path.join(output_dir, "tsne_clusters.png")
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Cluster Label")
    plt.title(title)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.savefig(filename)
    plt.show()

    print(f"{method.upper()} visualization saved to {filename}")

def cluster_and_visualize_predicted_genes_(graph, non_labeled_nodes, node_names, output_dir, num_clusters=5):
    """
    Performs clustering and visualization on the predicted cancer driver genes.

    Args:
        graph (DGLGraph): The graph containing embeddings.
        non_labeled_nodes (list): Indices of non-labeled (predicted) nodes.
        node_names (list): List of gene names.
        output_dir (str): Directory to save results.
        num_clusters (int): Number of clusters to form.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract embeddings for predicted nodes
    embeddings = graph.ndata['feat'][non_labeled_nodes].cpu().numpy()

    # Perform KMeans clustering
    cluster_labels = perform_kmeans_clustering(embeddings, num_clusters)

    # Save cluster assignments
    cluster_output_file = os.path.join(output_dir, "clustered_predicted_genes.csv")
    save_cluster_assignments([node_names[i] for i in non_labeled_nodes], cluster_labels, cluster_output_file)

    # Visualize clusters using PCA and t-SNE
    plot_pca_tsne(embeddings, cluster_labels, output_dir, method='pca')
    plot_pca_tsne(embeddings, cluster_labels, output_dir, method='tsne')

def plot_venn_diagram():
    """Plot a large 5-set Venn diagram with optimized margins and layout, including a bottom legend."""
    # Define file paths
    file_paths = {
        "EGCN": "results/ACGNN_CPDB_predicted_driver_genes_epo1027_2048.csv",
        "HGDC": "results/HGDC_CPDB_predicted_driver_genes_epo1027_2048.csv",
        "EMOGI": "results/EMOGI_CPDB_predicted_driver_genes_epo1027_2048.csv",
        "MTGCN": "results/MTGCN_CPDB_predicted_driver_genes_epo1027_2048.csv",
        "GCN": "results/GCN_CPDB_predicted_driver_genes_epo1027_2048.csv",
    }

    # Read data from files
    gene_sets = {model: read_genes(path) for model, path in file_paths.items()}

    # Create the Venn diagram
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size for a larger plot
    venn_plot = venn(gene_sets, ax=ax)

    # Create a bottom legend
    colors = ["purple", "blue", "cyan", "lightgreen", "yellow"]
    labels = list(gene_sets.keys())
    ##handles = [plt.Line2D([0], [0], color=col, lw=8) for col in colors]
    # Create a bottom legend with larger bars
    handles = [plt.Line2D([0], [0], color=col, lw=12) for col in colors]  # Increased lw for thicker legend bars

    ax.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1),
        
        ncol=5, frameon=False, fontsize=14
    )

    # **Minimize margins**
    plt.tight_layout(pad=0.2)  # Reduce padding
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.15)  # Adjust bottom margin for the legend

    # Save and show the plot
    plt.savefig("results/venn_diagram.png", bbox_inches="tight", dpi=300)
    plt.show()

def read_genes(file_path):
    """Read gene names from a CSV file and return as a set."""
    try:
        df = pd.read_csv(file_path)  # Ensure correct delimiter if needed
        return set(df["Gene"])  # Extract unique gene names
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()

def plot_venn_diagram_ori():
    """Plot a large 5-set Venn diagram with optimized margins and layout."""
    # Define file paths
    file_paths = {
        "EGCN": "results/ACGNN_CPDB_predicted_driver_genes_epo1027_2048.csv",
        "HGDC": "results/HGDC_CPDB_predicted_driver_genes_epo1027_2048.csv",
        "EMOGI": "results/EMOGI_CPDB_predicted_driver_genes_epo1027_2048.csv",
        "MTGCN": "results/MTGCN_CPDB_predicted_driver_genes_epo1027_2048.csv",
        "GCN": "results/GCN_CPDB_predicted_driver_genes_epo1027_2048.csv",
    }

    # Read data from files
    gene_sets = {model: read_genes(path) for model, path in file_paths.items()}

    # Create the Venn diagram
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size for a larger plot
    venn_plot = venn(gene_sets, ax=ax)

    # Create a tight legend
    colors = ["purple", "blue", "cyan", "lightgreen", "yellow"]
    labels = list(gene_sets.keys())
    handles = [plt.Line2D([0], [0], color=col, lw=8) for col in colors]

    ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=5, frameon=False, fontsize=14
    )

    # **Minimize margins**
    plt.tight_layout(pad=0.2)  # Reduce padding
    plt.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.04)  # Minimize side margins

    # Save and show the plot
    plt.savefig("results/venn_diagram.png", bbox_inches="tight", dpi=300)
    plt.show()

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


def save_overall_metrics(total_time, average_time_per_epoch, average_cpu_usage, average_gpu_usage, args, output_dir):
    """
    Save the overall performance metrics to a CSV file.

    Args:
    - total_time: Total training time in seconds.
    - average_time_per_epoch: Average time per epoch in seconds.
    - average_cpu_usage: Average CPU usage in MB.
    - average_gpu_usage: Average GPU usage in MB.
    - args: Argument object containing model and network type.
    - output_dir: The directory where the results will be saved.
    """
    # Save overall metrics
    df_overall_metrics = pd.DataFrame([{
        "Model Type": args.model_type,
        "Total Time": f"{total_time:.4f}s",
        "Average Time per Epoch": f"{average_time_per_epoch:.4f}s",
        "Average CPU Usage (MB)": f"{average_cpu_usage:.2f}",
        "Average GPU Usage (MB)": f"{average_gpu_usage:.2f}"
    }])
    
    # Define path to save the CSV
    overall_metrics_csv_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_overall_performance_epo{args.num_epochs}_2048.csv')
    
    # Save to CSV
    df_overall_metrics.to_csv(overall_metrics_csv_path, index=False)
    print(f"Overall performance metrics saved to {overall_metrics_csv_path}")

def calculate_and_save_prediction_stats(non_labeled_nodes, labels, node_names, scores, args):
    """
    Calculate prediction statistics and save them to a CSV file.

    Parameters:
    - non_labeled_nodes: List of nodes without labels
    - labels: List of ground truth labels for the nodes
    - node_names: List of node names corresponding to the nodes
    - scores: List of predicted scores for the nodes
    - args: Arguments containing model and network type, score threshold, and number of epochs
    """
    # Calculate statistics
    non_labeled_nodes_count = len(non_labeled_nodes)
    ground_truth_driver_nodes = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_non_driver_nodes = [i for i, label in enumerate(labels) if label == 0]

    predicted_driver_nodes = [node_names[i] for i in non_labeled_nodes if scores[i] >= args.score_threshold]

    # Prepare data to save to CSV
    stats_output_file = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_prediction_stats_{args.num_epochs}.csv')
    
    with open(stats_output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Non-Labeled Nodes Count', 'Driver Genes', 'Non-Driver Genes', 'Total Testing Nodes', 'Predicted Driver Genes'])
        csvwriter.writerow([
            non_labeled_nodes_count,
            len(ground_truth_driver_nodes),
            len(ground_truth_non_driver_nodes),
            len(ground_truth_driver_nodes) + len(ground_truth_non_driver_nodes),
            len(predicted_driver_nodes)
        ])

    print(f"Prediction statistics saved to {stats_output_file}")

def plot_degree_distributions(sorted_degree_counts_above, sorted_degree_counts_below, args, output_dir):
    """
    Generates a box plot comparing interaction degrees of PCGs vs. Other Genes with KCGs.
    
    Parameters:
    - sorted_degree_counts_above: List of degrees for PCGs.
    - sorted_degree_counts_below: List of degrees for other genes.
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the plot.
    """

    print("Generating box plot for degree distributions...")

    degree_data = [sorted_degree_counts_above, sorted_degree_counts_below]

    plt.figure(figsize=(3, 4))

    # Create the box plot
    boxplot = plt.boxplot(
        degree_data,
        vert=True,
        patch_artist=True,  # Allows customization of box color
        flierprops=dict(marker='o', markerfacecolor='grey', markeredgecolor='grey', markersize=5, alpha=0.2),  # Outliers
        boxprops=dict(color='black'),  # Box border color
        medianprops=dict(color='blue', linewidth=2),  # Median line style
        whiskerprops=dict(color='black', linewidth=1.5),  # Whiskers
        capprops=dict(color='black', linewidth=1.5)  # Caps
    )

    # Customize frame
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # Remove top frame line
    ax.spines['right'].set_visible(False)  # Remove right frame line

    # X-axis labels
    plt.xticks([1, 2], ['PCGs', 'Other'], fontsize=8)  
    plt.yticks(fontsize=8)
    plt.ylabel('Interaction Degrees with KCGs', fontsize=10, labelpad=10) 

    # Assign different colors to box plots
    colors = ['green', 'skyblue']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_degree_distributions_epo{args.num_epochs}_2048.png')
    plt.savefig(output_plot_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    print(f"Box plot saved to {output_plot_path}")

def generate_kde_and_curves(logits, node_names, degree_counts_above, degree_counts_below, labels, train_mask, args):
    """
    Generates KDE plot comparing ACGNN score ranks with KCG interaction ranks, 
    computes Spearman correlation, and saves the KDE plot.
    Also computes and saves ROC and PR curves.

    Parameters:
    - logits: Tensor of model outputs before applying sigmoid.
    - node_names: List of node names.
    - degree_counts_above, degree_counts_below: Dictionaries mapping nodes to degree counts.
    - labels: Ground truth labels.
    - train_mask: Boolean mask indicating training samples.
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save plots.
    """

    print("Preparing data for KDE plot...")
    
    # Convert logits to probabilities
    scores = torch.sigmoid(logits).cpu().numpy()

    # Compute degree ranks
    degrees = [
        degree_counts_above.get(node_names[i], 0) +
        degree_counts_below.get(node_names[i], 0)
        for i in range(len(node_names))
    ]

    # Create DataFrame
    plot_data = pd.DataFrame({
        "Prob_pos_ranked": pd.Series(scores).rank(),
        "Degree_ranked": pd.Series(degrees).rank()
    })

    # KDE Plot
    print("Generating KDE plot...")
    plt.figure(figsize=(4, 4))
    sns.kdeplot(
        x=plot_data["Prob_pos_ranked"],
        y=plot_data["Degree_ranked"],
        cmap="Reds", fill=True,
        alpha=0.7, levels=50, thresh=0.05
    )

    # Spearman correlation
    correlation, p_value = scipy.stats.spearmanr(
        plot_data["Prob_pos_ranked"], plot_data["Degree_ranked"]
    )

    # Labels and formatting
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('ACGNN score rank', fontsize=10, labelpad=10)
    plt.ylabel('KCG interaction rank', fontsize=12, labelpad=10)

    # Add correlation text
    legend_text = f"Spearman R: {correlation:.4f}\nP-value: {p_value:.3e}"
    plt.text(
        0.05, 0.95, legend_text,
        fontsize=8, transform=plt.gca().transAxes,
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    kde_output_path = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_kde_plot_epo{args.num_epochs}_2048.png')
    plt.savefig(kde_output_path, bbox_inches='tight')
    print(f"KDE plot saved to {kde_output_path}")

    plt.tight_layout()
    plt.show()

    # Extract labeled scores and labels
    labeled_scores = scores[train_mask.cpu().numpy()]
    labeled_labels = labels[train_mask.cpu().numpy()]

    # Convert to NumPy arrays if necessary
    labeled_scores_np = labeled_scores.cpu().detach().numpy() if isinstance(labeled_scores, torch.Tensor) else labeled_scores
    labeled_labels_np = labeled_labels.cpu().detach().numpy() if isinstance(labeled_labels, torch.Tensor) else labeled_labels

    # Save ROC and PR curves
    output_file_roc = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_epo{args.num_epochs}_2048_roc_curves.png')
    output_file_pr = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_epo{args.num_epochs}_2048_pr_curves.png')

    plot_roc_curve(labeled_labels_np, labeled_scores_np, output_file_roc)
    plot_pr_curve(labeled_labels_np, labeled_scores_np, output_file_pr)

    print(f"ROC curve saved to {output_file_roc}")
    print(f"PR curve saved to {output_file_pr}")

def plot_model_performance(args):
    """
    Generates and saves a scatter plot comparing AUROC and AUPRC values 
    for different models across multiple networks.

    Parameters:
    - models: List of model names.
    - networks: List of network names.
    - auroc: 2D list of AUROC scores (rows: models, cols: networks).
    - auprc: 2D list of AUPRC scores (rows: models, cols: networks).
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the plot.
    """


    # Define models and networks
    models = ["ACGNN", "HGDC", "EMOGI", "MTGCN", "GCN", "GAT", "GraphSAGE", "GIN", "ChebNet"]
    networks = ["CPDB", "STRING", "HIPPIE"]

    # AUPRC values for ONGene and OncoKB for each model (rows: models, cols: networks)
    auroc = [
        [0.9652, 0.9578, 0.9297],  # ACGNN ACGNN & 0.9652 & 0.9783 & 0.9578 & 0.9738 & 0.9297 & 0.9597 \\
        [0.6776, 0.7133, 0.6525],  # HGDC
        [0.6735, 0.8184, 0.6672],  # EMOGI
        [0.6862, 0.7130, 0.6762],  # MTGCN
        [0.6915, 0.6688, 0.6708],  # GCN
        [0.6670, 0.8166, 0.6478],  # GAT
        [0.6664, 0.6166, 0.6571],  # GraphSAGE
        [0.5836, 0.5173, 0.5844],  # GIN
        [0.8017, 0.8777, 0.7409]   # ChebNet
    ]

    auprc = [
        [0.9783, 0.9738, 0.9597],  # ACGNN
        [0.7288, 0.7740, 0.7634],  # HGDC
        [0.7230, 0.8737, 0.7960],  # EMOGI
        [0.7712, 0.7878, 0.7785],  # MTGCN
        [0.7730, 0.7681, 0.7675],  # GCN
        [0.7086, 0.8791, 0.7496],  # GAT
        [0.7522, 0.7182, 0.7624],  # GraphSAGE
        [0.6405, 0.5918, 0.6791],  # GIN
        [0.8622, 0.9159, 0.8443]   # ChebNet
    ]

    # Compute averages for each model
    avg_auroc = np.mean(auroc, axis=1)
    avg_auprc = np.mean(auprc, axis=1)

    # Define colors for models and unique shapes for networks
    colors = ['red', 'grey', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink']
    network_markers = ['P', '^', 's']  # One shape for each network
    avg_marker = 'o'  # Marker for average points

    # Create the plot
    plt.figure(figsize=(8, 7))

    # Plot individual points for each model and network
    for i, model in enumerate(models):
        for j, network in enumerate(networks):
            plt.scatter(auprc[i][j], auroc[i][j], color=colors[i], 
                        marker=network_markers[j], s=90, alpha=0.6)

    # Add average points for each model
    for i, model in enumerate(models):
        plt.scatter(avg_auprc[i], avg_auroc[i], color=colors[i], marker=avg_marker, 
                    s=240, edgecolor='none', alpha=0.5)

    # Create legends for models (colors) and networks (shapes)
    model_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                            markersize=14, label=models[i], alpha=0.5) for i in range(len(models))]
    network_legend = [Line2D([0], [0], marker=network_markers[i], color='k', linestyle='None', 
                            markersize=8, label=networks[i]) for i in range(len(networks))]

    # Add legends
    network_legend_artist = plt.legend(handles=network_legend, loc='lower right', title="Networks", fontsize=12, title_fontsize=14, frameon=True)
    plt.gca().add_artist(network_legend_artist)
    plt.legend(handles=model_legend, loc='upper left', fontsize=12, frameon=True)

    # Labels and title
    plt.ylabel("AUPRC", fontsize=14)
    plt.xlabel("AUROC", fontsize=14)

    # Save the plot
    comp_output_path = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_comp_plot_epo{args.num_epochs}_2048.png')
    plt.savefig(comp_output_path, bbox_inches='tight')
    
    print(f"Comparison plot saved to {comp_output_path}")

    # Show plot
    plt.tight_layout()
    plt.show()

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

def save_predicted_scores(scores, labels, nodes, args):
    """
    Saves predicted scores and labels to a CSV file.

    Parameters:
    - scores: List of predicted scores.
    - labels: List of ground-truth labels.
    - nodes: Dictionary of node names.
    - args: Arguments containing model configuration.
    """
    # Initialize variables to calculate average scores and standard deviations
    label_scores = {0: [], 1: [], 2: [], 3: []}  # Groups for each label

    # Define CSV file path
    csv_file_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Save results to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Node Name', 'Score', 'Label'])  # Header

        for i, score in enumerate(scores):
            label = int(labels[i].item())  # Ensure label is an integer
            
            if label in [1, 0]:  # Ground-truth labels
                writer.writerow([list(nodes.keys())[i], score, label])
                label_scores[label].append(score)
            elif label == -1 and score >= args.score_threshold:  # Predicted driver genes
                writer.writerow([list(nodes.keys())[i], score, 2])
                label_scores[2].append(score)
            else:  # Non-labeled nodes or other
                writer.writerow([list(nodes.keys())[i], score, 3])
                label_scores[3].append(score)

    print(f"Predicted scores and labels saved to {csv_file_path}")

    return label_scores  # Returning for further analysis if needed

def save_average_scores(label_scores, args):
    """
    Calculates and saves the average score, standard deviation, and number of nodes per label.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    # Define CSV file path
    average_scores_file = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_average_scores_by_label_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(average_scores_file), exist_ok=True)

    # Save average scores to CSV
    with open(average_scores_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Average Score', 'Standard Deviation', 'Number of Nodes'])  # Header

        for label, scores_list in label_scores.items():
            if scores_list:  # Check if the list is not empty
                avg_score = np.mean(scores_list)
                std_dev = np.std(scores_list)
                num_nodes = len(scores_list)
            else:
                avg_score = 0.0  # Default if no nodes in the label group
                std_dev = 0.0
                num_nodes = 0

            writer.writerow([label, avg_score, std_dev, num_nodes])

    print(f"Average scores by label saved to {average_scores_file}")

def plot_average_scores(label_scores, args):
    """
    Plots average scores with error bars and saves the figure.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    labels_list = []
    avg_scores = []
    std_devs = []

    for label, scores_list in label_scores.items():
        if scores_list:
            labels_list.append(label)
            avg_scores.append(np.mean(scores_list))
            std_devs.append(np.std(scores_list))

    if not labels_list:
        print("No valid scores to plot.")
        return

    # Define plot save path
    plot_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_average_scores_with_error_bars_threshold{args.score_threshold}_epo{args.num_epochs}.png'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.bar(labels_list, avg_scores, yerr=std_devs, capsize=5, color='skyblue', alpha=0.7)
    plt.xlabel('Label')
    plt.ylabel('Average Score')
    plt.title('Average Scores by Label with Error Bars')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Save and close plot
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Error bar plot saved to {plot_path}")

def plot_score_distributions(label_scores, args):
    """
    Plots score distributions for each label and saves the figures.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    for label, scores_list in label_scores.items():
        if scores_list:
            plt.figure(figsize=(8, 6))
            plt.hist(scores_list, bins=20, alpha=0.7, color='#98f5e1', edgecolor='black')

            # Set labels and tick sizes
            plt.xlabel('Score', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Customize grid and tick appearance
            plt.tick_params(axis='both', which='major', length=6, width=2, direction='inout', grid_color='gray', grid_alpha=0.5)
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            # Define plot save path
            plot_path = os.path.join(
                'results/gene_prediction/',
                f'{args.model_type}_{args.net_type}_score_distribution_label{label}_threshold{args.score_threshold}_epo{args.num_epochs}.png'
            )

            # Ensure directory exists
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

            # Save and close plot
            plt.savefig(plot_path)
            plt.close()

            print(f"Score distribution for label {label} saved to {plot_path}")

def save_performance_metrics(epoch_times, cpu_usages, gpu_usages, args):
    """
    Saves performance metrics per epoch, including time per epoch, CPU, and GPU usage.

    Parameters:
    - epoch_times: List of epoch durations (in seconds).
    - cpu_usages: List of CPU memory usage per epoch (in MB).
    - gpu_usages: List of GPU memory usage per epoch (in MB).
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the metrics CSV file.
    """

    # Compute total and average performance metrics
    total_time = sum(epoch_times)
    avg_time_per_epoch = total_time / args.num_epochs
    avg_cpu_usage = sum(cpu_usages) / args.num_epochs
    avg_gpu_usage = sum(gpu_usages) / args.num_epochs

    # Create DataFrame with per-epoch metrics
    df_metrics = pd.DataFrame({
        "Epoch": range(1, args.num_epochs + 1),
        "Time per Epoch (s)": epoch_times,
        "CPU Usage (MB)": cpu_usages,
        "GPU Usage (MB)": gpu_usages
    })

    # Define CSV path
    metrics_csv_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_performance_metrics_epo{args.num_epochs}_2048.csv'
    )

    # Save to CSV
    df_metrics.to_csv(metrics_csv_path, index=False)

    print(f"Epoch performance metrics saved to {metrics_csv_path}")

    # Print summary statistics
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {avg_time_per_epoch:.2f} seconds")
    print(f"Average CPU Usage: {avg_cpu_usage:.2f} MB")
    print(f"Average GPU Usage: {avg_gpu_usage:.2f} MB")

    
def get_neighbors_gene_names_direct_graph(graph, node_names, name_to_index, target_genes):
    neighbors_dict = {}

    for gene in target_genes:
        if gene not in name_to_index:
            print(f"⚠️ Gene {gene} not found in graph.")
            continue

        node_id = name_to_index[gene]
        # Get successors and predecessors, combine them
        neighbors = graph.successors(node_id).tolist() + graph.predecessors(node_id).tolist()
        unique_neighbors = list(set(neighbors))  # remove duplicates

        # Get the actual gene names, skip self-loops
        neighbor_gene_names = [node_names[n] for n in unique_neighbors if n != node_id]
        neighbors_dict[gene] = neighbor_gene_names

    return neighbors_dict

def plot_gene_feature_contributions_no_probability_added(gene_name, relevance_vector, feature_names, output_path=None):
    """
    Plot a heatmap of feature contributions for a single gene.

    Parameters:
        gene_name (str): The name of the gene (e.g. 'NRAS').
        relevance_vector (np.ndarray): 1D array of relevance scores for each feature (shape: 64, assuming 4 omics × 16 cancers).
        feature_names (list of str): List of all feature names like 'BRCA_mf', 'BRCA_cna', ...
        output_path (str, optional): Path to save the plot.
    """
    assert len(relevance_vector) == 64, "Expected 64 feature contributions (4 omics × 16 cancers)."

    # Create DataFrame
    df = pd.DataFrame({'Feature': feature_names, 'Relevance': relevance_vector})
    ##print('df--------------------------------------------------------------------- ',df)
    barplot_path = output_path.replace(".png", "_omics_barplot.png")
    plot_omics_barplot(df, barplot_path)

    df[['Cancer', 'Omics']] = df['Feature'].str.split('_', expand=True)

    # Pivot to wide format
    heatmap_data = df.pivot(index='Omics', columns='Cancer', values='Relevance')
    heatmap_data = heatmap_data.loc[['SNV', 'CNA', 'Meth', 'Expr']] if 'SNV' in df['Omics'].values else heatmap_data

    # Plot
    plt.figure(figsize=(8, 2.5))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar=False, linewidths=0.3, linecolor='gray')
    plt.title('Feature Contributions', fontsize=12)
    plt.yticks(rotation=0)
    plt.xticks(rotation=60, ha='right')
    plt.xlabel('')
    plt.ylabel('')
    plt.suptitle(gene_name, fontsize=18, fontweight='bold', x=0.05, y=1.15, ha='left')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
