
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
from .models import ACGNN, HGDC, EMOGI, MTGCN, GCN, GAT, GraphSAGE, GIN, ChebNet, FocalLoss
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

'''CLUSTER_COLORS = {
    0: '#0077B6',  1: '#0000FF',  2: '#00B4D8',  3: '#48EAC4',
    4: '#F1C0E8',  5: '#B9FBC0',  6: '#32CD32',  7: '#bee1e6',
    8: '#8A2BE2',  9: '#E377C2', 10: '#8EECF5', 11: '#A3C4F3'
}'''

# ✅ 9 fixed cluster colors
CLUSTER_COLORS = {
    0: '#0077B6',  1: '#00B4D8',  2: '#F1C0E8',
    3: '#B9FBC0',  4: '#32CD32', 5: '#8A2BE2',
    6: '#E377C2',  7: '#8EECF5', 8: '#A3C4F3'
}

def train(args):
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
    best_k = 9##find_optimal_k(node_features, k_range=(5, 20), plot_path=plot_path, save_best_k_path=best_k_path)

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
    topk_indices = torch.topk(scores, topk).indices

    # ✅ Filter relevance scores and cluster labels
    relevance_scores_topk = relevance_scores[topk_indices]
    cluster_labels_topk = graph.ndata['cluster'][topk_indices]

    # ✅ Update node names for top 1000
    node_names_topk = [node_names[i] for i in topk_indices.cpu().numpy()]

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
        scores[topk_indices], 
        tsne_plot_path,
        args
    )

    print(f"t-SNE plot saved to {tsne_plot_path}")
 
    tsne_plot_path = os.path.join(output_dir, f"{args.model_type}_{args.net_type}_top1000_predicted_genes_tsne_epo{args.num_epochs}.png")
    ##plot_top_predicted_genes_tsne(graph, node_names, torch.sigmoid(logits), tsne_plot_path)
    plot_top_predicted_genes_tsne(
        graph, 
        node_names_topk, 
        scores[topk_indices], 
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

def apply_spectral_biclustering(graph, embeddings, n_clusters=12, normalize=True):
    """
    Applies Spectral Biclustering to node embeddings and assigns cluster labels to the graph.

    Args:
        graph (dgl.DGLGraph): DGL graph object.
        embeddings (torch.Tensor): Node embedding tensor (num_nodes, feature_dim).
        n_clusters (int): Number of row clusters (genes).
        normalize (bool): Whether to normalize embeddings before clustering.

    Returns:
        DGLGraph: Graph with 'cluster' node data field updated.
    """
    print(f"Running Spectral Biclustering with {n_clusters} row clusters...")

    node_features = embeddings.cpu().numpy()

    if normalize:
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)

    bicluster = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state=42)
    bicluster.fit(node_features)

    cluster_labels = bicluster.row_labels_
    graph.ndata['cluster'] = torch.tensor(cluster_labels, dtype=torch.long)

    print("Spectral Biclustering complete. Cluster labels assigned to graph.")
    return graph

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

def extract_summary_features_np(features_np):
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

def plot_spectral_biclustering_heatmap(
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
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )
    ax_curve.set_ylim(ax.get_ylim())  # 🔥 Sync y-axis limits


    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1, 1,
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
        color='black',
        linewidth=0.6
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

def plot_spectral_biclustering_heatmap_no_space_subplot(
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

    fig = plt.figure(figsize=(16, 18))
    '''gs = fig.add_gridspec(nrows=10, ncols=1)
    ax = fig.add_subplot(gs[:-1, 0])       # heatmap
    ax_legend = fig.add_subplot(gs[-1, 0]) # legend

    fig = plt.figure(figsize=(18, 16))  # make taller and wider'''
    gs = fig.add_gridspec(nrows=10, ncols=2)  # more columns

    ax = fig.add_subplot(gs[:-1, :18])        # heatmap
    ax_bar = fig.add_subplot(gs[:-1, 18:])    # ✅ contribution bar
    ax_legend = fig.add_subplot(gs[-1, :0])    # legend

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

def plot_spectral_biclustering_heatmap_(
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
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )
    ax_curve.set_ylim(ax.get_ylim())  # 🔥 Sync y-axis limits


    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1, 1,
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
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=12)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # ✅ Right-hand sum curve
    row_contributions = sorted_scores.sum(axis=1)
    ##row_contributions = row_contributions / np.max(row_contributions) ##skip normalization to max
    ## to keep the natural scale differences.

    # ✅ Normalize to [0, 1] scale just for drawing, but retain original shape
    normalized_contribs = row_contributions / (row_contributions.max() + 1e-8)

    # Make curve narrower by applying a scale factor
    curve_scale = 0.3  # Change this for narrower/wider curves

    ax_curve.fill_betweenx(
        np.arange(len(row_contributions)),
        0,
        normalized_contribs * curve_scale,
        color='gray',
        alpha=0.6
    )
    ax_curve.plot(
        normalized_contribs * curve_scale,
        np.arange(len(row_contributions)),
        color='gray',
        linewidth=0.6
    )

    ax_curve.set_xlim(0, curve_scale)
    ax_curve.set_xticks([])
    ax_curve.set_yticks([])
    ax_curve.set_facecolor("white")
    ax_curve.spines[:].set_visible(False)

    # ✅ Add top axis with [0, 1] scale
    ax_top = ax_curve.twiny()
    ax_top.set_xlim(0, 1)
    ax_top.set_xticks([0, 1])
    ax_top.set_xticklabels(["0", "1"], fontsize=9)
    ax_top.xaxis.set_ticks_position('top')
    ax_top.spines["top"].set_position(("axes", 1.02))
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)


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

def plot_spectral_biclustering_heatmap_in_between_pa(
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


    # fig = plt.figure(figsize=(16, 12))
    # gs = fig.add_gridspec(nrows=10, ncols=1)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(nrows=10, ncols=40)
    ax = fig.add_subplot(gs[:-1, 1:30])        # heatmap
    ax_curve = fig.add_subplot(gs[:-1, 30:32], sharey=ax)  # sum curve
    ax_cbar = fig.add_subplot(gs[:-1, 32])     # colorbar
    ax_legend = fig.add_subplot(gs[-1, 1:30])  # legend

    ##ax = fig.add_subplot(gs[:-1, 0])       # heatmap
    ##ax_legend = fig.add_subplot(gs[-1, 0]) # legend

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
        cbar_ax=ax_cbar,  # 👈 use the custom colorbar axis
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

    # ✅ Add cluster labels with counts on y-axis, near axis
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -5, center_y, f"{count}",  # just number of genes
            va='center', ha='right', fontsize=9, fontweight='bold'
        )

    # ---- 🔹 Colored x-axis labels ----
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=6)
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

    # 🔹 Plot sum curve (sum of contributions per gene)
    curve_data = sorted_scores.sum(axis=1)
    ax_curve.plot(curve_data, np.arange(len(curve_data)), color='black')
    ax_curve.invert_yaxis()
    ax_curve.axis('off')


    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_fun(
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
    gs = fig.add_gridspec(
        nrows=20, ncols=18,  # Increase rows for better layout granularity
        height_ratios=[0.5] + [1]*15 + [0.5, 0.5, 1, 1],  # Top bar is row 0 (narrow)
        wspace=0
    )


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
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )
    ax_curve.set_ylim(ax.get_ylim())  # 🔥 Sync y-axis limits


    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1, 1,
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
        linewidth=0.6
    )

    normalized_contribs = row_contributions / (row_contributions.max() + 1e-8)
    # Make curve narrower by applying a scale factor
    curve_scale = 0.3  # Change this for narrower/wider curves

    ax_curve.fill_betweenx(
        np.arange(len(row_contributions)),
        0,
        normalized_contribs * curve_scale,
        color='gray',
        alpha=0.6
    )
    ax_curve.plot(
        normalized_contribs * curve_scale,
        np.arange(len(row_contributions)),
        color='gray',
        linewidth=0.6
    )

    ax_curve.set_xlim(0, curve_scale)
    ax_curve.set_xticks([])
    ax_curve.set_yticks([])
    ax_curve.set_facecolor("white")
    ax_curve.spines[:].set_visible(False)

    # ✅ Add top axis with [0, 1] scale
    ax_top = ax_curve.twiny()
    ax_top.set_xlim(0, 1)
    ax_top.set_xticks([0, 1])
    ax_top.set_xticklabels(["0", "1"], fontsize=9)
    ax_top.xaxis.set_ticks_position('top')
    ax_top.spines["top"].set_position(("axes", 1.02))
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)



    # ✅ Legend (omics types)
    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(handles=omics_patches, loc="center", ncol=len(omics_patches), fontsize=12, frameon=False)
 
    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False,
        markerscale=2,  # Increase the size of the patches
        fontsize=12  # Increase the font size of the labels
    )


    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    # ✅ Horizontal contribution bar to the right of the filled curve with narrower size, min 0, and max 1
    ax_contrib_cbar = fig.add_subplot(gs[1:16, 17:18])  # Adjust the position of the colorbar right next to the filled curve
    norm = Normalize(vmin=0, vmax=1)

    # Create a horizontal colorbar with the desired settings
    cbar = ColorbarBase(
        ax_contrib_cbar,
        cmap=plt.cm.Greys,
        norm=norm,
        orientation='horizontal'
    )

    # Set min and max ticks (0 and 1)
    cbar.set_ticks([0, 1])  # Manually setting ticks to 0 and 1

    # Make the colorbar narrower by adjusting its width
    ax_contrib_cbar.set_aspect(10)  # Adjust the aspect ratio to make the colorbar narrower

    # Adjust label sizes for clarity
    cbar.ax.tick_params(labelsize=8)  # Set font size for tick labels


    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")

def plot_spectral_biclustering_heatmap_curve_better(
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
    gs = fig.add_gridspec(nrows=18, ncols=19, wspace=0)


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
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )
    ax_curve.set_ylim(ax.get_ylim())  # 🔥 Sync y-axis limits


    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1, 1,
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
        color='black',
        linewidth=0.6
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

def plot_spectral_biclustering_heatmap_comment_out_magnify_normalization(
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
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )
    ax_curve.set_ylim(ax.get_ylim())  # 🔥 Sync y-axis limits


    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1, 1,
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
        color='black',
        linewidth=0.6
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

def plot_spectral_biclustering_heatmap_pa(
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
    row_min = sorted_scores.min(axis=1, keepdims=True)
    row_max = sorted_scores.max(axis=1, keepdims=True)
    sorted_scores = (sorted_scores - row_min) / (row_max - row_min + 1e-8)

    # ✅ Main heatmap
    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )
    ax_curve.set_ylim(ax.get_ylim())  # 🔥 Sync y-axis limits


    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1, i), 2, 1,
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
        color='black',
        linewidth=0.6
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

def plot_spectral_biclustering_heatmap_connected_curve(
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
    row_min = sorted_scores.min(axis=1, keepdims=True)
    row_max = sorted_scores.max(axis=1, keepdims=True)
    sorted_scores = (sorted_scores - row_min) / (row_max - row_min + 1e-8)

    # ✅ Main heatmap
    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )
    ax_curve.set_ylim(ax.get_ylim())  # 🔥 Sync y-axis limits


    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1, i), 2, 1,
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
    row_contributions = row_contributions / np.max(row_contributions)
    offset = 0.15
    ax_curve.fill_betweenx(
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
    )
    ax_curve.set_xlim(0, offset)
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

def plot_spectral_biclustering_heatmap_still_leftward(
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
    gs = fig.add_gridspec(nrows=18, ncols=18)

    ax = fig.add_subplot(gs[1:16, :16])         # Main heatmap
    ax_curve = fig.add_subplot(gs[1:16, 16:17]) # Sum curve
    ax_legend = fig.add_subplot(gs[-1, :])      # Legend at bottom

    # Cluster info for boundaries and counts
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # Normalize rows
    row_min = sorted_scores.min(axis=1, keepdims=True)
    row_max = sorted_scores.max(axis=1, keepdims=True)
    sorted_scores = (sorted_scores - row_min) / (row_max - row_min + 1e-8)

    # ✅ Main heatmap
    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )

    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-3, i), 3, 1,
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
    row_contributions = row_contributions / np.max(row_contributions)
    offset = 0.15
    ax_curve.fill_betweenx(
        np.arange(len(row_contributions)),
        0,
        row_contributions * offset,
        color='gray',
        alpha=0.6
    )
    ax_curve.plot(
        row_contributions * offset,
        np.arange(len(row_contributions)),
        color='black',
        linewidth=0.6
    )
    ax_curve.set_xlim(0, offset)
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

def plot_spectral_biclustering_heatmap_funny_ridiculous_plot(
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

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(nrows=12, ncols=20)

    ax_legend = fig.add_subplot(gs[0, :])
    ax = fig.add_subplot(gs[1:11, :16])         # Main heatmap
    ax_curve = fig.add_subplot(gs[1:11, 16:18]) # Sum curve
    ax_bar = fig.add_subplot(gs[11, :16])       # Bottom feature type bar

    # Cluster info
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # Normalize rows
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

    # ✅ Left color bar for clusters
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1, i), 1, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # ✅ Cluster counts on the left
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # ✅ X-axis labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=10)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # ✅ Right-hand sum curve tightly hugging edge
    row_contributions = sorted_scores.sum(axis=1)
    row_contributions = row_contributions / np.max(row_contributions)

    offset = 0.15
    ax_curve.fill_betweenx(
        np.arange(len(row_contributions)),
        1 - row_contributions * offset,
        1,
        color='gray',
        alpha=0.6
    )
    ax_curve.plot(
        1 - row_contributions * offset,
        np.arange(len(row_contributions)),
        color='black',
        linewidth=0.6
    )
    ax_curve.set_xlim(1 - offset, 1)
    ax_curve.set_xticks([])
    ax_curve.set_yticks([])
    ax_curve.set_facecolor("white")
    ax_curve.spines[:].set_visible(False)


    # ✅ Feature type bar (bottom)
    feature_rgba = [to_rgba(color) for color in feature_colors]
    ax_bar.imshow([feature_rgba], aspect='auto')

    ax_bar.set_xticks(np.arange(len(feature_names)))
    ax_bar.set_xticklabels(pretty_labels, rotation=90, fontsize=10)
    ax_bar.set_yticks([])
    ax_bar.set_ylabel("Feature type", fontsize=12)

    for label, color in zip(ax_bar.get_xticklabels(), feature_colors):
        label.set_color(color)


    # ✅ Legend
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


def plot_spectral_biclustering_heatmap_orr(
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

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(nrows=10, ncols=1)
    ax = fig.add_subplot(gs[:-1, 0])       # heatmap
    ax_legend = fig.add_subplot(gs[-1, 0]) # legend

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

    # ✅ Add cluster labels with counts on y-axis, near axis
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -5, center_y, f"{count}",  # just number of genes
            va='center', ha='right', fontsize=9, fontweight='bold'
        )

    # ---- 🔹 Colored x-axis labels ----
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=6)
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

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(nrows=12, ncols=20)

    ax_legend = fig.add_subplot(gs[0, :])
    ax = fig.add_subplot(gs[1:11, :16])         # Heatmap
    ax_curve = fig.add_subplot(gs[1:11, 16:18]) # Silhouette curve
    ax_bar = fig.add_subplot(gs[11, :16])       # Feature bar

    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # Normalize per row
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
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=11)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # ✅ Compute row-wise sum of contributions
    row_contributions = sorted_scores.sum(axis=1)
    row_contributions = row_contributions / np.max(row_contributions)

    # ✅ Fill curve tightly next to heatmap, short outward reach
    offset = 0.15
    ax_curve.fill_betweenx(
        np.arange(len(row_contributions)),
        1 - row_contributions * offset,  # Hug heatmap, extend slightly
        1,
        color='gray',
        alpha=0.6
    )
    ax_curve.plot(
        1 - row_contributions * offset,
        np.arange(len(row_contributions)),
        color='black',
        linewidth=0.6
    )
    ax_curve.set_xlim(1 - offset, 1)
    ax_curve.set_xticks([])
    ax_curve.set_yticks([])
    ax_curve.set_facecolor("white")
    ax_curve.spines[:].set_visible(False)

    # ✅ Feature bar at bottom
    ax_bar.imshow([feature_colors], aspect='auto')
    ax_bar.set_xticks(np.arange(len(feature_names)))
    ax_bar.set_xticklabels(pretty_labels, rotation=90, fontsize=10)
    ax_bar.set_yticks([])
    ax_bar.set_ylabel("Feature type", fontsize=12)
    for label, color in zip(ax_bar.get_xticklabels(), feature_colors):
        label.set_color(color)

    # ✅ Omics legend on top
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

def plot_tsne_predicted_genes(graph, node_names, scores, output_path, args):
    cluster_ids = graph.ndata['cluster'].cpu().numpy()
    embeddings = graph.ndata['feat'].cpu().numpy()
    scores = scores.cpu().numpy()

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
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=f"Cluster {c}", markersize=10)
               for c, color in CLUSTER_COLORS.items() if any(cluster_ids[predicted_mask] == c)]
    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE of Top 2 Predicted Genes per Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

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





def train_bicluster_pass(args):
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
    ax.set_title("Spectral Biclustering of LRP Relevance Scores")

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
    ax.set_title("Spectral Biclustering of LRP Relevance Scores")

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
    ax.set_title("Spectral Biclustering of LRP Relevance Scores")

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
    ax.set_title("Spectral Biclustering of LRP Relevance Scores")

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
    ax.set_title("Spectral Biclustering of LRP Relevance Scores")

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
    ax.set_xticklabels(omics_splits, rotation=45, ha='right', fontsize=12)
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

def plot_top_predicted_genes_tsne(graph, node_names, scores, output_path, top_k=1000):
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
            # Highlight the top 1 with a yellow circle (half the size of the original dot)
            plt.scatter(x, y, s=75, edgecolors='yellow', alpha=1.0, linewidth=2, marker='o', color='yellow')
            # Change label text color to red
            plt.text(x, y, name, fontsize=9, fontweight='bold', ha='center', va='center', color='red')

    plt.title("t-SNE of Top 1000 Predicted Genes by Cluster", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    # Remove legend
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Top predicted gene t-SNE plot saved to:\n{output_path}")

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

def train_ori(args):
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

    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    # Save cluster labels into graph.ndata for later use
    graph.ndata['cluster'] = torch.tensor(cluster_labels)

    # --- Begin LRP-related section ---
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    all_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]
    if all_gene_indices:
        avg_degree = graph.ndata['degree'][all_gene_indices].float().mean().item()
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
    cluster_to_genes = {cluster_id: [] for cluster_id in total_counts}
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
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    all_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if all_gene_indices:
        avg_degree = graph.ndata['degree'][all_gene_indices].float().mean().item()
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
    cluster_to_genes = {cluster_id: [] for cluster_id in total_counts}
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

    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    '''output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)'''

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    all_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if all_gene_indices:
        avg_degree = graph.ndata['degree'][all_gene_indices].float().mean().item()
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
    cluster_to_genes = {cluster_id: [] for cluster_id in total_counts}
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

    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
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
    save_predicted_genes_by_cluster(predicted_cancer_genes, total_counts, output_path_cluster_genes)

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
    plt.xlabel('Relevance Score')
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

def save_predicted_genes_by_cluster(predicted_cancer_genes, total_counts, output_path):
    """
    Save the predicted cancer genes by cluster.
    """
    cluster_to_genes = {cluster_id: [] for cluster_id in total_counts}
    for gene in predicted_cancer_genes:
        cluster_id = gene % len(total_counts)  # Assuming cluster_id is determined by gene's index
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
    plt.xlabel("Relevance Score")
    plt.title(f"Top {top_k} Feature Importances for Node {node_name}")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(relevance_vector, feature_names=None, node_name=None, top_k=20):
    relevance_vector = relevance_vector.detach().cpu().numpy()
    norm_scores = normalize(torch.tensor(relevance_vector)).numpy()

    top_indices = np.argsort(norm_scores)[-top_k:][::-1]
    top_scores = norm_scores[top_indices]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(relevance_vector))]
    top_names = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(10, 5))
    plt.barh(top_names[::-1], top_scores[::-1], color='skyblue')
    plt.xlabel("Relevance Score")
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

def compute_lrp_scores(model, graph, features, node_indices=None):
    model.eval()
    features.requires_grad = True

    with torch.enable_grad():
        logits = model(graph, features)
        probs = torch.sigmoid(logits.squeeze())

        # Select the nodes to analyze (e.g., predicted cancer genes)
        if node_indices is None:
            node_indices = torch.nonzero((probs > 0.5)).squeeze()

        relevance_scores = torch.zeros_like(features)

        for idx in node_indices:
            model.zero_grad()
            probs[idx].backward(retain_graph=True)
            relevance_scores[idx] = features.grad[idx].detach()

    return relevance_scores

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
    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )
    '''
    output_path_genes_clusters = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_clusters_epo{args.num_epochs}.png')
    output_path_predicted_genes = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_genes.csv')

    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
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
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )
    
    # === KCG Plotting ===
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')#('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
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
    cluster_to_genes = {cluster_id: [] for cluster_id in total_counts}
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
    cluster_labels_dict, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, top_predicted_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels_tensor, output_path=output_path_predicted
    )

    # KCG plot
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')##('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
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
    
    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, 
        output_path_genes_clusters
    )
    
    output_path_legend = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_legend_epo{args.num_epochs}.png')

    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)
            
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    output_path_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_ground_truth_epo{args.num_epochs}.png')
    
    plot_pcg_cancer_genes_ground_truth(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, ground_truth_cancer_genes=ground_truth_cancer_genes,
        node_names=node_names, cluster_labels=cluster_labels, output_path=output_path_percent,
        ##cluster_colors=CLUSTER_COLORS
    )

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
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

def train_8_5_pass(args):
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
    
    '''cluster_labels, pred_counts, total_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters, output_path_predicted_cancer_genes,
        cluster_colors=CLUSTER_COLORS
    )'''

    '''CLUSTER_COLORS = {
        0: '#0077B6',  1: '#0000FF',  2: '#00B4D8',  3: '#48EAC4',
        4: '#F1C0E8',  5: '#B9FBC0',  6: '#32CD32',  7: '#FFD700',
        8: '#8A2BE2',  9: '#E377C2', 10: '#8EECF5', 11: '#A3C4F3'
    }
    cluster_colors = [CLUSTER_COLORS[i] for i in range(12)]'''
    
    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, 
        output_path_genes_clusters
    )
    
    output_path_legend = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_legend_epo{args.num_epochs}.png')

    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)
            
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    output_path_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_ground_truth_epo{args.num_epochs}.png')
    
    plot_pcg_cancer_genes_ground_truth(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, ground_truth_cancer_genes=ground_truth_cancer_genes,
        node_names=node_names, cluster_labels=cluster_labels, output_path=output_path_percent,
        ##cluster_colors=CLUSTER_COLORS
    )

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
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

    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_all_predicted_cluster_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
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

    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, top_1000_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/intogen_23444.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_pcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
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

    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    '''output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)'''

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    all_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if all_gene_indices:
        avg_degree = graph.ndata['degree'][all_gene_indices].float().mean().item()
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
    cluster_to_genes = {cluster_id: [] for cluster_id in total_counts}
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

    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/ncg_8886.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    all_gene_indices = [name_to_index[name] for name in predicted_cancer_genes if name in name_to_index]

    if all_gene_indices:
        avg_degree = graph.ndata['degree'][all_gene_indices].float().mean().item()
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

    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, top_1000_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    # PCG counts (from pred_counts earlier)
    pcg_counts = pred_counts  # already computed above

    # Plot percentages
    '''output_path_kcg_pcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_pcg_percent_epo{args.num_epochs}.png')
    plot_kcg_pcg_percentages(
        clusters=total_counts.keys(),
        kcg_counts=kcg_counts,
        pcg_counts=pcg_counts,
        total_counts=total_counts,
        output_path=output_path_kcg_pcg
    )'''


    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    '''output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )'''

    # Plot KCG percentage
    output_path_kcg_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_percent_epo{args.num_epochs}.png')
    plot_percentage_per_cluster(
        counts=kcg_counts,
        total_counts=total_counts,
        clusters=total_counts.keys(),
        label="KCGs",
        color="orange",
        output_path=output_path_kcg_percent
    )

    # Plot PCG percentage
    output_path_pcg_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_percent_epo{args.num_epochs}.png')
    plot_percentage_per_cluster(
        counts=pred_counts,
        total_counts=total_counts,
        clusters=total_counts.keys(),
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

    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, top_1000_genes, node_names, output_path_genes_clusters
    )

    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_predicted
    )

    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    name_to_index = {name: idx for idx, name in enumerate(node_names)}
    gt_indices = set(name_to_index[name] for name in ground_truth_cancer_genes if name in name_to_index)

    kcg_counts = {}
    for cluster_id in total_counts:
        cluster_nodes = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        kcg_in_cluster = [i for i in cluster_nodes if i in gt_indices]
        kcg_counts[cluster_id] = len(kcg_in_cluster)

    output_path_kcg = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_kcg_epo{args.num_epochs}.png')
    plot_kcg_cancer_genes(
        clusters=total_counts.keys(), kcg_count=kcg_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_kcg
    )

    top_1000_gene_indices = [name_to_index[name] for name in top_1000_genes if name in name_to_index]

    if top_1000_gene_indices:
        avg_degree = graph.ndata['degree'][top_1000_gene_indices].float().mean().item()
        print(f"Average degree of top 1000 predicted nodes: {avg_degree:.2f}")
    else:
        print("No top 1000 nodes predicted above the threshold.")

def plot_percentage_per_cluster(counts, total_counts, clusters, label, color, output_path):
    """
    Plot the percentage of a specific gene type (KCG or PCG) per cluster.

    Args:
        counts (dict): Count of genes (KCG or PCG) per cluster.
        total_counts (dict): Total number of genes per cluster.
        clusters (iterable): Cluster IDs.
        label (str): Label for the bar (e.g., "KCGs" or "PCGs").
        color (str): Color of the bars.
        output_path (str): Where to save the plot.
    """
    clusters = list(clusters)
    percentages = [100 * counts.get(c, 0) / total_counts[c] for c in clusters]

    plt.figure(figsize=(12, 6))
    plt.bar(clusters, percentages, color=color)
    plt.xticks(clusters, [f"Cluster {c}" for c in clusters], rotation=45)
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
    total_counts = [total_genes_per_cluster[c] for c in cluster_list]
    kcg_counts = [kcg_per_cluster.get(c, 0) for c in cluster_list]

    x = np.arange(len(cluster_list))
    width = 0.4

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, total_counts, width, label='Total Genes', color='lightgray')
    plt.bar(x + width / 2, kcg_counts, width, label='Known Cancer Genes', color='firebrick')

    plt.xlabel('Cluster')
    plt.ylabel('Number of Genes')
    ##plt.title('Known Cancer Genes Distribution by Cluster')
    plt.ylim(0, 0.4)
    plt.xticks(ticks=x, labels=[f'Cluster {c}' for c in cluster_list], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

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

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pcg_cancer_genes(clusters, predicted_cancer_genes_count, total_genes_per_cluster, node_names, cluster_labels, output_path):
    """
    Plots the percentage of predicted cancer genes per cluster using a fixed color scheme.
    """
    # Convert to NumPy arrays for safe division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_counts = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes per cluster
    percent_predicted = np.divide(predicted_counts, total_counts, where=total_counts > 0)  

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
    
    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, top_1000_genes, node_names, output_path_genes_clusters
    )
    
    output_path_legend = os.path.join(output_dir, 'legend.png')
    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)
    
    '''output_path_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_cluster_ground_truth_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes_ground_truth(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
        cluster_labels=cluster_labels, output_path=output_path_percent
    )'''
    
    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_top1000_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
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
    
    cluster_labels, total_counts, pred_counts = cluster_and_visualize_predicted_genes(
        graph, predicted_cancer_genes, node_names, 
        output_path_genes_clusters
    )
    
    output_path_legend = os.path.join(output_dir, f'legend.png')

    save_cluster_legend(output_path_legend, CLUSTER_COLORS, num_clusters=12)
            
    ground_truth_cancer_genes = load_ground_truth_cancer_genes('data/796_drivers.txt')
    output_path_percent = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_cluster_ground_truth_epo{args.num_epochs}.png')
    
    plot_pcg_cancer_genes_ground_truth(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, ground_truth_cancer_genes=ground_truth_cancer_genes,
        node_names=node_names, cluster_labels=cluster_labels, output_path=output_path_percent,
        ##cluster_colors=CLUSTER_COLORS
    )

    output_path_predicted = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_pcg_epo{args.num_epochs}.png')
    plot_pcg_cancer_genes(
        clusters=total_counts.keys(), predicted_cancer_genes_count=pred_counts,
        total_genes_per_cluster=total_counts, node_names=node_names,
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

def plot_kcg_pcg_percentages(clusters, kcg_counts, pcg_counts, total_counts, output_path):
    """
    Plots a bar chart showing the percentage of KCGs and PCGs in each cluster.

    Args:
        clusters (iterable): Cluster IDs.
        kcg_counts (dict): Count of known cancer genes per cluster.
        pcg_counts (dict): Count of predicted cancer genes per cluster.
        total_counts (dict): Total genes per cluster.
        output_path (str): Path to save the figure.
    """
    clusters = list(clusters)
    kcg_percents = [100 * kcg_counts.get(c, 0) / total_counts[c] for c in clusters]
    pcg_percents = [100 * pcg_counts.get(c, 0) / total_counts[c] for c in clusters]

    x = np.arange(len(clusters))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, pcg_percents, width, label='Predicted Cancer Genes (PCGs)', color='skyblue')
    plt.bar(x + width/2, kcg_percents, width, label='Known Cancer Genes (KCGs)', color='orange')

    plt.xticks(x, [f"Cluster {c}" for c in clusters], rotation=45)
    plt.ylabel("Percentage (%)")
    plt.title("Percentage of PCGs and KCGs in Each Cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"KCG/PCG percentage plot saved to {output_path}")

def save_cluster_legend(output_path_legend, cluster_colors, num_clusters=12):
    """
    Creates and saves a separate legend image for cluster colors in two rows (Cluster 1 to 12).

    Args:
        output_path_legend (str): Path to save the legend image.
        cluster_colors (list): List of colors for each cluster.
        num_clusters (int): Number of clusters.
    """
    fig, ax = plt.subplots(figsize=(10, 2))  # Taller figure for two-row legend

    # Create legend handles labeled from Cluster 1 to Cluster 12
    legend_patches = [mpatches.Patch(color=cluster_colors[i], label=f"Cluster {i + 1}") 
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

from matplotlib.patches import Patch

def plot_interactions_with_pcgs(data, output_path):
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
    plt.ylim(0, 400)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.close()

    print(f"Plot saved to {output_path}")

def plot_interactions_with_kcgs(data, output_path):
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
    plt.ylim(0, 400)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  
    plt.close()

    print(f"Plot saved to {output_path}")

def cluster_and_visualize_predicted_genes(graph, predicted_cancer_genes, node_names, 
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
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=12)
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
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=12)
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
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=12)
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
    total_counts = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes per cluster
    percent_predicted = np.divide(predicted_counts, total_counts, where=total_counts > 0)  

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
    total_counts = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes safely
    percent_predicted = np.divide(predicted_counts, total_counts, where=total_counts > 0)

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


# Define global cluster colors
'''CLUSTER_COLORS = {
    0: '#0077B6',  1: '#0000FF',  2: '#00B4D8',  3: '#48EAC4',
    4: '#F1C0E8',  5: '#B9FBC0',  6: '#32CD32',  7: '#FFD700',
    8: '#8A2BE2',  9: '#E377C2', 10: '#8EECF5', 11: '#A3C4F3'
}'''

def plot_pcg_cancer_genes_ground_truth(clusters, predicted_cancer_genes_count, total_genes_per_cluster, ground_truth_cancer_genes, node_names, cluster_labels, output_path):
    """
    Plots the percentage of predicted cancer genes in each cluster.
    The number on top of each bar represents the count of ground truth cancer genes in that cluster.
    """
    # Convert to NumPy arrays for safer division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_counts = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes safely
    percent_predicted = np.divide(predicted_counts, total_counts, where=total_counts > 0)

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
    total_counts = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes per cluster
    percent_predicted = np.divide(predicted_counts, total_counts, where=total_counts > 0)  

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

    plot_pcg_cancer_genes(
        clusters=total_genes_per_cluster.keys(),  # Cluster IDs
        predicted_cancer_genes_count=predicted_cancer_genes_count,
        total_genes_per_cluster=total_genes_per_cluster,
        node_names=node_names,
        cluster_labels=cluster_labels,
        output_path=output_path_percent  # Removed ground_truth_cancer_genes
    )

            
    ##plot_pcg_cancer_genes(cluster_labels, predicted_cancer_genes_count, total_genes_per_cluster, output_path_percent)

    '''# Perform clustering and store cluster labels
    cluster_labels, predicted_counts, total_counts = cluster_and_visualize_predicted_genes(
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
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=12)
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

import numpy as np
import matplotlib.pyplot as plt

def plot_pcg_cancer_genes_ori(clusters, predicted_cancer_genes_count, total_genes_per_cluster, node_names, cluster_labels, output_path):
    """
    Plots the percentage of predicted cancer genes per cluster.
    """
    # Convert to NumPy arrays for safe division
    clusters = np.array(list(total_genes_per_cluster.keys()))  # Cluster IDs
    total_counts = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes per cluster
    percent_predicted = np.divide(predicted_counts, total_counts, where=total_counts > 0)  

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
    total_counts = np.array(list(total_genes_per_cluster.values()))  # Total genes per cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Compute percentage of predicted cancer genes safely
    percent_predicted = np.divide(predicted_counts, total_counts, where=total_counts > 0)

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
    total_counts = np.array(list(total_genes_per_cluster.values()))  # Total genes in each cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Perform element-wise division safely
    percent_predicted = np.divide(predicted_counts, total_counts, where=total_counts > 0)  

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
    total_counts = np.array(list(total_genes_per_cluster.values()))  # Total genes in each cluster
    predicted_counts = np.array([predicted_cancer_genes_count.get(cluster, 0) for cluster in clusters])

    # Perform element-wise division safely
    percent_predicted = np.divide(predicted_counts, total_counts, where=total_counts > 0)  


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

