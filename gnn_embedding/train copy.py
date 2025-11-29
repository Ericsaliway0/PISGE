import copy
import json
import os
import csv
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
# from sklearn import metrics
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import dataset
import model, utils, network
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats

CLUSTER_COLORS = {
    0: '#0077B6',   1: '#0000FF',   2: '#00B4D8',   3: '#48EAC4',
    4: '#F1C0E8',   5: '#B9FBC0',   6: '#32CD32',   7: '#bee1e6',
    8: '#8A2BE2',   9: '#E377C2',  10: '#8EECF5',  11: '#A3C4F3',
    12: '#FFB347', 13: '#FFD700',  14: '#FF69B4',  15: '#CD5C5C',
    16: '#7FFFD4', 17: '#FF7F50',  18: '#C71585',  19: '#20B2AA',
    20: '#6A5ACD', 21: '#40E0D0',  22: '#FF8C00',  23: '#DC143C',
    24: '#9ACD32'
}

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, weight=None):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if weight is not None:
            focal_loss = focal_loss * weight
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train(hyperparams=None, data_path='gnn_embedding/data/emb', plot=True):

    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    hidden_feats = hyperparams['hidden_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    weight_decay = hyperparams['weight_decay']
    dropout = hyperparams['dropout']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    contrastive_weight = hyperparams.get('contrastive_weight', 2.0)

    # Dataset and DataLoader
    ds = dataset.GeneDataset(data_path)
    ds_train = [ds[1]]
    ds_valid = [ds[0]]
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # Model and optimizer
    net = model.GATModel(
        in_feats=in_feats, out_feats=out_feats,
        num_layers=num_layers, num_heads=num_heads,
        do_train=True
    ).to(device)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer = optim.Adam(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_model = model.GATModel(
        in_feats=in_feats, out_feats=out_feats,
        num_layers=num_layers, num_heads=num_heads,
        do_train=True
    )
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    loss_per_epoch_train, f1_per_epoch_train = [], []
    loss_per_epoch_valid, f1_per_epoch_valid = [], []
    best_train_loss, best_valid_loss = float('inf'), float('inf')
    best_f1_score = 0.0
    max_f1_scores_train = []
    max_f1_scores_valid = []
    results_path = 'gnn_embedding/results/node_embeddings/'
    os.makedirs(results_path, exist_ok=True)
    
    
    
    model_path = os.path.join(data_path, 'models')
    model_path = os.path.join(model_path, f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth')

    # Initial embeddings visualization
    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)
    save_path_heatmap_initial = os.path.join(results_path, f'heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial = os.path.join(results_path, f'matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial = os.path.join(results_path, f'pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial = os.path.join(results_path, f't-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')

    for data in dl_train:
        graph, _ = data
        node_embeddings_initial = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.graph_nx.nodes)}
        first_node_stId_in_cluster_initial = {}
        first_node_embedding_in_cluster_initial = {}
        stid_dic_initial = {}

        for node in nx_graph.graph_nx.nodes:
            emb = node_embeddings_initial[node_to_index_initial[node]]
            stid_dic_initial[node] = emb.reshape(-1)

        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

        for node, cluster in zip(nx_graph.graph_nx.nodes, cluster_labels_initial):
            if cluster not in first_node_stId_in_cluster_initial:
                first_node_stId_in_cluster_initial[cluster] = node
                first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index_initial[node]]

        stid_list = list(first_node_stId_in_cluster_initial.values())
        embedding_list_initial = list(first_node_embedding_in_cluster_initial.values())
        embedding_list_initial = [emb.reshape(-1) if emb.ndim > 1 else emb for emb in embedding_list_initial]
        create_heatmap_with_stid(embedding_list_initial, stid_list, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list_initial, stid_list, save_path_matrix_initial)
        break

    visualize_embeddings_tsne(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_t_SNE_initial)
    visualize_embeddings_pca(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_pca_initial)

    # -------------------------------
    # Training loop
    # -------------------------------
    net.to(device)
    best_model = copy.deepcopy(net)
    best_valid_loss = float('inf')
    best_f1_score = 0.0

    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=False) as pbar:
        for epoch in range(num_epochs):
            net.train()
            loss_per_graph, f1_per_graph = [], []

            # -------------------------------
            # Training Phase
            # -------------------------------
            for data in dl_train:
                graph, name = data
                name = name[0]

                # --- Supervised ---
                logits = net(graph).view(-1)
                labels = graph.ndata["significance"].float().squeeze()

                num_pos = labels.sum().item()
                num_neg = labels.shape[0] - num_pos
                weight = torch.tensor([num_neg / labels.shape[0], num_pos / labels.shape[0]]).to(device)
                weight_ = weight[labels.long()].clone()  # <-- clone avoids in-place error

                supervised_loss = (criterion(logits, labels) * weight_).mean()


                # --- Contrastive loss ---
                contrastive_loss = net.compute_contrastive_loss(
                    graph,
                    feat_mask_ratio=hyperparams['feat_mask_ratio'],
                    edge_drop_ratio=hyperparams['edge_drop_ratio']
                )


                # --- Total loss ---
                total_loss = supervised_loss + hyperparams.get('contrastive_weight', 2.0) * contrastive_loss

                # --- Backpropagation ---
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()


                # --- Metrics ---
                preds = (logits.sigmoid() > 0.5).int()
                f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)

                loss_per_graph.append(total_loss.item())
                f1_per_graph.append(f1)

            # --- Epoch summary ---
            running_loss = np.mean(loss_per_graph)
            running_f1 = np.mean(f1_per_graph)
            loss_per_epoch_train.append(running_loss)
            f1_per_epoch_train.append(running_f1)

            pbar.set_postfix({"loss": running_loss, "f1": running_f1})
            pbar.update(1)

            # -------------------------------
            # Validation Phase
            # -------------------------------
            net.eval()
            with torch.no_grad():
                loss_per_graph, f1_per_graph = [], []

                for data in dl_valid:
                    graph, name = data
                    name = name[0]

                    logits = net(graph).view(-1)
                    labels = graph.ndata["significance"].float().squeeze()

                    # Safe weights
                    num_pos = labels.sum().item()
                    num_neg = labels.shape[0] - num_pos
                    weight_tensor = torch.tensor([num_neg / labels.shape[0], num_pos / labels.shape[0]], device=device)
                    weight_ = weight_tensor[labels.long()].clone()

                    # Supervised loss
                    supervised_loss = (criterion(logits, labels) * weight_).mean()

                    # Contrastive loss
                    contrastive_loss = net.compute_contrastive_loss(graph)

                    # Total loss
                    total_loss = supervised_loss + contrastive_weight * contrastive_loss

                    preds = (logits.sigmoid() > 0.5).int()
                    f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)

                    loss_per_graph.append(total_loss.item())
                    f1_per_graph.append(f1)

                # Epoch validation metrics
                running_loss_valid = np.mean(loss_per_graph)
                running_f1_valid = np.mean(f1_per_graph)
                loss_per_epoch_valid.append(running_loss_valid)
                f1_per_epoch_valid.append(running_f1_valid)

                # Track max F1 scores
                max_f1_train = max(f1_per_epoch_train)
                max_f1_valid = max(f1_per_epoch_valid)
                max_f1_scores_train.append(max_f1_train)
                max_f1_scores_valid.append(max_f1_valid)

                # Update best model
                if running_loss_valid < best_valid_loss:
                    best_valid_loss = running_loss_valid
                    best_f1_score = running_f1_valid
                    best_model.load_state_dict(copy.deepcopy(net.state_dict()))
                    print(f"✅ New Best Model Found | F1: {best_f1_score:.4f} | Valid Loss: {best_valid_loss:.4f}")

            # Progress display
            pbar.update(1)
            print(f"Epoch {epoch + 1:03d} | Train F1: {running_f1:.4f} | Valid F1: {running_f1_valid:.4f} | "
                  f"Max Train F1: {max_f1_train:.4f} | Max Valid F1: {max_f1_valid:.4f}")

    print(f"Training Complete ✅ | Best Valid F1: {best_f1_score:.4f} | Best Valid Loss: {best_valid_loss:.4f}")
    # return best_model, loss_per_epoch_train, f1_per_epoch_train, loss_per_epoch_valid, f1_per_epoch_valid


    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)  # Flatten 
    print('cluster_labels=========================\n', cluster_labels)

    # ================================================
    # GCL Visualization: Before vs After Contrastive Training
    # ================================================
    visualize_gcl_comparison(
        emb_before=all_embeddings_initial,
        emb_after=all_embeddings,
        cluster_labels=cluster_labels,
        save_dir=results_path
    )

    # emb_before, emb_after: numpy arrays (N,D) you already have
    # cluster_labels: use cluster_labels from your spectral biclustering / kmeans (same order)
    # node_labels: optional ground-truth labels if available

    metrics = evaluate_gcl(emb_before=all_embeddings_initial, emb_after=all_embeddings, cluster_labels=cluster_labels, node_labels=None)
    pd.Series(metrics).to_csv('gnn_embedding/results/node_embeddings/gcl_eval_summary.csv')
    print(metrics)

    
    loss_path = os.path.join(results_path, f'loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    cos_sim = np.dot(all_embeddings, all_embeddings.T)
    norms = np.linalg.norm(all_embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    if plot:
        loss_path = os.path.join(results_path, f'loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        f1_path = os.path.join(results_path, f'f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        max_f1_path = os.path.join(results_path, f'max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        matrix_path = os.path.join(results_path, f'matrix_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
 
        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    torch.save(best_model.state_dict(), model_path)

    save_path_pca = os.path.join(results_path, f'pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_t_SNE = os.path.join(results_path, f't-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_heatmap_= os.path.join(results_path, f'heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_matrix = os.path.join(results_path, f'matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    
    cluster_stId_dict = {}  # Dictionary to store clusters and corresponding stIds
    significant_stIds = []  # List to store significant stIds
    clusters_with_significant_stId = {}  # Dictionary to store clusters and corresponding significant stIds
    clusters_node_info = {}  # Dictionary to store node info for each cluster
    
    for data in dl_train:
        graph, _ = data
        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        assert len(cluster_labels) == len(nx_graph.graph_nx.nodes), "Cluster labels and number of nodes must match"
        node_to_index = {node: idx for idx, node in enumerate(nx_graph.graph_nx.nodes)}
        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}

        stid_dic = {}

        # Populate stid_dic with node stIds mapped to flattened embeddings
        for node in nx_graph.graph_nx.nodes:
            emb = node_embeddings[node_to_index[node]]
            stid_dic[node] = emb.reshape(-1)  # flatten (1, 32) → (32,)  
        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')
                
        for node, cluster in zip(nx_graph.graph_nx.nodes, cluster_labels):
            if cluster not in first_node_stId_in_cluster:
                # first_node_stId_in_cluster[cluster] = nx_graph.graph_nx.nodes[node]============================================================
                first_node_stId_in_cluster[cluster] = node
                first_node_embedding_in_cluster[cluster] = node_embeddings[node_to_index[node]]
                
            # Populate cluster_stId_dict
            if cluster not in cluster_stId_dict:
                cluster_stId_dict[cluster] = []
            cluster_stId_dict[cluster].append(nx_graph.graph_nx.nodes[node])

            # Populate clusters_with_significant_stId
            if cluster not in clusters_with_significant_stId:
                clusters_with_significant_stId[cluster] = []
            if nx_graph.graph_nx.nodes[node] in significant_stIds:
                clusters_with_significant_stId[cluster].append(nx_graph.graph_nx.nodes[node])
            
            # Populate clusters_node_info with node information for each cluster
            if cluster not in clusters_node_info:
                clusters_node_info[cluster] = []
            node_info = {
                'stId': nx_graph.graph_nx.nodes[node],
                'significance': graph.ndata['significance'][node_to_index[node]].item(),
                'other_info': nx_graph.graph_nx.nodes[node]  # Add other relevant info if necessary
            }
            clusters_node_info[cluster].append(node_info)
        
        print(first_node_stId_in_cluster)
        stid_list = list(first_node_stId_in_cluster.values())
        embedding_list = list(first_node_embedding_in_cluster.values())
        embedding_list = [emb.reshape(-1) if emb.ndim > 1 else emb for emb in embedding_list]
        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)
        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        # Call the function to plot cosine similarity matrix for cluster representatives with similarity values
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break
        
    csv_save_path_initial = os.path.join(results_path, f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv')
    ##csv_save_path_initial = os.path.join('gat/gat/data/', f'inhibition_gene_embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv')
    stid_df_initial.to_csv(csv_save_path_initial, index_label='Gene')
    csv_save_path_final = os.path.join(results_path, f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv')
    stid_df_final.to_csv(csv_save_path_final, index_label='Gene')
    

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, stid_list, save_path_pca)
    silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)

    print(f"Silhouette Score%%%%%%%%%%%%###########################: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    summary = f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {max_f1_train}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"

    save_file = os.path.join(results_path, f'head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file, 'w') as f:
        f.write(summary)

    return model_path

def create_heatmap_with_genes_small_font(all_embeddings, stid_list, save_path):
    """
    Create a clustermap heatmap for node embeddings, aligned with GeneSymbols.

    Parameters:
    - all_embeddings: np.array, shape (num_nodes, embedding_dim)
    - stid_list: list of GeneSymbols corresponding to the embeddings
    - save_path: str, path to save the heatmap
    """
    if len(all_embeddings) == 0 or len(stid_list) == 0:
        print(f"⚠️ Heatmap skipped: empty embeddings or gene list. save_path={save_path}")
        return

    # Ensure embeddings and GeneSymbols match in length
    if len(all_embeddings) != len(stid_list):
        print(f"⚠️ Warning: Embeddings ({len(all_embeddings)}) and GeneSymbols ({len(stid_list)}) lengths mismatch.")
        min_len = min(len(all_embeddings), len(stid_list))
        all_embeddings = all_embeddings[:min_len]
        stid_list = stid_list[:min_len]

    # Build DataFrame
    heatmap_data = pd.DataFrame(all_embeddings, index=stid_list)

    # Check if DataFrame is empty
    if heatmap_data.shape[0] == 0 or heatmap_data.shape[1] == 0:
        print(f"⚠️ Heatmap skipped: empty DataFrame. save_path={save_path}")
        return

    # Plot clustermap
    try:
        ax = sns.clustermap(
            heatmap_data,
            cmap='tab20',
            standard_scale=1,
            figsize=(10, 10)
        )
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Heatmap saved to {save_path}")
    except Exception as e:
        print(f"❌ Failed to create heatmap: {e}")

def create_heatmap_with_genes(all_embeddings, stid_list, save_path):
    """
    Create a clustermap heatmap for node embeddings, aligned with GeneSymbols.
    """

    if len(all_embeddings) == 0 or len(stid_list) == 0:
        print(f"⚠️ Heatmap skipped: empty embeddings or gene list. save_path={save_path}")
        return

    if len(all_embeddings) != len(stid_list):
        print(f"⚠️ Warning: Embeddings ({len(all_embeddings)}) and GeneSymbols ({len(stid_list)}) lengths mismatch.")
        min_len = min(len(all_embeddings), len(stid_list))
        all_embeddings = all_embeddings[:min_len]
        stid_list = stid_list[:min_len]

    # Build DataFrame
    heatmap_data = pd.DataFrame(all_embeddings, index=stid_list)

    if heatmap_data.shape[0] == 0 or heatmap_data.shape[1] == 0:
        print(f"⚠️ Heatmap skipped: empty DataFrame. save_path={save_path}")
        return

    try:
        # Create clustermap
        cg = sns.clustermap(
            heatmap_data,
            cmap='tab20',
            standard_scale=1,
            figsize=(12, 12)
        )

        # Set big font sizes for labels
        cg.ax_heatmap.set_xticklabels(
            cg.ax_heatmap.get_xticklabels(),
            fontsize=14, rotation=90
        )
        cg.ax_heatmap.set_yticklabels(
            cg.ax_heatmap.get_yticklabels(),
            fontsize=14
        )

        # Save figure
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"✅ Heatmap saved to {save_path}")
    except Exception as e:
        print(f"❌ Failed to create heatmap: {e}")


def evaluate_gcl(
    emb_before, emb_after,
    cluster_labels=None,
    node_labels=None,
    random_state=42,
    save_csv=True,
    output_dir="gnn_embedding/results/gcl_eval",
    filename=None
):
    """
    emb_before, emb_after: np.array shape (N, D)
    cluster_labels: array-like (N,) — optional cluster IDs you computed (spectral biclustering or KMeans)
    node_labels: array-like (N,) — optional ground-truth labels if available (for purity/NMI/linear probe)
    Returns dict of metrics and prints summary.
    """

    results = {}

    # ---- 1) clustering quality (unsupervised) ----
    try:
        results['silhouette_before'] = float(silhouette_score(emb_before, cluster_labels)) if cluster_labels is not None else None
    except Exception:
        results['silhouette_before'] = None
    try:
        results['silhouette_after'] = float(silhouette_score(emb_after, cluster_labels)) if cluster_labels is not None else None
    except Exception:
        results['silhouette_after'] = None

    try:
        results['db_before'] = float(davies_bouldin_score(emb_before, cluster_labels)) if cluster_labels is not None else None
    except Exception:
        results['db_before'] = None
    try:
        results['db_after'] = float(davies_bouldin_score(emb_after, cluster_labels)) if cluster_labels is not None else None
    except Exception:
        results['db_after'] = None

    # ---- 2) intra / inter cluster cosine similarity ----
    def intra_inter_stats(emb, clusters):
        S = cosine_similarity(emb)
        clusters = np.asarray(clusters)
        n = len(clusters)
        intra_vals = []
        inter_vals = []
        for i in range(n):
            for j in range(i+1, n):
                if clusters[i] == clusters[j]:
                    intra_vals.append(S[i, j])
                else:
                    inter_vals.append(S[i, j])
        intra_mean = float(np.mean(intra_vals)) if len(intra_vals) else np.nan
        inter_mean = float(np.mean(inter_vals)) if len(inter_vals) else np.nan
        return intra_mean, inter_mean, np.array(intra_vals), np.array(inter_vals)

    if cluster_labels is not None:
        ib_mean, ob_mean, ib_vals, ob_vals = intra_inter_stats(emb_before, cluster_labels)
        ia_mean, oa_mean, ia_vals, oa_vals = intra_inter_stats(emb_after, cluster_labels)

        results.update({
            'intra_mean_before': ib_mean,
            'inter_mean_before': ob_mean,
            'intra_mean_after': ia_mean,
            'inter_mean_after': oa_mean,
            'intra_gain': (ia_mean - ib_mean) if (not np.isnan(ib_mean) and not np.isnan(ia_mean)) else None,
            'inter_change': (oa_mean - ob_mean) if (not np.isnan(ob_mean) and not np.isnan(oa_mean)) else None
        })

        try:
            m = min(len(ib_vals), len(ia_vals))
            if m > 10:
                t_stat, p_val = stats.ttest_rel(ia_vals[:m], ib_vals[:m])
            else:
                t_stat, p_val = np.nan, np.nan
            results['intra_t_stat'] = float(t_stat) if not np.isnan(t_stat) else None
            results['intra_p_val'] = float(p_val) if not np.isnan(p_val) else None
        except Exception:
            results['intra_t_stat'] = None
            results['intra_p_val'] = None
    else:
        cos_b = cosine_similarity(emb_before)
        cos_a = cosine_similarity(emb_after)
        results['mean_pairwise_before'] = float(np.mean(cos_b))
        results['mean_pairwise_after'] = float(np.mean(cos_a))
        results['mean_pairwise_change'] = results['mean_pairwise_after'] - results['mean_pairwise_before']

    # ---- 3) clustering agreement ----
    if node_labels is not None and cluster_labels is not None:
        results['NMI'] = normalized_mutual_info_score(node_labels, cluster_labels)
        results['ARI'] = adjusted_rand_score(node_labels, cluster_labels)

    # ---- 4) linear separability ----
    if node_labels is not None:
        Xb = emb_before
        Xa = emb_after
        y = np.asarray(node_labels)
        scaler = StandardScaler()
        Xb = scaler.fit_transform(Xb)
        Xa = scaler.fit_transform(Xa)

        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        acc_b = cross_val_score(clf, Xb, y, cv=cv, scoring='accuracy').mean()
        acc_a = cross_val_score(clf, Xa, y, cv=cv, scoring='accuracy').mean()
        auc_b = cross_val_score(clf, Xb, y, cv=cv, scoring='roc_auc').mean() if len(np.unique(y)) == 2 else None

        results['linear_acc_before'] = float(acc_b)
        results['linear_acc_after']  = float(acc_a)
        results['linear_acc_gain'] = float(acc_a - acc_b)
        results['linear_auc_before'] = float(auc_b) if auc_b is not None else None

    # ---- 5) quick summary print ----
    print("=== GCL Evaluation Summary ===")
    print("Silhouette before/after:", results.get('silhouette_before'), results.get('silhouette_after'))
    print("DB index before/after (lower better):", results.get('db_before'), results.get('db_after'))
    if cluster_labels is not None:
        print("Intra mean before/after:", results['intra_mean_before'], results['intra_mean_after'])
        print("Inter mean before/after:", results['inter_mean_before'], results['inter_mean_after'])
        print("Intra t-test p-value:", results.get('intra_p_val'))
    if node_labels is not None:
        print("Linear probe accuracy before/after:", results.get('linear_acc_before'), results.get('linear_acc_after'))

    # ---- 6) save to CSV ----
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"gcl_eval_results_{timestamp}.csv"
        csv_path = os.path.join(output_dir, filename)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in results.items():
                writer.writerow([k, v])

        print(f"\n✅ GCL evaluation results saved to: {csv_path}\n")

    return results

def visualize_gcl_comparison(emb_before, emb_after, cluster_labels, save_dir):
    """
    Compare embeddings before vs. after GCL contrastive training.
    Saves t-SNE and cosine similarity heatmaps.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- Normalize ----
    emb_before = nn.functional.normalize(torch.tensor(emb_before), dim=1).cpu().numpy()
    emb_after  = nn.functional.normalize(torch.tensor(emb_after),  dim=1).cpu().numpy()

    # ---- Combine for consistent t-SNE projection ----
    combined = np.concatenate([emb_before, emb_after], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    tsne_emb = tsne.fit_transform(combined)
    n = len(emb_before)
    tsne_before = tsne_emb[:n]
    tsne_after  = tsne_emb[n:]

    # ---- t-SNE plot ----
    # fig, axs = plt.subplots(1, 2, figsize=(12,6))
    # sns.scatterplot(x=tsne_before[:,0], y=tsne_before[:,1], hue=cluster_labels,
    #                 palette="tab10", legend=False, s=25, ax=axs[0])
    # axs[0].set_title("Before Contrastive Training")
    # axs[0].set_xticks([]); axs[0].set_yticks([])

    # sns.scatterplot(x=tsne_after[:,0], y=tsne_after[:,1], hue=cluster_labels,
    #                 palette="tab10", legend=False, s=25, ax=axs[1])
    # axs[1].set_title("After Contrastive Training")
    # axs[1].set_xticks([]); axs[1].set_yticks([])

    # plt.suptitle("t-SNE: GCL Embeddings Before vs. After", fontsize=14)
    # plt.tight_layout()
    # tsne_path = os.path.join(save_dir, "gcl_tsne_before_vs_after.png")
    # plt.savefig(tsne_path, dpi=300)
    # plt.close()

    # convert the color dict to a palette list matching unique cluster indices
    unique_clusters = sorted(set(cluster_labels))
    palette = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in unique_clusters]

    # map cluster_labels -> colors explicitly
    cluster_color_map = [CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)] for c in cluster_labels]

    # ---- t-SNE plot ----
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # BEFORE contrastive training
    sns.scatterplot(
        x=tsne_before[:, 0], y=tsne_before[:, 1],
        hue=cluster_labels,
        palette=CLUSTER_COLORS,
        legend=False, s=25, ax=axs[0]
    )
    axs[0].set_title("Before Contrastive Training", fontsize=13)
    axs[0].set_xticks([]); axs[0].set_yticks([])

    # AFTER contrastive training
    sns.scatterplot(
        x=tsne_after[:, 0], y=tsne_after[:, 1],
        hue=cluster_labels,
        palette=CLUSTER_COLORS,
        legend=False, s=25, ax=axs[1]
    )
    axs[1].set_title("After Contrastive Training", fontsize=13)
    axs[1].set_xticks([]); axs[1].set_yticks([])

    # Shared main title
    plt.suptitle("t-SNE: GCL Embeddings Before vs. After", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    tsne_path = os.path.join(save_dir, "gcl_tsne_before_vs_after.png")
    plt.savefig(tsne_path, dpi=300)
    plt.show()

    # ---- Cosine similarity ----
    cos_before = cosine_similarity(emb_before)
    cos_after  = cosine_similarity(emb_after)

    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    sns.heatmap(cos_before, cmap='Spectral', center=0, square=True, ax=axs[0])
    axs[0].set_title("Cosine Similarity: Before GCL")

    sns.heatmap(cos_after, cmap='Spectral', center=0, square=True, ax=axs[1])
    axs[1].set_title("Cosine Similarity: After GCL")

    plt.suptitle("Pairwise Similarity Before vs. After Contrastive Training", fontsize=14)
    plt.tight_layout()
    cos_path = os.path.join(save_dir, "gcl_cosine_before_vs_after.png")
    plt.savefig(cos_path, dpi=300)
    plt.close()

    print(f"✅ GCL comparison visualizations saved to: {save_dir}")

def plot_cosine_similarity_matrix_for_clusters_with_values_small_legend(embeddings, stids, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Compute cosine similarity
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    # Figure size
    plt.figure(figsize=(22, 20), dpi=100)

    vmin = cos_sim.min()
    vmax = cos_sim.max()

    # Create the heatmap with larger annotation font size
    ax = sns.heatmap(
        cos_sim,
        cmap="Spectral",
        annot=True,
        fmt=".3f",
        annot_kws={"size": 11},  # larger numbers
        xticklabels=stids,
        yticklabels=stids,
        cbar_kws={"shrink": 0.3, "aspect": 18, "ticks": [vmin, vmax]}
    )

    # X-axis on top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Font sizes for ticks
    plt.xticks(rotation=-30, fontsize=18, ha='right')
    plt.yticks(fontsize=18, rotation=0, ha='right')

    # Title below plot
    ax.text(
        x=0.5, y=-0.03, s="Gene-gene similarities",
        fontsize=32, ha='center', va='top',
        transform=ax.transAxes
    )

    # Save figure with tight layout
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_cosine_similarity_matrix_for_clusters_with_values_ori(embeddings, stids, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter

    # Compute cosine similarity
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    # Figure size
    plt.figure(figsize=(22, 20), dpi=100)

    vmin = cos_sim.min()
    vmax = cos_sim.max()

    # Create the heatmap with annotations
    ax = sns.heatmap(
        cos_sim,
        cmap="Spectral",
        annot=True,
        fmt=".3f",
        annot_kws={"size": 11},  # numbers inside heatmap
        xticklabels=stids,
        yticklabels=stids,
        cbar_kws={"shrink": 0.3, "aspect": 18}
    )

    # Format colorbar ticks
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)  # bigger font
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # enforce .3f

    # X-axis on top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Font sizes for tick labels
    plt.xticks(rotation=-30, fontsize=18, ha='right')
    plt.yticks(fontsize=18, rotation=0, ha='right')

    # Title below plot
    ax.text(
        x=0.5, y=-0.03, s="Gene-gene similarities",
        fontsize=32, ha='center', va='top',
        transform=ax.transAxes
    )

    # Save figure
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, stids, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter

    # Ensure embeddings are 2D [num_clusters, embedding_dim]
    embeddings = np.array(embeddings)
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)  # flatten last two dims

    # Compute cosine similarity
    cos_sim = np.dot(embeddings, embeddings.T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    # Figure size
    plt.figure(figsize=(22, 20), dpi=100)

    vmin = cos_sim.min()
    vmax = cos_sim.max()

    # Create the heatmap with annotations
    ax = sns.heatmap(
        cos_sim,
        cmap="Spectral",
        annot=True,
        fmt=".3f",
        annot_kws={"size": 11},  # numbers inside heatmap
        xticklabels=stids,
        yticklabels=stids,
        cbar_kws={"shrink": 0.3, "aspect": 18}
    )

    # Format colorbar ticks
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)  # bigger font
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # enforce .3f

    # X-axis on top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Font sizes for tick labels
    plt.xticks(rotation=-30, fontsize=18, ha='right')
    plt.yticks(fontsize=18, rotation=0, ha='right')

    # Title below plot
    ax.text(
        x=0.5, y=-0.03, s="Gene-gene similarities",
        fontsize=32, ha='center', va='top',
        transform=ax.transAxes
    )

    # Save figure
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def create_pathway_map(reactome_file, output_file):
    """
    Extracts gene IDs with the same pathway STID and saves them to a new CSV file.

    Parameters:
    reactome_file (str): Path to the NCBI2Reactome.csv file.
    output_file (str): Path to save the output CSV file.
    """
    pathway_map = {}  # Dictionary to store gene IDs for each pathway STID

    # Read the NCBI2Reactome.csv file and populate the pathway_map
    with open(reactome_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            gene_id = row[0]
            pathway_stid = row[1]
            pathway_map.setdefault(pathway_stid, []).append(gene_id)

    # Write the pathway_map to the output CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pathway STID", "Gene IDs"])  # Write header
        for pathway_stid, gene_ids in pathway_map.items():
            writer.writerow([pathway_stid, ",".join(gene_ids)])
    
    return pathway_map
        
def save_to_neo4j(graph, stid_dic, stid_mapping, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, uri, user, password):
    from neo4j import GraphDatabase

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    session = driver.session()

    # Clean the database
    session.run("MATCH (n) DETACH DELETE n")

    try:
        # Create nodes with embeddings and additional attributes
        for node_id in stid_dic:
            embedding = stid_dic[node_id].tolist()  
            stId = stid_mapping[node_id]  # Access stId based on node_id
            name = graph.graph_nx.nodes[node_id]['name']
            weight = graph.graph_nx.nodes[node_id]['weight']
            significance = graph.graph_nx.nodes[node_id]['significance']
            session.run(
                "CREATE (n:Pathway {embedding: $embedding, stId: $stId, name: $name, weight: $weight, significance: $significance})",
                embedding=embedding, stId=stId, name=name, weight=weight, significance=significance
            )

            # Create gene nodes and relationships
            ##genes = get_genes_by_pathway_stid(node_id, reactome_file, gene_names_file)
            genes = pathway_map.get(node_id, [])


            ##print('stid_to_gene_info=========================-----------------------------\n', genes)
    
            # Create gene nodes and relationships
            for gene_id in genes:
                gene_name = gene_id_to_name_mapping.get(gene_id, None)
                gene_symbol = gene_id_to_symbol_mapping.get(gene_id, None)
                if gene_name:  # Only create node if gene name exists
                    session.run(
                        "MERGE (g:Gene {id: $gene_id, name: $gene_name, symbol: $gene_symbol})",
                        gene_id=gene_id, gene_name=gene_name, gene_symbol = gene_symbol
                    )
                    session.run(
                        "MATCH (p:Pathway {stId: $stId}), (g:Gene {id: $gene_id}) "
                        "MERGE (p)-[:INVOLVES]->(g)",
                        stId=stId, gene_id=gene_id
                    )
                
                session.run(
                    "MATCH (p:Pathway {stId: $stId}), (g:Gene {id: $gene_id}) "
                    "MERGE (p)-[:INVOLVES]->(g)",
                    stId=stId, gene_id=gene_id
                )
                
        # Create relationships using the stId mapping
        for source, target in graph.graph_nx.edges():
            source_stId = stid_mapping[source]
            target_stId = stid_mapping[target]
            session.run(
                "MATCH (a {stId: $source_stId}), (b {stId: $target_stId}) "
                "CREATE (a)-[:CONNECTED]->(b)",
                source_stId=source_stId, target_stId=target_stId
            )

    finally:
        session.close()
        driver.close()

def read_gene_names(file_path):
    """
    Reads the gene names from a CSV file and returns a dictionary mapping gene IDs to gene names.

    Parameters:
    file_path (str): Path to the gene names CSV file.

    Returns:
    dict: A dictionary mapping gene IDs to gene names.
    """
    gene_id_to_name_mapping = {}
    gene_id_to_symbol_mapping = {}

    # Read the gene names CSV file and populate the dictionary
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gene_id = row['NCBI_Gene_ID']
            gene_name = row['Name']
            gene_symbol = row['Approved symbol']
            gene_id_to_name_mapping[gene_id] = gene_name
            gene_id_to_symbol_mapping[gene_id] = gene_symbol

    return gene_id_to_name_mapping, gene_id_to_symbol_mapping

def create_heatmap_with_stid_ori(embedding_list, stid_list, save_path):
    # Convert the embedding list to a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list)
    
    # Create a clustermap
    ax = sns.clustermap(heatmap_data, cmap='tab20', standard_scale=1, figsize=(10, 10))
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Save the clustermap to a file
    plt.savefig(save_path)

    plt.close()

def create_heatmap_with_stid(embedding_list, stid_list, save_path):
    # Ensure embedding_list is a 2D numpy array
    embedding_array = np.array(embedding_list)
    if embedding_array.ndim == 3:
        embedding_array = np.squeeze(embedding_array)  # removes dimensions of size 1
        
    # Convert the embedding list to a DataFrame
    heatmap_data = pd.DataFrame(embedding_array, index=stid_list)
    
    # Create a clustermap with a continuous color palette
    ax = sns.clustermap(
        heatmap_data,
        cmap='Spectral',          # continuous colormap (options: 'cubehelix', 'Spectral', 'twilight', 'magma', 'cividis', 'viridis', 'plasma', 'coolwarm', 'RdYlBu_r', 'bone', 'Greys', etc.)
        standard_scale=1,        # scale features (columns)
        figsize=(12, 12),
        linewidths=0.0,
        cbar_kws={"label": "Scaled Feature Value"}
    )
    # ax = sns.clustermap(
    #     heatmap_data,
    #     cmap='tab20',
    #     standard_scale=1,
    #     figsize=(12, 12)  # slightly larger figure for bigger fonts
    # )
    
    # Increase font sizes for clarity
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=12)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=14)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=14)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=12)  # Color bar labels
    ax.ax_row_dendrogram.set_ylabel(ax.ax_row_dendrogram.get_ylabel(), fontsize=12)
    ax.ax_col_dendrogram.set_xlabel(ax.ax_col_dendrogram.get_xlabel(), fontsize=12)
    
    # Save the clustermap to a file
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def calculate_cluster_labels(net, dataloader, device, num_clusters=25):
    all_embeddings = []
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, _ = data
            embeddings = net.get_node_embeddings(graph.to(device))
            all_embeddings.append(embeddings)
    # all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # # Use KMeans clustering to assign cluster labels
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # cluster_labels = kmeans.fit_predict(all_embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # Flatten [num_heads, out_feats] → single feature dimension if needed
    if all_embeddings.ndim == 3:
        all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)

    return all_embeddings, cluster_labels

def visualize_embeddings_pca_ori(embeddings, cluster_labels, stid_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(20, 20), dpi=100)

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    # plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)

    plt.close()

def visualize_embeddings_pca_(embeddings, cluster_labels, stid_list, save_path):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Force square figure size
    plt.figure(figsize=(20, 20), dpi=100)

    # Style
    sns.set(style="whitegrid")

    # Unique clusters (sorted)
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)

    # Scatter plot with your custom colors
    for cluster in sorted_clusters:
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f'{stid_list[cluster]}',
            s=40,  # larger dots
            color=CLUSTER_COLORS.get(cluster, "#808080"),  # fallback gray
            edgecolor='k',
            linewidth=0.5
        )

    # Labels and title with bigger fonts
    plt.xlabel('PC1', fontsize=24)
    plt.ylabel('PC2', fontsize=24)
    plt.title('PCA of Embeddings', fontsize=28)

    # Tick label size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)
    ax.set_aspect('equal', adjustable='box')

    # Custom legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=CLUSTER_COLORS.get(cluster, "#808080"),
                   markersize=12, label=stid_list[cluster])
        for cluster in sorted_clusters
    ]
    plt.legend(
        handles=handles,
        title='Label',
        bbox_to_anchor=(1.02, 0.5),
        loc='center left',
        borderaxespad=0.,
        fontsize=16,
        title_fontsize=18,
        handlelength=0.8,
        handletextpad=0.8
    )

    plt.savefig(save_path, dpi=100, bbox_inches=None)
    plt.close()

def visualize_embeddings_pca(embeddings, cluster_labels, stid_list, save_path):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Square figure
    plt.figure(figsize=(20, 20), dpi=100)

    # White background without grid
    sns.set_style("white")

    # Unique clusters (sorted)
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)

    # Scatter plot with CLUSTER_COLORS
    for cluster in sorted_clusters:
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f'{stid_list[cluster]}',
            s=40,  # larger dots
            color=CLUSTER_COLORS.get(cluster, "#808080"),  # fallback gray
            edgecolor='k',
            linewidth=0.5
        )

    # Labels and title
    plt.xlabel('PC1', fontsize=24)
    plt.ylabel('PC2', fontsize=24)
    plt.title('PCA of Embeddings', fontsize=28)

    # Tick labels
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Axes square
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Custom legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=CLUSTER_COLORS.get(cluster, "#808080"),
                   markersize=12, label=stid_list[cluster])
        for cluster in sorted_clusters
    ]
    plt.legend(
        handles=handles,
        title='Label',
        bbox_to_anchor=(1.02, 0.5),
        loc='center left',
        borderaxespad=0.,
        fontsize=16,
        title_fontsize=18,
        handlelength=0.8,
        handletextpad=0.8
    )

    # Save with tight layout
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def visualize_embeddings_tsne_ORI(embeddings, cluster_labels, stid_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.title('T-SNE of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_embeddings_tsne(embeddings, cluster_labels, stid_list, save_path):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Square figure with larger size
    plt.figure(figsize=(20, 20), dpi=100)

    # White background without grid
    sns.set_style("white")

    # Unique clusters (sorted)
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)

    # Scatter plot with CLUSTER_COLORS
    for cluster in sorted_clusters:
        cluster_points = embeddings_2d[cluster_labels == cluster]
        color = CLUSTER_COLORS.get(cluster, '#808080')  # fallback gray
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=stid_list[cluster],
            s=40,  # larger dots
            color=color,
            edgecolor='k',
            linewidth=0.5
        )

    # Labels and title with bigger fonts
    plt.xlabel('Dim1', fontsize=24)
    plt.ylabel('Dim2', fontsize=24)
    plt.title('T-SNE of Embeddings', fontsize=28)

    # Tick labels
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Axes square
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Custom legend with larger fonts
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=CLUSTER_COLORS.get(cluster, '#808080'),
                   markersize=12, label=stid_list[cluster])
        for cluster in sorted_clusters
    ]
    plt.legend(
        handles=handles,
        title='Label',
        bbox_to_anchor=(1.02, 0.5),
        loc='center left',
        borderaxespad=0.,
        fontsize=16,
        title_fontsize=18,
        handlelength=0.8,
        handletextpad=0.8
    )

    # Save with tight layout
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def export_to_cytoscape(node_embeddings, cluster_labels, stid_list, output_path):
    # Create a DataFrame for Cytoscape export
    data = {
        'Node': stid_list,
        'Cluster': cluster_labels,
        'Embedding': list(node_embeddings)
    }
    df = pd.DataFrame(data)
    
    # Expand the embedding column into separate columns
    embeddings_df = pd.DataFrame(node_embeddings, columns=[f'Embed_{i}' for i in range(node_embeddings.shape[1])])
    df = df.drop('Embedding', axis=1).join(embeddings_df)

    # Save to CSV for Cytoscape import
    df.to_csv(output_path, index=False)
    print(f"Data exported to {output_path} for Cytoscape visualization.")

def draw_loss_plot_(train_loss, valid_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility
    
    plt.savefig(f'{save_path}')
    plt.close()

def draw_loss_plot(train_loss, valid_loss, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set white background without grid
    sns.set_style("white")

    plt.figure(figsize=(12, 8), dpi=100)
    plt.plot(train_loss, label='Train', linewidth=2)
    plt.plot(valid_loss, label='Validation', linewidth=2)
    
    # Labels and title with larger fonts
    plt.title('Loss over epochs', fontsize=24)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    
    # Tick labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Legend with bigger font
    plt.legend(fontsize=18)

    # Axes
    ax = plt.gca()
    ax.set_aspect('auto')  # line plots do not need square
    ax.set_facecolor('white')  # ensure white background

    # Save figure with tight layout
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def draw_max_f1_plot(max_train_f1, max_valid_f1, save_path):
    plt.figure()
    plt.plot(max_train_f1, label='train')
    plt.plot(max_valid_f1, label='validation')
    plt.title('Max F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

def draw_f1_plot_(train_f1, valid_f1, save_path):
    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')
    plt.title('F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    plt.savefig(f'{save_path}')
    plt.close()

def draw_f1_plot(train_f1, valid_f1, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set white background without grid
    sns.set_style("white")

    plt.figure(figsize=(12, 8), dpi=100)
    plt.plot(train_f1, label='Train', linewidth=2)
    plt.plot(valid_f1, label='Validation', linewidth=2)
    
    # Labels and title with larger fonts
    plt.title('F1-score over epochs', fontsize=24)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('F1-score', fontsize=20)
    
    # Tick labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Legend with bigger font
    plt.legend(fontsize=18)

    # Axes square aspect (optional)
    ax = plt.gca()
    ax.set_aspect('auto')  # F1 plot doesn't need perfect square
    ax.set_facecolor('white')  # ensure white background

    # Save figure with tight layout
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    hyperparams = {
        'num_epochs': 100,
        'out_feats': 128,
        'num_layers': 2,
        'lr': 0.001,
        'batch_size': 1,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    train(hyperparams=hyperparams)
