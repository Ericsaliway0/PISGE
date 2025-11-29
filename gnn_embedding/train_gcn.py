import copy
import json
import os
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
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
    ##feat_drop = hyperparams['feat_drop']
    in_feats = hyperparams['in_feats']
    hidden_feats = hyperparams['hidden_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    neo4j_uri = "neo4j+s://bb7d3bb8.databases.neo4j.io"
    neo4j_user = "neo4j"
    neo4j_password = "0vZCoYqO6E9YkZRSFsdKPwHcziXu1-b0h8O9edAzWjM"

    reactome_file_path = "gnn_embedding/data/NCBI2Reactome.csv"
    output_file_path = "gnn_embedding/data/NCBI_pathway_map.csv"
    gene_names_file_path = "gnn_embedding/data/gene_names.csv"
    pathway_map = create_pathway_map(reactome_file_path, output_file_path)
    gene_id_to_name_mapping, gene_id_to_symbol_mapping = read_gene_names(gene_names_file_path)
    
    model_path = os.path.join(data_path, 'models')
    model_path = os.path.join(model_path, f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth')
    
    ds = dataset.GeneDataset(data_path)
    '''ds_train = [ds[0]]
    ds_valid = [ds[1]]'''
    ds_train = [ds[1]]
    ds_valid = [ds[0]]
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # net = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True).to(device)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # best_model = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True)
    # best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    # === Model Initialization ===
    net = model.GCNModel(
        in_feats=in_feats,
        hidden_feats=hidden_feats,   # <-- add this parameter
        out_feats=out_feats,
        num_layers=num_layers,
        dropout=dropout,             # optional, can set to 0.0
        do_train=True
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # === Best Model Tracking ===
    best_model = model.GCNModel(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        num_layers=num_layers,
        dropout=dropout,
        do_train=True
    )

    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []

    ## criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    weight = torch.tensor([0.00001, 0.99999]).to(device)


    best_train_loss, best_valid_loss = float('inf'), float('inf')
    best_f1_score = 0.0

    max_f1_scores_train = []
    max_f1_scores_valid = []
    
    results_path = 'gnn_embedding/results/node_embeddings/'
    os.makedirs(results_path, exist_ok=True)

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)  # Flatten 
    save_path_heatmap_initial= os.path.join(results_path, f'heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial= os.path.join(results_path, f'matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial = os.path.join(results_path, f'pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial = os.path.join(results_path, f't-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
        
    for data in dl_train:
        graph, _ = data
        node_embeddings_initial= best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))
        
        print('nx_graph.graph_nx.nodes=====================', len(cluster_labels_initial))
        print('cluster_labels_initial=====================', len(nx_graph.graph_nx.nodes))

        assert len(cluster_labels_initial) == len(nx_graph.graph_nx.nodes), "Cluster labels and number of nodes must match"
        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.graph_nx.nodes)}
        first_node_stId_in_cluster_initial= {}
        first_node_embedding_in_cluster_initial= {}

        stid_dic_initial= {}

        # Populate stid_dic with node stIds mapped to embeddings
        for node in nx_graph.graph_nx.nodes:
            # stid_dic_initial[nx_graph.graph_nx.nodes[node]['stId']] = node_embeddings_initial[node_to_index_initial[node]]
            stid_dic_initial[node] = node_embeddings_initial[node_to_index_initial[node]]

        # Convert stid_dic_initial to a DataFrame
        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

            
        for node, cluster in zip(nx_graph.graph_nx.nodes, cluster_labels_initial):
            if cluster not in first_node_stId_in_cluster_initial:
                first_node_stId_in_cluster_initial[cluster] = node
                # first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node]
                first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index_initial[node]]
                

        print('first_node_stId_in_cluster_initial-------------------------------\n', first_node_stId_in_cluster_initial)
        stid_list = list(first_node_stId_in_cluster_initial.values())
        embedding_list_initial = list(first_node_embedding_in_cluster_initial.values())
        create_heatmap_with_stid(embedding_list_initial, stid_list, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list_initial, stid_list, save_path_matrix_initial)

        break

    visualize_embeddings_tsne(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_t_SNE_initial)
    visualize_embeddings_pca(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_pca_initial)
    silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
    davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)
      
    # Start training  
    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=False) as pbar:
        for epoch in range(num_epochs):
            loss_per_graph = []
            f1_per_graph = [] 
            net.train()
            for data in dl_train:
                graph, name = data
                name = name[0]
                logits = net(graph).view(-1)
                labels = graph.ndata['significance'].float().squeeze()  # removes extra dimension
                num_pos = labels.sum().item()
                num_neg = labels.shape[0] - num_pos
                # weight = torch.tensor([num_neg/labels.shape[0], num_pos/labels.shape[0]]).to(device)

                weight_ = weight[labels.data.view(-1).long()].view_as(labels)

                loss = criterion(logits, labels)
                loss_weighted = loss * weight_
                loss_weighted = loss_weighted.mean()

                # Update parameters
                optimizer.zero_grad()
                loss_weighted.backward()
                optimizer.step()
                
                # Append output metrics
                loss_per_graph.append(loss_weighted.item())
                ##preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                preds = (logits.sigmoid() > 0.5).int()
                # labels = labels.squeeze(1).int()
                labels = labels.int()
                f1 = metrics.f1_score(labels, preds)
                f1_per_graph.append(f1)
                print("logits[:10], preds[:10], labels[:10]===========================================\n", logits[:10], preds[:10], labels[:10])


            running_loss = np.array(loss_per_graph).mean()
            running_f1 = np.array(f1_per_graph).mean()
            loss_per_epoch_train.append(running_loss)
            f1_per_epoch_train.append(running_f1)

            # Validation iteration
            with torch.no_grad():
                loss_per_graph = []
                f1_per_graph = []
                net.eval()
                for data in dl_valid:
                    graph, name = data
                    name = name[0]
                    logits = net(graph).view(-1)
                    # labels = graph.ndata['significance'].unsqueeze(-1)
                    labels = graph.ndata['significance'].float().squeeze()  # removes extra dimension
                    
                    weight_ = weight[labels.data.view(-1).long()].view_as(labels)
                    
                    loss = criterion(logits, labels)
                    loss_weighted = loss * weight_
                    loss_weighted = loss_weighted.mean()
                    
                    loss_per_graph.append(loss_weighted.item())
                    ##preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                    preds = (logits.sigmoid() > 0.5).int()
                    # labels = labels.squeeze(1).int()
                    labels = labels.int()
                    
                    
                    f1 = metrics.f1_score(labels, preds)
                    f1_per_graph.append(f1)

                running_loss = np.array(loss_per_graph).mean()
                running_f1 = np.array(f1_per_graph).mean()
                loss_per_epoch_valid.append(running_loss)
                f1_per_epoch_valid.append(running_f1)
                
                max_f1_train = max(f1_per_epoch_train)
                max_f1_valid = max(f1_per_epoch_valid)
                max_f1_scores_train.append(max_f1_train)
                max_f1_scores_valid.append(max_f1_valid)

                if running_loss < best_valid_loss:
                    best_train_loss = running_loss
                    best_valid_loss = running_loss
                    best_f1_score = running_f1
                    best_model.load_state_dict(copy.deepcopy(net.state_dict()))
                    print(f"Best F1 Score: {best_f1_score}")

            pbar.update(1)
            print(f"Epoch {epoch + 1} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}")

    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)  # Flatten 
    print('cluster_labels=========================\n', cluster_labels)
    
    
    loss_path = os.path.join(results_path, f'loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    # protein_embeddings_initial = pd.DataFrame.from_dict(all_embeddings_initial)##, orient='index')
    # protein_embeddings_initial.to_csv('gat/data/protein_embeddings_initial.csv', index_label='protein2')

    # protein_embeddings_final = pd.DataFrame.from_dict(all_embeddings)##, orient='index')
    # protein_embeddings_final.to_csv('gat/data/protein_embeddings_final.csv', index_label='protein2')

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

        # Populate stid_dic with node stIds mapped to embeddings
        for node in nx_graph.graph_nx.nodes:
            stid_dic[node] = node_embeddings[node_to_index[node]]       
            # # Check if the node's significance is 'significant' and add its stId to the list
            # if graph.ndata['significance'][node_to_index[node]].item() == 'significant':
            #     significant_stIds.append(nx_graph.graph_nx.nodes[node])
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

def train_(hyperparams=None, data_path='gnn_embedding/data/emb', plot=True):
    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    # === Pathway & Gene mapping ===
    reactome_file_path = "gnn_embedding/data/NCBI2Reactome.csv"
    output_file_path = "gnn_embedding/data/NCBI_pathway_map.csv"
    gene_names_file_path = "gnn_embedding/data/gene_names.csv"
    pathway_map = create_pathway_map(reactome_file_path, output_file_path)
    gene_id_to_name_mapping, gene_id_to_symbol_mapping = read_gene_names(gene_names_file_path)

    # === Dataset & DataLoader ===
    ds = dataset.GeneDataset(data_path)
    ds_train = [ds[1]]
    ds_valid = [ds[0]]
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # === Model & Optimizer ===
    # net = model.GATModel(
    #     in_feats=in_feats,
    #     hidden_size=out_feats,
    #     out_feats=out_feats,
    #     num_heads=num_heads,
    #     num_layers=num_layers,
    #     num_classes=1
    # ).to(device)
    
    

    net = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # best_model = model.GATModel(
    #     in_feats=in_feats,
    #     hidden_size=out_feats,
    #     out_feats=out_feats,
    #     num_heads=num_heads,
    #     num_layers=num_layers,
    #     num_classes=1
    # ).to(device)
    
    

    best_model = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    # === Loss Function ===

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    # criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    # === Pre-training Embedding Analysis ===
    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(best_model, dl_train, device)

    # if plot:
    #     plot_cluster_heatmap(all_embeddings_initial, cluster_labels_initial, "heatmap_pre_training.png")
    #     plot_cosine_similarity_heatmap(all_embeddings_initial, "cosine_similarity_heatmap_pre_training.png")
    #     plot_cluster_pca(all_embeddings_initial, cluster_labels_initial, "pca_plot_pre_training.png")
    #     plot_cluster_tsne(all_embeddings_initial, cluster_labels_initial, "tsne_plot_pre_training.png")

    # Save pre-training embeddings to CSV
    for data in dl_train:
        graph, _ = data
        node_embeddings_initial = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'emb/raw', 'emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        node_embeddings_initial = node_embeddings_initial.reshape(node_embeddings_initial.shape[0], -1)
        stid_dic_initial = {}
        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_initial[stId] = node_embeddings_initial[node]

        csv_save_path_initial = os.path.join(
            data_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv'
        )
        pd.DataFrame.from_dict(stid_dic_initial, orient='index').to_csv(csv_save_path_initial, index_label='stId')
        print(f"Saved pre-training embeddings to {csv_save_path_initial}")
        break

    # Evaluate clustering metrics (pre-training)
    sil_score_pre = silhouette_score(all_embeddings_initial, cluster_labels_initial)
    db_score_pre = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    print(f"Pre-training clustering quality: Silhouette Score = {sil_score_pre:.4f}, DB Score = {db_score_pre:.4f}")

    # === Training Loop ===
    losses_train, losses_valid = [], []
    f1_scores_train, f1_scores_valid = [], []
    min_loss = float('inf')

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            # Training
            net.train()
            loss_all = []
            f1_all = []
            for batched_graph, labels in dl_train:
                batched_graph, labels = batched_graph.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = net(batched_graph)
                pos_weight = (labels == 0).sum().float() / (labels == 1).sum().float()
                loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1),
                    labels.float(),
                    pos_weight=pos_weight,
                    reduction='mean'
                )
                loss.backward()
                optimizer.step()
                preds = torch.sigmoid(logits.view(-1)) > 0.5
                f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=1)
                loss_all.append(loss.item())
                f1_all.append(f1)

            losses_train.append(sum(loss_all) / len(loss_all))
            f1_scores_train.append(sum(f1_all) / len(f1_all))

            # Validation
            net.eval()
            loss_all = []
            f1_all = []
            with torch.no_grad():
                for batched_graph, labels in dl_valid:
                    batched_graph, labels = batched_graph.to(device), labels.to(device)
                    logits = net(batched_graph)
                    loss = criterion(logits.view(-1), labels.float())
                    preds = torch.sigmoid(logits.view(-1)) > 0.5
                    f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=1)
                    loss_all.append(loss.item())
                    f1_all.append(f1)

            loss_valid = sum(loss_all) / len(loss_all)
            f1_valid = sum(f1_all) / len(f1_all)
            losses_valid.append(loss_valid)
            f1_scores_valid.append(f1_valid)

            if loss_valid < min_loss:
                min_loss = loss_valid
                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

            pbar.set_postfix({'loss': loss_valid, 'f1': f1_valid})
            pbar.update(1)

    # === Post-training Embedding Analysis ===
    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)

    # if plot:
    #     plot_cluster_heatmap(all_embeddings, cluster_labels, "heatmap_post_training.png")
    #     plot_cosine_similarity_heatmap(all_embeddings, "cosine_similarity_heatmap_post_training.png")
    #     plot_cluster_pca(all_embeddings, cluster_labels, "pca_plot_post_training.png")
    #     plot_cluster_tsne(all_embeddings, cluster_labels, "tsne_plot_post_training.png")

    # Save post-training embeddings to CSV
    for data in dl_train:
        graph, _ = data
        node_embeddings_final = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'emb/raw', 'emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        node_embeddings_final = node_embeddings_final.reshape(node_embeddings_final.shape[0], -1)
        stid_dic_final = {}
        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_final[stId] = node_embeddings_final[node]

        csv_save_path_final = os.path.join(
            data_path,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv'
        )
        pd.DataFrame.from_dict(stid_dic_final, orient='index').to_csv(csv_save_path_final, index_label='stId')
        print(f"Saved post-training embeddings to {csv_save_path_final}")
        break

    # Evaluate clustering metrics (post-training)
    sil_score_post = silhouette_score(all_embeddings, cluster_labels)
    db_score_post = davies_bouldin_score(all_embeddings, cluster_labels)
    print(f"Post-training clustering quality: Silhouette Score = {sil_score_post:.4f}, DB Score = {db_score_post:.4f}")

    # === Save Loss/F1 Curves ===
    plt.figure()
    plt.plot(losses_train, label='Train Loss')
    plt.plot(losses_valid, label='Valid Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()

    plt.figure()
    plt.plot(f1_scores_train, label='Train F1')
    plt.plot(f1_scores_valid, label='Valid F1')
    plt.legend()
    plt.savefig('f1_curve.png')
    plt.close()

    # === Save Best Model ===
    model_path = os.path.join(data_path, "gat_model_best.pth")
    torch.save(best_model.state_dict(), model_path)
    print(f"Best model saved to {model_path}")

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

def plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, stids, save_path):
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
    # Convert the embedding list to a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list)
    
    # Create a clustermap with a continuous color palette
    ax = sns.clustermap(
        heatmap_data,
        cmap='RdYlBu_r',          # continuous colormap (options: 'viridis', 'plasma', 'coolwarm', 'RdYlBu_r', etc.)
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
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
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
