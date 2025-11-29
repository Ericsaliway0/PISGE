import os
import pickle
import networkx as nx
import torch
import pandas as pd
from py2neo import Graph, Node, Relationship
import network
import dataset
import model
import train
from sklearn.model_selection import train_test_split
import dgl
import numpy as np


# -------------------------
# Network Creation
# -------------------------
class Network:
    def __init__(self, kge=None):
        self.kge = kge
        self.graph_nx = nx.DiGraph()  # directed graph

def create_network_from_genes_0_1(
    gene_list,
    kge,
    save_dir=None,
    csv_file='data/processed/pathway_gene_type_and_connected_driver_gene_first10000.csv'
):
    """
    Build a directed NetworkX graph from a gene list.
    Node weight = edge pvalue, node significance = any connected edge significant.
    """
    graph = Network(kge=kge)

    if len(gene_list) == 0:
        print(f"⚠️ Gene list is empty for {kge}. Skipping network creation.")
        return graph

    df = pd.read_csv(csv_file)
    df = df[df['Gene1'].notna() & df['Gene2'].notna()]
    df_filtered = df[df['Gene1'].isin(gene_list) & df['Gene2'].isin(gene_list)]

    # Add nodes and edges
    for _, row in df_filtered.iterrows():
        gene1 = row['Gene1']
        gene2 = row['Gene2']
        pval = float(row.get('pvalue', 1.0))
        sig = int(row.get('significance', 0))

        # Add nodes with initial attributes
        graph.graph_nx.add_node(gene1, weight=pval, significance=sig)
        graph.graph_nx.add_node(gene2, weight=pval, significance=sig)

        # Add directed edge
        graph.graph_nx.add_edge(gene1, gene2, pvalue=pval, significance=sig)

    # Propagate node significance: node is significant if any connected edge is significant
    for node in graph.graph_nx.nodes():
        edge_sigs = [data['significance'] for _, _, data in graph.graph_nx.edges(node, data=True)]
        graph.graph_nx.nodes[node]['significance'] = int(any(edge_sigs))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{kge}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(graph, f)

    print(f"✅ Created network '{kge}' with {graph.graph_nx.number_of_nodes()} nodes "
          f"and {graph.graph_nx.number_of_edges()} edges")
    return graph

def create_network_from_genes(
    gene_list,
    kge,
    save_dir=None,
    csv_file='data/processed/gene_gene_pairs_with_pathwayA_enrichment_unique_200_limitation.csv'
):
    """
    Build a directed NetworkX graph from a gene list.
    Node weight = edge pvalue, node significance = any connected edge significant.
    """
    graph = Network(kge=kge)

    if len(gene_list) == 0:
        print(f"⚠️ Gene list is empty for {kge}. Skipping network creation.")
        return graph

    df = pd.read_csv(csv_file)
    df = df[df['Gene1'].notna() & df['Gene2'].notna()]
    df_filtered = df[df['Gene1'].isin(gene_list) & df['Gene2'].isin(gene_list)]

    # Add nodes and edges
    for _, row in df_filtered.iterrows():
        gene1 = row['Gene1']
        gene2 = row['Gene2']
        pval = float(row.get('pvalue', 1.0))

        # Convert significance to 0/1
        sig_val = row.get('significance', 0)
        if isinstance(sig_val, str):
            sig_val = 1 if sig_val.lower() == 'significant' else 0
        sig = int(sig_val)

        # Add nodes with initial attributes
        graph.graph_nx.add_node(gene1, weight=pval, significance=sig)
        graph.graph_nx.add_node(gene2, weight=pval, significance=sig)

        # Add directed edge
        graph.graph_nx.add_edge(gene1, gene2, pvalue=pval, significance=sig)

    # Propagate node significance: node is significant if any connected edge is significant
    for node in graph.graph_nx.nodes():
        edge_sigs = [data['significance'] for _, _, data in graph.graph_nx.edges(node, data=True)]
        graph.graph_nx.nodes[node]['significance'] = int(any(edge_sigs))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{kge}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(graph, f)

    print(f"✅ Created network '{kge}' with {graph.graph_nx.number_of_nodes()} nodes "
          f"and {graph.graph_nx.number_of_edges()} edges")
    return graph

# -------------------------
# Full Pipeline
# -------------------------
def create_embedding_with_genes(save=True, data_dir='gnn_embedding/data/emb'):
    """
    Build train/test networks from CSV and convert them to DGLGraphs.
    """
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    csv_path = 'data/processed/gene_gene_pairs_with_pathwayA_enrichment_unique_200_limitation.csv'
    df = pd.read_csv(csv_path)
    symbols = list(set(df['Gene1'].dropna().tolist() + df['Gene2'].dropna().tolist()))

    emb_train, emb_test = train_test_split(symbols, test_size=0.3, random_state=42)
    save_dir = os.path.join(data_dir, 'raw')
    os.makedirs(save_dir, exist_ok=True)

    # Build networks
    graph_train = create_network_from_genes(emb_train, 'emb_train', save_dir, csv_file=csv_path)
    # g = create_network_from_genes(gene_list, 'emb_train', save_dir)
    num_pos = sum([n['significance'] for _, n in graph_train.graph_nx.nodes(data=True)])
    print(f"Number of positives in network: {num_pos}======================================++++++++++++++")

    graph_test = create_network_from_genes(emb_test, 'emb_test', save_dir, csv_file=csv_path)

    # Ensure significance is numeric 0/1
    for g in [graph_train, graph_test]:
        for node in g.graph_nx.nodes():
            sig = g.graph_nx.nodes[node]['significance']
            if isinstance(sig, str):
                g.graph_nx.nodes[node]['significance'] = 1 if sig.lower() == 'significant' else 0

    # Convert to DGL
    dgl_train = convert_to_dgl(graph_train)
    dgl_test = convert_to_dgl(graph_test)

    if save:
        print(f"✅ Saved train/test DGL graphs to {save_dir}")

    return dgl_train, dgl_test


def create_embeddings(load_model=True, save=True, data_dir='gnn_embedding/data/emb', 
                      hyperparams=None, plot=True, do_pca_tsne=False):
    import dgl
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    data = dataset.GeneDataset(root=data_dir)
    print(f"✅ Loaded dataset with {len(data)} graphs")

    # Inspect all graphs
    inspect_gene_dataset(data, "GeneDataset")


    emb_dir = os.path.join(data_dir, 'embeddings')
    os.makedirs(emb_dir, exist_ok=True)
    
    net = model.GCNModel(
        in_feats=in_feats,
        hidden_feats=hidden_feats,
        out_feats=out_feats,
        num_layers=num_layers,
        dropout=dropout,
        do_train=True
    ).to(device)

    # net = model.GATModel(
    #     in_feats=hyperparams['in_feats'],
    #     out_feats=hyperparams['out_feats'],
    #     num_layers=hyperparams['num_layers'],
    #     num_heads=hyperparams['num_heads']
    # ).to(device)

    model_path = os.path.join(data_dir, 'models/model.pth')

    # Load or train model
    if load_model and os.path.isfile(model_path):
        print(f"🔄 Loading existing model from {model_path}")
        net.load_state_dict(torch.load(model_path))
    else:
        print("🚀 Training new model...")
        model_path = train.train(hyperparams=hyperparams, data_path=data_dir)

        print(f"✅ Model trained and saved at {model_path}")
        net.load_state_dict(torch.load(model_path))

    net.eval()

    # Collect embeddings (2D node-level per graph)
    embedding_dict = {}
    with torch.no_grad():
        for idx in range(len(data)):
            g, name = data[idx]
            g = g.to(device)
            emb = net(g)  # shape [num_nodes, out_feats]

            embedding_dict[name] = emb.cpu().numpy()

            if save:
                emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
                torch.save(emb.cpu(), emb_path)

    # Try aligned matrix (only if embeddings are 1D per graph)
    all_embeddings_aligned = None
    try:
        all_embeddings_aligned = np.array([
            embedding_dict[g] for g in embedding_dict if embedding_dict[g].ndim == 1
        ])
        if all_embeddings_aligned.size > 0:
            print(f"🔍 Embedding matrix shape: {all_embeddings_aligned.shape}")
    except Exception:
        print("⚠️ Skipping aligned matrix since embeddings are 2D per graph")

    # --- Heatmap ---
    if plot and all_embeddings_aligned is not None and all_embeddings_aligned.ndim == 2 \
       and all_embeddings_aligned.shape[1] > 1 and len(embedding_dict) > 2:
        save_path_heatmap = os.path.join(
            data_dir, 
            f"heatmap_GeneSymbol_dim{hyperparams['out_feats']}_lay{hyperparams['num_layers']}_epo{hyperparams['num_epochs']}.png"
        )
        try:
            create_heatmap_with_genes(all_embeddings_aligned, list(embedding_dict.keys()), save_path_heatmap)
            print(f"✅ Heatmap saved to {save_path_heatmap}")
        except Exception as e:
            print(f"❌ Failed to create heatmap: {e}")
    else:
        print("⚠️ Skipping heatmap: embeddings are per-node (2D) and not directly comparable")

    # --- PCA & t-SNE ---
    if do_pca_tsne:
        # Stack all node embeddings across graphs
        X = np.vstack([emb for emb in embedding_dict.values()])
        y = np.hstack([
            g.ndata['significance'].cpu().numpy()
            for g, _ in data
        ])[:len(X)]  # binary labels for coloring

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7)
        plt.colorbar(scatter, label="Label")
        plt.title("PCA of Node Embeddings")

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        X_tsne = tsne.fit_transform(X)

        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="coolwarm", alpha=0.7)
        plt.colorbar(scatter, label="Label")
        plt.title("t-SNE of Node Embeddings")

        plt.tight_layout()
        plt.show()

    return embedding_dict


def inspect_gene_dataset(dataset, dataset_name="GeneDataset"):
    """
    Inspect all graphs in a dataset and report number of positive nodes per graph.
    
    Parameters:
        dataset: GeneDataset or PathwayDataset (subclass of DGLDataset)
        dataset_name: str, name for printing
    """
    print(f"\n🔍 Checking {dataset_name}\n" + "="*80)
    total_graphs = len(dataset)
    total_nodes = 0
    total_positives = 0

    for i in range(total_graphs):
        graph, name = dataset[i]   # your __getitem__ returns (graph, name)
        num_nodes = graph.num_nodes()
        num_positives = int(graph.ndata["significance"].sum().item())
        
        total_nodes += num_nodes
        total_positives += num_positives

        print(f"Graph {i} ({name}): {num_positives} positives / {num_nodes} nodes")

    print("-"*80)
    print(f"✅ Checked {total_graphs} graphs in {dataset_name}_____________+++++++++++++++\n")
    print(f"📊 Total positives: {total_positives} / {total_nodes} nodes ({100*total_positives/total_nodes:.2f}%)\n")

def convert_to_dgl(network):
    nx_graph = network.graph_nx

    # Ensure all nodes have significance and weight
    for node in nx_graph.nodes():
        edge_significance = [
            nx_graph.edges[node, nbr].get('significance', 0)
            for nbr in nx_graph.neighbors(node)
        ]
        nx_graph.nodes[node]['significance'] = int(any(edge_significance))
        nx_graph.nodes[node]['weight'] = nx_graph.nodes[node].get('weight', 1.0)

    # Ensure all edges have required attributes
    for u, v in nx_graph.edges():
        data = nx_graph.edges[u, v]
        data.setdefault('significance', 0)
        data.setdefault('pvalue', 1.0)

    # Convert to DGL
    dgl_graph = dgl.from_networkx(
        nx_graph,
        node_attrs=['significance', 'weight'],  # <-- include significance here
        edge_attrs=['significance', 'pvalue']
    )
    return dgl_graph
