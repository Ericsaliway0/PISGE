import os
import pickle
import urllib.request
import json
from collections import defaultdict, namedtuple
from datetime import datetime
import networkx as nx
from py2neo import Graph, Node, Relationship
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import marker
import network
import dataset
import model
import train
import pandas as pd
from sklearn.model_selection import train_test_split

def get_stid_mapping(graph):
    stid_mapping = {}  # Mapping of node_id to stId
    for node_id, data in graph.graph_nx.nodes(data=True):
        stId = data['stId']
        stid_mapping[node_id] = stId  # Store the mapping
    return stid_mapping  # Return the stId mapping

def save_graph_to_neo4j(graph):
    from py2neo import Graph, Node, Relationship

    neo4j_url = "neo4j+s://7ffb183d.databases.neo4j.io"
    user = "neo4j"
    password = "BGc2jKUI44h_awhU5gEp8NScyuyx-iSSkTbFHEHJRpY"
    
    neo4j_graph = Graph(neo4j_url, auth=(user, password))

    # Clear the existing graph
    neo4j_graph.delete_all()

    # Create nodes
    nodes = {}
    for node_id, data in graph.graph_nx.nodes(data=True):
        stId = data['stId']
        node = Node("Pathway", stId=stId, name=data['name'], weight=data['weight'], significance=data['significance'])
        nodes[node_id] = node
        neo4j_graph.create(node)

    # Create relationships
    for source, target in graph.graph_nx.edges():
        relationship = Relationship(nodes[source], "parent-child", nodes[target])
        neo4j_graph.create(relationship)

def create_network_from_markers(marker_list, p_value, kge):
    enrichment_analysis = marker.Marker(marker_list, p_value)
    graph = network.Network(enrichment_analysis.result, kge)
    return graph

def save_to_disk(graph, save_dir):
    print('save_dir--------------------', save_dir)
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    save_path = os.path.join(save_dir, graph.kge + '.pkl')
    pickle.dump(graph.graph_nx, open(save_path, 'wb'))

def save_stid_to_csv(graph, save_dir):
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    stid_data = {'stId': [node['stId'] for node in graph.graph_nx.nodes.values()]}
    df = pd.DataFrame(stid_data)
    csv_path = os.path.join(save_dir, 'stId_nodes.csv')
    df.to_csv(csv_path, index=False)

def create_embedding_with_markers(p_value=0.05, save=True, data_dir='embedding/data/emb'):
    emb_train = ['MS4A1', 'CD8A', 'CD4', 'KRT19', 'PCNA', 'CD68', 'PDCD1', 'PTRPC', 'KRT8', 'HER2', 'FOXP3', 'KRT5', 'H3F3A', 'H3F3B', 'RPS6', 'ESR1', 'CD44', 'KRT17', 'PDPN', 'PECAM1', 'GZMB', 'VIM', 'pAb', 'RB1', 'CD3', 'ACTA2', 'PARP1', 'H2AFX', 'CDH1', 'KRT7', 'KRT14', 'COL4A1', 'LMNA', 'H3K27', 'CD274', 'MKI67', 'PGR', 'LMNB1', 'H3K4', 'LMNB2', 'COL1A1', 'CD34', 'AR', 'HIF1A', 'FOXP3']
    emb_test = ['AKT1', 'BMP2', 'BMP4', 'MAPK1', 'MAPK3', 'BRD4', 'CASP3', 'NCAM1', 'MTOR']
    
    graph_train = create_network_from_markers(emb_train, p_value, 'emb_train')
    graph_test = create_network_from_markers_(emb_test, p_value, 'emb_test')

    if save:
        save_dir = os.path.join(data_dir, 'raw')
        save_to_disk(graph_train, save_dir)
        save_to_disk(graph_test, save_dir)

    return graph_train, graph_test

def create_network_from_markers(marker_list, p_value, kge, save_dir="results/enrichment"):
    """
    Create a pathway/gene network from marker list and enrichment, 
    and save enrichment p-values to CSV.

    Parameters
    ----------
    marker_list : list
        List of input marker genes
    p_value : float
        p-value threshold for enrichment
    kge : str
        Identifier for network/embedding run
    save_dir : str
        Directory to save enrichment results
    """
    import os
    import pandas as pd

    # Run enrichment analysis
    enrichment_analysis = marker.Marker(marker_list, p_value)

    # Save enrichment results to CSV
    os.makedirs(save_dir, exist_ok=True)
    enrichment_csv = os.path.join(save_dir, f"enrichment_{kge}.csv")

    if isinstance(enrichment_analysis.result, pd.DataFrame):
        enrichment_analysis.result.to_csv(enrichment_csv, index=False)
    else:
        # fallback if it's a list/dict
        pd.DataFrame(enrichment_analysis.result).to_csv(enrichment_csv, index=False)

    print(f"✅ Enrichment results with p-values saved to {enrichment_csv}")

    # Build network
    graph = network.Network(enrichment_analysis.result, kge)
    return graph

def create_embedding_with_markers(p_value=0.05, save=True, data_dir='embedding/data/emb'):
    # Read symbols from the CSV file
    csv_path = 'embedding/data/genes_pathways.csv'
    data = pd.read_csv(csv_path)
    symbols = data['symbol'].tolist()
    
    # Split the symbols into train and test sets
    emb_train, emb_test = train_test_split(symbols, test_size=0.3, random_state=42)
    ##print('emb_train=========================\n', emb_train)

    # Create networks for train and test sets
    graph_train = create_network_from_markers(emb_train, p_value, 'emb_train')
    graph_test = create_network_from_markers(emb_test, p_value, 'emb_test')
    graph_all = create_network_from_markers(symbols, p_value, 'emb_all')

    if save:
        save_dir = os.path.join(data_dir, 'raw')
        save_to_disk(graph_train, save_dir)
        save_to_disk(graph_test, save_dir)
        # save_to_disk(graph_all, save_dir)

    return graph_train, graph_test

# def create_network_from_markers(marker_list, p_value, kge, save_dir="embedding/results/enrichment"):
#     """
#     Create a pathway/gene network from marker list and enrichment,
#     and return both the network graph and expanded enrichment DataFrame.
#     """
#     import os
#     import pandas as pd

#     enrichment_analysis = marker.Marker(marker_list, p_value)

#     # Convert enrichment results to DataFrame
#     if isinstance(enrichment_analysis.result, pd.DataFrame):
#         df_enrichment = enrichment_analysis.result.copy()
#     else:
#         df_enrichment = pd.DataFrame(enrichment_analysis.result)

#     # Expand to marker × pathway combinations
#     expanded_rows = []
#     for _, row in df_enrichment.iterrows():
#         pathway_info = row.to_dict()
#         for gene in marker_list:
#             expanded_rows.append({**pathway_info, "Marker": gene})

#     df_expanded = pd.DataFrame(expanded_rows)

#     # Build network (still needed downstream)
#     graph = network.Network(enrichment_analysis.result, kge)
#     return graph, df_expanded

# def create_embedding_with_markers_x(p_value=0.05, save=True, data_dir='embedding/data/emb'):
#     """
#     Run enrichment and create embeddings from markers.
#     Save a single enrichment CSV with all markers × pathways.
#     """
#     import os
#     import pandas as pd
#     from sklearn.model_selection import train_test_split

#     # Read symbols from the CSV file
#     csv_path = 'embedding/data/genes_pathways.csv'
#     data = pd.read_csv(csv_path)
#     symbols = data['symbol'].tolist()

#     # Split the symbols into train and test sets
#     emb_train, emb_test = train_test_split(symbols, test_size=0.3, random_state=42)

#     # Create networks and enrichment results
#     graph_train, df_train = create_network_from_markers(emb_train, p_value, 'emb_train')
#     graph_test, df_test = create_network_from_markers(emb_test, p_value, 'emb_test')

#     # Merge both into a single DataFrame
#     df_all = pd.concat([df_train, df_test], ignore_index=True)

#     if save:
#         save_dir = os.path.join(data_dir, 'raw')
#         os.makedirs(save_dir, exist_ok=True)

#         # Save graphs (if needed downstream)
#         save_to_disk(graph_train, save_dir)
#         save_to_disk(graph_test, save_dir)

#         # Save combined enrichment results
#         enrichment_csv = os.path.join(save_dir, "enrichment_all.csv")
#         df_all.to_csv(enrichment_csv, index=False)
#         print(f"✅ Combined enrichment (train + test) saved to {enrichment_csv}")

#     return graph_train, graph_test#, df_all

# def create_network_from_markers(marker_list, p_value, kge, save_dir="embedding/results/enrichment"):
#     """
#     Create a pathway/gene network from marker list and enrichment,
#     save enrichment p-values and stats to CSV, and build the network.

#     Parameters
#     ----------
#     marker_list : list
#         List of input marker genes
#     p_value : float
#         p-value threshold for enrichment
#     kge : str
#         Identifier for network/embedding run
#     save_dir : str
#         Directory to save enrichment results
#     """
#     import os
#     import pandas as pd

#     # Run enrichment analysis
#     enrichment_analysis = marker.Marker(marker_list, p_value)

#     # Expand to Marker × Pathway with all available statistics
#     rows = []
#     if isinstance(enrichment_analysis.result, pd.DataFrame):
#         stats_cols = [c for c in enrichment_analysis.result.columns if c != "PathwayID"]
#         for _, row in enrichment_analysis.result.iterrows():
#             pathway = row["PathwayID"]
#             for gene in marker_list:
#                 row_dict = {"Marker": gene, "Pathway": pathway}
#                 for col in stats_cols:
#                     row_dict[col] = row[col]
#                 rows.append(row_dict)
#         df_expanded = pd.DataFrame(rows)
#     else:
#         # fallback if result is dict/list
#         df_expanded = pd.DataFrame(enrichment_analysis.result)

#     # Save enrichment results to CSV
#     os.makedirs(save_dir, exist_ok=True)
#     enrichment_csv = os.path.join(save_dir, f"enrichment_{kge}.csv")
#     df_expanded.to_csv(enrichment_csv, index=False)
#     print(f"✅ Enrichment results with statistics saved to {enrichment_csv}")

#     # Build network
#     graph = network.Network(enrichment_analysis.result, kge)
#     return graph#, df_expanded


def create_embeddings(load_model=True, save=True, data_dir='embedding/data/emb', hyperparams=None, plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset.PathwayDataset(data_dir)
    emb_dir = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    if not os.path.isdir(emb_dir):
        os.mkdir(emb_dir)

    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']

    net = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads).to(device)

    if load_model:
        model_path = os.path.abspath(os.path.join(data_dir, 'models/model.pth'))
        net.load_state_dict(torch.load(model_path))
    else:
        model_path = train.train(hyperparams=hyperparams, data_path=data_dir, plot=plot)
        net.load_state_dict(torch.load(model_path))

    embedding_dict = {}
    
    for idx in range(len(data)):
        graph, name = data[idx]
        graph = graph.to(device)  # Move graph to the same device as net
        
        with torch.no_grad():
            embedding = net(graph)
        embedding_dict[name] = embedding
        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding.cpu(), emb_path)

    return embedding_dict
