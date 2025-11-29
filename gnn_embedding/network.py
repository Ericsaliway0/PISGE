import pandas as pd
import networkx as nx
import random
import os
import pickle
import dgl
import networkx as nx
from dgl.data import DGLDataset

class GeneDataset(DGLDataset):
    """
    DGL Dataset for gene–gene networks built from pickled Network objects.
    Node features: weight, significance
    Matches the feature structure of PathwayDataset (no edge features).
    """

    def __init__(self, root='embedding/data/emb'):
        self.raw_dir = os.path.join(root, 'raw')
        self.save_dir = os.path.join(root, 'processed')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        super().__init__(name='gene_graph', raw_dir=self.raw_dir, save_dir=self.save_dir)

    def has_cache(self):
        raw_files = os.listdir(self.raw_dir)
        processed_files = os.listdir(self.save_dir)
        return len(processed_files) == len(raw_files) and len(raw_files) > 0

    def __len__(self):
        return len(os.listdir(self.save_dir))

    def __getitem__(self, idx):
        names = sorted(os.listdir(self.save_dir))
        name = names[idx]
        (graphs,), _ = dgl.load_graphs(os.path.join(self.save_dir, name))
        return graphs, name

    def process(self):
        """
        Convert raw Network pickled objects into DGLGraphs with numeric node attributes.
        Node attrs: weight, significance
        (No edge attributes, to match PathwayDataset)
        """
        for graph_file in os.listdir(self.raw_dir):
            graph_path = os.path.join(self.raw_dir, graph_file)

            # Load the full Network object
            network_obj = pickle.load(open(graph_path, 'rb'))
            nx_graph = network_obj.graph_nx

            # Ensure directed
            if not nx_graph.is_directed():
                nx_graph = nx_graph.to_directed()

            # Clean node attributes
            for node in nx_graph.nodes:
                # Weight → float
                weight = nx_graph.nodes[node].get("weight", 1.0)
                try:
                    nx_graph.nodes[node]["weight"] = float(weight)
                except Exception:
                    nx_graph.nodes[node]["weight"] = 1.0

                # Significance → already set by Network.add_node_significance()
                sig = nx_graph.nodes[node].get("significance", 0)
                nx_graph.nodes[node]["significance"] = 1 if sig else 0

            # Debug: count positives before conversion
            num_pos = sum(nx_graph.nodes[n]["significance"] for n in nx_graph.nodes)
            print(f"🔍 {graph_file}: {num_pos} positives / {nx_graph.number_of_nodes()} nodes")

            # Convert to DGLGraph (node features only)
            dgl_graph = dgl.from_networkx(
                nx_graph,
                node_attrs=["weight", "significance"]
            )

            # Save DGLGraph
            save_path = os.path.join(self.save_dir, f"{graph_file[:-4]}.dgl")
            dgl.save_graphs(save_path, dgl_graph)

            print(f"✅ Processed {graph_file}: {dgl_graph.num_nodes()} nodes, {dgl_graph.num_edges()} edges")

class Network:
    def __init__(self, csv_file=None, kge=None, max_pairs=2, seed=42):
        self.kge = kge
        self.max_pairs = max_pairs
        self.seed = seed
        self.graph_nx = nx.Graph()
        if csv_file:
            self.graph_nx = self.to_gene_networkx(csv_file)
            self.add_node_significance()

    def to_gene_networkx(self, file_path):
        df = pd.read_csv(file_path)
        graph_nx = nx.Graph()
        grouped = df.groupby(["PathwayA", "PathwayB"])
        for (pA, pB), group in grouped:
            rows = group.sample(n=min(len(group), self.max_pairs), random_state=self.seed)
            for _, row in rows.iterrows():
                gene1, gene2 = row["Gene1"], row["Gene2"]
                pval = float(row["pvalue"])
                sig = int(row["significance"])
                gtype = row.get("gene_type", None)

                if pd.notna(gene1):
                    graph_nx.add_node(gene1, pathway=pA, gene_type=gtype, weight=pval)
                if pd.notna(gene2):
                    graph_nx.add_node(gene2, pathway=pB, gene_type=gtype, weight=pval)
                if pd.notna(gene1) and pd.notna(gene2):
                    graph_nx.add_edge(gene1, gene2, pvalue=pval, significance=sig)

        print(f"✅ Built gene network: {graph_nx.number_of_nodes()} nodes, {graph_nx.number_of_edges()} edges")
        return graph_nx

    def add_node_significance(self):
        for node in self.graph_nx.nodes():
            edge_significance = [
                data['significance'] for _, _, data in self.graph_nx.edges(node, data=True)
            ]
            self.graph_nx.nodes[node]['significance'] = int(any(edge_significance))

# import math
# import json
# import urllib.request
# from collections import defaultdict, namedtuple
# from datetime import datetime
# import networkx as nx
# from py2neo import Graph, Node, Relationship
# from networkx.algorithms.traversal.depth_first_search import dfs_tree


# class Network:
    
#     Info = namedtuple('Info', ['name', 'species', 'type', 'diagram'])

#     def __init__(self, ea_result=None, kge=None):
#         self.txt_url = 'https://reactome.org/download/current/ReactomePathwaysRelation.txt'
#         self.json_url = 'https://reactome.org/ContentService/data/eventsHierarchy/9606'
#         if kge is not None:
#             self.kge = kge
#         else:
#             time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
#             kge = time_now
#         self.txt_adjacency = self.parse_txt()
#         self.json_adjacency, self.pathway_info = self.parse_json()
#         if ea_result is not None:
#             self.weights = self.set_weights(ea_result)
#         else:
#             self.weights = None
#         self.name_to_id = self.set_name_to_id()
#         self.graph_nx = self.to_networkx()

#         # Save name_to_id and sorted stids to text files
#         self.save_name_to_id()
#         self.save_sorted_stids()
        
#     def parse_txt(self):
#         txt_adjacency = defaultdict(list)
#         found = False
#         with urllib.request.urlopen(self.txt_url) as f:
#             lines = f.readlines()
#             for line in lines:
#                 line = line.decode('utf-8')
#                 stid1, stid2 = line.strip().split()
#                 if not 'R-HSA' in stid1:
#                     if found:
#                         break
#                     else:
#                         continue
#                 txt_adjacency[stid1].append(stid2)
#         txt_adjacency = dict(txt_adjacency)
#         return txt_adjacency

#     def parse_json(self):
#         with urllib.request.urlopen(self.json_url) as f:
#             tree_list = json.load(f)
#         json_adjacency = defaultdict(list)
#         pathway_info = {}
#         for tree in tree_list:
#             self.recursive(tree, json_adjacency, pathway_info)
#         json_adjacency = dict(json_adjacency)
#         return json_adjacency, pathway_info

#     def recursive(self, tree, json_adjacency, pathway_info):
#         id = tree['stId']
#         try:
#             pathway_info[id] = Network.Info(tree['name'], tree['species'], tree['type'], tree['diagram'])
#         except KeyError:
#             pathway_info[id] = Network.Info(tree['name'], tree['species'], tree['type'], None)
#         try:
#             children = tree['children']
#         except KeyError:
#             return
#         for child in children:
#             json_adjacency[id].append(child['stId'])
#             self.recursive(child, json_adjacency, pathway_info)

#     def set_weights(self, ea_result):
#         weights = {}
#         for stid in self.pathway_info.keys():
#             if stid in ea_result.keys():
#                 weights[stid] = ea_result[stid].copy()
#             else:
#                 weights[stid] = {'p_value': 1.0, 'significance': 'not-found'}
#         return weights

#     def set_node_attributes(self):
#         stids, names, weights, significances = {}, {}, {}, {}
#         for stid in self.pathway_info.keys():
#             stids[stid] = stid
#             names[stid] = self.pathway_info[stid].name
#             weights[stid] = 1.0 if self.weights is None else self.weights[stid]['p_value']
#             significances[stid] = 'not-found' if self.weights is None else self.weights[stid]['significance']
#         return stids, names, weights, significances

#     def set_name_to_id(self):
#         name_to_id = {}
#         for id, info in self.pathway_info.items():
#             name_to_id[info.name] = id
#         return name_to_id

#     def save_name_to_id(self):
#         file_path = 'embedding/data/emb/info/name_to_id.txt'
#         with open(file_path, 'w') as f:
#             for name, id in self.name_to_id.items():
#                 f.write(f"{name}: {id}\n")

#     def save_sorted_stids(self):
#         file_path = 'embedding/data/emb/info/sorted_stids.txt'
#         stids = sorted(self.pathway_info.keys())
#         with open(file_path, 'w') as f:
#             for stid in stids:
#                 f.write(f"{stid}\n")

#     def to_networkx(self, type='json'):
#         graph_nx = nx.DiGraph()
#         graph = self.json_adjacency if type == 'json' else self.txt_adjacency
#         for key, values in graph.items():
#             for value in values:
#                 graph_nx.add_edge(key, value)

#         stids, names, weights, significances = self.set_node_attributes()

#         nx.set_node_attributes(graph_nx, stids, 'stId')
#         nx.set_node_attributes(graph_nx, names, 'name')
#         nx.set_node_attributes(graph_nx, weights, 'weight')
#         nx.set_node_attributes(graph_nx, significances, 'significance')

#         return graph_nx

#     def add_significance_by_stid(self, stid_list):
#         for stid in stid_list:
#             try:
#                 self.graph_nx.nodes[stid]['significance'] = 'significant'
#                 self.graph_nx.nodes[stid]['weight'] = 0.0
#             except KeyError:
#                 continue

#     def save_to_neo4j(self):
#         # Clear the existing graph
#         self.neo4j_graph.delete_all()

#         # Create nodes
#         nodes = {}
#         for node_id, data in self.graph_nx.nodes(data=True):
#             node = Node("Pathway", stId=data['stId'], name=data['name'], weight=data['weight'], significance=data['significance'])
#             nodes[node_id] = node
#             self.neo4j_graph.create(node)

#         # Create relationships
#         for source, target in self.graph_nx.edges():
#             relationship = Relationship(nodes[source], "RELATED_TO", nodes[target])
#             self.neo4j_graph.create(relationship)
