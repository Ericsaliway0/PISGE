import pandas as pd
import networkx as nx

import pandas as pd
import networkx as nx
import random

class Network_ori:
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
                gene1 = row["Gene1"]
                gene2 = row["Gene2"]
                pval = float(row["pvalue"])
                sig = int(row["significance"])
                gtype = row.get("gene_type", None)

                # Add nodes
                if pd.notna(gene1):
                    graph_nx.add_node(gene1, pathway=pA, gene_type=gtype, weight=pval)
                if pd.notna(gene2):
                    graph_nx.add_node(gene2, pathway=pB, gene_type=gtype, weight=pval)

                # Add edge (exclude string attributes for DGL)
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
