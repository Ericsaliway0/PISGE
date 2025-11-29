import os
import pickle
import dgl
from dgl.data import DGLDataset

class GeneDataset(DGLDataset):
    """
    DGL Dataset for gene–gene networks built from pickled NetworkX graphs.
    Node features: weight, significance
    Matches the feature structure of PathwayDataset (no edge features).
    """

    def __init__(self, root='embedding/data/emb'):
        raw_dir = os.path.join(root, 'raw')
        save_dir = os.path.join(root, 'processed')
        model_dir = os.path.join(root, 'models')

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # store them separately if you need to reuse later
        self._raw_dir = raw_dir
        self._save_dir = save_dir

        super().__init__(name='gene_graph', raw_dir=raw_dir, save_dir=save_dir)

    def has_cache(self):
        raw_files = os.listdir(self._raw_dir)
        processed_files = os.listdir(self._save_dir)
        return len(processed_files) == len(raw_files) and len(raw_files) > 0

    def __len__(self):
        return len(os.listdir(self._save_dir))

    def __getitem__(self, idx):
        names = sorted(os.listdir(self._save_dir))
        name = names[idx]
        (graphs,), _ = dgl.load_graphs(os.path.join(self._save_dir, name))
        return graphs, name

    def process(self):
        for graph_file in os.listdir(self._raw_dir):
            graph_path = os.path.join(self._raw_dir, graph_file)

            nx_graph = pickle.load(open(graph_path, 'rb')).graph_nx

            if not nx_graph.is_directed():
                nx_graph = nx_graph.to_directed()

            for node in nx_graph.nodes:
                weight = nx_graph.nodes[node].get("weight", 1.0)
                try:
                    nx_graph.nodes[node]["weight"] = float(weight)
                except:
                    nx_graph.nodes[node]["weight"] = 1.0

                sig = nx_graph.nodes[node].get("significance", 0)
                if isinstance(sig, (int, float)):
                    nx_graph.nodes[node]["significance"] = 1 if sig != 0 else 0
                else:
                    nx_graph.nodes[node]["significance"] = (
                        1 if str(sig).lower() == "significant" else 0
                    )

            dgl_graph = dgl.from_networkx(
                nx_graph,
                node_attrs=["weight", "significance"]
            )

            save_path = os.path.join(self._save_dir, f"{graph_file[:-4]}.dgl")
            dgl.save_graphs(save_path, dgl_graph)

            print(f"✅ Processed graph {graph_file}: {dgl_graph.num_nodes()} nodes, {dgl_graph.num_edges()} edges")
