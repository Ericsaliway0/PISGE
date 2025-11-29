import numpy as np
import torch
import torch.nn as nn
import dgl
##from dgl.nn import GATConv


import torch
import torch.nn as nn
from torch import Tensor
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
from dgl.base import DGLError
from typing import Callable, Optional, Tuple, Union

class GATConv(nn.Module):
    def __init__(self,
                 in_feats: Union[int, Tuple[int, int]],
                 out_feats: int,
                 num_heads: int,
                 feat_drop: float = 0.,
                 attn_drop: float = 0.,
                 negative_slope: float = 0.2,
                 residual: bool = False,
                 activation: Optional[Callable] = None,
                 allow_zero_in_degree: bool = False,
                 bias: bool = True) -> None:
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.res_fc is not None and not isinstance(self.res_fc, nn.Identity):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value: bool) -> None:
        """Set the flag to allow zero in-degree for the graph."""
        self._allow_zero_in_degree = set_value

    def forward(self, graph: DGLGraph, feat: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
        """Forward computation."""
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting `allow_zero_in_degree` '
                               'to `True` when constructing this module will '
                               'suppress this check and let the users handle '
                               'it by themselves.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if hasattr(self, 'fc_src'):
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)

            graph.srcdata.update({'ft': feat_src, 'el': (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)})
            graph.dstdata.update({'er': (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval

            if self.bias is not None:
                rst = rst + self.bias.view(1, -1, self._out_feats)

            if self.activation:
                rst = self.activation(rst)

            return rst
 
class _GATModel(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=1, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True, num_layers=1, do_train=False):
        super(GATModel, self).__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, in_feats)
        
        # Initial GAT layer
        self.conv_0 = GATConv(
            in_feats=in_feats, 
            out_feats=out_feats, 
            num_heads=num_heads, 
            feat_drop=feat_drop, 
            attn_drop=attn_drop, 
            negative_slope=negative_slope, 
            residual=residual, 
            activation=activation, 
            allow_zero_in_degree=allow_zero_in_degree, 
            bias=bias
        )
        
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        
        # Additional GAT layers
        self.layers = nn.ModuleList([
            GATConv(
                in_feats=out_feats * num_heads, 
                out_feats=out_feats, 
                num_heads=num_heads, 
                feat_drop=feat_drop, 
                attn_drop=attn_drop, 
                negative_slope=negative_slope, 
                residual=residual, 
                activation=activation, 
                allow_zero_in_degree=allow_zero_in_degree, 
                bias=bias
            )
            for _ in range(num_layers - 1)
        ])
        
        # Final prediction layer
        self.predict = nn.Linear(out_feats * num_heads, 1)

    def forward(self, graph):
        # Preprocess the input features
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        
        # Add self-loops to the graph
        graph = dgl.add_self_loop(graph)
        
        # Initial GAT layer
        embedding = self.conv_0(graph, features).flatten(1)
        
        # Subsequent GAT layers
        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding).flatten(1)
        
        if not self.do_train:
            return embedding.detach()
        
        # Predict logits for training
        logits = self.predict(embedding.mean(dim=1)).squeeze(-1)
        return logits

    def get_node_embeddings(self, graph):
        # Preprocess the input features
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        
        # Add self-loops to the graph
        graph = dgl.add_self_loop(graph)
        
        # Initial GAT layer
        embedding = self.conv_0(graph, features).flatten(1)
        
        # Subsequent GAT layers
        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding).flatten(1)
        
        return embedding
 
class GATModel_(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=1, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True, num_layers=1, do_train=False):
        super(GATModel, self).__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, in_feats)
        self.conv_0 = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=residual, activation=activation, allow_zero_in_degree=allow_zero_in_degree, bias=bias)
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList([GATConv(in_feats=out_feats * num_heads, out_feats=out_feats, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=residual, activation=activation, allow_zero_in_degree=allow_zero_in_degree, bias=bias)
                                     for _ in range(num_layers - 1)])
        self.predict = nn.Linear(out_feats * num_heads, 1)
        

    def forward(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)
        
        if not self.do_train:
            return embedding.detach()
        
        logits = self.predict(embedding.mean(dim=1)).squeeze(-1)  # Adjust the shape to match the target
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding).flatten(1)

        ##embedding = np.vstack(embedding).detach().numpy()
        return embedding

class GATModel_GCL(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=1, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=True, activation=None, allow_zero_in_degree=False, bias=True, num_layers=1, do_train=False):
        super(GATModel, self).__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, in_feats)
        self.conv_0 = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=residual, activation=activation, allow_zero_in_degree=allow_zero_in_degree, bias=bias)
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList([GATConv(in_feats=out_feats * num_heads, out_feats=out_feats, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=residual, activation=activation, allow_zero_in_degree=allow_zero_in_degree, bias=bias)
                                     for _ in range(num_layers - 1)])
        self.predict = nn.Linear(out_feats * num_heads, 1)
        

    def forward(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)
        
        if not self.do_train:
            return embedding.detach()
        
        logits = self.predict(embedding.mean(dim=1)).squeeze(-1)  # Adjust the shape to match the target
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding).flatten(1)

        ##embedding = np.vstack(embedding).detach().numpy()
        return embedding

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv


class GCNModel(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_feats,
        out_feats,
        activation=F.relu,
        dropout=0.0,
        residual=True,
        num_layers=2,
        do_train=False
    ):
        super(GCNModel, self).__init__()
        self.do_train = do_train
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)

        # Linear transform from raw node weight to input feature space
        self.linear = nn.Linear(1, in_feats)

        # First GCN layer
        self.conv_0 = GraphConv(in_feats, hidden_feats, norm='both', weight=True, bias=True)

        # Additional GCN layers
        self.layers = nn.ModuleList([
            GraphConv(hidden_feats, hidden_feats, norm='both', weight=True, bias=True)
            for _ in range(num_layers - 2)
        ])

        # Final GCN layer → output feature space
        self.conv_out = GraphConv(hidden_feats, out_feats, norm='both', weight=True, bias=True)

        # Prediction head (used only if training for a task)
        self.predict = nn.Linear(out_feats, 1)

    def forward(self, graph):
        # Input features: use node weights as initial signals
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)

        # Ensure graph has self-loops
        graph = dgl.add_self_loop(graph)

        # Layer 1
        h = self.conv_0(graph, features)
        h = self.activation(h)
        h = self.dropout(h)

        # Hidden layers
        for conv in self.layers:
            h_res = h
            h = conv(graph, h)
            h = self.activation(h)
            if self.residual:
                h = h + h_res
            h = self.dropout(h)

        # Final layer
        embedding = self.conv_out(graph, h)

        if not self.do_train:
            return embedding.detach()

        # Prediction head
        logits = self.predict(embedding).squeeze(-1)
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)

        h = self.conv_0(graph, features)
        h = self.activation(h)

        for conv in self.layers:
            h = conv(graph, h)
            h = self.activation(h)

        embedding = self.conv_out(graph, h)
        return embedding.detach()

# model.eval()
# embeddings = model.get_node_embeddings(graph_bio)

# -----------------------------
# GATModel (safe for autograd)
# -----------------------------
class GATModel(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=1, feat_drop=0.0, attn_drop=0.0,
                 negative_slope=0.2, residual=True, activation=None, allow_zero_in_degree=False,
                 bias=True, num_layers=1, do_train=False):
        super(GATModel, self).__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, in_feats)
        self.conv_0 = GATConv(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
            bias=bias
        )
        self.relu = nn.LeakyReLU(inplace=False)  # avoid in-place
        self.layers = nn.ModuleList([
            GATConv(
                in_feats=out_feats * num_heads,
                out_feats=out_feats,
                num_heads=num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=activation,
                allow_zero_in_degree=allow_zero_in_degree,
                bias=bias
            ) for _ in range(num_layers - 1)
        ])
        self.predict = nn.Linear(out_feats * num_heads, 1)

    def forward(self, graph):
        # --- Node features ---
        weights = graph.ndata['weight'].unsqueeze(-1).clone()  # clone to avoid in-place issues
        features = self.linear(weights)
        g = dgl.add_self_loop(graph)
        embedding = self.conv_0(g, features)

        for conv in self.layers:
            embedding = self.relu(embedding)  # safe
            embedding = conv(g, embedding)

        if not self.do_train:
            return embedding.detach()  # safe: no grad

        logits = self.predict(embedding.mean(dim=1)).squeeze(-1)
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1).clone()
        features = self.linear(weights)
        g = dgl.add_self_loop(graph)
        embedding = self.conv_0(g, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(g, embedding)

        return embedding  # shape: [num_nodes, out_feats * num_heads]

    # -----------------------------
    # Contrastive loss (graph augmentation)
    # -----------------------------
    def compute_contrastive_loss(self, graph, tau=0.5, feat_mask_ratio=0.1, edge_drop_ratio=0.2):
        # --- Generate two augmented views ---
        g1 = self.augment_graph(graph, feat_mask_ratio=feat_mask_ratio, edge_drop_ratio=edge_drop_ratio)
        g2 = self.augment_graph(graph, feat_mask_ratio=feat_mask_ratio, edge_drop_ratio=edge_drop_ratio)

        # --- Node embeddings ---
        z1 = self.get_node_embeddings(g1)  # [N, D] or [N, 1, D]
        z2 = self.get_node_embeddings(g2)  # [N, D] or [N, 1, D]

        # --- Flatten if necessary ---
        if z1.dim() > 2:
            z1 = z1.reshape(z1.size(0), -1)
        if z2.dim() > 2:
            z2 = z2.reshape(z2.size(0), -1)

        # --- Normalize embeddings ---
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # --- Cosine similarity matrix ---
        sim_matrix = torch.matmul(z1, z2.T) / tau
        N = z1.size(0)
        labels = torch.arange(N, device=z1.device)

        # --- Contrastive loss ---
        loss1 = nn.CrossEntropyLoss()(sim_matrix, labels)
        loss2 = nn.CrossEntropyLoss()(sim_matrix.T, labels)
        loss = (loss1 + loss2) / 2.0
        return loss

    # ---------------------------------------------------------
    # Graph augmentations (feature masking + edge dropout)
    # ---------------------------------------------------------
    def augment_graph(self, g, feat_mask_ratio=0.1, edge_drop_ratio=0.2):
        import torch

        g_aug = g.clone()

        # --- Edge dropout ---
        num_edges = g_aug.num_edges()
        keep_prob = 1.0 - edge_drop_ratio
        num_keep = int(num_edges * keep_prob)
        keep_idx = torch.randperm(num_edges)[:num_keep]

        # Use relabel_nodes=False instead of preserve_nodes
        g_aug = dgl.edge_subgraph(g_aug, keep_idx, relabel_nodes=False)

        # --- Node feature masking ---
        feat = g_aug.ndata['weight'].clone()
        num_mask = int(feat.size(0) * feat_mask_ratio)
        mask_idx = torch.randperm(feat.size(0))[:num_mask]
        mask = torch.ones_like(feat)
        mask[mask_idx] = 0.0
        g_aug.ndata['weight'] = feat * mask

        return g_aug

    # ---------------------------------------------------------
    # Contrastive loss (InfoNCE)
    # ---------------------------------------------------------
    def contrastive_loss(self, z1, z2):
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim_matrix = torch.matmul(z1, z2.T) / self.tau
        sim_pos = torch.diag(sim_matrix)
        loss = -torch.log(sim_pos / torch.sum(torch.exp(sim_matrix), dim=1))
        return loss.mean()

    # ---------------------------------------------------------
    # Compute contrastive loss between two graph views
    # ---------------------------------------------------------
    # def compute_contrastive_loss_(self, g):
        g1 = self.augment_graph(g)
        g2 = self.augment_graph(g)

        features1 = g1.ndata['weight'].unsqueeze(-1)
        features2 = g2.ndata['weight'].unsqueeze(-1)

        z1 = self.conv_0(g1, self.linear(features1))
        z2 = self.conv_0(g2, self.linear(features2))

        for conv in self.layers:
            z1 = self.relu(z1)
            z1 = conv(g1, z1)
            z2 = self.relu(z2)
            z2 = conv(g2, z2)

        z1 = z1.flatten(1)
        z2 = z2.flatten(1)
        return self.contrastive_loss(z1, z2)