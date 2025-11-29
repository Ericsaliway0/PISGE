import os
import pickle
import torch
import dgl
import utils
import model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create embeddings and save to disk.')

    # ----------------------------------------------------------
    # Directories and Output
    # ----------------------------------------------------------
    parser.add_argument('--data_dir', type=str, default='gnn_embedding/data/emb',
                        help='Directory containing the dataset.')
    parser.add_argument('--output_file', type=str, default='gnn_embedding/data/emb/embeddings.pkl',
                        help='Path to save the final embeddings.')
    parser.add_argument('--save', action='store_true',
                        help='Flag to save embeddings to disk (default: False).')

    # ----------------------------------------------------------
    # Model and Training Hyperparameters
    # ----------------------------------------------------------
    parser.add_argument('--model_type', type=str, default='GAT',
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'ChebNet', 'ChebNetII'],
                        help='Type of GNN model to use.')
    parser.add_argument('--num_epochs', type=int, default=20000,
                        help='Number of epochs for training.')
    parser.add_argument('--in_feats', type=int, default=2,
                        help='Number of input features.')
    parser.add_argument('--hidden_feats', type=int, default=128,
                        help='Number of hidden features.')
    parser.add_argument('--out_feats', type=int, default=128,
                        help='Number of output features.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in the model.')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='Number of attention heads (for GAT-type models).')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for optimizer (L2 regularization).')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate.')

    # ----------------------------------------------------------
    # Contrastive Learning Hyperparameters
    # ----------------------------------------------------------
    parser.add_argument('--feat_mask_ratio', type=float, default=0.4,
                        help='Fraction of node features to mask during graph augmentation.')
    parser.add_argument('--edge_drop_ratio', type=float, default=0.3,
                        help='Fraction of edges to randomly drop during graph augmentation.')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                        help='Weight of contrastive loss in total training loss.')
    # parser.add_argument('--feat_mask_ratio', type=float, default=0.4,
    #                     help='Fraction of node features to mask during graph augmentation.')
    # parser.add_argument('--edge_drop_ratio', type=float, default=0.3,
    #                     help='Fraction of edges to randomly drop during graph augmentation.')
    # parser.add_argument('--contrastive_weight', type=float, default=1.0,
    #                     help='Weight of contrastive loss in total training loss.')

    # ----------------------------------------------------------
    # Visualization and Debugging
    # ----------------------------------------------------------
    parser.add_argument('--print_embeddings', action='store_true',
                        help='Print the embeddings dictionary after creation.')
    parser.add_argument('--plot', action='store_true',
                        help='Enable PCA/t-SNE visualization of embeddings.')
    parser.add_argument('--do_pca_tsne', action='store_true',
                        help='Perform PCA and t-SNE on node embeddings.')

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Aggregate hyperparameters into a dictionary
    # ----------------------------------------------------------
    # hyperparams = {
    #     'in_feats': args.in_feats,
    #     'hidden_feats': args.hidden_feats,
    #     'out_feats': args.out_feats,
    #     'num_layers': args.num_layers,
    #     'num_heads': args.num_heads,
    #     'learning_rate': args.lr,
    #     'weight_decay': args.weight_decay,
    #     'dropout': args.dropout,
    #     'num_epochs': args.num_epochs,
    #     'feat_mask_ratio': args.feat_mask_ratio,
    #     'edge_drop_ratio': args.edge_drop_ratio,
    #     'contrastive_weight': args.contrastive_weight,
    # }

    # -------------------------
    # Build networks directly from gene symbols
    # -------------------------
    # graph_train, graph_test = utils.create_embedding_with_genes(
    #     save=args.save,
    #     data_dir=args.data_dir
    # )

    def check_dataset_for_positives(dataset):
        """
        Ensure each graph in the dataset has at least one positive node.
        Returns True if all graphs have positives, False otherwise.
        """
        all_ok = True
        for i, graph in enumerate(dataset):
            # get significance values from NetworkX graph inside your Network wrapper
            if hasattr(graph, 'graph_nx'):
                labels = [graph.graph_nx.nodes[n]['significance'] for n in graph.graph_nx.nodes()]
            else:  # if it's already a DGLGraph
                labels = graph.ndata['significance'].cpu().numpy()
            
            num_positives = sum(labels)
            if num_positives == 0:
                print(f"⚠️ Graph {i} ({getattr(graph, 'kge', 'unknown')}) has NO positive nodes!")
                all_ok = False
            else:
                print(f"Graph {i} ({getattr(graph, 'kge', 'unknown')}) has {num_positives} positives out of {len(labels)} nodes")
        return all_ok


    # ds_train = [graph_train]  # or PathwayDataset(...)
    # ds_test = [graph_test]

    # print("Checking training graphs:==========================================================================================")
    # check_dataset_for_positives(ds_train)

    # print("Checking test graphs:")
    # check_dataset_for_positives(ds_test)

    # -------------------------
    # Training hyperparameters
    # -------------------------
    hyperparameters = {
        'num_epochs': args.num_epochs,
        'in_feats': args.in_feats,
        'out_feats': args.out_feats,
        'hidden_feats': args.hidden_feats,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'batch_size': args.batch_size,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': args.lr,
        "weight_decay": args.weight_decay,
        'dropout': args.dropout,
        'feat_mask_ratio': args.feat_mask_ratio,
        'edge_drop_ratio': args.edge_drop_ratio,
        'contrastive_weight': args.contrastive_weight,
    }


    # -------------------------
    # Create embeddings using GAT
    # -------------------------
    embedding_dict = utils.create_embeddings(
        data_dir=args.data_dir,
        load_model=False,
        hyperparams=hyperparameters,
        output_dir='gnn_embedding/results/node_embeddings'
    )

    # Print embeddings if requested
    if args.print_embeddings:
        print(embedding_dict)

    # Save embeddings to pickle file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(embedding_dict, f)
    print(f"✅ Embeddings saved to {args.output_file}")


if __name__ == '__main__':
    main()
    


## gene_pathway_embedding_ % python gnn_embedding/gat_embedding.py --model_type GAT --out_feats 32 --num_layers 2 --num_heads 1 --batch_size 1 --lr 0.0001 --num_epochs 100
