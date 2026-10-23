import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Pehli GNN layer
        self.convs.append(HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum'))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Intermediate layers (agar num_layers > 2 hain)
        for _ in range(num_layers - 2):
            self.convs.append(HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            }, aggr='sum'))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        # Final output layer
        if num_layers > 1:
            self.convs.append(HeteroConv({
                edge_type: SAGEConv((-1, -1), out_channels)
                for edge_type in metadata[1]
            }, aggr='sum'))
        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x_dict, edge_index_dict):
        # Hidden layers pass with BatchNorm, LeakyReLU, and Dropout
        for i in range(len(self.convs) - 1):
            x_dict = self.convs[i](x_dict, edge_index_dict)
            x_dict = {key: self.dropout(self.activation(self.bns[i](x))) for key, x in x_dict.items()}
        
        # Final layer (no activation/dropout on final output embeddings)
        if self.num_layers > 1:
            x_dict = self.convs[-1](x_dict, edge_index_dict)
            
        return x_dict

class LinkPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_dict, edge_index, src_type='entity', tgt_type='entity'):
        row, col = edge_index
        z_src = z_dict[src_type][row]
        z_tgt = z_dict[tgt_type][col]
        return (z_src * z_tgt).sum(dim=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Advanced GraphRx GNN Architecture")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden channel dimension size")
    parser.add_argument("--out_dim", type=int, default=32, help="Output embedding dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    args = parser.parse_args()

    processed_path = "data/processed_graph.pt"
    if os.path.exists(processed_path):
        print(f"Loading cached graph for Advanced Model test (layers={args.num_layers}, hidden={args.hidden_dim}, out={args.out_dim})...")
        graph = torch.load(processed_path, weights_only=False)
        
        model = HeteroGraphSAGE(
            metadata=graph.metadata(), 
            hidden_channels=args.hidden_dim, 
            out_channels=args.out_dim, 
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        predictor = LinkPredictor()
        print("Advanced HeteroGraphSAGE initialized successfully!")
        
        model.eval()
        with torch.no_grad():
            out_dict = model(graph.x_dict, graph.edge_index_dict)
            sample_edges = graph['entity', 'CpD', 'entity'].edge_index[:, :5]
            scores = predictor(out_dict, sample_edges)
            
        print("Forward pass successful with Advanced Architecture!")
        print(f"Output embedding shape for entity: {out_dict['entity'].shape}")
        print("Sample scores shape:", scores.shape)
    else:
        print("Processed graph not found!")