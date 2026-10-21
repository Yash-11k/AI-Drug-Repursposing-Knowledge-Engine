import os
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), out_channels)
            for edge_type in metadata[1]
        }, aggr='sum')
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_dict, edge_index_dict):
        # Pehla GNN layer + ReLU + Dropout
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.dropout(x.relu()) for key, x in x_dict.items()}
        
        # Dusra GNN layer (final embeddings ke liye)
        x_dict = self.conv2(x_dict, edge_index_dict)
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
    processed_path = "data/processed_graph.pt"
    if os.path.exists(processed_path):
        print("Loading cached graph for model test with dropout...")
        graph = torch.load(processed_path, weights_only=False)
        
        model = HeteroGraphSAGE(metadata=graph.metadata(), hidden_channels=32, out_channels=16, dropout=0.2)
        predictor = LinkPredictor()
        print("Model with Dropout and LinkPredictor initialized successfully!")
        
        model.eval()
        with torch.no_grad():
            out_dict = model(graph.x_dict, graph.edge_index_dict)
            sample_edges = graph['entity', 'CpD', 'entity'].edge_index[:, :5]
            scores = predictor(out_dict, sample_edges)
            
        print("Forward pass with Dropout successful!")
        print("Sample scores shape:", scores.shape)
    else:
        print("Processed graph not found!")