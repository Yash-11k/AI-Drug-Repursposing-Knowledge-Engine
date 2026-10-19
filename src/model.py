import os
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), out_channels)
            for edge_type in metadata[1]
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

if __name__ == "__main__":
    processed_path = "data/processed_graph.pt"
    if os.path.exists(processed_path):
        print("Loading cached graph for model test...")
        # weights_only=False yahan add kiya gaya hai
        graph = torch.load(processed_path, weights_only=False)
        
        # Model initialize karna
        model = HeteroGraphSAGE(metadata=graph.metadata(), hidden_channels=32, out_channels=16)
        print("Model initialized successfully!")
        
        # Forward pass test
        model.eval()
        with torch.no_grad():
            out_dict = model(graph.x_dict, graph.edge_index_dict)
            
        print("Forward pass successful!")
        print("Output embeddings shape for entity:", out_dict['entity'].shape)
    else:
        print("Processed graph not found! Pehle graph_builder.py run karein.")