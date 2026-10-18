import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        
        # HeteroConv har ek edge type (metaedge) ke liye independent GraphSAGE convolution banata hai
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), out_channels)
            for edge_type in metadata[1]
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        # Pehla GNN layer + ReLU activation
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        # Dusra GNN layer (final embeddings ke liye)
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        return x_dict

if __name__ == "__main__":
    print("HeteroGraphSAGE model architecture template ready!")