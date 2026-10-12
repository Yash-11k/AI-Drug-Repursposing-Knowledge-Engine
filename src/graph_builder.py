import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData

def build_hetero_graph():
    file_path = "data/hetionet-v1.0-edges.sif.gz"
    
    if not os.path.exists(file_path):
        print("Error: Dataset file nahi mili!")
        return None
    
    print("Loading Hetionet edges with Pandas...")
    df = pd.read_csv(file_path, compression='gzip', sep='\t')
    
    print(f"Total edges loaded: {len(df)}")
    
    # Hetionet nodes ke identifiers se unke types nikalte hain (e.g., Compound, Disease)
    # Aam taur par Hetionet nodes format: type::name hota hai ya hum simple prefix/mapping use karte hain
    # Let's extract unique source and target node types based on their structure or prefixes if available
    # For Hetionet, source/target nodes contain identifiers. Let's map unique node indices.
    
    unique_sources = df['source'].unique()
    unique_targets = df['target'].unique()
    all_nodes = pd.unique(pd.concat([df['source'], df['target']]))
    
    print(f"Total unique nodes: {len(all_nodes)}")
    
    # Node to index mapping dictionary banana
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    data = HeteroData()
    
    # Filhal hum ek generalized node type 'node' aur metaedge relations ke sath tensors banate hain
    # Har metaedge ke liye edge_index map karna
    print("Mapping edges to PyTorch tensors...")
    
    # Sample ke taur par hum edge types (metaedges) ko group karte hain
    grouped = df.groupby('metaedge')
    
    for metaedge, group in grouped:
        src_indices = [node_to_idx[s] for s in group['source']]
        tgt_indices = [node_to_idx[t] for t in group['target']]
        
        # PyTorch tensor format (2, num_edges)
        edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
        
        # HeteroData mein relation store karna (e.g., ('node', metaedge, 'node'))
        data['node', metaedge, 'node'].edge_index = edge_index

    print("Hetero graph successfully built with all metaedge relations!")
    print(data)
    return data

if __name__ == "__main__":
    graph = build_hetero_graph()