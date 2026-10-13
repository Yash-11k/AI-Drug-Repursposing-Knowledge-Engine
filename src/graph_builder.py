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
    
    data = HeteroData()
    
    # Hetionet mein nodes apne prefix ke sath hote hain ya hum unhein categorize kar sakte hain
    # Aam taur par source/target format mein node type identify hoti hai (e.g., Compound, Disease)
    # Let's extract unique node types and map them properly
    
    # Simple heuristic to extract node type from prefix or name if structured, 
    # Otherwise we map them based on unique identifiers.
    # Let's create dictionaries for mapping source and target nodes by their types.
    
    unique_sources = df['source'].unique()
    unique_targets = df['target'].unique()
    all_nodes = pd.unique(pd.concat([df['source'], df['target']]))
    
    print(f"Total unique nodes across network: {len(all_nodes)}")
    
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    # Group by metaedge to create typed relations
    grouped = df.groupby('metaedge')
    
    for metaedge, group in grouped:
        src_indices = [node_to_idx[s] for s in group['source']]
        tgt_indices = [node_to_idx[t] for t in group['target']]
        
        edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
        
        # Using clear typed relations for HeteroData
        # e.g., ('compound', metaedge, 'disease') or generic entity types
        data['entity', metaedge, 'entity'].edge_index = edge_index

    # Add dummy node features (Identity matrix or random features for GNN input)
    num_nodes = len(all_nodes)
    feature_dim = 64  # Standard hidden dimension size
    data['entity'].x = torch.eye(num_nodes) if num_nodes < 50000 else torch.randn((num_nodes, feature_dim))
    
    print("Enhanced HeteroData Graph successfully built!")
    print(data)
    return data

if __name__ == "__main__":
    graph = build_hetero_graph()