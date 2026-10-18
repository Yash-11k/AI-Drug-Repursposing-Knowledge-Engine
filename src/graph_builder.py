import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData

def build_hetero_graph():
    processed_path = "data/processed_graph.pt"
    
    # Agar pehle se processed graph saved hai, toh direct load kar lo (Fast!)
    if os.path.exists(processed_path):
        print("Loading pre-built HeteroData graph from cache...")
        return torch.load(processed_path)
    
    file_path = "data/hetionet-v1.0-edges.sif.gz"
    if not os.path.exists(file_path):
        print("Error: Dataset file nahi mili!")
        return None
    
    print("Loading Hetionet edges with Pandas...")
    df = pd.read_csv(file_path, compression='gzip', sep='\t')
    print(f"Total edges loaded: {len(df)}")
    
    data = HeteroData()
    all_nodes = pd.unique(pd.concat([df['source'], df['target']]))
    num_nodes = len(all_nodes)
    print(f"Total unique nodes across network: {num_nodes}")
    
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    grouped = df.groupby('metaedge')
    for metaedge, group in grouped:
        src_indices = [node_to_idx[s] for s in group['source']]
        tgt_indices = [node_to_idx[t] for t in group['target']]
        
        edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
        data['entity', metaedge, 'entity'].edge_index = edge_index

    feature_dim = 64
    data['entity'].x = torch.randn((num_nodes, feature_dim))
    
    # Future-proof caching: Graph ko disk par save karna
    os.makedirs("data", exist_ok=True)
    torch.save(data, processed_path)
    print(f"Graph successfully built and cached at {processed_path}!")
    
    return data

if __name__ == "__main__":
    graph = build_hetero_graph()
    print(graph)