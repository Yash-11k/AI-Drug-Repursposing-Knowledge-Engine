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
    
    # Hetionet mein node types aur relation types nikalne ke liye metaedge parsing
    # Format aam taur par: SourceNodeType__MetaEdgeName__TargetNodeType ya similar hota hai
    # Let's inspect unique metaedges
    print("Sample metaedges:", df['metaedge'].unique()[:5])
    
    data = HeteroData()
    
    # Node types (jaise Compound/Drug, Disease, Gene, Pathway) extract karna
    # Hetionet nodes usually source/target names se identify hote hain (e.g., Cg, Dd etc.)
    # Hum ek simple mapping banate hain:
    
    sources = df['source'].astype(str).values
    targets = df['target'].astype(str).values
    relations = df['metaedge'].astype(str).values
    
    # Unique nodes collect karna
    unique_nodes = pd.unique(pd.concat([df['source'], df['target']]))
    print(f"Total unique nodes across network: {len(unique_nodes)}")
    
    print("Graph builder skeleton ready for tensor conversion!")
    return data

if __name__ == "__main__":
    graph = build_hetero_graph()