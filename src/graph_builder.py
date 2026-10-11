import os
import pandas as pd
import torch
from torch_geometric.data import HeteroData

def build_hetero_graph():
    file_path = "data/hetionet-v1.0-edges.sif.gz"
    
    if not os.path.exists(file_path):
        print("Error: Dataset file nahi mili! Pehle data_loader.py run karein.")
        return None
    
    print("Loading Hetionet edges with Pandas...")
    df = pd.read_csv(file_path, compression='gzip', sep='\t')
    
    print(f"Total edges loaded: {len(df)}")
    print("Columns:", df.columns.tolist())
    
    # HeteroData object initialize karna
    data = HeteroData()
    
    # Hetionet mein aam taur par columns: source, metaedge, target hote hain
    # Hum yahan check karke nodes aur edges map karenge
    if 'source' in df.columns and 'target' in df.columns and 'metaedge' in df.columns:
        print("Dataset columns are valid for Graph Construction.")
        # Future steps: Unique node mapping aur edges torch tensors mein convert karna
    
    return data

if __name__ == "__main__":
    graph = build_hetero_graph()
    print("Graph object status:", graph)