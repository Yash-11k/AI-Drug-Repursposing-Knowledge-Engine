import os
import pandas as pd
import urllib.request

def download_hetionet():
    """Hetionet dataset ko download karta hai agar local par na ho."""
    os.makedirs("data", exist_ok=True)
    
    # Hetionet edge (hetnet) JSON/TSV link ya public URL
    # Hum standard hetionet edge list ka URL use karenge
    edge_url = "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
    file_path = "data/hetionet-v1.0-edges.sif.gz"
    
    if not os.path.exists(file_path):
        print("Downloading Hetionet dataset...")
        urllib.request.urlretrieve(edge_url, file_path)
        print("Download complete!")
    else:
        print("Dataset already exists locally.")
        
    # Pandas se read karna
    print("Loading edges into Pandas DataFrame...")
    df = pd.read_csv(file_path, compression='gzip', sep='\t')
    return df

if __name__ == "__main__":
    df = download_hetionet()
    print("Edge data preview:")
    print(df.head())