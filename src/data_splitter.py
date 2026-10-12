import torch
import random

def create_train_val_test_splits(data, target_edge_type=('node', 'CpD', 'node'), val_ratio=0.1, test_ratio=0.1):
    """
    Drug-disease ya target relation ke edges ko train, validation aur test mein split karta hai.
    Sath hi negative sampling ke liye helper structure banata hai.
    """
    if target_edge_type not in data.edge_types:
        print(f"Warning: Edge type {target_edge_type} graph mein nahi mila. Available: {data.edge_types}")
        return None

    edge_index = data[target_edge_type].edge_index
    num_edges = edge_index.size(1)
    
    print(f"Total edges for {target_edge_type}: {num_edges}")
    
    # Indices shuffle karna
    perm = torch.randperm(num_edges)
    edge_index = edge_index[:, perm]
    
    # Split sizes calculate karna
    test_size = int(num_edges * test_ratio)
    val_size = int(num_edges * val_ratio)
    train_size = num_edges - (test_size + val_size)
    
    # Splits divide karna
    train_edge_index = edge_index[:, :train_size]
    val_edge_index = edge_index[:, train_size:train_size + val_size]
    test_edge_index = edge_index[:, train_size + val_size:]
    
    print(f"Split completed successfully!")
    print(f" - Train edges: {train_edge_index.size(1)}")
    print(f" - Val edges:   {val_edge_index.size(1)}")
    print(f" - Test edges:  {test_edge_index.size(1)}")
    
    return {
        'train': train_edge_index,
        'val': val_edge_index,
        'test': test_edge_index
    }

if __name__ == "__main__":
    # Test karne ke liye graph builder call kar sakte hain ya mock data check kar sakte hain
    print("Data splitter script ready!")