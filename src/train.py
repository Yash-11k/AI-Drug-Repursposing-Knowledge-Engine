import os
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import numpy as np
from model import HeteroGraphSAGE, LinkPredictor

def train_graph_rx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load cached graph
    processed_path = "data/processed_graph.pt"
    if not os.path.exists(processed_path):
        print("Processed graph not found! Please run graph_builder.py first.")
        return

    print("Loading processed graph...")
    graph = torch.load(processed_path, weights_only=False)
    graph = graph.to(device)

    # 2. Train/Val/Test Edge Split
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        edge_types=[('entity', 'CpD', 'entity')],
        rev_edge_types=[('entity', 'rev_CpD', 'entity')]
    )
    
    train_data, val_data, test_data = transform(graph)
    print("Graph split into Train, Validation, and Test sets successfully!")

    # 3. Initialize Model and Predictor
    model = HeteroGraphSAGE(
        metadata=train_data.metadata(),
        hidden_channels=64,
        out_channels=32,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    predictor = LinkPredictor().to(device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), 
        lr=0.001, 
        weight_decay=1e-5
    )

    os.makedirs("models", exist_ok=True)
    best_val_auc = 0.0

    # 4. Training Function with Gradient Clipping
    def train():
        model.train()
        predictor.train()
        
        optimizer.zero_grad()
        z_dict = model(train_data.x_dict, train_data.edge_index_dict)
        
        pos_edge_index = train_data['entity', 'CpD', 'entity'].edge_index
        pos_pred = predictor(z_dict, pos_edge_index, src_type='entity', tgt_type='entity')
        
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=train_data['entity'].num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        neg_pred = predictor(z_dict, neg_edge_index, src_type='entity', tgt_type='entity')
        
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        target = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)
        loss = F.binary_cross_entropy_with_logits(pred, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(predictor.parameters()), max_norm=1.0)
        optimizer.step()
        return loss.item()

    # 5. Evaluation Function
    @torch.no_grad()
    def evaluate(data):
        model.eval()
        predictor.eval()
        z_dict = model(train_data.x_dict, train_data.edge_index_dict)
        
        pos_edge_index = data['entity', 'CpD', 'entity'].edge_index
        pos_pred = predictor(z_dict, pos_edge_index, src_type='entity', tgt_type='entity').sigmoid()
        
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data['entity'].num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        neg_pred = predictor(z_dict, neg_edge_index, src_type='entity', tgt_type='entity').sigmoid()
        
        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
        target = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0).cpu().numpy()
        
        auc = roc_auc_score(target, pred)
        ap = average_precision_score(target, pred)
        return auc, ap

    print("Starting Training and Validation Loop for 30 Epochs...")
    for epoch in range(1, 31):
        loss = train()
        val_auc, val_ap = evaluate(val_data)
        print(f"Epoch: {epoch:02d} | Loss: {loss:.4f} | Val ROC-AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")
        
        # Save best checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'val_auc': val_auc
            }, "models/best_gnn.pt")

    print(f"Training completed! Best Validation ROC-AUC: {best_val_auc:.4f}. Checkpoint saved.")

    # 6. Train XGBoost Classifier on GNN Embeddings (Synopsis Requirement)
    print("\nTraining XGBoost Link Scorer on GNN Embeddings...")
    model.eval()
    with torch.no_grad():
        z_dict = model(train_data.x_dict, train_data.edge_index_dict)
        entity_emb = z_dict['entity'].cpu().numpy()

    # Prepare training samples for XGBoost using edge indices and node embeddings
    pos_edges = train_data['entity', 'CpD', 'entity'].edge_index.cpu().numpy()
    neg_edges = negative_sampling(
        train_data['entity', 'CpD', 'entity'].edge_index,
        num_nodes=train_data['entity'].num_nodes,
        num_neg_samples=pos_edges.shape[1]
    ).cpu().numpy()

    # Feature engineering: Element-wise product of source and target node embeddings
    def get_features(edges, embeddings):
        src_emb = embeddings[edges[0]]
        tgt_emb = embeddings[edges[1]]
        return src_emb * tgt_emb  # Hadamard product

    X_pos = get_features(pos_edges, entity_emb)
    X_neg = get_features(neg_edges, entity_emb)
    X_train = np.vstack([X_pos, X_neg])
    y_train = np.hstack([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])

    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Evaluate XGBoost on Test Set
    test_pos_edges = test_data['entity', 'CpD', 'entity'].edge_index.cpu().numpy()
    test_neg_edges = negative_sampling(
        test_data['entity', 'CpD', 'entity'].edge_index,
        num_nodes=test_data['entity'].num_nodes,
        num_neg_samples=test_pos_edges.shape[1]
    ).cpu().numpy()

    X_test_pos = get_features(test_pos_edges, entity_emb)
    X_test_neg = get_features(test_neg_edges, entity_emb)
    X_test = np.vstack([X_test_pos, X_test_neg])
    y_test = np.hstack([np.ones(X_test_pos.shape[0]), np.zeros(X_test_neg.shape[0])])

    test_preds = xgb_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    test_ap = average_precision_score(y_test, test_preds)

    print(f"XGBoost Final Test ROC-AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")
    print("Module 3 completed successfully!")

if __name__ == "__main__":
    train_graph_rx()