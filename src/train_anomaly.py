import torch
import pandas as pd
import numpy as np
from tgn_model import TemporalGraphEncoder, EdgePredictor
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from datetime import datetime

# Helper function to convert transaction amount to a tensor feature
def prepare_edge_features(df):
    amounts = torch.tensor(df['amount'].values, dtype=torch.float).view(-1, 1)
    # Simple normalization
    amounts = (amounts - amounts.mean()) / amounts.std()
    return amounts

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    try:
        df = pd.read_csv('synthetic_fraud_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        print("Please run synthetic_data.py first.")
        return

    num_nodes = max(df['src_id'].max(), df['dst_id'].max()) + 1
    
    # 2. Hyperparameters
    NODE_DIM = 16
    EDGE_DIM = 1
    MEMORY_DIM = 32
    EMBED_DIM = 32
    BATCH_SIZE = 200 # Processing 200 continuous edges at a time
    
    # 3. Model Initialization
    tgn = TemporalGraphEncoder(num_nodes, NODE_DIM, EDGE_DIM, MEMORY_DIM, EMBED_DIM).to(device)
    predictor = EdgePredictor(EMBED_DIM).to(device)
    
    optimizer = torch.optim.Adam(
        list(tgn.parameters()) + list(predictor.parameters()), lr=0.001
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    
    edge_attr = prepare_edge_features(df).to(device)
    
    src_nodes = torch.tensor(df['src_id'].values, dtype=torch.long).to(device)
    dst_nodes = torch.tensor(df['dst_id'].values, dtype=torch.long).to(device)
    times = torch.tensor((df['timestamp'] - df['timestamp'].min()).dt.total_seconds().values, dtype=torch.long).to(device)
    
    tgn.memory.reset_state() # Clear all node memories

    # --- Self-Supervised Link Prediction Training ---
    print("Starting TGN training...")
    tgn.train()
    predictor.train()
    
    num_batches = len(df) // BATCH_SIZE
    
    for epoch in range(1): # 1 Epoch is sufficient for continuous series demonstration
        total_loss = 0
        for i in tqdm(range(num_batches)):
            optimizer.zero_grad()
            
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(df))
            
            src_batch = src_nodes[start_idx:end_idx]
            dst_batch = dst_nodes[start_idx:end_idx]
            edge_batch = edge_attr[start_idx:end_idx]
            t_batch = times[start_idx:end_idx]
            
            # For simplicity in this demo, we infer immediately.
            # In a real PyG setup, we construct a 1-hop temporal NeighborLoader.
            # Here we just pass the direct edges.
            
            # Nodes involved in this batch
            n_id = torch.unique(torch.cat([src_batch, dst_batch]))
            
            # Relabel global node IDs to local indices for the sub-graph
            src_local = torch.searchsorted(n_id, src_batch)
            dst_local = torch.searchsorted(n_id, dst_batch)
            edge_index = torch.stack([src_local, dst_local], dim=0)
            
            # Forward GNN Pass
            z = tgn(n_id, edge_index, edge_batch)
            
            # Fetch embeddings for current batch
            z_src = z[src_local]
            z_dst = z[dst_local]
            
            # Positive Link Prediction (These edges exist)
            pos_out = predictor(z_src, z_dst)
            pos_loss = criterion(pos_out, torch.ones_like(pos_out))
            
            # Negative Sampling
            neg_dst_batch = torch.randint(0, num_nodes, (src_batch.size(0),), device=device)
            n_id_neg = torch.unique(torch.cat([src_batch, neg_dst_batch]))
            
            src_local_neg = torch.searchsorted(n_id_neg, src_batch)
            dst_local_neg = torch.searchsorted(n_id_neg, neg_dst_batch)
            edge_index_neg = torch.stack([src_local_neg, dst_local_neg], dim=0)
            
            z_neg = tgn(n_id_neg, edge_index_neg, edge_batch)
            
            z_dst_neg = z_neg[dst_local_neg]
            neg_out = predictor(z_src, z_dst_neg)
            neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
            
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            
            # --- POST-UPDATE MEMORY ---
            # Now that we've used memory t-1 to predict edge t, we update memory to t.
            # Message format requires raw_msg_dim = memory_dim + edge_dim
            # We mock the node state component with zeros or just z_src to match dimensions
            raw_msg = torch.cat([z_src.detach(), edge_batch], dim=-1)
            tgn.memory.update_state(src_batch, dst_batch, t_batch, raw_msg)
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch} Loss: {total_loss / num_batches:.4f}")

    # --- Zero-Day Anomaly Detection (Isolation Forest) ---
    print("\\nExtracting Final Dynamic Node Embeddings...")
    tgn.eval()
    with torch.no_grad():
        final_n_id = torch.arange(num_nodes, device=device)
        # We pass an empty edge list to just extract pure memory states
        final_embeddings = tgn(final_n_id, torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0, EDGE_DIM), device=device))
        
    X = final_embeddings.cpu().numpy()
    
    print("Fitting Isolation Forest...")
    clf = IsolationForest(contamination=0.02, random_state=42, n_estimators=200)
    
    # -1 identifies anomalies, 1 normal
    predictions = clf.fit_predict(X)
    anomaly_scores = clf.decision_function(X) # lower means more anomalous
    
    # Map predictions back to nodes
    results = pd.DataFrame({
        'node_id': range(num_nodes),
        'score': anomaly_scores,
        'is_anomaly': predictions == -1
    })
    
    # Save the model
    torch.save(tgn.state_dict(), 'tgn_model.pt')
    results.to_csv('anomaly_scores.csv', index=False)
    
    print(f"Detected {results['is_anomaly'].sum()} anomalies.")
    print("Scores saved to anomaly_scores.csv")
    
if __name__ == '__main__':
    train()
