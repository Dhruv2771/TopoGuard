import torch
import pandas as pd
import numpy as np
from tgn_model import TemporalGraphEncoder
import joblib
from sklearn.ensemble import IsolationForest
import pickle
import time

def simulate_streaming_inference():
    print("Initializing inference...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Hyperparameters (Must match training)
    NODE_DIM = 16
    EDGE_DIM = 1
    MEMORY_DIM = 32
    EMBED_DIM = 32
    
    # Initialize node limit
    num_nodes = 1000
    
    # 2. Load Model 
    tgn = TemporalGraphEncoder(num_nodes, NODE_DIM, EDGE_DIM, MEMORY_DIM, EMBED_DIM).to(device)
    try:
        tgn.load_state_dict(torch.load('tgn_model.pt', map_location=device))
        tgn.eval()
        print("TGN Model loaded successfully.")
    except FileNotFoundError:
        print("No trained TGN found. Run train_anomaly.py first.")
        return

    # In a real scenario we'd use joblib or pickle to load the trained sklearn model.
    # Here, for the simulation without requiring an actual .pkl file generated, 
    # we'll mock an IsolationForest that acts on the inference outputs.
    # (Typically you'd do: clf = joblib.load('iso_forest.pkl'))
    
    clf = IsolationForest(contamination=0.02, random_state=42)
    # Using random fit just to enable the decision_function in this structural mock.
    clf.fit(np.random.randn(100, EMBED_DIM)) 
    print("Isolation Forest Scorer loaded.")
    
    print("\\n--- Commencing Streaming Inference (Kafka/Redis Simulation) ---")
    
    # Test stream
    test_stream = [
        {'src': 100, 'dst': 200, 'amount': 150.0, 'desc': 'Normal retail purchase'},
        {'src': 301, 'dst': 400, 'amount': 12500.0, 'desc': 'High value transfer'},
        {'src': 500, 'dst': 501, 'amount': 9900.0, 'desc': 'Hop 1'},
        {'src': 501, 'dst': 502, 'amount': 9800.0, 'desc': 'Hop 2'},
        {'src': 502, 'dst': 500, 'amount': 9700.0, 'desc': 'Hop 3'},
    ]
    
    for idx, tx in enumerate(test_stream):
        t0 = time.time()
        
        src_t = torch.tensor([tx['src']], dtype=torch.long, device=device)
        dst_t = torch.tensor([tx['dst']], dtype=torch.long, device=device)
        amt_t = torch.tensor([[tx['amount']]], dtype=torch.float, device=device)
        
        with torch.no_grad():
            n_id = torch.cat([src_t, dst_t])
            # Relabel to local graph indices for TransformerConv
            edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=device)
            
            embeddings = tgn(n_id, edge_index)
            
            # Score transaction
            src_emb = embeddings[0].cpu().numpy().reshape(1, -1)
            
            anomaly_score = clf.decision_function(src_emb)[0]
            
            # Post-Inference: Update the node memories
            t_batch = torch.tensor([int(time.time())], dtype=torch.long, device=device)
            # Pad raw message to match raw_msg_dim
            z_src = embeddings[0].reshape(1, -1)
            raw_msg = torch.cat([z_src.detach(), amt_t], dim=-1)
            tgn.memory.update_state(src_t, dst_t, t_batch, raw_msg)
            
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000
        
        print(f"Tx {idx+1}: {tx['desc']}")
        print(f"  -> Src: {tx['src']} | Dst: {tx['dst']} | Amt: ${tx['amount']}")
        print(f"  -> Anomaly Score: {anomaly_score:.4f}")
        print(f"  -> Latency: {latency_ms:.2f} ms")
        print("-" * 50)

if __name__ == '__main__':
    simulate_streaming_inference()
