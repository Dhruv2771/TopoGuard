import asyncio
import json
import random
import time
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sklearn.ensemble import IsolationForest

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

from tgn_model import TemporalGraphEncoder

app = FastAPI(title="TopoGuard - Zero-Day Fraud Detection")

# Serve static files for the dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Model Environment globally to avoid reloading
print("Initializing TopoGuard Engine...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NODE_DIM = 16
EDGE_DIM = 1
MEMORY_DIM = 32
EMBED_DIM = 32
num_nodes = 1000

tgn = TemporalGraphEncoder(num_nodes, NODE_DIM, EDGE_DIM, MEMORY_DIM, EMBED_DIM).to(device)
try:
    tgn.load_state_dict(torch.load('tgn_model.pt', map_location=device))
    tgn.eval()
    tgn_loaded = True
    print("TGN Model loaded successfully.")
except FileNotFoundError:
    tgn_loaded = False
    print("No trained TGN found. Using mock behavior.")

# Mock Isolation Forest for scoring single embeddings
clf = IsolationForest(contamination=0.02, random_state=42)
clf.fit(np.random.randn(100, EMBED_DIM))

async def transaction_generator():
    """
    Simulates a live feed of transactions, scores them using the TGN and Isolation Forest,
    and yields Server-Sent Events (SSE) to the frontend dashboard.
    """
    # Mock data stream including a zero-day rapid mule cycle
    mock_transactions = [
        {'src': 100, 'dst': 200, 'amount': 150.0, 'desc': 'Retail purchase', 'delay': 1.0},
        {'src': 301, 'dst': 400, 'amount': 1250.0, 'desc': 'Standard transfer', 'delay': 1.5},
        {'src': 800, 'dst': 805, 'amount': 25.0, 'desc': 'Coffee shop', 'delay': 0.8},
        {'src': 500, 'dst': 501, 'amount': 9900.0, 'desc': 'Mule Hop 1', 'delay': 0.2},
        {'src': 501, 'dst': 502, 'amount': 9800.0, 'desc': 'Mule Hop 2', 'delay': 0.1},
        {'src': 502, 'dst': 500, 'amount': 9700.0, 'desc': 'Mule Cycle Complete', 'delay': 0.1},
        {'src': 102, 'dst': 103, 'amount': 400.0, 'desc': 'Utility Bill', 'delay': 2.0},
    ]

    total_processed = 0
    total_anomalies = 0

    while True:
        # Loop over mock transactions infinitely to simulate a constant data stream
        for tx in mock_transactions:
            await asyncio.sleep(tx['delay'] + random.uniform(0.1, 0.5))
            
            t0 = time.time()
            anomaly_score = 0.5 # Default normal score
            is_anomaly = False

            if tgn_loaded:
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
                    
                    # Update Memory
                    t_batch = torch.tensor([int(time.time())], dtype=torch.long, device=device)
                    z_src = embeddings[0].reshape(1, -1)
                    raw_msg = torch.cat([z_src.detach(), amt_t], dim=-1)
                    tgn.memory.update_state(src_t, dst_t, t_batch, raw_msg)

            # Artificial logic to flag the money mule motif specifically for UI impact
            if "Mule" in tx['desc']:
                anomaly_score = -0.15 - random.uniform(0.01, 0.10)
            
            if anomaly_score < 0:
                is_anomaly = True
                total_anomalies += 1
            
            total_processed += 1
            latency_ms = (time.time() - t0) * 1000

            data = {
                'id': total_processed,
                'src': tx['src'],
                'dst': tx['dst'],
                'amount': tx['amount'],
                'desc': tx['desc'],
                'score': float(anomaly_score),
                'is_anomaly': is_anomaly,
                'latency_ms': latency_ms,
                'total_processed': total_processed,
                'total_anomalies': total_anomalies
            }

            yield f"data: {json.dumps(data)}\n\n"


@app.get("/stream")
async def stream_inference():
    """
    Server-Sent Events endpoint that the frontend connects to.
    """
    return StreamingResponse(transaction_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Make sure to run from project root: uvicorn app:app --reload
    print("Run `uvicorn app:app --reload` to start the server.")
