import pandas as pd
import numpy as np
import networkx as nx
import random
from datetime import datetime, timedelta

def generate_synthetic_transactions(num_accounts=1000, num_transactions=10000, anomaly_fraction=0.01):
    """
    Generates synthetic transactional data with injected graph motifs.
    """
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating {num_transactions} transactions...")
    
    accounts = [f"ACC_{i}" for i in range(num_accounts)]
    
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    
    # Generate Normal Transactions
    transactions = []
    
    for _ in range(num_transactions):
        src = random.choice(accounts)
        dst = random.choice(accounts)
        while src == dst:
            dst = random.choice(accounts)
            
        time_offset = timedelta(minutes=random.randint(1, 60*24*30)) # 1 month spread
        timestamp = start_time + time_offset
        
        amount = np.random.lognormal(mean=4.0, sigma=1.0) # Typical log-normal amounts
        
        transactions.append({
            'src': src,
            'dst': dst,
            'timestamp': timestamp,
            'amount': amount,
            'is_fraud': 0
        })
        
    df = pd.DataFrame(transactions)
    
    # Inject Circular Trading Anomalies
    num_anomalies = int(num_transactions * anomaly_fraction)
    rings_to_create = max(1, num_anomalies // 5) # Assuming rings of 4-5 hops
    
    print(f"Injecting {rings_to_create} structural anomalies...")
    
    for _ in range(rings_to_create):
        # Pick 4 random accounts to act as mules
        ring_accounts = random.sample(accounts, 4)
        
        # Pick a random starting point in time
        ring_start = start_time + timedelta(minutes=random.randint(1, 60*24*28))
        
        mule_amount = np.random.uniform(5000, 15000) # Large, suspicious amounts
        
        # Create a rapid circular flow: A -> B -> C -> D -> A
        for i in range(len(ring_accounts)):
            src = ring_accounts[i]
            dst = ring_accounts[(i + 1) % len(ring_accounts)] # Wraps around to 0
            
            # Transfers happen rapidly
            tx_time = ring_start + timedelta(minutes=random.randint(1, 5) * i)
            
            # Amount decreases slightly to simulate fees or intermediate cuts
            current_amount = mule_amount * (0.98 ** i) 
            
            # Use append to DataFrame (inefficient but fine for simulation)
            new_tx = pd.DataFrame([{
                'src': src,
                'dst': dst,
                'timestamp': tx_time,
                'amount': current_amount,
                'is_fraud': 1 # Ground truth labels for validation
            }])
            df = pd.concat([df, new_tx], ignore_index=True)
            
    # Sort chronologically (Crucial for TGN)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Create simple node indices (required for PyG)
    unique_nodes = pd.concat([df['src'], df['dst']]).unique()
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}
    
    df['src_id'] = df['src'].map(node_mapping)
    df['dst_id'] = df['dst'].map(node_mapping)
    
    print(f"Dataset generated. Shape: {df.shape}. Fraud ratio: {df['is_fraud'].mean():.4f}")
    
    # Save to CSV
    df.to_csv('synthetic_fraud_data.csv', index=False)
    
    return df, node_mapping

if __name__ == '__main__':
    generate_synthetic_transactions()
    print("Saved to synthetic_fraud_data.csv")
