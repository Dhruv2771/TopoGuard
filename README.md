# TopoGuard: Zero-Day Fraud Detection

**TopoGuard** is a real-time zero-day fraud detection engine utilizing Temporal Graph Networks (TGN) via PyTorch Geometric and Isolation Forests to detect topological anomalies (e.g., money mule rings) without external API dependencies.

## Setup & Requirements

1. **Install Dependencies:**
   Ensure you have a Python environment setup (e.g., Python 3.10+ or Conda).
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data:**
   Generates a synthetic transaction edge list with injected fraud motifs (circular trading).
   ```bash
   python src/synthetic_data.py
   ```

3. **Train Model:**
   Runs self-supervised link prediction to learn temporal node embeddings, then fits an Isolation Forest to score spatial anomalies.
   ```bash
   python src/train_anomaly.py
   ```

4. **Run Inference:**
   Simulates real-time stream processing of edges to evaluate latency and inductive node reasoning.
   ```bash
   python src/inference.py
   ```
