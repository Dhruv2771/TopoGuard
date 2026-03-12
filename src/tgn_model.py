import torch
from torch.nn import Linear, GRUCell
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from torch_geometric.nn import TGNMemory, TransformerConv

class TemporalGraphEncoder(torch.nn.Module):
    def __init__(self, num_nodes, node_dim, edge_dim, memory_dim, embed_dim):
        super().__init__()
        
        # 1. TGN Memory Module
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=memory_dim + edge_dim, # We simplify message dim
            memory_dim=memory_dim,
            time_dim=memory_dim,
            message_module=IdentityMessage(memory_dim + edge_dim, memory_dim, memory_dim),
            aggregator_module=LastAggregator()
        )
        
        # Initial node features
        self.node_embedding = torch.nn.Embedding(num_nodes, node_dim)

        # 2. Attention-based Message Passing Layer
        self.conv = TransformerConv(
            in_channels=memory_dim + node_dim, 
            out_channels=embed_dim, 
            heads=2, 
            concat=False, # averages heads
        )

        self.memory_dim = memory_dim

    def forward(self, n_id, edge_index, edge_attr=None):
        """
        n_id: Node indices in the current batch calculation
        edge_index: The local bipartite temporal sub-graph
        edge_attr: Edge features
        """
        # Retrieve node memory state
        # Detach state to prevent backprop through time leakage
        z, last_update = self.memory(n_id)
        z = z.detach()
        
        # Combine static node embedding with dynamic memory
        x = self.node_embedding(n_id)
        x = torch.cat([x, z], dim=-1) # shape: (len(n_id), node_dim + memory_dim)

        # Message Passing on the temporal subgraph using Graph Attention
        node_embeddings = self.conv(x, edge_index)
        
        return node_embeddings

    def update_memory(self, msg_store):
        pass

    
class EdgePredictor(torch.nn.Module):
    """ MLP for link prediction """
    def __init__(self, embed_dim):
        super().__init__()
        self.lin1 = Linear(embed_dim * 2, embed_dim)
        self.lin2 = Linear(embed_dim, 1)

    def forward(self, z_src, z_dst):
        h = torch.cat([z_src, z_dst], dim=-1)
        h = self.lin1(h).relu()
        return self.lin2(h)
