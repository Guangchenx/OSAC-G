import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphTransformerLayer import GraphTransformerLayer

class GraphTransformerActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_transformer_layers, num_heads):
        super(GraphTransformerActorNetwork, self).__init__()

        # Initial embedding layer
        self.embedding = nn.Linear(state_dim, hidden_size)

        # Stack of Graph Transformer Layers
        self.graph_transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_size, num_heads) for _ in range(num_transformer_layers)
        ])

        # Output layers for the mean and standard deviation of the action distribution
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, state, adjacency_matrix):
        x = F.relu(self.embedding(state))

        # Pass through each Graph Transformer Layer
        for layer in self.graph_transformer_layers:
            x = layer(x, adjacency_matrix)

        # Compute mean and standard deviation of action distribution
        mean = self.mean_layer(x)
        std = self.std_layer(x)
        std = torch.clamp(std, min=1e-6, max=1e+6)  # Ensure std is positive and finite

        return mean, std
