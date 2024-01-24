import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(CriticNetwork, self).__init__()

        self.embedding_layer = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        # Embedding layer
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.embedding_layer(x))

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value

