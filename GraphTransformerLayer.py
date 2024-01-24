import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformerLayer(nn.Module):
    def __init__(self, node_dim, num_heads, dropout_rate=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.node_dim = node_dim
        self.dropout = nn.Dropout(p=dropout_rate)
        self.size_per_head = self.node_dim // self.num_heads

        self.query_transform = nn.Linear(node_dim, node_dim)
        self.key_transform = nn.Linear(node_dim, node_dim)
        self.value_transform = nn.Linear(node_dim, node_dim)

        self.fc_out = nn.Linear(node_dim, node_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        scale = query.size(-1) ** 0.5
        softmax_attention = F.softmax(matmul_qk / scale, dim=-1)
        output = torch.matmul(softmax_attention, value)
        return output

    def forward(self, node_features, adjacency_matrix):
        batch_size = node_features.size(0)

        # Apply linear transformation and split into heads
        query = self.query_transform(node_features).view(batch_size, -1, self.num_heads, self.size_per_head)
        key = self.key_transform(node_features).view(batch_size, -1, self.num_heads, self.size_per_head)
        value = self.value_transform(node_features).view(batch_size, -1, self.num_heads, self.size_per_head)

        # Transpose to get dimensions batch_size * num_heads * seq_length * size_per_head
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply scaled dot product attention
        attention = self.scaled_dot_product_attention(query, key, value)

        # Concatenate heads and put through final linear layer
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.node_dim)
        attention_output = self.fc_out(attention)

        return attention_output
