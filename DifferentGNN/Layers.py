import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

"""Head for sentence-level classification tasks."""
class PredictionClassification(nn.Module):
    def __init__(self, config, args, input_size=None, num_classes=2):
        super().__init__()
        # self.dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        if input_size is None:
            input_size = args.hidden_size
        self.dense = nn.Linear(input_size, args.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(args.hidden_size, num_classes)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x.double(), self.weight.double())
        output = torch.matmul(adj.double(), support.double())
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


"""Graph Embedding Layer Using GCN"""
class GraphEmbedding(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, dropout, pool='uni',act=nn.functional.relu):
        super(GraphEmbedding, self).__init__()
        self.gnn = GraphConvolution(feature_dim_size, hidden_size, dropout, act=act)
        self.out_dim = hidden_size
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.posEmb = SinusoidalPositionalEmbedding()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act
        self.pool = pool

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.gnn(x, adj) * mask  # Residual Connection, can use a weighted sum
        soft_att = torch.sigmoid(self.soft_att(x.double()).double())
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        if self.pool == 'sum':
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.pool == 'mean':
            graph_embeddings = torch.mean(x, 1)
        elif self.pool == 'max':
            graph_embeddings, _ = torch.max(x, 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)
        graph_embeddings = graph_embeddings.unsqueeze(1)
        graph_embeddings = self.posEmb(graph_embeddings)
        graph_embeddings = graph_embeddings.squeeze(1)
        return graph_embeddings


"""Graph Embedding Layer Using GGNN"""
class GraphEmbeddingGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, dropout, pool='uni', act=nn.functional.relu):
        super(GraphEmbeddingGGNN, self).__init__()
        self.out_dim = hidden_size
        self.posEmb = SinusoidalPositionalEmbedding()
        self.pool = pool



        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).double()
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size).double()
        self.z1 = nn.Linear(hidden_size, hidden_size).double()
        self.r0 = nn.Linear(hidden_size, hidden_size).double()
        self.r1 = nn.Linear(hidden_size, hidden_size).double()
        self.h0 = nn.Linear(hidden_size, hidden_size).double()
        self.h1 = nn.Linear(hidden_size, hidden_size).double()
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.double())
        z1 = self.z1(x.double())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.double()) + self.r1(x.double()))
        # update embeddings
        h = self.act(self.h0(a.double()) + self.h1(r.double() * x.double()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.double())
        x = x * mask
        x = self.gatedGNN(x.double(), adj.double()) * mask.double()
        soft_att = torch.sigmoid(self.soft_att(x.double()).double())
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        if self.pool == 'sum':
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.pool == 'mean':
            graph_embeddings = torch.mean(x, 1)
        elif self.pool == 'max':
            graph_embeddings, _ = torch.max(x, 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)
        graph_embeddings = graph_embeddings.unsqueeze(1)
        graph_embeddings = self.posEmb(graph_embeddings)
        graph_embeddings = graph_embeddings.squeeze(1)
        return graph_embeddings


"""Sinusoidal Positional Embedding Layers"""
class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, x, position_ids=None):
        seq_len = x.size(1)
        output_dim = x.size(2)  # Dynamically get the output_dim

        if self.custom_position_ids:
            if position_ids is None:
                raise ValueError("custom_position_ids is set to True, but position_ids were not provided.")
        else:
            position_ids = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(0)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float32, device=x.device)
        indices = torch.pow(10000.0, -2 * indices / output_dim)

        pos_embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        pos_embeddings = torch.cat([
            torch.sin(pos_embeddings).unsqueeze(-1),
            torch.cos(pos_embeddings).unsqueeze(-1)
        ], dim=-1)
        pos_embeddings = pos_embeddings.view(-1, seq_len, output_dim)

        if self.merge_mode == 'add':
            return x + pos_embeddings
        elif self.merge_mode == 'mul':
            return x * pos_embeddings
        else:
            if not self.custom_position_ids:
                pos_embeddings = pos_embeddings.expand_as(x)
            return torch.cat([x, pos_embeddings], dim=-1)


