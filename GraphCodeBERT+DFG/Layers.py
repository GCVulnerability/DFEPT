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
        self.dense = nn.Linear(input_size, 256)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(256, num_classes)

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


"""Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 图注意力层 """
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


"""Graph Embedding Layer Using GCN"""
class GraphEmbedding(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, dropout, act=nn.functional.relu):
        super(GraphEmbedding, self).__init__()
        self.gnn = GraphConvolution(feature_dim_size, hidden_size, dropout, act=act)
        # self.graphAtt = GraphAttentionLayer(feature_dim_size, hidden_size, dropout=dropout, concat=True)
        self.out_dim = hidden_size
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.posEmb = SinusoidalPositionalEmbedding()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.gnn(x, adj) * mask  # Residual Connection, can use a weighted sum
        # x = self.graphAtt(x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x.double()).double())
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
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


"""Muti Head Attention of Transformer"""
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.projection_dim = embedding_dim // num_heads
        self.query_dense = nn.Linear(embedding_dim, embedding_dim)
        self.key_dense = nn.Linear(embedding_dim, embedding_dim)
        self.value_dense = nn.Linear(embedding_dim, embedding_dim)
        self.combine_heads = nn.Linear(embedding_dim, embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def attention(self, query, key, value):
        score = torch.matmul(query, key.transpose(-2, -1))
        dim_key = torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32))
        scaled_score = score / dim_key

        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, _ = self.attention(query, key, value)
        attention = attention.permute(0, 2, 1, 3)

        concat_attention = attention.contiguous().view(batch_size, -1, self.embedding_dim)
        output = self.combine_heads(concat_attention)
        output = self.dropout_layer(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, intermediate_dim):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = nn.Linear(embedding_dim, intermediate_dim)
        self.dense2 = nn.Linear(intermediate_dim, embedding_dim)

    def forward(self, inputs):
        x = F.relu(self.dense1(inputs))
        x = self.dense2(x)
        return x





"""Graph Muti Head Attention"""
class GraphMutiHeadAttentionLayer(nn.Module):
    def __init__(self, n_feat, n_hid, n_heads, dropout=0.1, alpha=0.2):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GraphMutiHeadAttentionLayer, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_dim = n_hid * n_heads

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        return x