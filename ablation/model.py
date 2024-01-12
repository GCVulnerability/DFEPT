# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from utils import *
from Layers import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelWithoutLLM(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelWithoutLLM, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.graphEmb = GraphEmbedding(feature_dim_size=args.feature_dim_size, hidden_size=args.hidden_size, dropout=config.hidden_dropout_prob)


        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)
        self.classifier = PredictionClassification(config, args, input_size=self.graphEmb.out_dim)

    def forward(self, input_ids=None, labels=None):
        # build DFG
        adj, x_feature = build_dfg(input_ids.cpu().detach().numpy(), self.w_embeddings, self.tokenizer)
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)
        g_emb = self.graphEmb(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double())
        outputs = self.classifier(g_emb)
        # Apply dropout
        outputs = self.dropout(outputs)

        logits = outputs
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob



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

class ModelwithoutSin(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelwithoutSin, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.graphEmb = GraphEmbeddingWithoutSin(feature_dim_size=args.feature_dim_size, hidden_size=args.hidden_size, dropout=config.hidden_dropout_prob)


        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)
        self.classifier = PredictionClassification(config, args, input_size=self.graphEmb.out_dim + 768)

    def forward(self, input_ids=None, labels=None):
        # build DFG
        adj, x_feature = build_dfg(input_ids.cpu().detach().numpy(), self.w_embeddings, self.tokenizer)
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        # print(adj.shape)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)
        # print(adj_feature.shape)
        g_emb = self.graphEmb(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double())
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
        outputs = self.classifier(torch.cat((vec,g_emb), dim=1))

        # Apply dropout
        outputs = self.dropout(outputs)

        logits = outputs
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
