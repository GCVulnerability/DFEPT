# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from utils import *
from Layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.graphEmb = GraphEmbeddingGGNN(feature_dim_size=args.feature_dim_size, hidden_size=args.hidden_size, dropout=config.hidden_dropout_prob, pool='max')


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


