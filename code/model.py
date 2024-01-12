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
        self.graphEmb = GraphEmbedding(feature_dim_size=args.feature_dim_size, hidden_size=args.hidden_size, dropout=config.hidden_dropout_prob)


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



        # outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
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

class DefectModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def forward(self, source_ids=None, labels=None):

        if self.args.model_type == 'codet5':
            vec = self.get_t5_vec(source_ids)
        elif self.args.model_type == 'bart':
            vec = self.get_bart_vec(source_ids)
        elif self.args.model_type == 'roberta':
            vec = self.get_roberta_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

class ModelT(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelT, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args


        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)
        self.classifier = PredictionClassification(config, args, input_size=768)

    def forward(self, input_ids=None, labels=None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
        outputs = self.classifier(vec)



        # outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
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

