import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os




class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class HyperGraphConvolution(Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(HyperGraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_channels))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, theta):
        a = torch.mm(x, self.weight)
        output = theta*a
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Machine(nn.Module):
    def __init__(self, n1,n2, dropout,
                 d_1=200, d_2=200, d_3=200,
                 ini_emb_mode='par',
                 embeddings_ae_or_one_hot=None):

        super(Machine, self).__init__()
        if ini_emb_mode == 'par':
            self.d_1 = d_1  # 200
            self.d_2 = d_1  # 200
            self.d_3 = d_1  # 200
            self.ini_embedding_1 = torch.nn.Parameter(torch.Tensor(n1, self.d_1))
            torch.nn.init.xavier_uniform_(self.ini_embedding_1)
            self.ini_embedding_2 = torch.nn.Parameter(torch.Tensor(n2, self.d_1))
            torch.nn.init.xavier_uniform_(self.ini_embedding_2)
        elif (ini_emb_mode == 'ae') or (ini_emb_mode == 'one_hot'):

            self.d_1 = embeddings_ae_or_one_hot.shape[1]  # 8560 for one_hot or 200 for ae, maybe
            self.d_2 = d_2  # 200
            self.d_3 = d_3  # 200
            self.ini_embeddings = embeddings_ae_or_one_hot
        elif ini_emb_mode == 'manual_fea':
            self.d_1 = embeddings_ae_or_one_hot.shape[1]  # 36
            self.d_2 = self.d_1  # 36
            self.d_3 = d_3  # 16
            self.ini_embeddings = embeddings_ae_or_one_hot

        else:
            print('WONG INI_EMB_MODE!')

        self.dropout = dropout

        self.gcn_k = 5
        self.gcn1 = GraphConvolution(self.d_1, self.d_2)  # (200, 100)
        self.gcn2 = GraphConvolution(self.d_2 * self.gcn_k, self.d_3)
        self.hcn1 = HyperGraphConvolution(self.d_3, self.d_3)
        self.hcn2 = HyperGraphConvolution(self.d_3, self.d_3)

    def forward(self, adj1,adj2,theta=1):

        x_list = []
        for k in range(self.gcn_k):
            x = F.relu(self.gcn1(self.ini_embedding_1, adj1))
            # x = F.dropout(x, self.dropout)
            x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.gcn2(x, adj1)
        x = F.relu(self.hcn1(x, theta))
        x = F.dropout(x, self.dropout)
        embedding_1 = self.hcn2(x, theta)
        
        x_list = []
        for k in range(self.gcn_k):
            x = F.relu(self.gcn1(self.ini_embedding_2, adj2))
            # x = F.dropout(x, self.dropout)
            x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.gcn2(x, adj2)
        x = F.relu(self.hcn1(x, theta))
        x = F.dropout(x, self.dropout)
        embedding_2 = self.hcn2(x, theta)
        
        return embedding_1,  embedding_2

    def anchor_loss(self, embedding_1,  embedding_2, seeds):

        
        dots_p = F.softmax(torch.mm(embedding_1, embedding_2.T),dim=1)
        val = dots_p[seeds[0],seeds[1]]
        loss = torch.sum(-torch.log(val + 10e-12))

        return loss
    
    def acc(self, S, y):
       
        Sn = S.detach().numpy()
        row, col = linear_sum_assignment(-Sn)
        pred = torch.tensor(col)
        
        correct = sum(pred[y[0]] == y[1])
        return correct 



def MGCN(G1,G2,seeds):
    n1 = G1.shape[0]
    n2 = G2.shape[0]
    Seeds = torch.zeros([n1,n2])
    Seeds[seeds[0],seeds[1]] = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Machine(n1,n2,
                    dropout=0.001,
                    d_1=20, d_2=0, d_3=0,
                    ini_emb_mode='par').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)

    model.train()

    for epoch in range(100):
        optimizer.zero_grad()

        embedding_1,  embedding_2 = model(G1,G2)
        loss = model.anchor_loss(embedding_1,  embedding_2,seeds)
        loss.backward()
        optimizer.step()
    
    embedding_1,  embedding_2 = model(G1,G2)
    dots_p = F.softmax(torch.mm(embedding_1, embedding_2.T),dim=1)+Seeds*1000000
    results = torch.argmax(dots_p, dim=1)    
    return results    
        
        