import torch
from torch.nn import Sequential as Seq, Linear as Lin
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.inits import reset
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import time
#torch.multiprocessing.set_start_method("spawn")
import torch.multiprocessing as mp

try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None


def masked_softmax(src):
    srcmax1 = src - torch.max(src,dim=1,keepdim = True)[0]
    out1 = torch.softmax(srcmax1,dim = 1)
    
    srcmax2 = src - torch.max(src,dim=0,keepdim = True)[0]
    out2 = torch.softmax(srcmax2,dim = 0)
    
    return (out1+out2)/2

def Agg(G1, G2, Si,queue):
    queue.put(torch.mm(G1,torch.mm(Si,G2)).reshape(-1,1))
    

class SeedGNN(torch.nn.Module):
    
    def __init__(self, num_layers, hid):
        super(SeedGNN, self).__init__()
        self.hid = hid
        self.num_layers = num_layers
        
        self.mlp = torch.nn.ModuleList([Seq(
                Lin(1, hid-1),
            )])
        self.readout = torch.nn.ModuleList([Seq(
                Lin(1, 1),
            )])
        # self.alpha = torch.nn.Parameter(torch.zeros(num_layers))
        
        for i in range(1,num_layers):
            self.mlp.append(Seq(
                Lin(hid, hid-1),
            ))
            self.readout.append(Seq(
                Lin(hid, 1),
            ))

    def reset_parameters(self):
        for i in range(self.num_layers):
            reset(self.mlp[i])
            reset(self.readout[i])

    def forward(self, G1, G2, seeds, noisy = False):
        
        Y_total = []
        n1 = G1.shape[0]
        n2 = G2.shape[0]
        
        Seeds = torch.zeros([n1,n2])
        Seeds[seeds[0],seeds[1]] = 1
        
        # S = -torch.ones([n1,n2])/n1
        # S[seeds[0],seeds[1]] = 1
        S = Seeds.unsqueeze(-1)
        
        for layeri in range(self.num_layers):
            
            H = torch.einsum("abh,bc->ach",torch.einsum("ij,jkh->ikh",G1,S),G2)
            if layeri < self.num_layers-1:
                X = self.mlp[layeri](H)/1000
                
            Match = self.readout[layeri](H).squeeze(-1)
            Matchnorm = masked_softmax(Match)
            Matchnorm[seeds[0],:]=0
            Matchnorm[:,seeds[1]]=0
            Matchnorm[seeds[0],seeds[1]]=1
            Y_total.append(Matchnorm)
            
            Matchn = Matchnorm.detach().numpy()
            row,col = linear_sum_assignment(-Matchn)
            NewSeeds = torch.zeros([n1,n2])
            NewSeeds[row,col] = 10
            
            Z = (Matchnorm*NewSeeds).unsqueeze(-1)
    
            S = torch.cat([X,Z],dim = 2)
        
       
        return Y_total[-1], Y_total
        
    def loss(self, S, y):
        
        nll = 0
        EPS = 1e-12
        k = 1
        for Si in S:
            val = Si[y[0], y[1]]
            nll += torch.sum(-torch.log(val + EPS))
            
        return nll

    def acc(self, S, y):
       
        Sn = S.detach().numpy()
        row, col = linear_sum_assignment(-Sn)
        pred = torch.tensor(col)
        
        correct = sum(pred[y[0]] == y[1])
        return correct 

class SeedGNN_hun(torch.nn.Module):
    
    def __init__(self, num_layers, hid):
        super(SeedGNN_hun, self).__init__()
        self.hid = hid
        self.num_layers = num_layers
        
        self.mlp = torch.nn.ModuleList([Seq(
                Lin(1, hid-1),
            )])
        self.readout = torch.nn.ModuleList([Seq(
                Lin(1, 1),
            )])
        # self.alpha = torch.nn.Parameter(torch.zeros(num_layers))
        
        for i in range(1,num_layers):
            self.mlp.append(Seq(
                Lin(hid, hid-1),
            ))
            self.readout.append(Seq(
                Lin(hid, 1),
            ))

    def reset_parameters(self):
        for i in range(self.num_layers):
            reset(self.mlp[i])
            reset(self.readout[i])

    def forward(self, G1, G2, seeds, noisy = False):
        
        Y_total = []
        n1 = G1.shape[0]
        n2 = G2.shape[0]
        
        Seeds = torch.zeros([n1,n2])
        Seeds[seeds[0],seeds[1]] = 1
        
        # S = -torch.ones([n1,n2])/n1
        # S[seeds[0],seeds[1]] = 1
        S = Seeds
        
        for layeri in range(self.num_layers):
            H = []
            if layeri == 0:
                H = [torch.sparse.mm(G2,torch.sparse.mm(G1,S).T).T.reshape(-1,1)]
            else:
                for hidi in range(self.hid):
                    Si = S[:,hidi].reshape(n1,n2)
                    H.append(torch.sparse.mm(G2,torch.sparse.mm(G1,Si).T).T.reshape(-1,1))
                
                
            H = torch.cat(H, dim = 1)
            if layeri < self.num_layers-1:
                X = self.mlp[layeri](H)/1000
            match = self.readout[layeri](H)
            Match = match.reshape(n1,n2)
            Match = Match
            Matchnorm = masked_softmax(Match)*10
            Matchnorm[seeds[0],:]=0
            Matchnorm[:,seeds[1]]=0
            Matchnorm[seeds[0],seeds[1]]=10
            Y_total.append(Matchnorm)
            
            Matchn = Matchnorm.detach().numpy()
            
            row,col = linear_sum_assignment(-Matchn)
            NewSeeds = torch.zeros([n1,n2])
            NewSeeds[row,col] = Matchnorm[row,col]
            
            Z = (NewSeeds).view(-1,1)
    
            S = torch.cat([X,Z],dim=1)
        
       
        return Y_total[-1], Y_total
        
    def loss(self, S, y):
        
        nll = 0
        EPS = 1e-12
        k = 1
        for Si in S:
            val = Si[y[0], y[1]]
            nll += torch.sum(-torch.log(val + EPS))
            
        return nll

    def acc(self, S, y):
       
        Sn = S.detach().numpy()
        row, col = linear_sum_assignment(-Sn)
        pred = torch.tensor(col)
        
        correct = sum(pred[y[0]] == y[1])
        return correct 
    
class SeedGNN_per(torch.nn.Module):
    
    def __init__(self, num_layers, hid):
        super(SeedGNN_per, self).__init__()
        self.hid = hid
        self.num_layers = num_layers
        
        self.mlp = torch.nn.ModuleList([Seq(
                Lin(1, hid-1),
            )])
        self.readout = torch.nn.ModuleList([Seq(
                Lin(1, 1),
            )])
        # self.alpha = torch.nn.Parameter(torch.zeros(num_layers))
        
        for i in range(1,num_layers):
            self.mlp.append(Seq(
                Lin(hid, hid-1),
            ))
            self.readout.append(Seq(
                Lin(hid, 1),
            ))

    def reset_parameters(self):
        for i in range(self.num_layers):
            reset(self.mlp[i])
            reset(self.readout[i])

    def forward(self, G1, G2, seeds, noisy = False):
        
        Y_total = []
        n1 = G1.shape[0]
        n2 = G2.shape[0]
        
        Seeds = torch.zeros([n1,n2])
        Seeds[seeds[0],seeds[1]] = 1
        
        # S = -torch.ones([n1,n2])/n1
        # S[seeds[0],seeds[1]] = 1
        S = Seeds
        
        for layeri in range(self.num_layers):
            H = []
            if layeri == 0:
                H = [torch.sparse.mm(G2,torch.sparse.mm(G1,S).T).T.reshape(-1,1)]
            else:
                for hidi in range(self.hid):
                    Si = S[:,hidi].reshape(n1,n2)
                    H.append(torch.sparse.mm(G2,torch.sparse.mm(G1,Si).T).T.reshape(-1,1))
                
                
            H = torch.cat(H, dim = 1)
            if layeri < self.num_layers-1:
                X = self.mlp[layeri](H)/1000
            match = self.readout[layeri](H)
            Match = match.reshape(n1,n2)
            Match = Match+Seeds
            Matchnorm = masked_softmax(Match)*10
            Matchnorm[seeds[0],:]=0
            Matchnorm[:,seeds[1]]=0
            Matchnorm[seeds[0],seeds[1]]=10
            Y_total.append(Matchnorm)
            
            Z = (Matchnorm).view(-1,1)
    
            S = torch.cat([X,Z],dim=1)
        
       
        return Y_total[-1], Y_total
        
    def loss(self, S, y):
        
        nll = 0
        EPS = 1e-12
        k = 1
        for Si in S:
            val = Si[y[0], y[1]]
            nll += torch.sum(-torch.log(val + EPS))
            
        return nll

    def acc(self, S, y):
       
        Sn = S.detach().numpy()
        row, col = linear_sum_assignment(-Sn)
        pred = torch.tensor(col)
        
        correct = sum(pred[y[0]] == y[1])
        return correct 
    
    
class SeedGNN_van(torch.nn.Module):
    
    def __init__(self, num_layers, hid):
        super(SeedGNN_van, self).__init__()
        self.hid = hid
        self.num_layers = num_layers
        
        self.mlp = torch.nn.ModuleList([Seq(
                Lin(1, hid-1),
            )])
        self.readout = torch.nn.ModuleList([Seq(
                Lin(1, 1),
            )])
        # self.alpha = torch.nn.Parameter(torch.zeros(num_layers))
        
        for i in range(1,num_layers):
            self.mlp.append(Seq(
                Lin(hid, hid-1),
            ))
            self.readout.append(Seq(
                Lin(hid, 1),
            ))

    def reset_parameters(self):
        for i in range(self.num_layers):
            reset(self.mlp[i])
            reset(self.readout[i])

    def forward(self, G1, G2, seeds, noisy = False):
        
        Y_total = []
        n1 = G1.shape[0]
        n2 = G2.shape[0]
        
        Seeds = torch.zeros([n1,n2])
        Seeds[seeds[0],seeds[1]] = 1
        
        # S = -torch.ones([n1,n2])/n1
        # S[seeds[0],seeds[1]] = 1
        S = Seeds
        
        for layeri in range(self.num_layers):
            H = []
            if layeri == 0:
                H = [torch.sparse.mm(G2,torch.sparse.mm(G1,S).T).T.reshape(-1,1)]
            else:
                for hidi in range(self.hid):
                    Si = S[:,hidi].reshape(n1,n2)
                    H.append(torch.sparse.mm(G2,torch.sparse.mm(G1,Si).T).T.reshape(-1,1))
                
                
            H = torch.cat(H, dim = 1)
            if layeri < self.num_layers-1:
                X = self.mlp[layeri](H)/1000
            match = self.readout[layeri](H)
            Match = match.reshape(n1,n2)
            Match = Match
            Matchnorm = masked_softmax(Match)
            Matchnorm[seeds[0],:]=0
            Matchnorm[:,seeds[1]]=0
            Matchnorm[seeds[0],seeds[1]]=1
            Y_total.append(Matchnorm)
            
            Matchn = Matchnorm.detach().numpy()
            row,col = linear_sum_assignment(-Matchn)
            NewSeeds = torch.zeros([n1,n2])
            NewSeeds[row,col] = 10
            
            Z = torch.zeros(n1*n2,1)
    
            S = torch.cat([X,Z],dim=1)
        
       
        return Y_total[-1], Y_total
        
    def loss(self, S, y):
        
        nll = 0
        EPS = 1e-12
        k = 1
        for Si in S:
            val = Si[y[0], y[1]]
            nll += torch.sum(-torch.log(val + EPS))
            
        return nll

    def acc(self, S, y):
       
        Sn = S.detach().numpy()
        row, col = linear_sum_assignment(-Sn)
        pred = torch.tensor(col)
        
        correct = sum(pred[y[0]] == y[1])
        return correct 
    
    
class SeedGNNx(torch.nn.Module):
    
    def __init__(self, num_layers, hid):
        super(SeedGNNx, self).__init__()
        self.hid = hid
        self.num_layers = num_layers
        self.mlp = torch.nn.ModuleList([Seq(
                Lin(1, hid-1),
            )])
        self.readout = torch.nn.ModuleList([Seq(
                Lin(1, 1),
            )])
        # self.alpha = torch.nn.Parameter(torch.zeros(num_layers))
        
        for i in range(1,num_layers):
            self.mlp.append(Seq(
                Lin(hid, hid-1),
            ))
            self.readout.append(Seq(
                Lin(hid, 1),
            ))

    def reset_parameters(self):
        for i in range(self.num_layers):
            reset(self.mlp[i])
            reset(self.readout[i])

    def forward(self, G1, G2, seeds, noisy = False):
        
        Y_total = []
        n1 = G1.shape[0]
        n2 = G2.shape[0]
        
        Seeds = torch.zeros([n1,n2])
        Seeds[seeds[0],seeds[1]] = 1
        
        # S = -torch.ones([n1,n2])/n1
        # S[seeds[0],seeds[1]] = 1
        S = Seeds
        
        for layeri in range(self.num_layers):
            
            H = torch.sparse.mm(G2,torch.sparse.mm(G1,S).T).T.reshape(-1,1)
                
            
            match = H
            Match = match.reshape(n1,n2)
            Match = Match
            Matchnorm = masked_softmax(Match)
            Matchnorm[seeds[0],:]=0
            Matchnorm[:,seeds[1]]=0
            Matchnorm[seeds[0],seeds[1]]=1
            Y_total.append(Matchnorm)
            
            Matchn = Matchnorm.detach().numpy()
            row,col = linear_sum_assignment(-Matchn)
            NewSeeds = torch.zeros([n1,n2])
            NewSeeds[row,col] = 1
            
            Z = NewSeeds
    
            S = Z
        
       
        return Y_total[-1], Y_total
        
    def loss(self, S, y):
        
        nll = 0
        EPS = 1e-12
        k = 1
        for Si in S:
            val = Si[y[0], y[1]]
            nll += torch.sum(-torch.log(val + EPS))
            
        return nll

    def acc(self, S, y):
       
        Sn = S.detach().numpy()
        row, col = linear_sum_assignment(-Sn)
        pred = torch.tensor(col)
        
        correct = sum(pred[y[0]] == y[1])
        return correct 
