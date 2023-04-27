import matplotlib.pyplot as plt

import copy
import os.path as osp
import numpy as np
from scipy import sparse as sp
import random

import argparse
import torch
from scipy.optimize import linear_sum_assignment
from SeedGNN import SeedGNN
import seaborn as sns
from GMAlgorithms import SynGraph, MultiHop, SGM2

# CompareMatchingResult

torch.set_printoptions(precision=4)
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--hid', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--pre_epochs', type=int, default=15)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--test_samples', type=int, default=100)
args, unknown = parser.parse_known_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SeedGNN(num_layers=args.num_layers, hid=args.hid).to(device)
path = "./model/SeedGNN-model.pth"
model.load_state_dict(torch.load(path))

def generate_y(num_nodes, truth):
    oneton = torch.arange(num_nodes)
    return [oneton, truth]

def masked_softmax(src,num_nodes):
    srcmax1 = src - torch.max(src,dim=1,keepdim = True)[0]
    out1 = torch.softmax(srcmax1,dim = 1)
    
    srcmax2 = src - torch.max(src,dim=0,keepdim = True)[0]
    out2 = torch.softmax(srcmax2,dim = 0)
    
    return (out1+out2)/2

def test(test_dataset):
    model.eval()

    total_correct = 0
    total_node = 0
    for data in test_dataset:
        
        
        G1 = data[0]
        G2 = data[1]
        seeds = data[2]
        truth = data[3]
        num_nodes = G1.shape[0]
        
        Y_L, Y_total = model(G1,G2,seeds)
        
        y = generate_y(num_nodes, truth)
        correct = model.acc(Y_L,y)
        total_correct += correct
        total_node += num_nodes
    return Y_total


def run(datasets):
    
    eyes = torch.eye(n)
    G12 = (( ((torch.sparse.mm(G1, G1.to_dense()))>0).float() - G1 - eyes)>0).float()
    G22 = (( ((torch.sparse.mm(G2, G2.to_dense()))>0).float() - G2 - eyes)>0).float()

    xticks = [i for i in range(0,n,int(n/10))]
    xticklabels = [i for i in range(0,n,int(n/10))]

    S_total = test(datasets)

    print('SeedGNN')
    i = 0
    for S in S_total:
        # fig = plt.figure(figsize=(5, 5))
        axi = plt.subplot(1,1,1)
    
        S = torch.rot90(-torch.pow(S-1,4)+1,3).detach().numpy()
    
        sns.heatmap(data=list(S),cbar=False,
                cmap=plt.get_cmap('Greys'),
                ax = axi)
        sns.despine(right=False, top=False, left=False)
        axi.set_yticks(xticks)
        axi.set_yticklabels(xticks)
        axi.set_xticks(xticks)
        axi.set_xticklabels(xticks)
        axi.set_aspect('equal')   
        i+=1
        # plt.savefig('./figure/ERmatrixgnn'+'{}'.format(i)+'.eps',bbox_inches ='tight')    
        plt.show()
    
    print('---------------------------------------')    
    print('1-hop')
    result = seeds
    j = 0
    for _ in range(6):
        Seeds = torch.zeros([n,n])
        Seeds[result[0],result[1]] = 1
        W1 = torch.sparse.mm(G2,torch.sparse.mm(G1,Seeds).T).T
        row, result = linear_sum_assignment(-W1)
        result = generate_y(n, result)
        axi = plt.subplot(1,1,1)
        W1 = masked_softmax(W1,n)
        W1 = (-torch.pow(W1-1,4)+1).detach().numpy()
        sns.heatmap(data=list(np.rot90(W1,3)),cbar=False,
                    cmap=plt.get_cmap('Greys'),
                    ax = axi)
        sns.despine(right=False, top=False, left=False)
        axi.set_yticks(xticks)
        axi.set_yticklabels(xticks)
        axi.set_xticks(xticks)
        axi.set_xticklabels(xticks)
        axi.set_aspect('equal')
        j+=1
        # plt.savefig('./figure/ERmatrix1hop'+'{}'.format(j)+'.eps',bbox_inches ='tight')
        plt.show()
    
    print('---------------------------------------')        
    print('2-hop')
    result = seeds
    j = 0
    for _ in range(3):
        Seeds = torch.zeros([n,n])
        Seeds[result[0],result[1]] = 1
        W1 = torch.sparse.mm(G22,torch.sparse.mm(G12,Seeds).T).T
        row, result = linear_sum_assignment(-W1)
        result = generate_y(n, result)
        axi = plt.subplot(1,1,1)
        W1 = masked_softmax(W1,n)
        W1 = (-torch.pow(W1-1,4)+1).detach().numpy()
        sns.heatmap(data=list(np.rot90(W1,3)),cbar=False,
                    cmap=plt.get_cmap('Greys'),
                    ax = axi)
        sns.despine(right=False, top=False, left=False)
        axi.set_yticks(xticks)
        axi.set_yticklabels(xticks)
        axi.set_xticks(xticks)
        axi.set_xticklabels(xticks)
        axi.set_aspect('equal')
        j+=1
        # plt.savefig('./figure/ERmatrix2hop'+'{}'.format(j)+'.eps',bbox_inches ='tight')
        plt.show()


n = 50
p = 0.4
s = 0.8
theta = 0.1
datasets = []
G1, G2, seeds, truth = SynGraph(n,p,s,theta,True)
datasets.append((G1, G2, seeds, truth))
print(f'Parameters: n={n}, p={p}, s={s}, theta={theta}')
run(datasets)

print('*********************************************')
n = 50
p = 0.1
s = 0.8
theta = 0.1
datasets = []
G1, G2, seeds, truth = SynGraph(n,p,s,theta,True)
datasets.append((G1, G2, seeds, truth))
print(f'Parameters: n={n}, p={p}, s={s}, theta={theta}')
run(datasets)