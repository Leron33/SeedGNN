import matplotlib.pyplot as plt
import copy
import os.path as osp
import numpy as np
from scipy import sparse as sp
import random

import argparse
import torch

from SeedGNN import SeedGNN
from GMAlgorithms import SynGraph

torch.set_printoptions(precision=4)
parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--pre_epochs', type=int, default=15)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--test_samples', type=int, default=100)
args, unknown = parser.parse_known_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def generate_y(num_nodes, truth):
    oneton = torch.arange(num_nodes)
    return [oneton, truth]


def test(test_dataset,var):
    model = SeedGNN(num_layers=args.num_layers, hid=args.hid).to(device)
    
    if var == 0:
        path = "./model/SeedGNN-model.pth"        
    elif var == 'p1':
        path = "./model/SeedGNNp1-model.pth"
    elif var == 'p2':
        path = "./model/SeedGNNp2-model.pth"
    elif var == 's1':
        path = "./model/SeedGNNs1-model.pth"
    elif var == 's2':
        path = "./model/SeedGNNs2-model.pth"
    elif var == 's3':
        path = "./model/SeedGNNs3-model.pth"
    elif var == 't1':
        path = "./model/SeedGNNt1-model.pth"
    elif var == 't2':
        path = "./model/SeedGNNt2-model.pth"
    
    model.load_state_dict(torch.load(path))
        
        
    model.eval()

    total_correct = 0
    total_node = 0
    for data in test_dataset:
        
        
        G1 = data[0]
        G2 = data[1]
        seeds = data[2]
        truth = data[3]
        num_nodes = G1.shape[0]
        
        Y_L, _ = model(G1,G2,seeds)
        
        y = generate_y(num_nodes, truth)
        correct = model.acc(Y_L,y)
        total_correct += correct
        total_node += num_nodes
    return total_correct/total_node

Itera = 50

n = 500
p = 0.04
s = 0.8
Theta = torch.linspace(0, 0.05, steps=11)

seedgnn = torch.zeros(len(Theta))
seedgnn_t1 = torch.zeros(len(Theta))
seedgnn_t2 = torch.zeros(len(Theta))

for thetai, theta in enumerate(Theta):
    datasets = []
    for itera in range(Itera):
        G1, G2, seeds, truth = SynGraph(n,p,s,theta)
        datasets = [(G1, G2, seeds, truth)]
            
    seedgnn[thetai] = test(datasets,0)        
    seedgnn_t1[thetai] = test(datasets,'t1')        
    seedgnn_t2[thetai] = test(datasets,'t2')

theta = [round(i,4) for i in Theta.tolist()]
seedgnn = [round(i,4) for i in (seedgnn).tolist()]
seedgnn_t1 = [round(i,4) for i in (seedgnn_t1).tolist()]
seedgnn_t2 = [round(i,4) for i in (seedgnn_t2).tolist()]

torch.set_printoptions(precision=4)
print(f'Parameters: n={n}, p={p}, s={s}')
print('Accuracy')
print('theta ='.ljust(10), theta)
print('SeedGNN = '.ljust(10),seedgnn)
print('SeedGNNt1 = '.ljust(10),seedgnn_t1)
print('SeedGNNt2 = '.ljust(10),seedgnn_t2)
print('-----------------------------------------------')

n = 500
s = 0.8
theta = 0.05
P = torch.linspace(0.02, 0.2, steps=10)
seedgnn = torch.zeros(len(P))
seedgnn_p1 = torch.zeros(len(P))
seedgnn_p2 = torch.zeros(len(P))

for pi, p in enumerate(P):
    datasets = []
    for itera in range(Itera):
        G1, G2, seeds, truth = SynGraph(n,p,s,theta)
        datasets = [(G1, G2, seeds, truth)]
            
    seedgnn[pi] = test(datasets,0)        
    seedgnn_p1[pi] = test(datasets,'p1')        
    seedgnn_p2[pi] = test(datasets,'p2')

P = [round(i,4) for i in P.tolist()]
seedgnn = [round(i,4) for i in (seedgnn).tolist()]
seedgnn_p1 = [round(i,4) for i in (seedgnn_p1).tolist()]
seedgnn_p2 = [round(i,4) for i in (seedgnn_p2).tolist()]

torch.set_printoptions(precision=4)
print(f'Parameters: n={n}, s={s}, theta={theta}')
print('Accuracy')
print('p ='.ljust(10), P)
print('SeedGNN = '.ljust(10),seedgnn)
print('SeedGNNp1 = '.ljust(10),seedgnn_p1)
print('SeedGNNp2 = '.ljust(10),seedgnn_p2)
print('-----------------------------------------------')

n = 500
S = torch.linspace(0.5, 1, steps= 6)
theta = 0.05
p = 0.08

seedgnn = torch.zeros(len(S))
seedgnn_s1 = torch.zeros(len(S))
seedgnn_s2 = torch.zeros(len(S))
seedgnn_s3 = torch.zeros(len(S))

for si, s in enumerate(S):
    datasets = []
    for itera in range(Itera):
        G1, G2, seeds, truth = SynGraph(n,p,s,theta)
        datasets = [(G1, G2, seeds, truth)]
            
    seedgnn[si] = test(datasets,0)        
    seedgnn_s1[si] = test(datasets,'s1')        
    seedgnn_s2[si] = test(datasets,'s2')
    seedgnn_s3[si] = test(datasets,'s3')
    
S = [round(i,4) for i in S.tolist()]
seedgnn = [round(i,4) for i in (seedgnn).tolist()]
seedgnn_s1 = [round(i,4) for i in (seedgnn_s1).tolist()]
seedgnn_s2 = [round(i,4) for i in (seedgnn_s2).tolist()]
seedgnn_s3 = [round(i,4) for i in (seedgnn_s3).tolist()]

torch.set_printoptions(precision=4)
print(f'Parameters: n={n}, p={p}, theta={theta}')
print('Accuracy')
print('s ='.ljust(10), S)
print('SeedGNN = '.ljust(10),seedgnn)
print('SeedGNNs1 = '.ljust(10),seedgnn_s1)
print('SeedGNNs2 = '.ljust(10),seedgnn_s2)
print('SeedGNNs3 = '.ljust(10),seedgnn_s3)