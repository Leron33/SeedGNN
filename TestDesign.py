import matplotlib.pyplot as plt
import copy
import os.path as osp
import numpy as np
from scipy import sparse as sp
import random

import argparse
import torch

from SeedGNN import SeedGNN, SeedGNN_hun, SeedGNN_per, SeedGNN_van, SeedGNNx
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
    if var == 0:
        model = SeedGNN(num_layers=args.num_layers, hid=args.hid).to(device)
        path = "./model/SeedGNN-model.pth"
        model.load_state_dict(torch.load(path))
    elif var == 1:
        model = SeedGNN_hun(num_layers=args.num_layers, hid=args.hid).to(device)
        path = "./model/SeedGNNhun-model.pth"
        model.load_state_dict(torch.load(path))
    elif var == 2:
        model = SeedGNN_per(num_layers=args.num_layers, hid=args.hid).to(device)
        path = "./model/SeedGNNper-model.pth"
        model.load_state_dict(torch.load(path))
    elif var == 3:
        model = SeedGNN_van(num_layers=args.num_layers, hid=args.hid).to(device)
        path = "./model/SeedGNNvan-model.pth"
        model.load_state_dict(torch.load(path))
    elif var == 4:
        model = SeedGNNx(num_layers=args.num_layers, hid=args.hid).to(device)
        path = "./model/SeedGNNx-model.pth"
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



n = 500
p = 0.04
s = 0.8
Theta = torch.linspace(0, 0.05, steps=11)
Itera = 10
seedgnn = torch.zeros(len(Theta))
seedgnn_hun = torch.zeros(len(Theta))
seedgnn_per = torch.zeros(len(Theta))
seedgnn_van = torch.zeros(len(Theta))
seedgnnx = torch.zeros(len(Theta))

for thetai, theta in enumerate(Theta):
    datasets = []
    for itera in range(Itera):
        G1, G2, seeds, truth = SynGraph(n,p,s,theta)
        datasets = [(G1, G2, seeds, truth)]
            
    seedgnn[thetai] = test(datasets,0)
        
    seedgnn_hun[thetai] = test(datasets,1)
        
    seedgnn_per[thetai] = test(datasets,2)
        
    seedgnn_van[thetai] = test(datasets,3)
        
    seedgnnx[thetai] = test(datasets,4)

theta = [round(i,4) for i in Theta.tolist()]
seedgnn = [round(i,4) for i in (seedgnn).tolist()]
seedgnn_hun = [round(i,4) for i in (seedgnn_hun).tolist()]
seedgnn_per = [round(i,4) for i in (seedgnn_per).tolist()]
seedgnn_van = [round(i,4) for i in (seedgnn_van).tolist()]
seedgnnx = [round(i,4) for i in (seedgnnx).tolist()]

torch.set_printoptions(precision=4)
print(f'Parameters: n={n}, p={p}, s={s}')
print('Accuracy')
print('theta ='.ljust(10), theta)
print('SeedGNN = '.ljust(10),seedgnn)
print('SeedGNN_hun = '.ljust(10),seedgnn_hun)
print('SeedGNN_per = '.ljust(10),seedgnn_per)
print('SeedGNN_van = '.ljust(10),seedgnn_van)
print('SeedGNNx = '.ljust(10),seedgnnx)