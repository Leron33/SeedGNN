import sys
import matplotlib.pyplot as plt
import copy
import os.path as osp
import numpy as np
from scipy import sparse as sp
import time

import argparse
import torch
import torch_geometric.transforms as T

from SeedGNN import SeedGNN
from MGCN import MGCN
from GMAlgorithms import ShrecGraph, MultiHop, PGM, SGM2

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

model = SeedGNN(num_layers=args.num_layers, hid=args.hid).to(device)

path = "./model/SeedGNN-model.pth"
model.load_state_dict(torch.load(path))


def generate_y(num_nodes, truth):
    oneton = torch.arange(num_nodes)
    return [oneton, truth]

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
        
        Y_L, _ = model(G1,G2,seeds)
        
        
        correct = model.acc(Y_L,truth)
        total_correct += correct
        total_node += num_nodes
    return total_correct/total_node

def run(L,Theta):
    
    Shrec_Filepath = './data/low_resolution/kid'
    
    seedgnn = torch.zeros(len(Theta))
    onehop6 = torch.zeros(len(Theta))
    twohop3 = torch.zeros(len(Theta))
    threehop2 = torch.zeros(len(Theta))
    pgm = torch.zeros(len(Theta))
    sgm = torch.zeros(len(Theta))
    mgcn = torch.zeros(len(Theta))
    
    graphi = 0
    
    for i in range(16,17):
        for j in range(17,18):
            for thetai, theta in enumerate(Theta):
                datasets = []
                G1, G2, seeds, truth = ShrecGraph(Shrec_Filepath,i,j,theta)
                datasets = [(G1, G2, seeds, truth)]
                
                n1 = G1.shape[0]
                n2 = G2.shape[0]
                
                eyes1 = torch.eye(n1)
                eyes2 = torch.eye(n2)
                G12 = (( ((torch.mm(G1, G1))>0).float() - G1 - eyes1)>0).float()
                G22 = (( ((torch.mm(G2, G2))>0).float() - G2 - eyes2)>0).float()
                G13 = (( ((torch.mm(G12, G1))>0).float() - G12 - G1 - eyes1)>0).float()
                G23 = (( ((torch.mm(G22, G2))>0).float() - G22 - G2 - eyes2)>0).float()
            
                # SeedGNN
                seedgnn[thetai] += test(datasets)
                    
                # result = seeds
                # for _ in range(L):
                #     result = MultiHop(G1,G2,result)
                # onehop6[thetai] += sum((result[1][truth[0]]==truth[1]).float())/n1
                
                # result = seeds
                # for _ in range(int(L/2)):
                #     result = MultiHop(G12,G22,result)
                # twohop3[thetai] += sum((result[1][truth[0]]==truth[1]).float())/n1

                # result = seeds        
                # for _ in range(int(L/3)):
                #     result = MultiHop(G13,G23,result)
                # threehop2[thetai] += sum((result[1][truth[0]]==truth[1]).float())/n1

                # result = PGM(G1,G2,seeds)
                # pgm[thetai] += sum((result[truth[0]]==truth[1]).float())/n1
        
                # result = SGM2(G1,G2,seeds)
                # sgm[thetai] += sum((result[truth[0]]==truth[1]).float())/n1
                
                # result = MGCN(G1,G2,seeds)
                # mgcn[thetai] = sum((result[truth[0]]==truth[1]).float())/n1
                
            graphi +=1

    theta = [round(i,4) for i in Theta.tolist()]
    seedgnn = [round(i,4) for i in (seedgnn/graphi).tolist()]
    onehop6 = [round(i,4) for i in (onehop6/graphi).tolist()]
    twohop3 = [round(i,4) for i in (twohop3/graphi).tolist()]
    threehop2 = [round(i,4) for i in (threehop2/graphi).tolist()]
    pgm = [round(i,4) for i in (pgm/graphi).tolist()]
    sgm = [round(i,4) for i in (sgm/graphi).tolist()]
    mgcn = [round(i,4) for i in (mgcn/graphi).tolist()]

    torch.set_printoptions(precision=4)
    print('Accuracy')
    print('theta ='.ljust(10), theta)
    print('SeedGNN = '.ljust(10),seedgnn)
    print('onehop6 ='.ljust(10), onehop6)
    print('twohop3 ='.ljust(10), twohop3)
    print('threehop2 ='.ljust(10), threehop2)
    print('pgm = '.ljust(10), pgm)
    print('sgm = '.ljust(10), sgm)
    print('mgcn = '.ljust(10), mgcn)


L = 6
Theta = torch.linspace(0, 0.1, steps=3)

start = time.time()

run(L,Theta)

end = time.time()
print('run: ',end-start)
