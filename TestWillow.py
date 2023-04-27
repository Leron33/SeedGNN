import matplotlib.pyplot as plt
import copy
import os
import os.path as osp
import numpy as np
from scipy import sparse as sp
import random
import scipy.io
from graspologic.match import GraphMatch as GMP

import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import WILLOWObjectClass as WILLOW
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from torch_geometric.data import DataLoader

from SeedGNN import SeedGNN
from GMAlgorithms import WillowGraph, MultiHop, PGM, SGM2

parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
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
        
        y = generate_y(num_nodes, truth)
        correct = model.acc(Y_L,y)
        total_correct += correct
        total_node += num_nodes
    return total_correct/total_node

def run():
    
    Shrec_Filepath = './data/low_resolution/kid'
    
    seedgnn = torch.zeros(5)
    graphi = 0
    
    transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
    ])

    path = './data/WILLOW'
    Datasets = [WILLOW(path, cat, transform) for cat in WILLOW.categories]

    k = 0
    for datasets in Datasets:
        test_dataset = []
        numgraphs=len(datasets)
        for i in range(20,numgraphs):
            for j in range(20,numgraphs):
                G1, G2, seeds, truth = WillowGraph(datasets[i],datasets[j],7,2)
                test_dataset.append((G1,G2,seeds,truth))
        
        seedgnn[k]= test(test_dataset)
        k +=1
    

    print('Accuracy')
    print(' '.join([category.ljust(13) for category in WILLOW.categories]))
    print(' '.join([f'{a:.3f}'.ljust(13) for a in seedgnn]))


run()