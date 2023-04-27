import sys
import os.path as osp
import numpy as np
import time

import argparse
import torch

from SeedGNN import SeedGNN
from MGCN import MGCN
from GMAlgorithms import SynGraph, MultiHop, PGM, SGM2



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

def run(n,p,s,L,Theta,Itera):
    seedgnn = torch.zeros(Itera,len(Theta))
    onehop6 = torch.zeros(Itera,len(Theta))
    twohop3 = torch.zeros(Itera,len(Theta))
    threehop2 = torch.zeros(Itera,len(Theta))
    pgm = torch.zeros(Itera,len(Theta))
    sgm = torch.zeros(Itera,len(Theta))
    mgcn = torch.zeros(len(Theta))

    for thetai, theta in enumerate(Theta):
        for itera in range(Itera):
            datasets = []
            G1, G2, seeds, truth = SynGraph(n,p,s,theta)
            datasets = [(G1, G2, seeds, truth)]
            eyes = torch.eye(n)
            G12 = (( ((torch.mm(G1, G1))>0).float() - G1 - eyes)>0).float()
            G22 = (( ((torch.mm(G2, G2))>0).float() - G2 - eyes)>0).float()
            G13 = (( ((torch.mm(G12, G1))>0).float() - G12 - G1 - eyes)>0).float()
            G23 = (( ((torch.mm(G22, G2))>0).float() - G22 - G2 - eyes)>0).float()
            
            # SeedGNN
            # seedgnn[itera,thetai] = test(datasets)
             
            # other algorithms
            # result = seeds
            # for _ in range(L):
            #     result = MultiHop(G1,G2,result)
            # onehop6[itera,thetai] = sum((result[1]==truth).float())/n
            
            # result = seeds
            # for _ in range(int(L/2)):
            #     result = MultiHop(G12,G22,result)
            # twohop3[itera,thetai] = sum((result[1]==truth).float())/n
            
            # result = seeds        
            # for _ in range(int(L/3)):
            #     result = MultiHop(G13,G23,result)
            # threehop2[itera,thetai] = sum((result[1]==truth).float())/n
            
            # result = PGM(G1,G2,seeds)
            # pgm[itera,thetai] = sum(result==truth)/n
        
            result = SGM2(G1,G2,seeds)
            sgm[itera,thetai] = sum((result==truth).float())/n
            
            # result = MGCN(G1,G2,seeds)
            # mgcn[thetai] = sum((result==truth).float())/n        

    seedgnnstd, seedgnn = torch.std_mean(seedgnn,dim=0,unbiased=False)
    onehop6std,onehop6 = torch.std_mean(onehop6,dim=0,unbiased=False)
    twohop3std,twohop3 = torch.std_mean(twohop3,dim=0,unbiased=False)
    threehop2std,threehop2 = torch.std_mean(threehop2,dim=0,unbiased=False)
    pgmstd,pgm = torch.std_mean(pgm,dim=0,unbiased=False)
    sgmstd,sgm = torch.std_mean(sgm,dim=0,unbiased=False)


    theta = [round(i,4) for i in Theta.tolist()]
    seedgnn = [round(i,4) for i in (seedgnn).tolist()]
    onehop6 = [round(i,4) for i in (onehop6).tolist()]
    twohop3 = [round(i,4) for i in (twohop3).tolist()]
    threehop2 = [round(i,4) for i in (threehop2).tolist()]
    pgm = [round(i,4) for i in (pgm).tolist()]
    sgm = [round(i,4) for i in (sgm).tolist()]
    mgcn = [round(i,4) for i in (mgcn).tolist()]

    torch.set_printoptions(precision=4)
    print(f'Parameters: n={n}, p={p}, s={s}, L={L}')
    print('Accuracy')
    print('theta ='.ljust(10), theta)
    print('SeedGNN = '.ljust(10),seedgnn)
    print('onehop6 ='.ljust(10), onehop6)
    print('twohop3 ='.ljust(10), twohop3)
    print('threehop2 ='.ljust(10), threehop2)
    print('pgm = '.ljust(10), pgm)
    print('sgm = '.ljust(10), sgm)
    print('mgcn = '.ljust(10), mgcn)

    # seedgnnstd = [round(i,4) for i in (seedgnnstd).tolist()]
    # onehop6std = [round(i,4) for i in (onehop6std).tolist()]
    # twohop3std = [round(i,4) for i in (twohop3std).tolist()]
    # threehop2std = [round(i,4) for i in (threehop2std).tolist()]
    # pgmstd = [round(i,4) for i in (pgmstd).tolist()]
    # sgmstd = [round(i,4) for i in (sgmstd).tolist()]

    # print('seedgnnstd ='.ljust(13), seedgnnstd)
    # print('onehop6std ='.ljust(13), onehop6std)
    # print('twohop3std ='.ljust(13), twohop3std)
    # print('threehop2std ='.ljust(10), threehop2std)
    # print('pgmstd = '.ljust(13), pgmstd)
    # print('sgmstd = '.ljust(13), sgmstd)


if __name__ ==  '__main__':
    
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
    for param in model.parameters():
        param.requires_grad = False
    
    start = time.time()
    L = 6
    n = 500
    p = 0.01
    s = 0.8
    Theta = torch.linspace(0, 0.2, steps=11)
    Itera = 1
    run(n,p,s,L,Theta,Itera)

    print('-----------------------------------------------')

    L = 6
    n = 500
    p = 0.2
    s = 0.8
    Theta = torch.linspace(0, 0.05, steps=11)
    Itera = 1
    run(n,p,s,L,Theta,Itera)

    end = time.time()
    print('run: ',end-start)