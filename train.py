import os
import os.path as osp
import random

import argparse
import torch
import torch_geometric.transforms as T

from SeedGNN import SeedGNN
from GMAlgorithms import SynGraph, facebookGraph

torch.set_printoptions(precision=4)

parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=200)

args, unknown = parser.parse_known_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SeedGNN(num_layers=args.num_layers, hid=args.hid).to(device)

def generate_y(num_nodes, truth):
    oneton = torch.arange(num_nodes)
    return [oneton, truth]

def train(train_dataset, optimizer):
    model.train()

    total_loss = 0
    num_examples = 0
    
    for i in range(0, len(train_dataset), args.batch_size):
        
        batch = train_dataset[i:i+args.batch_size]
        optimizer.zero_grad()
        batch_loss = 0
        
        for data in batch:
            optimizer.zero_grad()
        
            G1 = data[0]
            G2 = data[1]
            seeds = data[2]
            truth = data[3]
            num_nodes = G1.shape[0]
        
            Y_L, Y_total = model(G1,G2,seeds)
        
            y = generate_y(num_nodes, truth)
            loss = model.loss(Y_total, y)
            batch_loss += loss
            total_loss += loss
            num_examples +=1
           
        batch_loss.backward()
        optimizer.step()
    return total_loss.item()/num_examples


@torch.no_grad()
def test(test_dataset):
    model.eval()

    total_correct = 0
    num_test = 0
    for data in test_dataset:
        
        G1 = data[0]
        G2 = data[1]
        seeds = data[2]
        truth = data[3]
        num_nodes = G1.shape[0]
        
        Y_L, _ = model(G1,G2,seeds)
        
        y = generate_y(num_nodes, truth)
        correct = model.acc(Y_L,y)
        total_correct += correct/G1.shape[0]
        num_test += 1
    return total_correct/num_test


def run(datasets):
    
    # path = "./model/test5.pth"
    # model.load_state_dict(torch.load(path))
    model.reset_parameters()
    
    random.shuffle(datasets)
    train_dataset = datasets[:100]
    test_dataset = datasets[100:]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    
    for epoch in range(1, 1 + args.epochs):
        loss = train(train_dataset, optimizer)
        scheduler.step()
        
        if epoch%5 == 0:
            train_acc = test(train_dataset)
            test_acc = test(test_dataset)
            print(f'epoch {epoch:03d}: Loss: {loss:.8f}, Training Acc: {train_acc:.4f}, Testing Acc: {test_acc:.4f}')

    accs = 100 * test(test_dataset)
    
    print('Acc: ',accs)
    path = "./model/SeedGNN-model-trained.pth"
    torch.save(model.state_dict(), path)

    return accs


if __name__ == '__main__':
    print('Preparing training data...')
    datasets = []
    graph_para = [(100,0.1,0.6,0.1),(100,0.1,0.8,0.1),(100,0.1,1,0.1),
                  (100,0.3,0.6,0.1),(100,0.3,0.8,0.1),(100,0.3,1,0.1),
                  (100,0.5,0.6,0.1),(100,0.5,0.8,0.1),(100,0.5,1,0.1),
                  (100,0.1,0.6,0.1)]
    numgraphs = 10
    for n,p,s,theta in graph_para:
        for _ in range(numgraphs):
            GraphPair = SynGraph(n,p,s,theta)
            datasets.append(GraphPair)
    s = 0.8
    alpha = 0.9  
    theta = 0.1
    Facebook_Filepath = "./data/facebook100"
    filedirs = os.listdir(Facebook_Filepath)
    for realpath in filedirs[:10]:
        datasets.append(facebookGraph(Facebook_Filepath+'/'+realpath,s,alpha,theta,True))
    
    print('Done!')
    run(datasets)
    
