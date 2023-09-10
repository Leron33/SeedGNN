import sys

import numpy as np
from scipy import sparse as sp
from scipy.optimize import linear_sum_assignment
import torch
import scipy.io



def SynGraph(n, p, s, theta, degreesort = False):
    
    a = (torch.rand(n,n)<p).float()
    a = torch.triu(a,diagonal=1)
    adj = a+a.T
    sample = (torch.rand(n,n)<s).float()
    sample = torch.triu(sample,1)
    sample = sample + sample.T
    G1 = adj*sample
    sample = (torch.rand(n,n)<s).float()
    sample = torch.triu(sample,1)
    sample = sample + sample.T
    G2 = adj*sample
    
    if degreesort:
        degree = adj.sum(0)
        _,indices = torch.sort(degree,descending=True)
        G1 = G1[indices,:][:,indices]
        G2 = G2[indices,:][:,indices]
        truth = n-1-torch.arange(n)
    else:
        truth = torch.randperm(n)
        
    G1 = G1[truth,:][:,truth]

    
    numseeds = int(n*theta)
    indices = torch.randperm(n)[:numseeds]
    seeds = [indices, truth[indices]]
    
    return (G1, G2, seeds, truth)

def facebookGraph(realpath,s,alpha,theta,sample=True):
    
    mat = scipy.io.loadmat(realpath)
    
    adj = torch.tensor(mat['A'].toarray()).float()
    N = adj.shape[0]
    n = N
    if sample:
        n = 200
        indices = torch.randperm(N)[:n]
        adj = adj[indices,:][:,indices]
    
    sample = (torch.rand(n,n)<s).float()
    sample = torch.triu(sample,1)
    sample = sample + sample.T
    G1 = adj*sample

    sample = (torch.rand(n,n)<s).float()
    sample = torch.triu(sample,1)
    sample = sample + sample.T
    G2 = adj*sample    
    
    truth = torch.randperm(n)
        
    G1 = G1[truth,:][:,truth]
    
    
    numseeds = int(n*theta)
    indices = torch.randperm(n)[:numseeds]
    seeds = [indices, truth[indices]]
    
    return (G1, G2, seeds, truth)

def WillowGraph(datas, datat, numseeds, seed_method=1):
    n = 10
    
    G1 = torch.zeros([n,n])
    G1[datas.edge_index[0],datas.edge_index[1]] = 1
    x1 = datas.x

    G2 = torch.zeros([n,n])
    G2[datat.edge_index[0],datat.edge_index[1]] = 1
    x2 = datat.x
    
    truth = torch.randperm(n)
    # truth = torch.arange(n)
    G1 = G1[truth,:][:,truth]
    x1 = x1[truth,:]
    
    
    if seed_method == 2:
        indices = torch.randperm(n)[:numseeds]
        seeds = [indices, truth[indices]]
    elif seed_method == 1:
        W1 = torch.mm(x1,x2.T)
        row, col = linear_sum_assignment(-W1.detach().numpy())
        result = np.stack([row,col])
        seeds = torch.tensor(result)
    return G1, G2, seeds, truth

def triangulatoin2adj(realpath):
    f = open(realpath, 'r')
    f.readline() # OFF
    num_views, num_groups, num_edges = map(int, f.readline().split())
    view_data = []
    for view_id in range(num_views):
        view_data.append(list(map(float, f.readline().split())))    
    group_data = []
    for group_id in range(num_groups):
        group_data.append(list(map(int, f.readline().split()[1:])))
    
    f.close()
    
    
    adj = torch.zeros(num_views,num_views)
    for face in group_data:
        for k in range(3):
            kk = (k+1)%3
            adj[face[k]-1,face[kk]-1] = 1
    adj = torch.max(adj,adj.T)
    return adj

def merge_gt(path1,path2):
    f = open(path1, 'r')
    gt1 = []
    while True:
        line = f.readline()
        if not line:    
            break
        else:
            gt1.append(list(map(int, line.split())))
    f.close()
    
    f = open(path2, 'r')
    gt2 = []
    while True:
        line = f.readline()
        if not line:   
            break
        else:
            gt2.append(list(map(int, line.split())))
    f.close()

    maxnode = max(max(max(gt1)),max(max(gt2)))

    bin1=torch.zeros(maxnode)
    bin2=torch.zeros(maxnode)
    set1 = set()
    set2 = set()
    for maps in gt1:
        bin1[maps[1]-1]=maps[0]-1
        set1.add(maps[1]-1)
    for maps in gt2:
        bin2[maps[1]-1]=maps[0]-1
        set2.add(maps[1]-1)
    joint = set1.intersection(set2)
    gt = torch.zeros(2,len(joint))
    ni = 0
    for i in joint:
        gt[0][ni] = bin1[i]
        gt[1][ni] = bin2[i]
        ni += 1
    return gt.long()

def ShrecGraph(realpath,i,j,theta):
    path = realpath +str(i)+'.off'    
    G1 = triangulatoin2adj(path)

    path = realpath +str(j)+'.off'
    G2 = triangulatoin2adj(path)
    
    path1 = realpath +str(i)+'_ref.txt'    
    path2 = realpath +str(j)+'_ref.txt'  
    truth = merge_gt(path1,path2)
    
    n = truth.size(1)
    numseeds = int(n*theta)
    indices = torch.randperm(n)[:numseeds]
    seeds = [truth[0][indices], truth[1][indices]]
    
    return (G1, G2, seeds, truth)

def MultiHop(G1,G2,seeds):
    n1 = G1.shape[0]
    n2 = G2.shape[0]
    
    # Seeds Matrix
    Seeds = torch.zeros([n1,n2])
    Seeds[seeds[0],seeds[1]] = 1
    
    # Count D-hop witnesses
    W1 = torch.sparse.mm(G2,torch.sparse.mm(G1,Seeds).T).T
    
    # linear sum assignment
    row, col = linear_sum_assignment(-W1.detach().numpy())
    result = np.stack([row,col])
    return torch.tensor(result)

def PGM(G1,G2,seeds):
    n1 = G1.shape[0]
    n2 = G2.shape[0]
    
    # Seeds Matrix
    Seeds = torch.zeros([n1,n2])
    Seeds[seeds[0],seeds[1]] = 1
    
    for i in range(10):
        W1 = torch.sparse.mm(G2,torch.sparse.mm(G1,Seeds).T).T
        results = W1.argmax(1)
        newSeeds = Seeds
        for i in range(len(results)):
            if i not in seeds[0] and W1[i,results[i]] >=2:
                Seeds[i,results[i]] = 1
    return results


def SGM2(G1, G2, seeds, Itera = 20):
    
    m = len(seeds[0])
    n1 = G1.shape[0]
    n2 = G2.shape[0]
    
    unseed1 = torch.zeros(n1-m)
    unseed2 = torch.zeros(n2-m)
    
    j = 0
    k = 0
    for i in range(n1):
        if i not in seeds[0]:
            unseed1[j] = i
            j += 1
    for i in range(n2):
        if i not in seeds[1]:
            unseed2[k] = i
            k += 1
    unseed1 = unseed1.long()
    unseed2 = unseed2.long()
    
    A = G1
    B = G2
    
    G1index = torch.cat([seeds[0],unseed1])
    A = A[G1index,:][:,G1index]
    G2index = torch.cat([seeds[1],unseed2])
    B = B[G2index,:][:,G2index]
    Itera = 6
    
    A21 = A[m:n1,:m]
    A22 = A[m:n1,m:n1]
    
    B12 = B[:m,m:n2]
    B22 = B[m:n2,m:n2]
    
    P = torch.ones([n1-m,n2-m])/(n1-m)
    k = 1
    
    S_total = []
    
    while k < Itera:
        x = torch.mm(A21,B12)
        y = torch.mm(A22,torch.mm(P,B22))
        dP = x + y
        row, col = linear_sum_assignment(-dP.detach().numpy())
                
        Q = torch.zeros([n1-m,n2-m])
        Q[row,col] = 1
        
        z = torch.mm(A22,torch.mm(Q,B22))
        
        c = torch.sum(torch.diag(torch.mm(y,P.T)))
        d = torch.sum(torch.diag(torch.mm(y,Q.T)+torch.mm(z,P.T)))
        e = torch.sum(torch.diag(torch.mm(z,Q.T)))
        u = 2*torch.sum(torch.diag(torch.mm(P.T,x)))
        v = 2*torch.sum(torch.diag(torch.mm(Q.T,x)))
        if c-d+e!=0:
            alpha = -(d-2*e+u-v) / (2*(c-d+e))
        
        f0 = 0
        f1 = (c-d+e) + (d-2*e+u-v)
        
        if c-d+e<0:
            if 0 < alpha <1:
                P = alpha*P + (1-alpha)*Q
            elif alpha<=0:
                P = Q
            else:
                k = Itera
        else:
            if f0>f1:
                P = Q
            else:
                k = Itera
        
        k += 1
            
    row, col = linear_sum_assignment(-P.detach().numpy())
    
    result = torch.ones(n1)
    for i in range(m):
        result[seeds[0][i]] = seeds[1][i] 
    for i in range(n1-m):
        result[unseed1[i]] = unseed2[col[i]]
        
    return result
