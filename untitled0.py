import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils import *
import networkx as nx

parser = argparse.ArgumentParser(description = 'training GCN on Cora/Citeseer/Pubmed/Reddit datasets')

parser.add_argument('--dataset', type=str, default = 'Cora', help = 'Dataset name: Cora/Citeseer/Pubmed/Reddit')
parser.add_argument('--nhid', type=int, default = 256, help = 'Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default=100, help='Number of epochs')
parser.add_argument('--pool_num', type=int, default=1, help='Number of pool')
parser.add_argument('--batch_num', type=int, default=10, help='Maximum number of batch')
parser.add_argument('--batch_size', type=int, default=512, help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5, help='Number of GCN layers')
parser.add_argument('--n_stops', type=int, default=200, help='Stop after number of batches that f1 dont increases')
parser.add_argument('--samp_num', type=int, default=64, help='Number of sampled nodes per layer')
parser.add_argument('--sample_method', type=str, default='ladies', help='Sampled Algorithms: fastgcm/ladies/full')
parser.add_argument('--cude', type=int, default=0, help='available GPU ID')
args = parser.parse_args()


class GraphConvolution(nn.Module):    
    def __init__(self, n_in, n_out, bias=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)       
        
    def forward(self, x, adj):
        out = self.linear(x)
        return F.elu(torch.spmm(adj, out))



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout):
        super().__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat, nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid, nhid))
    
    def forward(self, x, adjs):
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx]))
        return x
        

class SuGCN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super().__init__()
        self.encode = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs):
        x = self.encode(feat, adjs)
        x = self.dropout(x)
        x = self.linear(x)
        return x


def fastgcn_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs = []
    
    
def ladies(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs = []
    
    for d in range(depth):
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = load_data(args.dataset)
adj_matrix = get_adj(edges, feat_data.shape[0])
lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))

print(adj_matrix)