#!/usr/bin/env python
# coding: utf-8
import torch_geometric.utils as tg
import networkx as nx
from utils import *
from tqdm import tqdm
import argparse
import scipy
import multiprocessing as mp
import torch_geometric

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

'''
Dataset arguments
'''
parser.add_argument('--dataset', type=str, default='cora',
                                help='Dataset name: Cora/CiteSeer/PubMed/Reddit')
parser.add_argument('--nhid', type=int, default=256,
                                help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 100,
                               help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default= 10,
                                help='Number of Pool')
parser.add_argument('--batch_num', type=int, default= 10,
                                help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512,
                                help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5,
                                help='Number of GCN layers')
parser.add_argument('--n_iters', type=int, default=1,
                                help='Number of iteration to run on a batch')
parser.add_argument('--n_stops', type=int, default=200,
                                help='Stop after number of batches that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=64,
                                help='Number of sampled nodes per layer')
parser.add_argument('--sample_method', type=str, default='ladies',
                               help='Sampled Algorithms: ladies/fastgcn/full')
parser.add_argument('--cuda', type=int, default=0,
                                help='Avaiable GPU ID')

args = parser.parse_args()
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
                
print(args.dataset, args.sample_method)
edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = load_data(args.dataset)

adj_matrix = get_adj(edges, feat_data.shape[0])

#adj_torch = tg.from_scipy_sparse_matrix(adj_matrix)

adj_torch =torch_geometric.data.Data(edge_index = torch.tensor([edges[:,0], edges[:,1]]))

adj_nx = tg.to_networkx(adj_torch)

eigen_cen = list(nx.eigenvector_centrality(adj_nx).values())
bet_cen = list(nx.betweenness_centrality(adj_nx).values())
clos_cen = list(nx.closeness_centrality(adj_nx).values())


lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
if type(feat_data) == scipy.sparse.lil.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device) 
else:
    feat_data = torch.FloatTensor(feat_data).to(device)
labels    = torch.LongTensor(labels).to(device) 

def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    pass

if args.sample_method == 'ladies':
    sampler = ladies_sampler
elif args.sample_method == 'fastgcn':
    sampler = fastgcn_sampler
elif args.sample_method == 'full':
    sampler = default_sampler

