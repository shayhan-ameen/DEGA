import os
import json
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
from GraphAugmentor import Augmentor
import torch
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0

    tvt_nids = pickle.load(open(f'graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'graphs/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'graphs/{args.dataset}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())

    params_all = json.load(open('best_parameters.json', 'r'))
    params = params_all[args.dataset][args.gnn]

    gnn = args.gnn
    layer_type = args.gnn
    jk = False
    if gnn == 'jknet':
        layer_type = 'gsage'
        gnn = 'gsage'
        jk = True
    feat_norm = 'row'
    if args.dataset == 'ppi':
        feat_norm = 'col'
    elif args.dataset in ('blogcatalog', 'flickr'):
        feat_norm = 'none'
    lr = 0.005 if layer_type == 'gat' else 0.01
    n_layers = 1
    if jk:
        n_layers = 3


    
    max_epoch=500
    epoch_iter = tqdm(range(max_epoch))
    accs = []
    for epoch in epoch_iter:
        model = Augmentor(adj_orig, features, labels, tvt_nids, cuda=gpu, gae=True, alpha=params['alpha'], beta=params['beta'], temperature=params['temp'], warmup=0, gnnlayer_type=gnn, jknet=jk, lr=lr, n_layers=n_layers, log=False, feat_norm=feat_norm)
        acc = model.fit(thresholds=params['thresholds'] , pretrain_ep=params['pretrain_ep'], pretrain_nc=params['pretrain_nc'])
        accs.append(acc)
        epoch_iter.set_description(f"# Epoch {epoch}: train_accuracy: {acc:.4f}")

    print(f'ALL Max F1: {np.max(accs)*100:.2f}, Micro F1: {np.mean(accs)*100:.2f}, Min F1: {np.min(accs)*100:.2f}, std: {np.std(accs):.6f}')
    accs.sort()
    accs= accs[-5:]
    print(f'Top 5 Max F1: {np.max(accs)*100:.2f}, Micro F1: {np.mean(accs)*100:.2f}, Min F1: {np.min(accs)*100:.2f}, std: {np.std(accs):.6f}')