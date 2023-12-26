import gc
import logging
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from sklearn.metrics import roc_auc_score, average_precision_score
from models import Dual_channel_auto_encoder, GNN, GNN_JK, GCNLayer, SAGELayer, GATLayer
from utils import MultipleOptimizer, RoundNoGradient, CeilNoGradient, scipysp_to_pytorchsp


class Augmentor(object):
    def __init__(self, adj_matrix, features, labels, tvt_nids, cuda=-1, hidden_size=128, emb_size=32, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, gae=False, beta=0.5, temperature=0.2, log=True, name='debug', warmup=3, gnnlayer_type='gcn', jknet=False, alpha=1, sample_type='add_sample', feat_norm='row'):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = epochs
        self.gae = gae
        self.beta = beta
        self.warmup = warmup
        self.feat_norm = feat_norm
        # create a logger, logs are saved to Augmentor-[name].log when name is not None
        if log:
            self.logger = self.get_logger(name)
        else:
            # disable logger if wanted
            # logging.disable(logging.CRITICAL)
            self.logger = logging.getLogger()
        # config device (force device to cpu when cuda is not available)
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda}' if cuda>=0 else 'cpu')
        # log all parameters to keep record
        all_vars = locals()
        self.log_parameters(all_vars)
        # fix random seeds if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # load data
        self.load_data(adj_matrix, features, labels, tvt_nids, gnnlayer_type)
        # setup the model
        self.model = Augmentor_model(self.features.size(1),
                                hidden_size,
                                emb_size,
                                self.out_size,
                                n_layers,
                                F.relu,
                                dropout,
                                self.device,
                                gnnlayer_type,
                                self.adj_orig.size(0), #dim_adj= self.adj_orig.size(0)
                                temperature=temperature,
                                gae=gae,
                                jknet=jknet,
                                alpha=alpha,
                                sample_type=sample_type)

    def load_data(self, adj_matrix, features, labels, tvt_nids, gnnlayer_type):
        """ preprocess data """
        # features (torch.FloatTensor)
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        # normalize feature matrix if needed
        if self.feat_norm == 'row':
            self.features = F.normalize(self.features, p=1, dim=1)
        elif self.feat_norm == 'col':
            self.features = self.col_normalization(self.features)
        # original adj_matrix for training auto encoder (torch.FloatTensor)
        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        adj_matrix.setdiag(1)
        self.adj_orig = scipysp_to_pytorchsp(adj_matrix).to_dense()
        # normalized adj_matrix used as input for ep_net (torch.sparse.FloatTensor)
        degrees = np.array(adj_matrix.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        self.adj_norm = scipysp_to_pytorchsp(adj_norm)
        # adj_matrix used as input for nc_net (torch.sparse.FloatTensor)
        if gnnlayer_type == 'gcn':
            self.adj = scipysp_to_pytorchsp(adj_norm)
        elif gnnlayer_type == 'gsage':
            adj_matrix_noselfloop = sp.coo_matrix(adj_matrix)
            # adj_matrix_noselfloop.setdiag(0)
            # adj_matrix_noselfloop.eliminate_zeros()
            adj_matrix_noselfloop = sp.coo_matrix(adj_matrix_noselfloop / adj_matrix_noselfloop.sum(1))
            self.adj = scipysp_to_pytorchsp(adj_matrix_noselfloop)
        elif gnnlayer_type == 'gat':
            # self.adj = scipysp_to_pytorchsp(adj_matrix)
            self.adj = torch.FloatTensor(adj_matrix.todense())
        # labels (torch.LongTensor) and train/validation/test nids (np.ndarray)
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]
        # number of classes
        if len(self.labels.size()) == 1:
            self.out_size = len(torch.unique(self.labels))
        else:
            self.out_size = labels.size(1)
        # sample the edges to evaluate edge prediction results
        # sample 10% (1% for large graph) of the edges and the same number of no-edges
        if labels.size(0) > 5000:
            edge_frac = 0.01
        else:
            edge_frac = 0.1
        adj_matrix = sp.csr_matrix(adj_matrix)
        n_edges_sample = int(edge_frac * adj_matrix.nnz / 2)
        # sample negative edges
        neg_edges = []
        added_edges = set()
        while len(neg_edges) < n_edges_sample:
            i = np.random.randint(0, adj_matrix.shape[0])
            j = np.random.randint(0, adj_matrix.shape[0])
            if i == j:
                continue
            if adj_matrix[i, j] > 0:
                continue
            if (i, j) in added_edges:
                continue
            neg_edges.append([i, j])
            added_edges.add((i, j))
            added_edges.add((j, i))
        neg_edges = np.asarray(neg_edges)
        # sample positive edges
        nz_upper = np.array(sp.triu(adj_matrix, k=1).nonzero()).T
        np.random.shuffle(nz_upper)
        pos_edges = nz_upper[:n_edges_sample]
        self.val_edges = np.concatenate((pos_edges, neg_edges), axis=0)
        self.edge_labels = np.array([1]*n_edges_sample + [0]*n_edges_sample)

    def pretrain_ep_net(self, model, adj, features, adj_orig, norm_w, pos_weight, n_epochs):
        """ pretrain the edge prediction network """
        optimizer = torch.optim.Adam(model.ep_net.parameters(),
                                     lr=self.lr)
        model.train()
        for epoch in range(n_epochs):
            adj_logits = model.ep_net(adj, features)
            loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
            self.logger.info('EPNet pretrain, Epoch [{:3}/{}]: loss {:.4f}, auc {:.4f}, ap {:.4f}'
                        .format(epoch+1, n_epochs, loss.item(), ep_auc, ep_ap))

    def pretrain_nc_net(self, model, adj, features, labels, n_epochs):
        """ pretrain the node classification network """
        optimizer = torch.optim.Adam(model.nc_net.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.
        for epoch in range(n_epochs):
            model.train()
            nc_logits = model.nc_net(adj, features)
            # losses
            loss = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(adj, features)
            val_acc = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
                self.logger.info('NCNet pretrain, Epoch [{:2}/{}]: loss {:.4f}, val acc {:.4f}, test acc {:.4f}'
                            .format(epoch+1, n_epochs, loss.item(), val_acc, test_acc))
            else:
                self.logger.info('NCNet pretrain, Epoch [{:2}/{}]: loss {:.4f}, val acc {:.4f}'
                            .format(epoch+1, n_epochs, loss.item(), val_acc))

    def fit(self, thresholds, pretrain_ep=200, pretrain_nc=20):
        """ train the model """
        # move data to device
        adj_norm = self.adj_norm.to(self.device)
        adj = self.adj.to(self.device)
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        adj_orig = self.adj_orig.to(self.device)
        model = self.model.to(self.device)
        # weights for log_lik loss when training EP net
        adj_t = self.adj_orig
        norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
        pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(self.device)
        # pretrain auto encoder if needed
        if pretrain_ep:
            self.pretrain_ep_net(model, adj_norm, features, adj_orig, norm_w, pos_weight, pretrain_ep)
        # pretrain GCN if needed
        if pretrain_nc:
            self.pretrain_nc_net(model, adj, features, labels, pretrain_nc)
        # optimizers
        optims = MultipleOptimizer(torch.optim.Adam(model.ep_net.parameters(),
                                                    lr=self.lr),
                                   torch.optim.Adam(model.nc_net.parameters(),
                                                    lr=self.lr,
                                                    weight_decay=self.weight_decay))
        # get the learning rate schedule for the optimizer of ep_net if needed
        if self.warmup:
            ep_lr_schedule = self.get_lr_schedule_by_sigmoid(self.n_epochs, self.lr, self.warmup)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()
        # keep record of the best validation accuracy for early stopping
        best_val_acc = 0.
        patience_step = 0
        # train model
        for epoch in range(self.n_epochs):
            # update the learning rate for ep_net if needed
            if self.warmup:
                optims.update_lr(0, ep_lr_schedule[epoch])

            model.train()
            nc_logits, adj_logits = model(adj_norm, adj_orig, features, thresholds)

            # losses
            loss = nc_loss = nc_criterion(nc_logits[self.train_nid], labels[self.train_nid])
            ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
            loss += self.beta * ep_loss
            optims.zero_grad()
            loss.backward()
            optims.step()
            # validate (without dropout)
            model.eval()
            with torch.no_grad():
                nc_logits_eval = model.nc_net(adj, features)
            val_acc = self.eval_node_cls(nc_logits_eval[self.val_nid], labels[self.val_nid])
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = self.eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
                self.logger.info('Epoch [{:3}/{}]: ep loss {:.4f}, nc loss {:.4f}, ep auc: {:.4f}, ep ap {:.4f}, val acc {:.4f}, test acc {:.4f}'
                            .format(epoch+1, self.n_epochs, ep_loss.item(), nc_loss.item(), ep_auc, ep_ap, val_acc, test_acc))
                patience_step = 0
            else:
                self.logger.info('Epoch [{:3}/{}]: ep loss {:.4f}, nc loss {:.4f}, ep auc: {:.4f}, ep ap {:.4f}, val acc {:.4f}'
                            .format(epoch+1, self.n_epochs, ep_loss.item(), nc_loss.item(), ep_auc, ep_ap, val_acc))
                patience_step += 1
                if patience_step == 100:
                    self.logger.info('Early stop!')
                    break
        # get final test result without early stop
        with torch.no_grad():
            nc_logits_eval = model.nc_net(adj, features)
        test_acc_final = self.eval_node_cls(nc_logits_eval[self.test_nid], labels[self.test_nid])
        # log both results
        self.logger.info('Final test acc with early stop: {:.4f}, without early stop: {:.4f}'
                    .format(test_acc, test_acc_final))
        # release RAM and GPU memory
        del adj, features, labels, adj_orig
        torch.cuda.empty_cache()
        gc.collect()
        return test_acc

    def log_parameters(self, all_vars):
        """ log all variables in the input dict excluding the following ones """
        del all_vars['self']
        del all_vars['adj_matrix']
        del all_vars['features']
        del all_vars['labels']
        del all_vars['tvt_nids']
        self.logger.info(f'Parameters: {all_vars}')

    @staticmethod
    def eval_edge_pred(adj_pred, val_edges, edge_labels):
        logits = adj_pred[val_edges.T]
        logits = np.nan_to_num(logits)
        roc_auc = roc_auc_score(edge_labels, logits)
        ap_score = average_precision_score(edge_labels, logits)
        return roc_auc, ap_score

    @staticmethod
    def eval_node_cls(nc_logits, labels):
        """ evaluate node classification results """
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(nc_logits))
            tp = len(torch.nonzero(preds * labels))
            tn = len(torch.nonzero((1-preds) * (1-labels)))
            fp = len(torch.nonzero(preds * (1-labels)))
            fn = len(torch.nonzero((1-preds) * labels))
            pre, rec, f1 = 0., 0., 0.
            if tp+fp > 0:
                pre = tp / (tp + fp)
            if tp+fn > 0:
                rec = tp / (tp + fn)
            if pre+rec > 0:
                fmeasure = (2 * pre * rec) / (pre + rec)
        else:
            preds = torch.argmax(nc_logits, dim=1)
            correct = torch.sum(preds == labels)
            fmeasure = correct.item() / len(labels)
        return fmeasure

    @staticmethod
    def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
        """ schedule the learning rate with the sigmoid function.
        The learning rate will start with near zero and end with near lr """
        factors = torch.FloatTensor(np.arange(n_epochs))
        factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
        factors = torch.sigmoid(factors)
        # range the factors to [0, 1]
        factors = (factors - factors[0]) / (factors[-1] - factors[0])
        lr_schedule = factors * lr
        return lr_schedule

    @staticmethod
    def get_logger(name):
        """ create a nice logger """
        logger = logging.getLogger(name)
        # clear handlers if they were created in other runs
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        # create console handler add add to logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler add add to logger when name is not None
        if name is not None:
            fh = logging.FileHandler(f'Augmentor-{name}.log')
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        return logger

    @staticmethod
    def col_normalization(features):
        """ column normalization for feature matrix """
        features = features.numpy()
        m = features.mean(axis=0)
        s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
        features -= m
        features /= s
        return torch.FloatTensor(features)


class Augmentor_model(nn.Module):
    def __init__(self,
                 dim_feats,
                 dim_h,
                 dim_z,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 device,
                 gnnlayer_type,
                 dim_adj,
                 temperature=1,
                 gae=False,
                 jknet=False,
                 alpha=1,
                 sample_type='add_sample'):
        super(Augmentor_model, self).__init__()
        self.device = device
        self.temperature = temperature
        self.gnnlayer_type = gnnlayer_type
        self.alpha = alpha
        self.sample_type=sample_type
        # edge prediction network
        self.ep_net = Dual_channel_auto_encoder(dim_feats, dim_h, dim_z, dim_adj, activation, gae=gae)
        # node classification network
        if jknet:
            self.nc_net = GNN_JK(dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type=gnnlayer_type)
        else:
            self.nc_net = GNN(dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type=gnnlayer_type)

    def dual_thresholding(self, adj_logits, adj_orig, alpha, thresholds):
        th1= thresholds[0]
        th2= thresholds[1]
        
        edge_probs = adj_logits / torch.max(adj_logits)
        edge_dif = torch.abs(edge_probs-adj_orig)
        rho = thresholds[0]-thresholds[1]

        edge_probs= torch.where((th1>edge_dif) & (edge_dif>th2), (1-rho)*edge_probs + rho*adj_orig, edge_probs) 
        edge_probs= torch.where(edge_dif>=th1, adj_orig, edge_probs) 

        # edge_probs= torch.where((th1>edge_dif) & (edge_dif>th2), (1-alpha)*edge_probs + alpha*adj_orig, edge_probs) 
        # edge_probs= torch.where(edge_dif>=th1, adj_orig, edge_probs)       
        
        #edge_probs = adj_logits / torch.max(adj_logits)
        #edge_probs = alpha*edge_probs + (1-alpha)*adj_orig
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def normalize_adj(self, adj):
        if self.gnnlayer_type == 'gcn':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
            D_norm = torch.diag(torch.pow(adj.sum(1), -0.5)).to(self.device)
            adj = D_norm @ adj @ D_norm
        elif self.gnnlayer_type == 'gat':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
        elif self.gnnlayer_type == 'gsage':
            # adj = adj + torch.diag(torch.ones(adj.size(0))).to(self.device)
            adj.fill_diagonal_(1)
            adj = F.normalize(adj, p=1, dim=1)
        return adj

    def forward(self, adj, adj_orig, features, thresholds):
        adj_logits = self.ep_net(adj, features)
        adj_new = self.dual_thresholding(adj_logits, adj_orig, self.alpha, thresholds)
        adj_new_normed = self.normalize_adj(adj_new)
        nc_logits = self.nc_net(adj_new_normed, features)
        return nc_logits, adj_logits

