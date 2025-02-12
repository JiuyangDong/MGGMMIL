import torch
import pandas as pd
import os
import itertools
import logging
import json
import itertools
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler, Dataset
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F 
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append('..')
from tqdm import tqdm
from torch.nn.modules.loss import _WeightedLoss
import glob

class WSIDataset(Dataset):
    def __init__(self, args, infold_cases, phase=None):
        self.args = args
        self.phase = phase
        self.infold_features = []
        KNNs = [int(K) for K in args.KNN.split('+')]
        self.adjacent_matrix = {K: [] for K in KNNs}
        self.edge_list = {K: [] for K in KNNs}

        for case in infold_cases:
            for patho_type in ['A', 'T']:
                path = os.path.join(args.data_dir, f'feats/{args.pretrain}/{args.wsi_type}_{patho_type}/{case}.pt')
                if os.path.exists(path):
                    self.infold_features.append(path)
                    KNNs = [int(K) for K in args.KNN.split('+')]
                    for K in KNNs:
                        self.adjacent_matrix[K].append(os.path.join(args.data_dir, f'adjacent_matrix/K={K}/{args.wsi_type}_{patho_type}/{case}.pt'))
                        self.edge_list[K].append(os.path.join(args.data_dir, f'edge_list/K={K}/{args.wsi_type}_{patho_type}/{case}.pt'))

    def __len__(self):
        return len(self.infold_features)
        
    def __getitem__(self, index):
        path = self.infold_features[index]


        if f'{self.args.wsi_type}_T' in path:
            label = 1
        else:
            label = 0
        
        fea = torch.load(path) 
        KNNs = [int(K) for K in self.args.KNN.split('+')]
        adjacent_matrixs, edge_lists = [], []
        for K in KNNs:
            edge_list = torch.load(self.edge_list[K][index])
            edge_lists.append(edge_list)

        return fea, label, path, adjacent_matrixs, edge_lists        

class MIL:
    def __init__(self, args):
        self.args = args

        self.train_loader, self.valid_loader, self.test_loader = self.init_data_wsi()

        self.model = self.init_model(args.baseline_model)
        print(self.model)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params / 1e6} M")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.baseline_lr, weight_decay=args.baseline_wd)

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.counter = 0
        self.patience = 20
        self.stop_epoch = 50
        self.best_loss = np.Inf
        self.flag = 1
        self.ckpt_name = os.path.join(self.args.ckpt_dir, 'best_epoch.pth')
        
        self.best_valid_metrics = None 
        
   

    def init_data_wsi(self):
        train_cases, valid_cases, test_cases = [], [], []

        with open(os.path.join(self.args.data_dir, f'splits/10-fold-100%-label/split_{self.args.k}_train.txt'), 'r') as f:
            for line in f.readlines():
                case = line.rstrip()
                train_cases.append(case)
        
        with open(os.path.join(self.args.data_dir, f'splits/10-fold-100%-label/split_{self.args.k}_valid.txt'), 'r') as f:
            for line in f.readlines():
                case = line.rstrip()
                valid_cases.append(case)

        with open(os.path.join(self.args.data_dir, f'splits/10-fold-100%-label/split_{self.args.k}_test.txt'), 'r') as f:
            for line in f.readlines():
                case = line.rstrip()
                test_cases.append(case)

        train_set = WSIDataset(self.args, train_cases, 'train')
        valid_set = WSIDataset(self.args, valid_cases, 'valid')
        test_set = WSIDataset(self.args, test_cases, 'test')
        
        logging.info("Case/WSI number for trainset = {}/{}".format(len(train_cases), len(train_set)))
        logging.info("Case/WSI number for validset = {}/{}".format(len(valid_cases), len(valid_set)))
        logging.info("Case/WSI number for testset = {}/{}".format(len(test_cases), len(test_set)))

        
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        return train_loader, valid_loader, test_loader



    def init_model(self, model_type):
        from models.mggm_mil import MGGMMIL
        model = MGGMMIL(args=self.args, I=self.args.feature_dim, L=self.args.L, D=self.args.D, n_classes=self.args.n_classes, layer=self.args.layer, mil_head=self.args.mil_head).to(self.args.device)

        return model

    def train(self):
        step = 0
        

        for epoch in range(1, self.args.n_epochs + 1):
            avg_train_loss = 0

            self.model.train()
            for i, (fea, label, path, adj, edge) in enumerate(tqdm(self.train_loader)):
                step += 1
                fea, label = fea.to(self.args.device), label.to(self.args.device)

                self.optimizer.zero_grad()
                loss = self.train_inference(fea, label, self.args.baseline_model, [a.to(self.args.device) for a in adj], [e.to(self.args.device) for e in edge])
                avg_train_loss += loss.item()                    

                loss.backward()

                self.optimizer.step()

            avg_train_loss /= (i + 1)

            logging.info("In step {} (epoch {}), average train loss = {:.4f}".format(step, epoch, avg_train_loss))
            
            self.valid(epoch)

            if self.flag == -1:
                break

        return self.best_valid_metrics

    def valid(self, epoch):
        avg_loss = 0
        self.model.eval()

        labels, probs = [], []

        for i, (fea, label, path, adj, edge) in enumerate(tqdm(self.valid_loader)):
            fea, label = fea.to(self.args.device), label.to(self.args.device)
            with torch.no_grad():
                loss, y_prob = self.test_inference(fea, label, self.args.baseline_model, [a.to(self.args.device) for a in adj], [e.to(self.args.device) for e in edge])

            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()
        avg_loss /= (i + 1)

        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info("loss = {:.4f}, auc = {:.4f}, acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".\
            format(avg_loss, auc, acc, precision, recall, f1))

        if epoch >= self.stop_epoch:
            if avg_loss < self.best_loss:
                self.counter = 0
                logging.info(f'Validation loss decreased ({self.best_loss:.4f} --> {avg_loss:.4f}).  Saving model ...')
                torch.save(self.model.state_dict(), self.ckpt_name)
                self.best_loss = avg_loss
                self.best_valid_metrics = [avg_loss, auc, acc, precision, recall, f1]
            else:
                self.counter += 1
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.flag = -1

    def test(self):
        avg_loss = 0
        self.model.load_state_dict(torch.load(self.ckpt_name))
        self.model.eval()

        labels, probs = [], []

        for i, (fea, label, path, adj, edge) in enumerate(tqdm(self.test_loader)):
            fea, label = fea.to(self.args.device), label.to(self.args.device)
            with torch.no_grad():
                loss, y_prob = self.test_inference(fea, label, self.args.baseline_model, [a.to(self.args.device) for a in adj], [e.to(self.args.device) for e in edge])

            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()
        avg_loss /= (i + 1)

        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info("loss = {:.4f}, auc = {:.4f}, acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".\
            format(avg_loss, auc, acc, precision, recall, f1))

        return avg_loss, auc, acc, precision, recall, f1

    def train_inference(self, fea, label, model_type, adj=None, edge=None):
        bag_logit = self.model(fea, edge)[0]

        loss = self.loss(bag_logit,label)
        return loss

    def test_inference(self, fea, label, model_type, adj=None, edge=None):
        bag_logit = self.model(fea, edge)[0]
        loss = self.loss(bag_logit, label)
        y_prob = F.softmax(bag_logit, dim=1)

        return loss, y_prob

    def cal_AUC(self, probs, labels, nclasses):
        '''
            probs(softmaxed): ndarray, [N, nclass] 
            labels(inte number): ndarray, [N, 1] 
        '''
        if nclasses == 2:
            auc_score = roc_auc_score(labels, probs[:, 1])
            return auc_score
        else:
            raise NotImplementedError

        
    

    def cal_ACC(self, probs, labels, nclasses):
        '''
            probs(softmaxed): ndarray, [N, nclass] 
            labels(inte number): ndarray, [N, 1] 
        '''
        log = [{"count": 0, "correct": 0} for i in range(nclasses)]
        pred_hat = np.argmax(probs, 1)
        labels = labels.astype(np.int32)

        if nclasses == 2:
            acc_score = accuracy_score(labels, pred_hat)
            precision = precision_score(labels, pred_hat, average='binary')
            recall = recall_score(labels, pred_hat, average='binary')
            f1 = f1_score(labels, pred_hat, average='binary')

            return acc_score, log, precision, recall, f1

        else:
            raise NotImplementedError
