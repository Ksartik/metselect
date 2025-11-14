import torch
from copy import deepcopy
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from utils.losses import Loss
import itertools
import time
from torch.optim import lr_scheduler


class Trainer:
    def __init__(self, scope_selection, model_parameters, dataset, train_config, model_config, class_weights, device):
        self.scope_selection = scope_selection
        self.num_layers = len(model_config.selected_layers) if 'selected_layers' in model_config and model_config.selected_layers is not None else model_config.num_layers

        #running average of the variances
        self.state_dict = None
        self.variance_max = None
        self.nvars_max = None
        self.max_val_metric = 0
        self.n_vars = 1
        
        self.es_patience = train_config.es_patience
        self.es_min_delta = train_config.es_min_delta
        self.es_counter = 0
        self.last_validation_loss = np.inf
        self.last_training_loss = np.inf
        self.min_loss_change = train_config.min_loss_change
        self.change_counter = 0
        self.max_validation_loss = np.inf
        
        self.device = device

        if train_config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam (model_parameters, lr=train_config.lr, weight_decay=train_config.weight_decay)
        elif train_config.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD (model_parameters, lr=train_config.lr, weight_decay=train_config.weight_decay)
        elif train_config.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop (model_parameters, lr=train_config.lr, weight_decay=train_config.weight_decay)
        
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, 
        #                                    patience=20, min_lr=0.00001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        # self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        # self.scheduler = lr_scheduler.ConstantLR(self.optimizer, factor=1, total_iters=500)
        
        self.criterion = Loss (dataset=dataset, fn_name=train_config.loss_fn)
            
        num_clusters = model_config['cluster_params']['num_clusters']
        self.num_clusters = num_clusters if type(num_clusters) is list else [num_clusters for _ in range(dataset.num_classes)]
        self.decoding = model_config.decoding

    def early_stop(self, model, training_loss, validation_loss):
        if validation_loss < self.max_validation_loss:
            self.max_validation_loss = validation_loss
            self.variance_max = model.vars.clone()
            self.nvars_max = deepcopy(self.n_vars)
            self.state_dict = deepcopy(model.state_dict())
            self.protos = model.protos.clone() if model.protos is not None else None
            self.es_counter = 0
        elif validation_loss >= self.max_validation_loss + self.es_min_delta:
            self.es_counter += 1
            if self.es_counter >= self.es_patience:
                return True
        if ((training_loss - self.last_training_loss).abs() <= self.min_loss_change):
            self.change_counter += 1
            if self.change_counter >= self.es_patience:
                return True
        else:
            self.change_counter = 0
        self.last_validation_loss = validation_loss
        self.last_training_loss = training_loss
        return False

    def store_best_params(self, model, val_metric):
        if val_metric > self.max_val_metric:
            if self.decoding == 'distance' or self.scope_selection.startswith('metscope'):
                self.variance_max = model.vars.clone()
                self.nvars_max = deepcopy(self.n_vars)
                self.protos = model.protos.clone() if model.protos is not None else None
            self.max_val_metric = val_metric
            self.state_dict = deepcopy(model.state_dict())
    
    def model_forward_and_loss (self, model, data, idx=None, **kwargs):
        if 'adj_t' in data.keys():
            loss = model.forward_and_loss(x=data.x, adj_t=data.adj_t, y=data.y, idx=idx, **kwargs)
        elif 'edge_index' in data.keys():
            loss = model.forward_and_loss(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, 
                                          edge_attr=data.edge_attr, y=data.y, idx=idx, **kwargs)
        else:
            loss = model.forward_and_loss(x=data.x, y=data.y, idx=idx, **kwargs)
        return loss

    def model_forward_and_predict (self, model, data, idx=None, **kwargs):
        if 'adj_t' in data.keys():
            y_hat = model.forward_and_predict(x=data.x, adj_t=data.adj_t, y=data.y, idx=idx, **kwargs)
        elif 'edge_index' in data.keys():
            y_hat = model.forward_and_predict(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, 
                                                    edge_attr=data.edge_attr, y=data.y, idx=idx, **kwargs)
        else:
            y_hat = model.forward_and_predict(x=data.x, y=data.y, idx=idx, **kwargs)
        return y_hat

    def train_data (self, model, data):
        model.train()
        self.optimizer.zero_grad()
        loss = self.model_forward_and_loss(model, data, idx=data.train_idx, criterion=self.criterion)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss #, y_hat, y

    def set_variance (self, model, data):
        z, z_layers = self.model_forward (model, data)
        z = z[data.train_idx]
        z_layers = z_layers[data.train_idx]
        y = data.y[data.train_idx]
        _, var = model.predict_distance(z, z_layers=z_layers, y=y)
        self.variance = var
        self.n_vars = 1
        
    def test_data (self, model, data, big_data=False):
        model.eval()
        with torch.no_grad():
            y_hat = torch.zeros(data.num_nodes, data.num_classes).to(self.device)
            y_hat[data.train_idx] = self.model_forward_and_predict(model, data, idx=data.train_idx)
            y_hat[data.val_idx] = self.model_forward_and_predict(model, data, idx=data.val_idx)
            y_hat[data.test_idx] = self.model_forward_and_predict(model, data, idx=data.test_idx)
            train_loss = self.criterion(y_hat[data.train_idx], data.y[data.train_idx], evaluate=True) 
            val_loss = self.criterion(y_hat[data.val_idx], data.y[data.val_idx], evaluate=True) 
            test_loss = self.criterion(y_hat[data.test_idx], data.y[data.test_idx], evaluate=True)
            # self.scheduler.step(val_loss)
            return y_hat, train_loss, val_loss, test_loss
        
    def test_data2 (self, model, data, big_data=False):
        model.eval()
        with torch.no_grad():
            z, z_layers = self.model_forward (model, data)
            y_hat = torch.zeros(data.num_nodes, data.num_classes).to(self.device)
            if 'metscope' in self.scope_selection:
                if big_data:
                    norm_dists, _ = model.predict_distance(z[data.train_idx], z_layers=z_layers[data.train_idx], 
                                                            var=model.vars/self.n_vars)
                    y_hat[data.train_idx] = model.predict_proba(norm_dists)
                    norm_dists, _ = model.predict_distance(z[data.val_idx], z_layers=z_layers[data.val_idx], 
                                                            var=model.vars/self.n_vars)
                    y_hat[data.val_idx] = model.predict_proba(norm_dists)
                    norm_dists, _ = model.predict_distance(z[data.test_idx], z_layers=z_layers[data.test_idx], 
                                                            var=model.vars/self.n_vars)
                    y_hat[data.test_idx] = model.predict_proba(norm_dists)
                else:
                    norm_dists, _ = model.predict_distance(z, z_layers=z_layers, var=model.vars/self.n_vars)
                    y_hat = model.predict_proba(norm_dists)
                train_loss = self.criterion(y_hat[data.train_idx], data.y[data.train_idx], evaluate=True) 
                val_loss = self.criterion(y_hat[data.val_idx], data.y[data.val_idx], evaluate=True) 
                test_loss = self.criterion(y_hat[data.test_idx], data.y[data.test_idx], evaluate=True) 
            else:
                y_hat = model.predict_proba(z)
                train_loss = self.criterion(y_hat[data.train_idx], data.y[data.train_idx], evaluate=True)
                val_loss = self.criterion(y_hat[data.val_idx], data.y[data.val_idx], evaluate=True) 
                test_loss = self.criterion(y_hat[data.test_idx], data.y[data.test_idx], evaluate=True)
            return y_hat, train_loss, val_loss, test_loss

    def test_data_closest (self, model, data, big_data=False):
        model.eval()
        with torch.no_grad():
            z, z_layers = self.model_forward (model, data)
            y_hat = torch.zeros(data.num_nodes, data.num_classes).to(self.device)
            norm_dists = torch.zeros (data.num_nodes, data.num_classes, z_layers.shape[1]).to(self.device)
            if big_data:
                norm_dists[data.train_idx], _ = model.predict_distance(z[data.train_idx], z_layers=z_layers[data.train_idx], 
                                                        var=self.variance/self.n_vars)
                y_hat[data.train_idx] = model.predict_proba(norm_dists[data.train_idx])
                norm_dists[data.val_idx], _ = model.predict_distance(z[data.val_idx], z_layers=z_layers[data.val_idx], 
                                                        var=self.variance/self.n_vars)
                y_hat[data.val_idx] = model.predict_proba(norm_dists[data.val_idx])
                norm_dists[data.test_idx], _ = model.predict_distance(z[data.test_idx], z_layers=z_layers[data.test_idx], 
                                                        var=self.variance/self.n_vars)
                y_hat[data.test_idx] = model.predict_proba(norm_dists[data.test_idx])
            else:
                norm_dists, _ = model.predict_distance(z, z_layers=z_layers, var=self.variance/self.n_vars)
                y_hat = model.predict_proba(norm_dists)
            pred_classes = y_hat.argmax(dim=-1)
            pred_idx = torch.arange(len(pred_classes)).to(pred_classes.device)
            min_layers = norm_dists[pred_idx, pred_classes, :].argmin(dim=-1)
            min_dists = norm_dists[pred_idx, :, min_layers]
            y_hat = model.predict_proba(min_dists[:, :, None])
            return min_dists, min_layers, y_hat
    
    def test_data_counterf_min (self, model, data, big_data=False):
        model.eval()
        with torch.no_grad():
            z, z_layers = self.model_forward (model, data)
            y_hat = torch.zeros(data.num_nodes, data.num_classes).to(self.device)
            norm_dists = torch.zeros (data.num_nodes, data.num_classes, z_layers.shape[1]).to(self.device)
            if big_data:
                norm_dists[data.train_idx], _ = model.predict_distance(z[data.train_idx], z_layers=z_layers[data.train_idx], 
                                                        var=self.variance/self.n_vars)
                y_hat[data.train_idx] = model.predict_proba(norm_dists[data.train_idx])
                norm_dists[data.val_idx], _ = model.predict_distance(z[data.val_idx], z_layers=z_layers[data.val_idx], 
                                                        var=self.variance/self.n_vars)
                y_hat[data.val_idx] = model.predict_proba(norm_dists[data.val_idx])
                norm_dists[data.test_idx], _ = model.predict_distance(z[data.test_idx], z_layers=z_layers[data.test_idx], 
                                                        var=self.variance/self.n_vars)
                y_hat[data.test_idx] = model.predict_proba(norm_dists[data.test_idx])
            else:
                norm_dists, _ = model.predict_distance(z, z_layers=z_layers, var=self.variance/self.n_vars)
                y_hat = model.predict_proba(norm_dists)
            pred_classes = y_hat.argmax(dim=-1)
            pred_idx = torch.arange(len(pred_classes)).to(pred_classes.device)
            min_layers = norm_dists[pred_idx, pred_classes, :].argmin(dim=-1)
            mask = torch.ones_like (norm_dists, dtype=bool)
            B, L, K = norm_dists.shape
            mask[pred_idx, :, min_layers] = 0
            nonmin_dists = norm_dists[mask].reshape(B, L, K-1)
            y_hat = model.predict_proba(nonmin_dists)
            return y_hat

    def get_dists (self, model, data):
        model.eval()
        with torch.no_grad():
            z, z_layers = self.model_forward (model, data)
            # train
            norm_dists, _ = model.predict_distance(z, z_layers=z_layers, var=self.variance/self.n_vars)
            return norm_dists