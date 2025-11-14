import torch
import torch.nn.functional as F
from models.base_model import BaseLayer, TransformDecoder
import torch 

INF = 10000
EPS = 1e-6

def get_cuda_memory (device):
    device_id = device if type(device) is int else device.index
    t = torch.cuda.get_device_properties(device_id).total_memory
    r = torch.cuda.memory_reserved(device_id)
    a = torch.cuda.memory_allocated(device_id)
    f = r-a  # free inside reserved
    return t // 1024**2, r // 1024**2, a // 1024**2, f // 1024**2

def pos_neg_dist(dists, y):
    pos_dist = dists[torch.arange(dists.shape[0]), y]
    notc_mask = torch.ones_like(dists, dtype=bool)
    notc_mask[torch.arange(y.shape[0]), y] = False
    neg_dist = dists[notc_mask].reshape(len(y), -1)
    return pos_dist, neg_dist

def calculate_mahla_dists(x, mean, var): # mean, var, B=1000):
    eigs_var, Q_var = torch.linalg.eigh(var)
    eigs_inv_var = torch.diag(1/torch.clip(eigs_var, min=EPS))
    # inv_var = Q_var @ eigs_inv_var @ Q_var.T # torch.linalg.inv(var)
    # print (torch.dist(Q_var @ inv_var @ Q_var.T, var))
    dists = []
    for c in range(mean.shape[0]):
        uQ = torch.einsum('bd, dx -> bx', (x - mean[c][None, :]), Q_var)
        dists.append(1/2 * torch.einsum('bd,dx,bx->b', uQ, eigs_inv_var, uQ))
    # for i in range(len(x)):
    #     uiQ = (x[i, None, :] - mean[None, :, :]) @ Q_var
    #     print (uiQ.shape)
    #     dist = 1/2 * (uiQ @ eigs_inv_var @ uiQ.T)
    #     dist = torch.einsum('cd,dx,cx->c', uiQ, eigs_inv_var, uiQ)
    # dists = 1/2 * torch.einsum('bd,dx,bcx->bc', uQ, eigs_inv_var, uQ)
    return torch.stack(dists).T

class PersonalizedScopeGNN (torch.nn.Module):
    def __init__ (self, task, dataset, scope_selection='final', model_name='gcn', decoding='linear', 
                  num_layers=2, hidden_dims=[64, 64], dropout_p=0.2, base_model_params=None, device='cpu', dist_metric='l2', k_nearest=-1, 
                  class_weights=None, temp_tau=1, protos_type='mean', variance_type=None, feat_transf='mlp', selected_layers=None, magnet_alpha=1.0, **kwargs):
        super (PersonalizedScopeGNN, self).__init__()

        self.scope_selection = scope_selection
        self.dist_metric = dist_metric
        self.protos_type = protos_type
        self.protos = None
        self.dist_transf = None
        self.k_nearest = k_nearest
        self.variance_type = variance_type
        self.selected_layers = [l for l in selected_layers if l <= num_layers] if selected_layers is not None else list(range(num_layers+1))
        self.num_proto_layers = len(self.selected_layers)
        
        self.class_weights = class_weights 
        self.num_layers = num_layers
        self.num_nodes = dataset.data.num_nodes
        self.num_classes = dataset.num_classes
        self.num_features = dataset.num_features
        self.emb_dim = hidden_dims[-1]

        self.task = task
        self.device = device
        
        if feat_transf == 'rand':
            self.init_encoder = BaseLayer('lin', 'none', self.num_features, hidden_dims[0], dropout_p=dropout_p)
            for param in self.init_encoder.parameters():
                param.requires_grad = False
        elif feat_transf in ['mlp', 'lin']:
            self.init_encoder = BaseLayer('lin', 'none', self.num_features, hidden_dims[0], dropout_p=dropout_p)
        elif feat_transf == 'pca':
            self.init_encoder = lambda x: torch.pca_lowrank(x, hidden_dims[0])[0]
        elif feat_transf == 'none':
            self.selected_layers.remove(0)
            self.init_encoder = None
        elif feat_transf in ['relu']:
            self.init_encoder = BaseLayer('lin', 'relu', self.num_features, hidden_dims[0], dropout_p=dropout_p)
        elif feat_transf in ['tg-lin']:
            self.init_encoder = BaseLayer('tg-lin', 'none', self.num_features, hidden_dims[0], dropout_p=dropout_p, 
                                          weight_initializer='glorot', bias=False)
        else:
            self.init_encoder = lambda x=None: torch.nn.Identity()(x)
            # 'relu'
        
        self.seq_model = model_name not in ['appnp-lin', 'sgconv', 'pool-gnn', 'multi-model']
        if model_name == 'pool-gnn':
            from models.base_model import GNN
            self.selected_layers.remove(0)
            for i in range(num_layers):
                model = base_model_params[i].pop('model_name', 'gcn')
            model = GNN (model_name=model,
                        in_channels=self.num_features, 
                        hidden_channels=hidden_dims[0], 
                        out_channels=hidden_dims[-1],
                        num_layers=base_model_params[0]['num_layers'])
            for i in range(num_layers):
                base_model_params[i]['model'] = model
                
        elif model_name == 'multi-model':
            self.selected_layers.remove(0)
            for i in range(num_layers):
                model =  base_model_params[i].pop('model_name', 'gcn')
                hidden_channels =  base_model_params[i].pop('hidden_channels', hidden_dims[i])
                if model == 'gcn':
                    from torch_geometric.nn import GCN
                    base_model_params[i]['model'] = GCN (in_channels=self.num_features, 
                                                         hidden_channels=hidden_channels,
                                                         out_channels=hidden_dims[-1], 
                                                         **base_model_params[i])
                elif model == 'gin':
                    from torch_geometric.nn import GIN
                    base_model_params[i]['model'] = GIN (in_channels=self.num_features, 
                                                         hidden_channels=hidden_channels,
                                                         out_channels=hidden_dims[-1], 
                                                         **base_model_params[i])
                elif model == 'gat':
                    from torch_geometric.nn import GAT
                    base_model_params[i]['model'] = GAT (in_channels=self.num_features, 
                                                         hidden_channels=hidden_channels,
                                                         out_channels=hidden_dims[-1], 
                                                         **base_model_params[i])
                elif model == 'sage':
                    from torch_geometric.nn import GraphSAGE
                    base_model_params[i]['model'] = GraphSAGE (in_channels=self.num_features, 
                                                               hidden_channels=hidden_channels,
                                                               out_channels=hidden_dims[-1], 
                                                               **base_model_params[i])
                elif model == 'mlp':
                    from torch_geometric.nn import MLP
                    base_model_params[i]['model'] = MLP (in_channels=self.num_features, 
                                                         hidden_channels=hidden_channels,
                                                         out_channels=hidden_dims[-1], 
                                                         **base_model_params[i])
                elif model == 'global-attn':
                    base_model_params[i]['model'] = torch.nn.MultiheadAttention(embed_dim=hidden_channels, 
                                                                                **base_model_params[i])
                    base_model_params[i]['lin'] = torch.nn.Linear(self.num_features, hidden_channels)
                    
                
        if model_name == 'gated-gcn': self.selected_layers.remove(0)
        
        self.base_model = []
        if self.seq_model:
            if self.num_layers > 0:
                self.base_model = [BaseLayer(model_name, 'relu', self.num_features, hidden_dims[0], **base_model_params[0])]
                for i in range(1, num_layers):
                    self.base_model.append(BaseLayer(model_name, 'relu', hidden_dims[i-1], hidden_dims[i], **base_model_params[i]))
        else:
            for i in range(num_layers):
                self.base_model.append(BaseLayer(model_name, 'none', self.num_features, hidden_dims[i], **base_model_params[i]))
        self.base_model = torch.nn.ModuleList (self.base_model)
        self.decoding = decoding

        if 'metselect' in scope_selection or decoding == 'distance':
            self.alpha = magnet_alpha
            # taking the points for each class in each layer (multi-view)
            # assume each layer is the same dimension
            self.protos = torch.zeros(self.num_classes, self.num_proto_layers, hidden_dims[-1]).to(device)
            self.vars = torch.zeros(self.num_proto_layers, hidden_dims[-1], hidden_dims[-1]).to(device)
            if protos_type == 'param':
                self.protos = torch.nn.Parameter(torch.rand(self.num_classes, self.num_proto_layers, self.init_nclusters, hidden_dims[-1]).to(device))
        if scope_selection == 'ndls':
            from NDLS.src.ndls import NDLS
            ndls = NDLS(num_layers=num_layers, epochs=100, lr=0.01)
            self.hops = ndls.get_hops(features=dataset.data.x, adj=dataset.data.adj_t, labels=dataset.data.y,
                                      idx_train=dataset.data.train_idx, idx_val=dataset.data.val_idx, 
                                      idx_test=dataset.data.test_idx).int()
        if decoding == 'linear':
            self.decoder = torch.nn.ModuleList([
                                TransformDecoder('lin', activation='none', in_channels=hidden_dims[-1], 
                                                    out_channels=self.num_classes) 
                                for _ in range(num_layers+1)
                            ])
            self.dist_transf = torch.nn.ModuleList([
                                    TransformDecoder('lin', activation='relu', in_channels=hidden_dims[-1], 
                                                    out_channels=hidden_dims[-1]) 
                                    for i in range(num_layers+1)
                                ])
            for param in self.dist_transf.parameters():
                param.requires_grad = False
        elif decoding == 'linear-distance':
        # and ('metselect' in scope_selection or scope_selection == 'ndls'):
            self.decoder = torch.nn.ModuleList([
                                TransformDecoder('lin', activation='none', in_channels=hidden_dims[-1], 
                                                out_channels=self.num_classes) 
                                for i in range(num_layers+1)
                            ])
            # transf = TransformDecoder('lin', activation='none', in_channels=hidden_dims[-1], 
            #                                         out_channels=hidden_dims[-1]) 
            self.dist_transf = torch.nn.ModuleList([TransformDecoder('lin', activation='none', 
                                                                     in_channels=hidden_dims[-1], 
                                                                     out_channels=hidden_dims[-1])  
                                                    for i in range(num_layers+1)])
            # for param in self.dist_transf.parameters():
            #     param.requires_grad = False
            # self.decoder = TransformDecoder('lin', activation='none', in_channels=hidden_dims[-1], 
            #                                         out_channels=self.num_classes)
            # self.decoder = torch.nn.ModuleList([self.decoder for _ in range(num_layers+1)])
        if scope_selection == 'wtd-sum':
            self.linear_selector = BaseLayer('lin', 'none', self.num_layers+1, 1).to(device)
        elif scope_selection == 'max-pool':
            from torch_geometric.nn import JumpingKnowledge
            self.pooler = JumpingKnowledge(mode='max')
        elif scope_selection == 'attn-pool':
            from torch_geometric.nn import JumpingKnowledge
            self.pooler = JumpingKnowledge(mode='lstm', channels=hidden_dims[-1], num_layers=2)
        elif scope_selection == 'attn-max':
            from torch_geometric.nn import JumpingKnowledge
            self.pooler = JumpingKnowledge(mode='lstm-max', channels=hidden_dims[-1], num_layers=2)
        elif scope_selection == 'cat-pool':
            from torch_geometric.nn import JumpingKnowledge
            self.pooler = JumpingKnowledge(mode='cat')
            self.decoder = torch.nn.ModuleList([
                                TransformDecoder('lin', activation='none', 
                                                 in_channels=len(self.selected_layers)*hidden_dims[-1], 
                                                 out_channels=self.num_classes) 
                                for _ in range(num_layers+1)
                            ])
        # else:
        #     self.decoder = TransformDecoder('lin', activation='none', in_channels=hidden_dims[-1], 
        #                                     out_channels=self.num_classes)
        # elif scope_selection in ['final']:
        #     self.base_model = self.base_model[:-1]
        #     in_dim = hidden_dims[i-1] if self.seq_model else self.num_features
        #     self.base_model.append(BaseLayer(model_name, 'none', in_dim, self.num_classes, **base_model_params[-1]))
        #     # self.decoder = TransformDecoder('id', in_channels=self.num_classes)
        #     self.decoder = TransformDecoder('lin', activation='none', in_channels=hidden_dims[-1], out_channels=self.num_classes)
        
    def update_means(self, z_nlayer, nlayer, y):
        with torch.no_grad():
            z = self.dist_transf[nlayer](z_nlayer) if self.dist_transf is not None else z_nlayer
            z = z_nlayer
            if self.protos_type == 'medoid':
                for c in y.unique(): 
                    c_medoid_id = torch.cdist(z, z[y == c]).sum(dim=-1).argmin(dim=0)
                    self.protos[c, nlayer, :] = z[c_medoid_id, :]
            elif self.protos_type == 'random_point':
                for c in y.unique(): 
                    i = torch.randint(0, len(z[y == c]), (1,)).item()
                    self.protos[c, nlayer, :] = z[y == c][i, :]
            else:
                for c in y.unique(): 
                    self.protos[c, nlayer, :] = z[y == c].mean(dim=0)

    def update_vars(self, z_nlayer, nlayer, y):
        with torch.no_grad():
            z = self.dist_transf[nlayer](z_nlayer) if self.dist_transf is not None else z_nlayer
            z = z_nlayer
            vec = z[torch.arange(len(y)), :] - self.protos[y, nlayer, :] # B, C, D
            self.vars[nlayer] = torch.einsum('bd,bx->dx', vec, vec) / (len(y)-1)
            # self.vars[nlayer] =  (vec[:, :, None] @ vec[:, None, :]).sum(dim=0)
            # for c in y.unique(): self.vars[nlayer] -= ((y == c).sum()) * self.protos[c, nlayer, :, None] @ self.protos[c, nlayer, None, :]
    
    def calculate_mahla_dists(self, x, nlayer): # mean, var, B=1000):
        if self.scope_selection == 'metselect-decoder':
            z = self.decoder[nlayer](x) # N x C
            return z.softmax(dim=-1) #/ self.decoder[nlayer].model[0].weight.norm(dim=-1)[None, :]
        x_transf = self.dist_transf[nlayer](x) if self.dist_transf is not None else x
        x_transf = x
        mean = self.protos[:, nlayer, :]
        var = self.vars[nlayer]
        eigs_var, Q_var = torch.linalg.eigh(var)
        eigs_inv_var = torch.diag(1/torch.clip(eigs_var, min=EPS))
        # inv_var = Q_var @ eigs_inv_var @ Q_var.T # torch.linalg.inv(var)
        # print (torch.dist(Q_var @ inv_var @ Q_var.T, var))
        dists = []
        for c in range(mean.shape[0]):
            uQ = torch.einsum('bd, dx -> bx', (x_transf - mean[c][None, :]), Q_var)
            dists.append(1/2 * torch.einsum('bd,dx,bx->b', uQ, eigs_inv_var, uQ))
        # for i in range(len(x)):
        #     uiQ = (x[i, None, :] - mean[None, :, :]) @ Q_var
        #     print (uiQ.shape)
        #     dist = 1/2 * (uiQ @ eigs_inv_var @ uiQ.T)
        #     dist = torch.einsum('cd,dx,cx->c', uiQ, eigs_inv_var, uiQ)
        # dists = 1/2 * torch.einsum('bd,dx,bcx->bc', uQ, eigs_inv_var, uQ)
        return torch.stack(dists).T
    
    def calculate_decoder_dists(self, x, nlayer): # mean, var, B=1000):
        # assume x: N x D
        z = self.decoder[nlayer](x) # N x C
        return z.norm(dim=1) / self.decoder[nlayer].weight.norm(dim=-1)[None, :]
        
    def forward(self, x, y=None, idx=None, adj_t=None, edge_index=None, edge_weight=None, edge_attr=None):
        pos_dists, neg_dists, all_dists = [], [], []
        x_layers = []
        x_0 = None
        if self.training:
            assert y is not None
            y = y[idx] if idx is not None else y
        if 'metselect' in self.scope_selection or self.decoding == 'distance':
            if 0 in self.selected_layers:
                x_0 = self.init_encoder(x=x) # encode the original features using a simple linear layer (and relu).
                x_layers = [x_0.clone()]
                x_0 = x_0[idx] if idx is not None else x_0
                if self.training:
                    self.update_means(x_0.detach(), 0, y)
                    self.update_vars(x_0.detach(), 0, y)
                # dists = calculate_mahla_dists(x_0, self.protos[:, 0, :], self.vars[0]) 
                dists = self.calculate_mahla_dists(x_0, 0)
                if self.training:
                    pos_dist, neg_dist = pos_neg_dist(dists, y)
                    pos_dists.append(pos_dist); neg_dists.append(neg_dist)
                else:
                    all_dists.append(dists)
                # pos_dists = pos_dist
                # neg_dists = torch.exp(-neg_dist).sum(dim=-1)
        elif self.scope_selection == 'ndls':
            x_0 = self.init_encoder(x=x) # encode the original features using a simple linear layer (and relu).
            if 0 in self.selected_layers: x_layers = [x_0.clone()]
            # x_ndls = x_0.clone()
        elif self.scope_selection in ['wtd-sum', 'max-pool', 'attn-pool', 'attn-max', 'cat-pool', 'max-warmup', 'lin-warmup']:
            x_0 = self.init_encoder(x=x)
            if 0 in self.selected_layers: x_layers = [x_0]
        if self.num_layers == 0:
            x_out = self.init_encoder(x=x)
        elif self.base_model[0].model_name == 'gated-gcn':
            x = self.init_encoder(x=x)
        x_in = x
        for i, layer in enumerate(self.base_model):
            # print (i, get_cuda_memory(x.device))
            if edge_index is not None:
                x_out = layer(x=x_in, edge_index=edge_index, edge_weight=edge_weight, 
                              edge_attr=edge_attr)
            elif adj_t is not None:
                try:
                    x_out = layer(x=x_in, adj_t=adj_t)
                except:
                    x_out = layer(x=x_in, edge_index=adj_t)
            x_out = layer.activation_fn(x_out)
            x_out = layer.dropout(x_out)
            x_in = x_out.clone() if self.seq_model else x_in
            if ('metselect' in self.scope_selection and (i+1) in self.selected_layers) or (self.decoding == 'distance'):
                # x_layers.append(x_out.clone())
                x_layers.append(x_out)
                x_out = x_out[idx] if idx is not None else x_out
                if self.training:
                    self.update_means(x_out.detach(), i+1, y)
                    self.update_vars(x_out.detach(), i+1, y)
                # dists = calculate_mahla_dists(x_out, self.protos[:, i+1, :], self.vars[i+1]) 
                dists = self.calculate_mahla_dists(x_out, i+1)
                if self.training:
                    pos_dist, neg_dist = pos_neg_dist(dists, y)
                    pos_dists.append(pos_dist); neg_dists.append(neg_dist)
                else:
                    all_dists.append(dists)
                # pos_dists = torch.min(pos_dists, pos_dist)
                # neg_dists += torch.exp(-neg_dist).sum(dim=-1)
            elif self.scope_selection == 'ndls':
                x_layers.append(x_out)
                # x_ndls[self.hops == i, :] = x_out[self.hops == i]
            elif self.scope_selection in ['wtd-sum', 'max-pool', 'attn-pool', 'attn-max', 'cat-pool', 'max-warmup', 'lin-warmup']:
                x_layers.append(x_out)
        if self.decoding == 'distance':
            return (pos_dists, neg_dists) if self.training else all_dists
        elif self.decoding == 'linear':
            if 'metselect' in self.scope_selection:
                if self.training:
                    all_dists = torch.stack(pos_dists)
                    all_dists -= torch.logsumexp(-torch.stack(neg_dists), dim=-1)
                    # all_dists = pos_dists 
                return (torch.stack(x_layers), self.get_metselect_layers(all_dists))
            elif self.scope_selection == 'ndls':
                return (torch.stack(x_layers), self.hops[idx])
            elif self.scope_selection in ['wtd-sum', 'max-pool', 'attn-pool', 'attn-max', 'cat-pool', 'max-warmup', 'lin-warmup']:
                return torch.stack(x_layers)
            else:
                return x_out
        elif self.decoding == 'linear-distance':
            assert('metselect' in self.scope_selection)
            if self.training:
                all_dists = torch.stack(pos_dists)
                all_dists -= torch.logsumexp(-torch.stack(neg_dists), dim=-1)
                # all_dists = pos_dists 
                return ((pos_dists, neg_dists), (torch.stack(x_layers), self.get_metselect_layers(all_dists)))
            else:
                return (torch.stack(x_layers), self.get_metselect_layers(all_dists))
        
            
    def get_metselect_layers (self, dists):
        assert 'metselect' in self.scope_selection
        try:
            dists = torch.stack(dists).transpose(0, 1)
        except:
            dists = dists.transpose(0, 1)
        if dists.ndim == 3:
            # dists: B x L x C
            logits = -dists
            logits -= torch.logsumexp(-dists, dim=-1)[:, :, None]
            pred_classes = logits.argmax(dim=-1, keepdim=True)
            dists = torch.gather(dists, -1, pred_classes)[:, :, 0]
            # logits = torch.logsumexp(-dists, dim=1)
            # logits -= torch.logsumexp(-dists.reshape(dists.shape[0], -1), dim=-1)[:, None]
            # pred_classes = logits.argmax(dim=-1)
            # dists = dists[torch.arange(len(pred_classes)), :, pred_classes]
        if self.scope_selection == 'metselect-min':
            return dists.argmin(dim=1)
        elif self.scope_selection == 'metselect-decoder':
            return dists.argmin(dim=1)
        elif self.scope_selection == 'metselect-min2':
            _, indices = torch.sort(dists, dim=1)
            return torch.where(indices == 1)[1]
        elif self.scope_selection == 'metselect-max':
            return dists.argmax(dim=1)
        else:
            raise NotImplementedError
    
    def forward_and_layers (self, x, idx=None, adj_t=None, edge_index=None, edge_weight=None, edge_attr=None, criterion=None, **args):
        if 'linear' in self.decoding:
            if self.scope_selection == 'final':
                return torch.zeros_like(idx) -1
        z = self.forward(x, idx=idx, adj_t=adj_t, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
        if 'linear' in self.decoding:
            if 'metselect' in self.scope_selection or self.scope_selection == 'ndls':
                _, layers = z
                return layers
    
    def forward_and_loss (self, x, y, idx=None, adj_t=None, edge_index=None, edge_weight=None, edge_attr=None, criterion=None, **args):
        z = self.forward(x, y, idx=idx, adj_t=adj_t, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
        if self.decoding == 'distance':
            pos_dists, neg_dists = z
            if self.scope_selection == 'final':
                pos_dists = pos_dists[-1]
                neg_dists = torch.logsumexp(-neg_dists[-1], dim=-1)[:, None]
            elif self.scope_selection == 'init':
                pos_dists = pos_dists[0]
                neg_dists = torch.logsumexp(-neg_dists[0], dim=-1)[:, None]
            elif self.scope_selection == 'ndls':
                pos_dists = torch.stack(pos_dists)
                pos_dists = pos_dists[self.hops[idx] if idx is not None else self.hops, torch.arange(len(idx))]
                neg_dists = torch.stack(neg_dists)[self.hops[idx] if idx is not None else self.hops, torch.arange(len(idx)), :]
                neg_dists = torch.logsumexp(-neg_dists, dim=-1)[:, None]
            elif 'metselect' in self.scope_selection:
                pos_dists = torch.stack(pos_dists).min(dim=0)[0] # (Nl+1) x B -> B
                neg_dists = torch.logsumexp(- torch.stack(neg_dists).transpose(0, 1).reshape((len(idx), -1)), dim=-1) # (Nl+1) x B x C-> B
            elif self.scope_selection == 'dist-warmup':
                loss = 0
                pos_dists = torch.stack(pos_dists).sum(dim=0)[0] # (Nl+1) x B -> B
                neg_dists = torch.logsumexp(- torch.stack(neg_dists).transpose(0, 1).reshape((len(idx), -1)), dim=-1) # (Nl+1) x B x C-> B
            loss = (self.alpha + pos_dists + neg_dists).relu().mean()
        elif self.decoding == 'linear':
            if self.scope_selection == 'final':
                assert criterion is not None
                y_hat = self.predict_proba(z[idx])
                loss = criterion(y_hat, y[idx])
            elif self.scope_selection == 'wtd-sum':
                x_out = self.linear_selector(x=z.transpose(0, 1).transpose(1, 2))[:, :, 0]
                y_hat = self.predict_proba(x_out[idx])
                loss = criterion(y_hat, y[idx])
            elif self.scope_selection in ['max-pool', 'attn-pool', 'attn-max', 'cat-pool']:
                # x_out = z.max(dim=0)[0]
                x_out = self.pooler([z[i, :, :] for i in range(z.shape[0])])
                y_hat = self.predict_proba(x_out[idx])
                loss = criterion(y_hat, y[idx])
            elif self.scope_selection == 'max-warmup':
                x_out = z.max(dim=0)[0]
                y_hat = self.predict_proba(x_out[idx])
                loss = criterion(y_hat, y[idx])
            elif self.scope_selection == 'lin-warmup':
                loss = 0
                for i in range(len(z)):
                    y_hat = self.predict_proba(z[i, idx])
                    loss += criterion(y_hat, y[idx])
            elif 'metselect' in self.scope_selection or self.scope_selection == 'ndls':
                x_out, layers = z
                y_hat = self.predict_proba(x_out[layers, idx, :], layers=layers)
                loss = criterion(y_hat, y[idx])
        elif self.decoding == 'linear-distance':
            (pos_dists, neg_dists), (x_out, layers) = z
            pos_dists = torch.stack(pos_dists).sum(dim=0)[0] # (Nl+1) x B -> B
            neg_dists = torch.logsumexp(- torch.stack(neg_dists).transpose(0, 1).reshape((len(idx), -1)), dim=-1) # (Nl+1) x B x C-> B
            loss = (self.alpha + pos_dists + neg_dists).relu().mean()
            y_hat = self.predict_proba(x_out[layers, idx, :], layers=layers)
            loss += criterion(y_hat, y[idx])
        else:
            raise NotImplementedError
        return loss #, running_means, running_vars

    def forward_and_predict (self, x, idx=None, adj_t=None, edge_index=None, edge_weight=None, edge_attr=None, **args):
        z = self.forward(x, idx=idx, adj_t=adj_t, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
        if self.decoding == 'distance':
            all_dists = torch.stack(z).transpose(0, 1)
            logits = torch.logsumexp(-all_dists, dim=1)
            logits -= torch.logsumexp(-all_dists.reshape(all_dists.shape[0], -1), dim=-1)[:, None]
            pred_classes = logits.argmax(dim=-1)
            pred_idx = torch.arange(len(pred_classes)).to(pred_classes.device)
            if self.scope_selection == 'metselect-min':
                min_layers = all_dists[pred_idx, :, pred_classes].argmin(dim=1)
                min_dists = all_dists[pred_idx, min_layers, :]
                logits = -min_dists - torch.logsumexp(-min_dists, dim=-1)[:, None]
            elif self.scope_selection == 'metselect-wo-min':
                all_dists = all_dists.transpose(1, 2)
                min_layers = all_dists[pred_idx, pred_classes, :].argmin(dim=-1)
                mask = torch.ones_like (all_dists, dtype=bool)
                B, L, K = all_dists.shape
                mask[pred_idx, :, min_layers] = 0
                nonmin_dists = all_dists[mask].reshape(B, L, K-1)
                # nonmin_dists = nonmin_dists[:, :, torch.randint()]
                logits = torch.logsumexp(-nonmin_dists, dim=-1)
                logits -= torch.logsumexp(-nonmin_dists.reshape(nonmin_dists.shape[0], -1), dim=-1)[:, None]
            elif self.scope_selection == 'metselect-min2':
                _, indices = torch.sort(all_dists[pred_idx, :, pred_classes], dim=1)
                min2_layers = torch.where(indices == 1)[1]
                min2_dists = all_dists[pred_idx, min2_layers, :]
                logits = -min2_dists - torch.logsumexp(-min2_dists, dim=-1)[:, None]
            elif self.scope_selection == 'metselect-max':
                max_layers = all_dists[pred_idx, :, pred_classes].argmax(dim=1)
                max_dists = all_dists[pred_idx, max_layers, :]
                logits = -max_dists - torch.logsumexp(-max_dists, dim=-1)[:, None]
            elif self.scope_selection == 'final':
                final_dists = all_dists[pred_idx, -1, :]
                logits = -final_dists - torch.logsumexp(-final_dists, dim=-1)[:, None]
            elif self.scope_selection == 'init':
                init_dists = all_dists[pred_idx, 0, :]
                logits = -init_dists - torch.logsumexp(-init_dists, dim=-1)[:, None]
            elif self.scope_selection == 'ndls':
                ndls_dists = all_dists[pred_idx, self.hops[idx] if idx is not None else self.hops, :]
                logits = -ndls_dists - torch.logsumexp(-ndls_dists, dim=-1)[:, None]
        elif 'linear' in self.decoding:
            if self.scope_selection == 'final':
                logits = self.predict_proba(z[idx] if idx is not None else z)
            elif self.scope_selection == 'wtd-sum':
                x_out = self.linear_selector(x=z.transpose(0, 1).transpose(1, 2))[:, :, 0]
                logits = self.predict_proba(x_out[idx] if idx is not None else x_out)
            elif self.scope_selection in ['max-pool', 'attn-pool', 'attn-max', 'cat-pool']:
                # x_out = z.max(dim=0)[0]
                x_out = self.pooler([z[i, :, :] for i in range(z.shape[0])])
                logits = self.predict_proba(x_out[idx] if idx is not None else x_out)
            elif self.scope_selection == 'max-warmup':
                x_out = z.max(dim=0)[0]
                logits = self.predict_proba(x_out[idx] if idx is not None else x_out)
            elif self.scope_selection == 'lin-warmup':
                x_out = z.max(dim=0)[0]
                logits = self.predict_proba(x_out[idx] if idx is not None else x_out)
            elif 'metselect' in self.scope_selection or self.scope_selection == 'ndls':
                x_out, layers = z
                logits = self.predict_proba(x_out[layers, idx] if idx is not None else x_out[layers, torch.arange(len(layers))], layers=layers)
            else:
                raise NotImplementedError
        # logits = torch.log(logits) - torch.log(logits.sum(dim=-1))[:, None]
        return logits


    def predict_proba (self, z, layers=None):
        # does not return probabilities but rather logits
        assert (self.decoder is not None)
        if layers is None:
            return F.log_softmax(self.decoder[-1](z, dim=-1), dim=-1)
        else:
            out = torch.zeros(len(z), self.num_classes).to(z.device)
            for i in range(self.num_layers+1):
                out[layers == i] = self.decoder[i](z[layers == i])
            return F.log_softmax(out, dim=-1)
            
        
    def predict_proba_old (self, z, decoded_probs=False):
        if not ('metselect' in self.scope_selection) or decoded_probs:
            if self.task == 'node_classification':
                logits = self.decoder(z)
                return F.log_softmax (logits, dim=-1)
            elif self.task == 'link_prediction':
                logits = self.decoder(z)
                return F.sigmoid (logits)
        else:
            # z = dists/(2*var)
            # logits[i, c] = log (sum_[j] exp(-z[i, c, j]) / sum_[c, j] exp(-z[i, c, j]))
            if self.k_nearest > 0:
                z_kmin, inds_kmin = z.reshape(z.shape[0], -1).sort()
                z_kmin, inds_kmin = z_kmin[:, :self.k_nearest], inds_kmin[:, :self.k_nearest]
                classes_kmin = inds_kmin // z.shape[1]
                logits = torch.zeros_like (z[:,:,0])
                for k in range(logits.shape[1]): logits[:, k] = (z_kmin * (classes_kmin == k)).sum(dim=1)
            else:
                logits = torch.logsumexp(-z, dim=-1)
                if self.class_weights is not None:
                    logits = logits * (torch.where (self.class_weights == 0, -torch.ones_like(self.class_weights)*INF, self.class_weights)[None, :])
                z_all = z.reshape(z.shape[0], -1)
                logits = logits - torch.logsumexp(-z_all, dim=-1)[:, None]
                # logits = -(z.min(dim=-1)[0])
            return logits
        
    def predict_distance (self, z, z_layers=None, y=None, var=None):
        if self.dist_metric.startswith('l') or self.dist_metric.startswith('cosine') or self.dist_metric.startswith('dot') or self.dist_metric.startswith('mahla'):
            if self.dist_metric.startswith('l'):
                ord_p = int(self.dist_metric.split('l')[1][0]) if 'inf' not in self.dist_metric else 'inf'
            # each layer output distances from mean representation in that layer for points of each class
            # z_layers: B x (Nl+1) x D, self.protos: L x (Nl+1) x D
            # means = self.magnet_batch_proto(y, x_layers=z_layers) if y is not None else self.protos
            means = self.protos
            assert (((var is None) and (y is not None)) or ((var is not None) and (y is None)))
            if 'mahla' in self.dist_metric:
                # covariance: B x L x (Nl+1) x D
                M_zz = z_layers[:, None, :, :] - means[None, :, :, :]
                if var is None:
                    Mz_y = M_zz[torch.arange(M_zz.shape[0]), y, :, :] # B x L x D
                    # var = torch.einsum('bld,blx->ldx', Mz_y, Mz_y).detach()/(Mz_y.shape[0]-1)
                    var = (Mz_y.transpose(0,1).transpose(1,2) @ Mz_y.transpose(0,1)).detach()/(Mz_y.shape[0]-1)
                # ivar = torch.cholesky_inverse(torch.cholesky(var))
                eigs, Q = torch.linalg.eigh(var)
                uQ = torch.einsum ('ijkl,klm->ijkm', M_zz, Q)
                L = torch.stack([torch.diag(1/torch.clip(eigs[k], min=EPS)) for k in range(eigs.shape[0])])
                dists = torch.einsum('ijkl,klm,ijkm->ijk', uQ, L, uQ)
                return dists if 'pow' in self.dist_metric else (dists + EPS).relu() ** 0.5, var
            elif 'cosine' in self.dist_metric:
                # dists: B x L x (Nl + 1)
                dists = 1 - F.cosine_similarity (z_layers[:, None, :, :], means[None, :, :, :], dim=-1)
            else:
                # dists: B x L x (Nl + 1)
                dists = (z_layers[:, None, :, :] - means[None, :, :, :]).norm(dim=-1, p=ord_p)
                dists = dists ** ord_p if 'pow' in self.dist_metric else dists
            #  'layer' in model_config.variance_type:
            if var is None:
                if self.class_weights is not None:
                    var = (self.class_weights[y][:, None] * dists[torch.arange(y.shape[0]), y, :]).sum(dim=0).detach()/(self.class_weights[y].sum() - 1)
                else:
                    var = dists[torch.arange(y.shape[0]), y, :].sum(dim=0).detach()/(dists.shape[0]-1)
            return dists/(2*var[None, None, :]), var
    
   
    def predict (self, p):
        if self.to_magnet:
            return p.argmax(dim=-1)
        if self.task == 'node_classification':
            return p.argmax(dim=-1)
        elif self.task == 'link_prediction':
            return (p > 0.5).int()
