import torch
import torch.nn.functional as F
from torch_geometric.nn import Sequential as TGSequential
from torch.nn import ModuleList

class GNN (torch.nn.Module):
    def __init__(self, model_name, in_channels, hidden_channels, out_channels, num_layers) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.init_encoder = BaseLayer('lin', 'relu', in_channels, hidden_channels)
        
        self.convs = ModuleList()
        self.convs.append(BaseLayer(model_name=model_name, activation='relu', 
                                    in_channels=in_channels, out_channels=hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(BaseLayer(model_name=model_name, activation='relu', 
                                        in_channels=hidden_channels, out_channels=hidden_channels))
        self.convs.append(BaseLayer(model_name=model_name, activation='relu', 
                                    in_channels=hidden_channels, out_channels=out_channels))
        
    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        z = x
        zs = [self.init_encoder(x=x)]
        for conv in self.convs:
            z = conv(x=z, edge_index=edge_index, edge_weight=edge_weight)
            zs.append(z)
        return zs
        
class TransformDecoder(torch.nn.Module):
    def __init__(self, fn_name, in_channels, **kwargs):
        super(TransformDecoder, self).__init__()
        self.fn_name = fn_name
        self.in_channels = in_channels
        if fn_name == 'lin':
            if kwargs['activation'] == 'relu':
                activation_fn = torch.nn.ReLU()
            elif kwargs['activation'] == 'sigmoid':
                activation_fn = torch.nn.Sigmoid()
            else:
                activation_fn = torch.nn.Identity()
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(in_channels, kwargs['out_channels']), activation_fn)
        elif fn_name == 'identity':
            self.model = torch.nn.Identity()

    def forward(self, x, *args, **kwargs):
        if self.fn_name == 'id':
            return x[kwargs['indices']] if 'indices' in kwargs else x
        elif self.fn_name == 'dot':
            return x[kwargs['indices'][0]] @ torch.transpose (x[kwargs['indices'][0]], -1, -2)
        else:
            return self.model (x)

class APPNPLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K=10, alpha=0.1, device='cpu', **kwargs):
        super (APPNPLinear, self).__init__()
        from torch_geometric.nn import APPNP
        self.device = device
        self.model = APPNP(K=K, alpha=alpha)
        self.lin = torch.nn.Linear (in_channels, out_channels)

    def forward(self, x, adj_t=None, edge_index=None, edge_weight=None, **args):
        if adj_t is not None:
            return self.lin(self.model(x, adj_t))
        else:
            return self.lin(self.model(x, edge_index=edge_index, edge_weight=edge_weight))
        

class BaseLayer(torch.nn.Module):
    def __init__(self, model_name, activation, in_channels, out_channels, 
                 device='cpu', dropout_p=0.0, **kwargs):
        super (BaseLayer, self).__init__()
        
        model_name = kwargs.pop('model') if model_name == 'seq' else model_name
        self.model_name = model_name
        if activation is None or activation == 'relu':
            self.activation_fn = torch.nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = torch.nn.Tanh()
        elif activation == 'sigmoid':
            self.activation_fn = torch.nn.Sigmoid()
        elif activation == 'none':
            self.activation_fn = torch.nn.Identity()

        if dropout_p > 0.0:
            self.dropout = torch.nn.Dropout(p=dropout_p)
        else:
            self.dropout = torch.nn.Identity()
        
        self.device = device
        if model_name == 'gcn': 
            from torch_geometric.nn import GCNConv
            self.layer = GCNConv (in_channels=in_channels, out_channels=out_channels, **kwargs)
        elif model_name == 'identity': 
            self.layer = torch.nn.Identity()
        elif model_name == 'sage': 
            from torch_geometric.nn import SAGEConv
            self.layer = SAGEConv (in_channels=in_channels, out_channels=out_channels, **kwargs)
        elif model_name == 'gcn2': 
            from torch_geometric.nn import GCN2Conv
            self.layer = GCN2Conv (channels=in_channels, alpha=0.1, **kwargs)
        elif model_name == 'gin': 
            from torch_geometric.nn import GINConv
            from torch_geometric.nn.models import MLP
            mlp = MLP([in_channels, out_channels, out_channels],)
            self.layer = GINConv (mlp, train_eps=True, **kwargs)
        elif model_name == 'sgconv': 
            from torch_geometric.nn import SGConv
            self.layer = SGConv (in_channels=in_channels, out_channels=out_channels, **kwargs)
        elif model_name == 'gat':
            from torch_geometric.nn import GATConv
            self.layer = GATConv (in_channels=in_channels, out_channels=out_channels, **kwargs)
        elif model_name == 'gtn':
            from torch_geometric.nn import TransformerConv
            self.layer = TransformerConv (in_channels=in_channels, out_channels=out_channels, **kwargs)
        elif model_name == 'tg-lin':
            from torch_geometric.nn.dense.linear import Linear
            self.layer = Linear (in_channels, out_channels, **kwargs)
        elif model_name == 'lin':
            self.layer = torch.nn.Linear (in_channels, out_channels, **kwargs)
        elif model_name == 'cheb':
            from torch_geometric.nn import ChebConv
            self.layer = ChebConv (in_channels, out_channels, **kwargs)
        elif model_name == 'appnp-lin':
            self.layer = APPNPLinear (in_channels, out_channels, **kwargs)
        elif model_name == 'weis':
            from torch_geometric.nn import GraphConv
            self.layer = GraphConv (in_channels, out_channels, **kwargs)
        elif model_name == 'gated-gcn':
            from torch_geometric.nn import GatedGraphConv
            self.layer = GatedGraphConv (out_channels=out_channels, num_layers=1, **kwargs)
        elif model_name == 'pool-gnn':
            from torch_geometric.nn import JumpingKnowledge
            assert (('num_layers' in kwargs) and ('jk' in kwargs))
            self.model = kwargs['model']
            if type(kwargs['jk']) is int:
                self.pooler = lambda x: x[kwargs['jk']]
            else:
                self.pooler = JumpingKnowledge(kwargs['jk'], out_channels, kwargs['num_layers']+1)
            if kwargs['jk'] == 'cat':
                self.pooler = torch.nn.Sequential(
                                    self.pooler, 
                                    torch.nn.Linear(out_channels*(kwargs['num_layers']+1), 
                                                    out_channels)
                                )
        elif self.model_name in ['dirgcn']:
            from torch_geometric.nn import DirGNNConv
            from torch_geometric.nn import GCNConv
            self.layer = DirGNNConv(GCNConv(in_channels=in_channels, out_channels=out_channels, **kwargs))
        elif model_name == 'mha':
            self.lin = torch.nn.Linear(in_channels, out_channels) if in_channels != out_channels else torch.nn.Identity()
            self.attn = torch.nn.MultiheadAttention(
                    out_channels,
                    kwargs['num_heads'],
                    batch_first=True,
                )
        elif model_name == 'multi-model':
            self.lin = kwargs.pop('lin', None)
            self.model = kwargs['model']


    def forward(self, **args):
        if self.model_name in ['gcn', 'sgconv', 'gated-gcn']:
            # handles edge weights but not edge_attr
            if 'adj_t' in args and args['adj_t'] is not None:
                z = self.layer(x=args['x'], edge_index=args['adj_t'])
            else:
                args['edge_weight'] = args['edge_weight'] if 'edge_weight' in args.keys() else None
                z = self.layer(x=args['x'], edge_index=args['edge_index'], edge_weight=args['edge_weight'])
        elif self.model_name in ['pool-gnn']:
            # handles edge weights but not edge_attr
            if 'adj_t' in args and args['adj_t'] is not None:
                z = self.model(x=args['x'], edge_index=args['adj_t'])
            else:
                args['edge_weight'] = args['edge_weight'] if 'edge_weight' in args.keys() else None
                z = self.model(x=args['x'], edge_index=args['edge_index'], edge_weight=args['edge_weight'])
            z = self.pooler(z)
        elif self.model_name == 'multi-model':
            # handles edge weights but not edge_attr
            try:
                z = self.model(x=args['x'], edge_index=args['adj_t'])
            except:
                try:
                    args['edge_weight'] = args['edge_weight'] if 'edge_weight' in args.keys() else None
                    z = self.model(x=args['x'], edge_index=args['edge_index'], edge_weight=args['edge_weight'])
                except:
                    try:
                        z = self.model(args['x'])
                    except:
                        # h, mask = to_dense_batch(args['x'], None)
                        h = self.lin(args['x'])
                        z, _ = self.model(h, h, h) #, key_padding_mask=~mask, need_weights=False)
        elif self.model_name in ['gat', 'sage', 'gtn', 'gin']:
            # do not handle edge weights but can handle edge_attr
            if 'adj_t' in args and args['adj_t'] is not None:
                z = self.layer(x=args['x'], edge_index=args['adj_t'])
            else:
                args['edge_attr'] = args['edge_attr'] if 'edge_attr' in args.keys() else None
                z = self.layer(x=args['x'], edge_index=args['edge_index'], edge_attr=args['edge_attr'])
        elif self.model_name in ['lin']:
            # handles only x
            z = self.layer(args['x'])
        elif self.model_name in ['mha']:
            from torch_geometric.utils import to_dense_batch
            z = self.lin(args['x'])
            # h, mask = to_dense_batch(z, batch=None)
            z, _ = self.attn(z, z, z, need_weights=False)
        elif self.model_name in ['dirgcn']:
            if 'adj_t' in args and args['adj_t'] is not None:
                z = self.layer(x=args['x'], edge_index=args['adj_t'].index)
            else:
                args['edge_weight'] = args['edge_weight'] if 'edge_weight' in args.keys() else None
                z = self.layer(x=args['x'], edge_index=args['edge_index'])
        else:
            z = self.layer(**args)
        return z