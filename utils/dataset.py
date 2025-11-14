from torch_geometric.data import Data
import torch
import torch_geometric.utils as tg_utils
import torch_geometric.transforms as T
import os
import numpy as np

def get_feature_mask(rate, n_nodes, n_features):
    return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()

def even_quantile_labels(vals, nclasses):
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    return label

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx

class Dataset ():
    def __init__(self, root='data/', name='', task='', split='public', nsplit=0, device='cpu', 
                 samples_per_class=20, transform_sparse=True, add_self_loops=False, **kwargs):
        self.name = name
        # self.small_tgdataset = name in ['Cora', 'Citeseer', 'Pubmed']
        self.task = task
        self.split = split
        self.nsplit = nsplit
        self_transf = T.AddSelfLoops() if add_self_loops else T.ToDevice(device='cpu')

        if name in ['Cora', 'Citeseer', 'Pubmed']:
            from torch_geometric.datasets import Planetoid
            if transform_sparse:
                dataset = Planetoid(root=root, name=name, split=split, num_train_per_class=samples_per_class, transform=T.Compose([T.ToUndirected(), T.ToSparseTensor(), T.NormalizeFeatures(), self_transf,]))
                self.data = dataset[0].to(device)
            else:
                dataset = Planetoid(root=root, name=name, split=split, num_train_per_class=samples_per_class, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), self_transf]))
                self.data = dataset[0].to(device)
        elif name == ['Cora', 'Citeseer', 'Pubmed']:
            from torch_geometric.datasets import Planetoid
            from torch_geometric.loader import DataLoader
            datasets = []
            for n in name:
                dataset = Planetoid(root=root, name=n, split=split, num_train_per_class=samples_per_class, 
                                    transform=T.Compose([T.ToUndirected(), T.ToSparseTensor(), T.NormalizeFeatures(), 
                                                         self_transf, ])) #T.SVDFeatureReduction(64)]))
                datasets.append(dataset[0].to(device))
            self.data = next(iter(DataLoader([dataset for dataset in datasets], batch_size=3)))
        elif name in ['Cornell', 'Texas', 'Wisconsin']:
            from torch_geometric.datasets import WebKB
            dataset = WebKB(root=root, name=name.lower(), transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf, ])) #T.SVDFeatureReduction(8)]))
            self.data = dataset[0].to(device)
        elif name == 'PolBlogs':
            from torch_geometric.datasets import PolBlogs
            dataset = PolBlogs(root=root, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)
        elif name.startswith('MixHop'):
            from torch_geometric.datasets import MixHopSyntheticDataset
            homophily = float(name.split("-")[1])
            dataset = MixHopSyntheticDataset(root=root, homophily=homophily, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)
        elif name == 'WebKB':
            from torch_geometric.datasets import WebKB
            from torch_geometric.loader import DataLoader
            datasets = [WebKB(root=root, name='wisconsin', transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf])),
                        WebKB(root=root, name='texas', transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf])),
                        WebKB(root=root, name='cornell', transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))]
            self.data = next(iter(DataLoader([dataset[0].to(device) for dataset in datasets], batch_size=3)))
            # self.data = dataset[0].to(device)
        elif name in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
            from torch_geometric.datasets import LINKXDataset
            dataset = LINKXDataset(root=root, name=name.lower(), transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)
        elif name in ["pokec"]:
            from NHLS.dataset import load_nc_dataset
            from torch_geometric.data import Data
            dataset = load_nc_dataset(name.split('|')[0]) #name.split('|')[1]
            split_idx = dataset.get_idx_split()
            train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            graph, y = dataset[0]
            self.data = Data(x=graph['node_feat'], edge_index=graph['edge_index'], y=y).to(device)
        elif name in ['arxiv-year']:
            from ogb.nodeproppred import PygNodePropPredDataset 
            dataset = PygNodePropPredDataset(root='data/', name='ogbn-arxiv', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor(), self_transf]))
            # split_idx = dataset.get_idx_split()
            data = dataset[0]
            data.y = torch.tensor(even_quantile_labels(data.node_year.flatten(), 5))
            train_idx, val_idx, test_idx = rand_train_test_idx(data.y)
            # data.y = data.node_year
            self.data = data.to(device)
        elif name == 'Actor':
            from torch_geometric.datasets import Actor
            dataset = Actor(root=f'{root}/Actor', transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)
        elif name in ['Chameleon', 'Squirrel']:
            from torch_geometric.datasets import WikipediaNetwork
            dataset = WikipediaNetwork(root=root, name=name, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)
        elif name in ['PascalVOC-SP', 'COCO-SP']:
            from torch_geometric.datasets import LRGBDataset
            dataset = LRGBDataset(root=root, name=name, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)
        elif name == 'Reddit':
            from torch_geometric.datasets import Reddit
            dataset = Reddit(root=root, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)
        elif name == 'Reddit2':
            from torch_geometric.datasets import Reddit
            dataset = Reddit(root=root, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0]
        elif name == 'Flickr':
            from torch_geometric.datasets import Flickr
            dataset = Flickr(root=root, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)
        elif name == 'AmazonProducts':
            from torch_geometric.datasets import AmazonProducts
            dataset = AmazonProducts(root=root, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)
        elif name == 'Yelp':
            from torch_geometric.datasets import Yelp
            dataset = Yelp(root=root, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)
        elif name in ['ogbn-arxiv', 'ogbn-mag', 'ogbn-products']:
            from ogb.nodeproppred import PygNodePropPredDataset 
            dataset = PygNodePropPredDataset(root='data/', name=name, transform=T.Compose([T.ToUndirected(), T.ToSparseTensor(), self_transf]))
            split_idx = dataset.get_idx_split()
            train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            self.data = dataset[0].to(device)
        elif name in ["IMDB-BINARY", "REDDIT-BINARY", "PROTEINS"]:
            from torch_geometric.datasets import TUDataset
            dataset = TUDataset(root=root, name=name, transform=T.Compose([T.ToUndirected(), T.NormalizeFeatures(), T.ToSparseTensor(), self_transf]))
            self.data = dataset[0].to(device)

        elif name in ['ogbn-proteins']:
            # proteins getting killed right now OOM. 
            from ogb.nodeproppred import PygNodePropPredDataset
            dataset = PygNodePropPredDataset(root='data/', name='ogbn-proteins', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor(attr='edge_attr'), self_transf]))
            data = dataset[0]
            # Move edge features to node features.
            data.x = data.adj_t.mean(dim=1)
            split_idx = dataset.get_idx_split()
            train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            self.data = data
        elif name.startswith('heart-'):
            import pandas as pd
            import yaml
            df = pd.read_csv(f"{root}/heart-disease/processed.{name.split('-')[1]}.data", header=None)

        # if 'edge_weight' not in self.data:
        #     self.data.edge_weight = torch.ones_like (self.data.edge_index[0], dtype=torch.float32)
        # if 'edge_attr' not in self.data:
        #     self.data.edge_attr = torch.ones_like (self.data.edge_index[0], dtype=torch.float32)[:, None]
        if (self.data.y.ndim > 1):
            # multiclass 
            if self.data.y.shape[1] == 1:
                # redundant (ogbn-arxiv)
                self.data.y = self.data.y[:, 0]
            else:
                self.data.y = self.data.y[:, 0]

        self.num_features = self.data.x.shape[1]
        self.num_classes = len(torch.unique(self.data.y))
        self.data.num_classes = self.num_classes
        self.device = device

        if not hasattr(self.data, 'train_mask'):
            self.data.train_idx, self.data.val_idx, self.data.test_idx = train_idx, val_idx, test_idx
        elif self.data.train_mask.ndim == 2:
            self.data.train_idx = torch.where(self.data.train_mask[:, nsplit])[0]
            self.data.val_idx = torch.where(self.data.val_mask[:, nsplit])[0]
            self.data.test_idx = torch.where(self.data.test_mask[:, nsplit])[0]
        else:
            self.data.train_idx = torch.where(self.data.train_mask)[0]
            self.data.val_idx = torch.where(self.data.val_mask)[0]
            self.data.test_idx = torch.where(self.data.test_mask)[0]
        # if name == 'ogbn-arxiv':
        #     import pandas as pd
        #     label_maps = torch.tensor(pd.read_csv("data/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv")["mapped"])
        #     self.data.y = label_maps[self.data.y]
        #     self.num_classes = len(self.data.y.unique())
            # print (df.loc[0, "mapped"], self.data.y.shape)
            # for j, yj in enumerate(self.data.y): self.data.y[j] = df.loc[yj, "mapped"]

    def poison_data(self, poisoning=False, link_p=0.0, feat_p=0.0, lambda_=0, ll_constraint=False):
        if (link_p == 0) and (feat_p == 0):
            return
        elif poisoning == 'mettack':
            if feat_p == 0.0:
                if os.path.exists (f"data/{self.name}/{self.split}/adj_{self.nsplit}_{link_p}.pt"):
                    adj = torch.load(f"data/{self.name}/{self.split}/adj_{self.nsplit}_{link_p}.pt", map_location=self.device)
                    adj = adj.to_sparse()
                    self.data.edge_index = adj.indices()
                    self.data.edge_weight = adj.values()
                    self.data = T.ToSparseTensor()(self.data)
                else:
                    from deeprobust.graph.defense import GCN
                    from deeprobust.graph.global_attack import Metattack
                    n_perturbations = int(link_p * self.data.adj_t.nnz()/2)
                    adj, features, labels = self.data.adj_t.to_dense().cpu(), self.data.x.cpu(), self.data.y.cpu()
                    idx_train, idx_val, idx_test = self.data.train_idx, self.data.val_idx, self.data.test_idx
                    # Setup Surrogate model
                    surrogate = GCN(nfeat=self.data.num_features, nclass=self.data.num_classes,
                                    nhid=64, dropout=0, with_relu=False, with_bias=False, device=self.device).to(self.device)
                    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
                    # Setup Attack Model
                    model = Metattack(surrogate, nnodes=self.data.num_nodes, feature_shape=self.data.num_features,
                                        attack_structure=True, attack_features=False, device=self.device, lambda_=lambda_).to(self.device)
                    # Attack
                    model.attack(features, adj, labels, idx_train, idx_test, 
                                 n_perturbations=n_perturbations, ll_constraint=ll_constraint)
                    self.data.adj_t = model.modified_adj.to_sparse(layout=torch.sparse_csr).to(self.device)
                    torch.save(self.data.adj_t, f"data/{self.name}/{self.split}/adj_{self.nsplit}_{link_p}.pt")
        elif poisoning == 'rand_miss':
            if link_p > 0.0:
                adj_coo = self.data.adj_t.to_torch_sparse_coo_tensor().coalesce()
                self.data.adj_t = None
                self.data.edge_index, _ = tg_utils.dropout_edge(adj_coo.indices(), p=link_p, force_undirected=True)
                self.data = T.ToSparseTensor()(self.data)
                # self.data.adj_t = tg_utils.to_torch_sparse_tensor(edge_index=edge_index)

            if feat_p > 0.0:
                feature_mask = get_feature_mask(rate=feat_p, n_nodes=self.data.num_nodes,
                                                n_features=self.data.num_features)
                self.data.x[~feature_mask] = 0.0