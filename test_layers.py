import torch
import numpy as np
import yaml
import logging
from utils.dataset import Dataset
from utils.evaluator import Evaluate
from tqdm import tqdm
from utils.parser import parse
import time
import pickle as pkl
from torch import nn
import argparse
import sys
import datetime as dt
import os

# torch.autograd.set_detect_anomaly(True)

# class CustomDataParallel(nn.Module):
#     def __init__(self, model):
#         super(CustomDataParallel, self).__init__()
#         self.model = nn.DataParallel(model).cuda()

#     def forward(self, *input, **kwargs):
#         return self.model(*input, **kwargs)

#     def __getattr__(self, name):
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             return getattr(self.model.module, name)
        
if __name__ == '__main__':
    args, save_dir, train_config, model_config = parse()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    
    logging.basicConfig(filename=f'{args.log_name}',
                        filemode='a',
                        format='%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(f'{args.log_name}')
    logger.info(f'Model: {model_config}')
    logger.info(f'Train: {train_config}')

    if '.out' in args.out_name:
        fout = open (f'{args.out_name}', 'a')
        fout.write(f'Model: {model_config}' + "\n")
        fout.write(f'Train: {train_config}' + "\n")
    else:
        print (f'Model: {model_config}')
        print (f'Train: {train_config}')

        
    dataset = Dataset(root='data/', name=train_config.dataset_name, task=train_config.task, 
                        split=train_config.split, nsplit=train_config.nsplit, device=device, 
                        samples_per_class=train_config.samples_per_class if "samples_per_class" in train_config else None,
                        add_self_loops=train_config.add_self_loops if 'add_self_loops' in train_config else False)
    # dataset.poison_data(poisoning=train_config.poisoning, link_p=train_config.link_p, feat_p=train_config.feat_p,)
    # dataset.set_weak_info(missing_link=train_config.missing_link, missing_feature=train_config.missing_feature)
    data = dataset.data
    data.num_classes = dataset.num_classes


    data_per_class = torch.tensor([(data.y[data.train_idx] == c).sum() for c in torch.unique(data.y)]).to(device)
    # self.class_weights = (data_per_class ** 2)/(data_per_class ** 2).max() if data_per_class is not None else None
    # class_weights = (data_per_class > 9000).float() if model_config.class_weighted else None
    class_weights = None

    if model_config.base_model_params is not None and type(model_config.base_model_params) is list and len(model_config.base_model_params) > model_config.num_layers:
        model_config.num_layers = len(model_config.base_model_params)
        model_config.hidden_dims = [model_config.hidden_dims[0]] * model_config.num_layers
    elif model_config.base_model_params is not None and type(model_config.base_model_params) is list and len(model_config.base_model_params) < model_config.num_layers:
        model_config.hidden_dims = [model_config.hidden_dims[0]] * model_config.num_layers
        model_config.base_model_params = [model_config.base_model_params[0]] * model_config.num_layers
    elif model_config.base_model_params is None or type(model_config.base_model_params) is not list:
        model_config.base_model_params = [{}] * model_config.num_layers

    from models.gnn_scope import PersonalizedScopeGNN
    from utils.trainer import Trainer
    model = PersonalizedScopeGNN (task=train_config.task, dataset=dataset, class_weights=class_weights, device=device, **model_config).to(device) 
    trainer = Trainer(scope_selection=model_config.scope_selection, model_parameters=model.parameters(), dataset=dataset, 
                        train_config=train_config, model_config=model_config, class_weights=class_weights, device=device)
    
    evaluators = []
    for metric in train_config.metrics:
        evaluators.append(Evaluate (dataset=dataset, task=train_config.task, metric_name=metric))
    
    model.load_state_dict (torch.load(f'{save_dir}/state_dict.pt', map_location=device))
    if model_config.decoding == 'distance' or model_config.scope_selection.startswith('metselect'):
        model.protos = torch.load(f'{save_dir}/protos.pt').to(device)
        trainer_params = pkl.load(open(f'{save_dir}/trainer_params.pkl', 'rb'))
        trainer.variance = trainer_params['variance'].to(device)
        trainer.n_vars = trainer_params['n_vars']
        
        model.vars = trainer.variance_max
        trainer.n_vars = trainer.nvars_max
        
    model.eval()
    with torch.no_grad():
        metrics = [[None for _ in train_config.metrics] for _ in range(3)]
        y_hat, _, _, _ = trainer.test_data(model, data, big_data=data.num_nodes>100000)
        layers0 = model.forward_and_layers(data.x, idx=data.train_idx, adj_t=data.adj_t)
        layers1 = model.forward_and_layers(data.x, idx=data.val_idx, adj_t=data.adj_t)
        layers2 = model.forward_and_layers(data.x, idx=data.test_idx, adj_t=data.adj_t)
        layers = torch.cat((layers0, layers1, layers2))
        unique, layer_props = torch.unique(layers, return_counts=True)
        layer_props = {x: y for x, y in zip(unique.cpu().tolist(), layer_props.cpu().tolist())}
        metrics[0] = [evaluator (y_hat[data.train_idx], data.y[data.train_idx]) for evaluator in evaluators]
        metrics[1] = [evaluator (y_hat[data.val_idx], data.y[data.val_idx]) for evaluator in evaluators]
        metrics[2] = [evaluator (y_hat[data.test_idx], data.y[data.test_idx]) for evaluator in evaluators]

        
    if '.out' in args.out_name:
        fout.write("\n")
        fout.write("  Train-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[0]) if metric != 'report']) + "\n")
        fout.write("  Val-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[1]) if metric != 'report']) + "\n")
        fout.write("  Test-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[2]) if metric != 'report']) + "\n")
        fout.write("  Layers-> " +  str(layer_props) + "\n")
        fout.write("-"*100 + "\n")
        fout.close()
    else:
        print ()
        print ("Train-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[0]) if metric != 'report']) + "\n")
        print ("Val-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[1]) if metric != 'report']) + "\n")
        print ("Test-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[2]) if metric != 'report']) + "\n")
        print ("  Layers-> " +  str(layer_props) + "\n")
        print ("-"*100)
    logger.info ("-"*100)

    if args.temp_fout is not None:
        with open (f'{args.temp_fout}', 'w+') as wf:
            wf.write("Train-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[0]) if metric != 'report']) + "\n")
            wf.write("Val-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[1]) if metric != 'report']) + "\n")
            wf.write("Test-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[2]) if metric != 'report']) + "\n")
            wf.write("  Layers-> " +  str(layer_props) + "\n")
            