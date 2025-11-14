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
import random

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
    random.seed(args.seed)
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
    dataset.poison_data(poisoning=train_config.poisoning, link_p=train_config.link_p, feat_p=train_config.feat_p,)
    # dataset.set_weak_info(missing_link=train_config.missing_link, missing_feature=train_config.missing_feature)
    data = dataset.data
    data.num_classes = dataset.num_classes


    data_per_class = torch.tensor([(data.y[data.train_idx] == c).sum() for c in torch.unique(data.y)]).to(device)
    # self.class_weights = (data_per_class ** 2)/(data_per_class ** 2).max() if data_per_class is not None else None
    # class_weights = (data_per_class > 9000).float() if model_config.class_weighted else None
    class_weights = None

    # if model_config.base_model_params is not None and type(model_config.base_model_params) is list and len(model_config.base_model_params) > model_config.num_layers:
    #     model_config.num_layers = len(model_config.base_model_params)
    #     model_config.hidden_dims = [model_config.hidden_dims[0]] * model_config.num_layers
    # elif model_config.base_model_params is not None and type(model_config.base_model_params) is list and len(model_config.base_model_params) < model_config.num_layers:
    #     model_config.hidden_dims = [model_config.hidden_dims[0]] * model_config.num_layers
    #     model_config.base_model_params = [model_config.base_model_params[0]] * model_config.num_layers
    # elif model_config.base_model_params is None or type(model_config.base_model_params) is not list:
    if 'base_model_params' not in model_config or model_config.base_model_params[0] == {}:
        model_config.base_model_params = [{}] * (model_config.num_layers)
    elif len(model_config.base_model_params) < model_config.num_layers:
        model_config.base_model_params = [model_config.base_model_params[0]] * (model_config.num_layers)
    
    if len(model_config.hidden_dims) < model_config.num_layers + 1:
        model_config.hidden_dims = [model_config.hidden_dims[0]] * (model_config.num_layers+1)
    
    from models.gnn_scope import PersonalizedScopeGNN
    from utils.trainer import Trainer
    model = PersonalizedScopeGNN (task=train_config.task, dataset=dataset, class_weights=class_weights, device=device, **model_config).to(device) 
    trainer = Trainer(scope_selection=model_config.scope_selection, model_parameters=model.parameters(), dataset=dataset, 
                        train_config=train_config, model_config=model_config, class_weights=class_weights, device=device)
        
    print ([name for name, param in model.named_parameters()])

    evaluators = []
    for metric in train_config.metrics:
        evaluators.append(Evaluate (dataset=dataset, task=train_config.task, metric_name=metric))

    train_losses = []
    val_losses = []

    epoch_bar = tqdm(total=train_config.num_epochs)
    start_time = time.time()
    # trainer.epoch_cluster_data(model, data)

    for epoch in range(1, train_config.num_epochs+1):
        epoch_bar.set_description(f'Epoch {epoch:02d}')

        if 'metselect' in model_config.scope_selection and model_config.warmup > 0:
            # warmup
            if (epoch < (model_config.warmup + 1)):
                model.scope_selection = 'lin-warmup'
                trainer.scope_selection = 'lin-warmup'
                # model.decoding = 'linear'
                # trainer.decoding = 'linear'
            else:
                model.scope_selection = model_config.scope_selection
                trainer.scope_selection = model_config.scope_selection
                model.decoding = model_config.decoding
                trainer.decoding = model_config.decoding
        
        metrics = [[None for _ in train_config.metrics] for _ in range(3)]

        train_mloss = trainer.train_data(model, data)

        y_hat, train_eloss, val_loss, test_loss = trainer.test_data(model, data, big_data=data.num_nodes>100000)

        metrics[0] = [evaluator (y_hat[data.train_idx], data.y[data.train_idx]) for evaluator in evaluators]
        metrics[1] = [evaluator (y_hat[data.val_idx], data.y[data.val_idx]) for evaluator in evaluators]
        metrics[2] = [evaluator (y_hat[data.test_idx], data.y[data.test_idx]) for evaluator in evaluators]

        train_losses.append(train_mloss)
        val_losses.append(val_loss)
        
        if epoch % 10 == 1:
            logger.info (f"Epoch: {epoch:02d} | " + \
                         f"Train Magnet Loss: {train_mloss:.5f} | " + \
                         f"Train CE Loss: {train_eloss:.5f} | " + \
                         ' | '.join([f'Train {met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[0]) if metric != 'report']) + \
                         f" | Val Loss: {val_loss:.5f} | " + \
                         ' | '.join([f'Val {met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[1]) if metric != 'report']) + \
                         f" | Test Loss: {test_loss:.5f} | " + \
                         ' | '.join([f'Test {met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[2]) if metric != 'report']))
            for i, metric in enumerate(train_config.metrics):
                if metric == 'report':
                    logger.info(f"Train: {metrics[0][i]}")
                    logger.info(f"Val: {metrics[1][i]}")
                    logger.info(f"Test: {metrics[2][i]}")
                    break


        if train_config.early_stopping and trainer.early_stop (model, train_mloss, val_loss):
            break
        else:
            trainer.store_best_params(model, metrics[1][0]) # acc

        epoch_bar.update(1)
        
    total_time = time.time() - start_time

    if trainer.state_dict is not None:
        model.load_state_dict (trainer.state_dict)
        if model_config.decoding == 'distance' or model_config.scope_selection.startswith('metselect'):
            model.protos = trainer.protos
            model.vars = trainer.variance_max
            trainer.n_vars = trainer.nvars_max

    if args.to_save:
        yaml.dump (yaml.load(open(f'{args.model_config}', 'r'), Loader=yaml.FullLoader), open(f'{save_dir}/model_config.yaml', 'w'))
        yaml.dump (yaml.load(open(f'{args.train_config}', 'r'), Loader=yaml.FullLoader), open(f'{save_dir}/train_config.yaml', 'w'))
        torch.save (model.state_dict(), f'{save_dir}/state_dict.pt')
        if model_config.decoding == 'distance' or model_config.scope_selection.startswith('metselect'):
            torch.save (model.protos, f'{save_dir}/protos.pt')
            pkl.dump ({'variance': trainer.variance_max, 'n_vars': trainer.nvars_max}, open(f'{save_dir}/trainer_params.pkl', 'wb'))

    model.eval()
    with torch.no_grad():
        metrics = [[None for _ in train_config.metrics] for _ in range(3)]
        y_hat, _, _, _ = trainer.test_data(model, data, big_data=data.num_nodes>100000)
        metrics[0] = [evaluator (y_hat[data.train_idx], data.y[data.train_idx]) for evaluator in evaluators]
        metrics[1] = [evaluator (y_hat[data.val_idx], data.y[data.val_idx]) for evaluator in evaluators]
        metrics[2] = [evaluator (y_hat[data.test_idx], data.y[data.test_idx]) for evaluator in evaluators]

        
    if '.out' in args.out_name:
        fout.write("\n")
        fout.write("  Train-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[0]) if metric != 'report']) + "\n")
        fout.write("  Val-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[1]) if metric != 'report']) + "\n")
        fout.write("  Test-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[2]) if metric != 'report']) + "\n")
        fout.write("  Time-> " +  str(total_time) + "\n")
        fout.write("  Nepochs-> " +  str(epoch) + "\n")
        fout.write("-"*100 + "\n")
        fout.close()
    else:
        print ()
        print ("Train-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[0]) if metric != 'report']) + "\n")
        print ("Val-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[1]) if metric != 'report']) + "\n")
        print ("Test-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[2]) if metric != 'report']) + "\n")
        print ("  Time-> " +  str(total_time) + "\n")
        print ("  Nepochs-> " +  str(epoch) + "\n")
        print ("-"*100)
    logger.info ("-"*100)

    if args.temp_fout is not None:
        with open (f'{args.temp_fout}', 'w+') as wf:
            wf.write("Train-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[0]) if metric != 'report']) + "\n")
            wf.write("Val-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[1]) if metric != 'report']) + "\n")
            wf.write("Test-> " + ' | '.join([f'{met_name}: {metric:.5f}' for met_name, metric in zip(train_config.metrics, metrics[2]) if metric != 'report']) + "\n")
            wf.write("Time-> " +  str(total_time) + "\n")
            wf.write("Nepochs-> " +  str(epoch) + "\n")
            
            