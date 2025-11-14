import yaml
from easydict import EasyDict as edict
import os
import argparse
import datetime as dt 
import sys
import copy
import csv
from collections import defaultdict
import itertools
import subprocess
from multiprocessing import Pool
import random

EPS_INV = 2

# fname = "results_graphsage.csv" # graph sages
# fname = "results_11_24.csv" # for sgconv and appnp
# fname = "results_11_10.csv" # for others
# fname = "results_12_19.csv" # for gcn different layers
# fname = "results_randsplit.csv" # for others
# fname = "results_poison.csv"
fname = "results_new_gcn_all.csv"

num_clusters = {
    # num_layers: clusters
    1: [[4, 7], [8, 7], [16, 7]],
    2: [[32, 32, 32], [128, 128, 128]], # [[16, 16], [32, 16], [64, 16]],
    # [[x, y, y] for x in range(1, 33) for y in range(1, 17)], # [[5, 5], [41, 4]], #[[2, 2, 7], [4, 4, 7], [8, 4, 7], [16, 4, 7], [32, 32, 7]],
    3: [[4, 4, 4, 7], [8, 4, 2, 7], [16, 8, 4, 7]]
}

# min_mu_dist = [1e-3, 1e-5, 1e-7]

trainConfigs_dict = {
    'task': ['node_classification'],
    'dataset_name': ['Cora', 'Citeseer', 'Pubmed', 'Cornell', 'Wisconsin', 
                     'Texas', 'Squirrel', 'Chameleon', 'Actor', 'ogbn-arxiv'],
    # , 'ogbn-arxiv'],
    'nnbrs': [[-1, -1]],
    'split': ['geom-gcn'],
    # 'split': ['random'],
    'nsplit': list(range(10)),
    # 'nsplit': [0], #, 7],
    # 'samples_per_class': [20, 16, 10, 8, 5, 2], #, 7],
    # 'lr': [0.01, 0.001, 0.0001, 0.00001], #, 0.01],
    'lr': [0.01, 0.001], #, 0.01],
    'num_epochs': [300], # [50], # [10, 20, 50],
    # 'poisoning': ['mettack'],
    # 'link_p': [0.02, 0.05, 0.1, 0.2, 0.5],
    # 'batched_cluster': [True],
}

modelConfigs_dict = {
    'model_name': ['gcn'],
    'base_model_params': [
        [{}, {}],
        # [{'aggr': 'add'}, {'aggr': 'add'}],
        # [{'aggr': 'max'}, {'aggr': 'max'}],
        # [{'aggr': 'min'}, {'aggr': 'min'}]
        # [{'K': 2, 'add_self_loops': False},
        #  {'K': 4, 'add_self_loops': False},
        # ],
        # [{'K': 0, 'add_self_loops': False},
        #  {'K': 1, 'add_self_loops': False},
        # ],
        # [{'K': 0, 'add_self_loops': False},
        #  {'K': 1, 'add_self_loops': False},
        #  {'K': 2, 'add_self_loops': False},
        #  {'K': 4, 'add_self_loops': False},
        #  {'K': 8, 'add_self_loops': False},
        # ],
        # [{'K': 0, 'add_self_loops': False},
        #  {'K': 1, 'add_self_loops': False},
        #  {'K': 2, 'add_self_loops': False},
        #  {'K': 4, 'add_self_loops': False},
        # ],
        # [{'K': 2, 'add_self_loops': False},
        #  {'K': 4, 'add_self_loops': False},
        #  {'K': 8, 'add_self_loops': False},
        # ],
    ],
    'scope_selection': ['metselect-min'], #'final', 'ndls'], #metselect-wo-min metselect , 'metselect-max'
    'decoding': ['linear'], 
    'num_layers': [2, 4],
    'hidden_dims': [64],
    'magnet_alpha': [1],
    'k_nearest': [-1],
    'warmup': [0, 10, 20, 30],
    'dist_metric': ['mahla'],
    'only_closest_layer': [False], #[True, False],
    'batch_hard_thresh': [100],
}

def update (d, u, v):
    if (type(u) is str):
        d[u] = v
        return d
    elif len(u) == 1:
        d[u[0]] = v
        return d
    else:
        if (type (d[u[0]]) is list) and (type(v) is list):
            if len(d[u[0]]) != len(v):
                dvals = [copy.deepcopy(d[u[0]][0]) for _ in range(len(v))]
            else:
                dvals = d[u[0]]
            dnewvals = []
            for di, vi in zip (dvals, v):
                di_new = update (di, u[1:], vi)
                dnewvals.append(di_new)
            d[u[0]] = dnewvals
        else:
            update(d.get(u[0], {}), u[1:], v)
        
def is_layered_key (x):
    return (x[0] in ['dim_red_params', 'cluster_params', 'proto_params', 'aug_params']) or (x == 'hidden_dims')

def is_cluster_key (x):
    return x == ('dim_red_params', 'out_dim')

def generate_setting (config, val_dict, sel_val_dict=defaultdict(lambda:None), curr_val=0):
    keys = list(val_dict.keys())
    if curr_val == len(keys):
        yield config, sel_val_dict
    else:
        curr_key = keys[curr_val]
        # print ([c['algorithm'] for c in config['dim_red_params']])
        if (((curr_key == ('dim_red_params', 'hidden_dim')) and all([c['algorithm'] == 'svd' for c in config['dim_red_params']])) or 
            ((curr_key == ('dim_red_params', 'in_dim')) and all([c['algorithm'] == 'svd' for c in config['dim_red_params']])) or
            ((curr_key == ('cluster_params', 'inv_temp')) and (all([not(c['soft_match']) for c in config['cluster_params']])))):
            # ignore
            yield from generate_setting (config, val_dict, sel_val_dict, curr_val+1)
        else:
            if is_cluster_key (curr_key):
                values = [val_dict[curr_key]([x['num_clusters'] for x in config['cluster_params']], config['hidden_dims'])]
            elif is_layered_key (curr_key):
                try:
                    values = val_dict[curr_key](config['num_layers'])
                except:
                    values = [[v]*(config['num_layers']+1) for v in val_dict[curr_key]]
            else:
                values = val_dict[curr_key]
            for v in values:
                sel_val_dict[curr_key] = v
                if curr_key == 'dataset_name': 
                    if v == 'ogbn-arxiv':
                        update (config, curr_key, v)
                        val_dict['nsplit'] = [0]
                    else:
                        update (config, curr_key, v)
                    yield from generate_setting (config, val_dict, sel_val_dict, curr_val+1)
                elif curr_key == 'to_magnet': 
                    if v:
                        update (config, curr_key, v)
                        yield from generate_setting (config, val_dict, sel_val_dict, curr_val+1)
                        # dataset, model, num_epochs, lr, hidden_dim, dropout num_layers come earlier
                    else:
                        update (config, curr_key, v)
                        # curr_key, v = 'decoder', 'id'
                        # update (config, curr_key, v)
                        # dataset, model, num_epochs, lr, hidden_dim, dropout num_layers come earlier
                        yield from generate_setting (config, val_dict, sel_val_dict, curr_val+1)
                        # yield config, sel_val_dict
                else:
                    update (config, curr_key, v)
                    yield from generate_setting (config, val_dict, sel_val_dict, curr_val+1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument ('--device', type=str, default='cpu')
    parser.add_argument ('--seed', type=int, default=42)
    args = parser.parse_args (sys.argv[1:])

    model_config = yaml.load(open(f'model_config.yaml', 'r'), Loader=yaml.FullLoader)
    train_config = yaml.load(open(f'train_config.yaml', 'r'), Loader=yaml.FullLoader)

    train_names = list(trainConfigs_dict.keys())
    model_names = list(modelConfigs_dict.keys())
    out_names = ['Train', 'Val', 'Test', 'nclusters', 'Time', 'Nepochs']
    temp_dir = f"temp" #{'|'.join(trainConfigs_dict['dataset_name'])}_{'|'.join(modelConfigs_dict['model_name'])}"
    temp_model_config = f'{temp_dir}/temp_model_config'
    temp_train_config = f'{temp_dir}/temp_train_config'

    try:
        os.makedirs(temp_dir)
    except:
        pass
    temp_outfile = f'{temp_dir}/temp'

    # with open (f"results/agglo_cluster{'|'.join(trainConfigs_dict['dataset_name'])}_{'|'.join(modelConfigs_dict['model_name'])}.csv", 'a') as wf:
    with open (f"results/{fname}", 'a') as wf:
        writer = csv.DictWriter(wf, fieldnames=train_names + model_names + out_names)
        writer.writeheader()
        max_processes = 10
        rows, processes = [], []
        free_pidx, active_proc_to_pidx = list(range(max_processes)), {}
        for train_setting, train_sets in generate_setting (train_config, trainConfigs_dict):
            for model_setting, model_sets in generate_setting (model_config, modelConfigs_dict):
                pidx = free_pidx.pop(0); row = {}
                yaml.dump(train_setting, open(f'{temp_train_config}_{pidx}.yaml', 'w'))
                for x in train_names: row[x] = train_setting[x]
                yaml.dump(model_setting, open(f"{temp_model_config}_{pidx}.yaml", 'w'))
                for x in model_names: row[x] = model_sets[x]
                # print (train_setting, model_setting)
                proc = subprocess.Popen (["python3", "train.py", "--train_config", f"{temp_train_config}_{pidx}.yaml", 
                                            "--model_config", f"{temp_model_config}_{pidx}.yaml", "--log_name", "temp_gcn.log", 
                                            "--out_name", f'temp_gcn.out', "--temp_fout", f'{temp_outfile}_{pidx}.out', 
                                            "--device", args.device, "--seed", f"{args.seed}"])
                active_proc_to_pidx[len(processes)] = pidx
                processes.append(proc); rows.append(row)
                if len(free_pidx) == 0:
                    os.wait()
                    # processes.difference_update(
                    #     [p for p in processes if p.poll() is not None])
                    new_processes, new_rows, new_active_proc_to_pidx = [], [], {}
                    for i, p in enumerate(processes):
                        row, pidx = rows[i], active_proc_to_pidx[i]
                        if p.poll() is not None:
                            with open (f'{temp_outfile}_{pidx}.out', 'r') as f:
                                for line in f:
                                    try:
                                        split_name, output = line[:-1].split('-> ', maxsplit=1)
                                    except:
                                        for name in out_names: row[name] = -1
                                    if 'Num Clusters' in line: 
                                        row['nclusters'] = output
                                    else:
                                        row[split_name] = output
                            writer.writerow(row)
                            free_pidx.append(pidx)
                        else:
                            new_active_proc_to_pidx[len(new_processes)] = active_proc_to_pidx[i]
                            new_rows.append(row); new_processes.append(p)
                    processes = new_processes
                    active_proc_to_pidx = new_active_proc_to_pidx
                    rows = new_rows
                    
        for i, p in enumerate(processes):
            row = rows[i]
            if p.poll() is None:
                p.wait()
                with open (f'{temp_outfile}_{active_proc_to_pidx[i]}.out', 'r') as f:
                    for line in f:
                        split_name, output = line[:-1].split('-> ', maxsplit=1)
                        if 'Num Clusters' in line: 
                            row['nclusters'] = output
                        else:
                            row[split_name] = output
                writer.writerow(row)
