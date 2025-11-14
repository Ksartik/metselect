import sys
import datetime as dt
import os
import yaml
from easydict import EasyDict as edict
import argparse

def parse (args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument ('--model_config', type=str, default='model_config.yaml')
        parser.add_argument ('--train_config', type=str, default='train_config.yaml')
        parser.add_argument ('--debug', action='store_true')
        parser.add_argument ('--log_name', type=str, default=f"log/{dt.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log")
        parser.add_argument ('--out_name', type=str, default=f"out/{dt.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.out")
        parser.add_argument ('--save_dir', type=str, default=f"/")
        parser.add_argument ('--temp_fout', type=str, default=None)
        parser.add_argument ('--to_save', action='store_true')
        parser.add_argument ('--device', type=str, default='cuda:0')
        parser.add_argument ('--seed', type=int, default=42)
        args = parser.parse_args (sys.argv[1:])
    
    if args.debug:
        args.log_name = 'log/debug.log'
        args.out_name = 'out/debug.out'
    
    model_config = edict(yaml.load(open(f'{args.model_config}', 'r'), Loader=yaml.FullLoader))
    train_config = edict(yaml.load(open(f'{args.train_config}', 'r'), Loader=yaml.FullLoader))
    
    save_dir = f'saved_models/{train_config.task}/{train_config.dataset_name}/{model_config.model_name}/{model_config.scope_selection}/{args.save_dir}'
    if model_config.base_model_params is not None and len(model_config.base_model_params) > model_config.num_layers and len(model_config.base_model_params[0]) > 0:
        if 'K' in model_config.base_model_params[0]:
            ks = ','.join(map(str, [param['K'] for param in model_config.base_model_params]))
            save_dir = f'saved_models/{train_config.task}/{train_config.dataset_name}/{model_config.model_name}_{ks}/{model_config.scope_selection}/{args.save_dir}'
    if args.to_save:
        os.makedirs(save_dir, exist_ok=True)
        args.log_name = f'{save_dir}/saved.log'
        args.out_name = f'{save_dir}/saved.out'

    return args, save_dir, train_config, model_config
