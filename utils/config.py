import yaml
import os
import argparse

from easydict import EasyDict
from typing import Optional, Dict
from .logger import print_log


def log_args_to_file(args: argparse.Namespace, pre: str='args', logger: Optional[str]=None) -> None:
    """
    Logs the attributes of the argparse Namespace object to a file.
    
    :param args: argparse Namespace containing the arguments.
    :param pre: Prefix for log messages.
    :param logger: Optional logger name.
    """
    for key, val in vars(args).items():
        print_log(f'{pre}.{key} : {val}', logger=logger)


def log_config_to_file(cfg: Dict, pre: str='cfg', logger: Optional[str]=None) -> None:
    """
    Logs the configuration dictionary to a file, handling nested EasyDict objects.
    
    :param cfg: Configuration dictionary.
    :param pre: Prefix for log messages.
    :param logger: Optional logger name.
    """
    for key, val in cfg.items():
        if isinstance(val, EasyDict):
            print_log(f'{pre}.{key} = edict()', logger = logger)
            log_config_to_file(val, pre=f'{pre}.{key}', logger=logger)
        else:
            print_log(f'{pre}.{key} : {val}', logger=logger)

def merge_new_config(cfg: Dict, new_cfg: Dict) -> Dict:
    """
    Merges a new configuration dictionary into an existing one.
    
    :param cfg: Existing configuration dictionary.
    :param new_cfg: New configuration dictionary.
    :return: Merged configuration dictionary.
    """
    for key, val in new_cfg.items():
        if key == '_base_':
            with open(new_cfg['_base_'], 'r') as f:
                val = yaml.safe_load(f)
                cfg[key] = EasyDict()
                merge_new_config(cfg[key], val)
        else:
                cfg[key] = val
    return cfg

def cfg_from_yaml_file(cfg_file: str) -> EasyDict:
    """
    Loads and returns a configuration from a YAML file.
    
    :param cfg_file: Path to the YAML configuration file.
    :return: Configuration as an EasyDict.
    """
    cfg = EasyDict()
    with open(cfg_file, 'r') as f:
        new_cfg = yaml.safe_load(f)
        merge_new_config(cfg, new_cfg)
    return cfg

def get_cfg(args: argparse.Namespace, logger: Optional[str]=None) -> EasyDict:
    """
    Retrieves the configuration either from a specified file or from a previous experiment.
    
    :param args: argparse Namespace containing the arguments.
    :param logger: Optional logger name.
    :return: Configuration as an EasyDict.
    """
    if args.resume_training:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log('Failed to resume training', logger=logger)
            raise FileNotFoundError(f'Config file not found: {cfg_path}')
        print_log(f'Resume yaml from {cfg_path}', logger=logger)
        args.config = cfg_path
    cfg = cfg_from_yaml_file(args.config)
    if not args.resume_training and args.local_rank == 0:
        save_experiment_config(args, logger)
    return cfg


def save_experiment_config(args: argparse.Namespace, logger: Optional[str]=None) -> None:
    """
    Saves the experiment configuration to a file.
    
    :param args: argparse Namespace containing the arguments.
    :param config: Configuration to save.
    :param logger: Optional logger name.
    """
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system(f'cp {args.config} {config_path}')
    print_log(f'Copy the Config file from {args.config} to {config_path}',logger = logger )
