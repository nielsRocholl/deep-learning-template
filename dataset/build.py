from typing import Optional, Dict, Any
from utils import registry


DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> Any:
    """
    Build a dataset, defined by the 'NAME' key in the cfg dictionary.

    :param cfg: A dictionary containing the configuration for the dataset.
    :param default_args: Optional default arguments for building the dataset.
    :return: A constructed dataset specified by the NAME key in the cfg dictionary.
    """
    
    return DATASETS.build(cfg, default_args=default_args)




