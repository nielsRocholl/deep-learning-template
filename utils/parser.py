import os
import argparse
import logging
from pathlib import Path


def get_args() -> argparse.Namespace:
    """
    This function defines a base argument parses which we can use in or main file to handle the input
    arguments provided by the user. 
    """
    parser = argparse.ArgumentParser(description="Deep Learning Experiment Argument Parser")

    parser.add_argument('--config', type=str, help='YAML configuration file', required=True)
    parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--test', action='store_true', help="Test a trained model")
    parser.add_argument('--finetune', action='store_true', help="Finetune a pretrained model")
    parser.add_argument('--local_rank', type=int, default=0, help="Rank of the process in distributed training")
    parser.add_argument('--start_at_ckpnt', type=str, default=None, help='Path of a checkpoint to resume training from')
    parser.add_argument('--eval_at_ckpnt', type=str, default=None, help='Path of a checkpoint to evaluate the model')
    parser.add_argument('--resume_training', action='store_true', help='Flag to resume interruped training')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')

    
    args = parser.parse_args()
    validate_args(args)
    set_local_rank_environment_variable(args)
    configure_and_create_paths(args)

    return args


def validate_args(args: argparse.Namespace) -> None:
    """
    Performs basic validation on the parsed arguments.
    Raise ValueError if validation fails
    """
    if args.start_at_ckpnt and args.resume_training:
        raise ValueError("--start_at_ckpnt and --resume_training cannot be active simultaneously")
    
    if args.config and not Path(args.config).exists():
        raise ValueError(f'Config file {args.config} does not exist')


def configure_and_create_paths(args: argparse.Namespace) -> None:
    """
    Configure path arguments and create necessary paths
    """
    base_path = Path('./experiments')
    config_path = Path(args.config)
    args.experiment_path = base_path / config_path.stem / config_path.parent.stem / args.exp_name
    args.tfboard_path = base_path / config_path.stem / config_path.parent.stem / 'TFBoard' / args.exp_name
    args.log_name = config_path.stem

    create_experiment_dir(args.experiment_path)
    create_experiment_dir(args.tfboard_path)


def set_local_rank_environment_variable(args: argparse.Namespace) -> None:
    """
    Sets the LOCAL_RANK environment variable based on the provided local rank.

    :param local_rank: Rank of the process in distributed training.
    """
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)


def create_experiment_dir(path: Path) -> None:
    """
    Creates the specified directory path if it doesn't already exist.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f'Created path at {path}')
    except Exception as e:
        logging.error(f'Error creating path {path}: {e}')