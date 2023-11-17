import time
import torch

from utils import parser
from utils.config import *
from utils.logger import *
from utils.misc import set_random_seed, initialize_wandb
from tools.model_trainer import fine_tune, pre_train
from tools.model_tester import test


def main():
    # args
    args = parser.get_args()
    # logger
    time_stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{time_stamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # config
    config = get_cfg(args=args, logger=logger)
    # CUDA
    args.use_gpu = True if config['device']['name'] == 'cuda' and torch.cuda.is_available() else False
    torch.backends.cudnn.benchmark = args.use_gpu
    # Weights and Biases
    initialize_wandb(args=args, config=config, project_name='Deep Learning Project')
    # log
    log_args_to_file(args=args, logger=logger)
    log_config_to_file(cfg=config, pre='config', logger=logger)
    # random seed 
    set_random_seed(logger=logger, seed=args.seed + args.local_rank, deterministic=args.deterministic)
    
    # run 
    if args.test:
        test()
    else:
        if args.finetune_model:
            fine_tune()
        else:
            pre_train()




    

if __name__ == '__main__':
    main()