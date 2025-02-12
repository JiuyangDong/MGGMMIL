import argparse
import torch
import csv
import random
import os
import numpy as np
import logging
from train_mil import MIL

def parse_args_and_save():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--wsi_type", type=str, default="AF")
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--pretrain", type=str, default="ResNet50_ImageNet")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--baseline_model", type=str, default="ABMIL")
    parser.add_argument("--baseline_lr", type=float, default=2e-5)
    parser.add_argument("--baseline_wd", type=float, default=1e-5)
    parser.add_argument("--L", type=int, default=512)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--mil_head", type=str, default='GAMBL')
    parser.add_argument("--KNN", type=str, default=8)

    args = parser.parse_args()
    
    args.device = torch.device('cuda')

    if 'ResNet18' in args.pretrain or 'PLIP' in args.pretrain:
        args.feature_dim = 512
    else:
        raise NotImplementedError

    args.n_groups = 4
    
    return args

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def set_loggers(stdout_txt):
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(stdout_txt)
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel('INFO')

def make_expdir_and_logs(args):

    exp_dir = os.path.join('experiments/{}/pre_{}/{}/lr={}/layer={}-KNN={}-MILHead={}'.
        format(args.wsi_type, args.pretrain, args.baseline_model, args.baseline_lr, args.layer, args.KNN, args.mil_head)
    )

    args.exp_dir = exp_dir
    args.log_dir = os.path.join(args.exp_dir, 'logs')

    if os.path.exists(args.exp_dir):
        pass
    else:    
        os.makedirs(args.exp_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_loggers(os.path.join(args.log_dir, "{}-stdout-fold{}.txt".format(args.phase, args.k)))

    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp dir = {}".format(args.exp_dir))
    logging.info("Writing log file to {}".format(os.path.join(args.exp_dir, 'logs')))

def main(args):
    if args.phase == 'train':
        csv_name = 'valid_metrics.csv'
    elif args.phase == 'test':
        csv_name = 'test_metrics.csv'
    else:
        raise NotImplementedError
        
    with open(os.path.join(args.log_dir, csv_name), 'a') as f:
        writer = csv.writer(f)

        if args.k == 0:
            if args.phase == 'train':
                writer.writerow(['fold', 'valid_loss', 'valid_auc', 'valid_acc', 'valid_precision', 'valid_recall', 'valid_f1'])
            else:
                writer.writerow(['fold', 'test_loss', 'test_auc', 'test_acc', 'test_precision', 'test_recall', 'test_f1'])

        args.ckpt_dir = os.path.join(args.exp_dir, f'ckpts/fold-{args.k}')
        if not os.path.exists(args.ckpt_dir):   
            os.makedirs(args.ckpt_dir)
                
        MIL_runner = MIL(args)
        if args.phase == 'train':
            loss, auc, acc, precision, recall, f1 = MIL_runner.train()
        elif args.phase == 'test':
            loss, auc, acc, precision, recall, f1 = MIL_runner.test()
        else:
            raise NotImplementedError

        writer.writerow(['{}'.format(args.k)] + [round(loss, 4), round(auc, 4), round(acc, 4), round(precision, 4), round(recall, 4), round(f1, 4)])

if __name__ == '__main__':
    args = parse_args_and_save()
    seed_torch(args.seed)
    make_expdir_and_logs(args)
    main(args)
