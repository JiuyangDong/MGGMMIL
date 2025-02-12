import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str)
parser.add_argument("--lr", type=float)
parser.add_argument("--pretrain", type=str, default='ResNet18_ImageNet')
parser.add_argument("--baseline_model", type=str, default='AFMIL-v1')
parser.add_argument("--wsi_type", type=str, default="AF")
parser.add_argument("--layer", type=int, default=2)
parser.add_argument("--KNN", type=str, default=8)
parser.add_argument("--L", type=int, default=512)
parser.add_argument("--mil_head", type=str, default='GAMBL')
parser.add_argument("--data_dir", type=str, default="")

args = parser.parse_args()

for k in range(1):
    cmd = 'CUDA_VISIBLE_DEVICES={} python baseline.py --phase train --wsi_type {} --pretrain {} --baseline_model {} --baseline_lr {} --k {} --layer {} --KNN {} --L {} --mil_head {} --data_dir {}'.\
        format(args.gpu_id, args.wsi_type, args.pretrain, args.baseline_model, args.lr, k, args.layer, args.KNN, args.L, args.mil_head, args.data_dir)
    os.system(cmd)

    # cmd = 'CUDA_VISIBLE_DEVICES={} python baseline.py --phase test --wsi_type {} --pretrain {} --baseline_model {} --baseline_lr {} --k {} --layer {} --KNN {} --L {} --mil_head {} --data_dir {}'.\
    #     format(args.gpu_id, args.wsi_type, args.pretrain, args.baseline_model, args.lr, k, args.layer, args.KNN, args.L, args.mil_head, args.data_dir)
    # os.system(cmd)
