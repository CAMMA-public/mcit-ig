"""
main.py to train/test
"""

import os
import math
import json
import torch
import numpy as np
import argparse
from lib.tool_tissue_parser import build_experiment
from lib.utils import parse_config, merge_a_into_b

def get_args():
    parser = argparse.ArgumentParser(description="MCIT-IG")
    parser.add_argument('--en', default="mcit_exp", type=str, 
                      help='experiment_name for Triplet Interaction Module')
    parser.add_argument('--cf', default="ivt_config.yaml", type=str, 
                      help='config yaml file to use for exps')
    parser.add_argument('--log_name', default='test_runs1.log', type=str,
                      help='log file name for storing per epoch results') 
    parser.add_argument("--wb", type=int, default=0, help='use wandb logger')
    parser.add_argument("--run", type=str, default='test', help='experiment run name')
    args = parser.parse_args()
    
    return args

# set seed
def set_seed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed) 

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def set_device_and_seed(args):
    if args.TRAIN.DEVICE == 'cuda' and torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    set_seed(args.TRAIN.SEED)

# train function
def train(args):
    set_device_and_seed(args)
    model = build_experiment(args)
    best_score, test_score, test_recall_score = model.train()
        
    return best_score, test_score, test_recall_score

def test_stage2(args):
    model = build_experiment(args)
    model.eval_stage2()

def visualize(args):
    model = build_experiment(args)
    model.visualize() 

def test_stage1(args):
    model = build_experiment(args)
    model.eval_stage1() 

def main():
    global args
    args_main = get_args()
    args = parse_config(args_main.cf).CONFIG
    print("Loading Config File :: ", args_main.cf)

    merge_a_into_b(args_main, args) # merge the yaml to the argspace values

    # specify the logfile
    args.logpath = os.path.join(args.TRAIN.CKPFOLDER, str(args.run), args.en)
    if not os.path.exists(args.logpath):
        os.makedirs(args.logpath)
        print("Experiment folder created successfully!!")

    args.logfile = os.path.join(args.logpath, f'{args.en}.log')
    print(f"[RUN] {args.run} Experiment files will be saved at >>> {args.logfile}")
    args.runfile = os.path.join(args.TRAIN.CKPFOLDER, f'run_{args.run}_summary.log')
    print(f"[RUN] {args.run} summary will be saved at >>> {args.runfile}")

    best_score, det_ivt_score = 0.0, 0.0
    if args.EXP.MODE == 1:
        best_score, det_ivt_score, det_recall_score = train(args)
        print(f"RUN: {str(args.run)} || EXP_NAME: {args.en} || SCORE (Target): {best_score:.5f} || (Triplet mAP): {det_ivt_score:.5f} || (Triplet Recall): {det_recall_score:.5f}", file=open(args.runfile, 'a+'))
    elif args.EXP.MODE == 2:
        test_stage2(args)
    elif args.EXP.MODE == 3:
        visualize(args) # TODO
    elif args.EXP.MODE == 4:
        test_stage1(args)
     
if __name__ == "__main__":
    main()