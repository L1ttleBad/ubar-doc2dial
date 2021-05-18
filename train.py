from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from torch.optim import Adam
import torch
import torch.nn as nn
import os
import random
import argparse
import time
import logging
import json
import numpy as np
from config import global_config as cfg







def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))  # this is the global cfg comes from config.py
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    # parse args first
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    if args.mode:
        cfg.mode = args.mode
    parse_arg_cfg(args)


    if not os.path.exists(cfg.experiment_path):
        os.mkdir(cfg.experiment_path)




if __name__ == '__main__':
    main()
