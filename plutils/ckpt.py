from typing import *
import yaml
import os
from dataclasses import dataclass
from collections import namedtuple
import re

import torch
from lightning.pytorch import loggers as pl_loggers, Trainer
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

from .common import logger

Checkpoint = namedtuple('checkpoint', ('step', 'top', 'model_path', 'predict_path'))


def global_step(path):
    fet = re.findall(r'epoch=0-step=(\d+)', path)
    if fet:
        return int(fet[0])
    return torch.load(path, map_location='cpu')['global_step']


def parse_ds_ckpt(path: str) -> Checkpoint:
    lightning_name = os.path.basename(path).replace('.ckpt', '.lightning.ckpt')
    predict_name = os.path.basename(path).replace('.ckpt', '.predict')
    model_path = os.path.join(os.path.dirname(path), lightning_name)
    if not os.path.exists(model_path):
        convert_zero_checkpoint_to_fp32_state_dict(path, model_path)
    return Checkpoint(
        global_step(os.path.join(path, 'checkpoint', 'mp_rank_00_model_states.pt')),
        path, model_path, os.path.join(os.path.dirname(path), predict_name)
    )


def parse_lightning_ckpt(path: str) -> Checkpoint:
    # a hack to prioritize deepspeed ckpt
    return Checkpoint(global_step(path)-1, path, path, path+'.predict')


def parse_checkpoint(path: str) -> Checkpoint:
    return (parse_lightning_ckpt if os.path.isfile(path) else parse_ds_ckpt)(path)


def find_ckpt(ckpt) -> Optional[Checkpoint]:
    if os.path.isfile(ckpt):
        return parse_lightning_ckpt(ckpt)
    if os.path.isdir(os.path.join(ckpt, 'ckpt')):
        ckpt = os.path.join(ckpt, 'ckpt')
    all_ckpt = list()
    if ckpt.endswith('.ckpt'):
        all_ckpt.append(parse_checkpoint(ckpt))
    else:
        for fn in os.listdir(ckpt):
            if fn.endswith('.ckpt'):
                all_ckpt.append(parse_checkpoint(os.path.join(ckpt, fn)))
    if len(all_ckpt) == 0:
        return None
    all_ckpt.sort()
    return all_ckpt[-1]


def process_ckpt(args):
    assert os.path.exists(args.ckpt), f'ckpt {args.ckpt} not exist'
    ckpt = find_ckpt(args.ckpt)
    logger.warning(f'Resuming from {ckpt.top}.')
    log_dir = os.path.dirname(os.path.dirname(ckpt.top))
    version = os.path.basename(log_dir)
    hparams = yaml.load(open(os.path.join(log_dir, 'hparams.yaml')), yaml.Loader)
    return hparams, log_dir, version, ckpt
