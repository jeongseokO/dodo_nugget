from typing import *
import logging

import torch
from lightning.pytorch.strategies import DDPStrategy
from .ds_strategy import MyDeepSpeedStrategy

from .common import logger


def gen_gpu_args(
        n_gpu: int, precision: str, strategy: str,
        deepspeed: Optional[int], offload_optim: bool, offload_param: bool
):
    if not bf16_available() and precision == 'bf16-mixed':
        precision = '16-mixed'
    # deepspeed stage is default to 2 if n_gpu > 0
    if torch.cuda.device_count() < n_gpu:
        logger.warning(f'Reduce the number of GPU to {torch.cuda.device_count()}')
        n_gpu = torch.cuda.device_count()
    if deepspeed is None:
        deepspeed = 2 if n_gpu > 1 else 0
    if strategy == 'deepspeed' and deepspeed > 0:
        logger.debug('Using DeepSpeed')
        return 'deepspeed_cpu' if offload_optim else 'deepspeed', {
            'accelerator': 'gpu', 'devices': n_gpu, 'precision': precision,
            'strategy': MyDeepSpeedStrategy(
                stage=deepspeed, offload_optimizer=offload_optim, offload_parameters=offload_param,
                logging_level=logging.ERROR
            ),
        }
    elif strategy == 'ddp' and n_gpu > 1:
        return 'ddp', {
            'strategy': DDPStrategy('gpu', find_unused_parameters=True),
            'devices': n_gpu, 'precision': precision,
        }
    elif n_gpu == 1:
        return 'single', {'devices': 1, 'accelerator': 'gpu', 'precision': precision}
    else:
        return 'cpu', {'accelerator': 'cpu'}


def bf16_available():
    if not torch.cuda.is_available():
        return False
    for name in ['A100', 'RTX 3090']:
        if name in torch.cuda.get_device_properties('cuda:0').name:
            return True
    return False
