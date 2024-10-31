import logging
import os
import warnings
import datasets
import re
from collections import defaultdict

import torch


def num_seq(nums):
    seq = []
    last_n = -99999
    for n in sorted(nums):
        if n == last_n + 1:
            seq[-1][1] = n
        else:
            seq.append([n, None])
        last_n = n
    return '[' + ','.join([str(s) if e is None else f'{s}-{e}' for s, e in seq]) + ']'


def param_names(params, trainable=True):
    if isinstance(params, torch.nn.Module):
        params = [n for n, p in params.named_parameters() if p.requires_grad or not trainable]
    seen, ret, nums = set(), [], defaultdict(list)
    for p in params:
        if re.findall(r'\d+', p):
            n = int(re.findall(r'\d+', p)[0])
            p = re.sub(r'(\d+)', '#', p)
            nums[p].append(n)
        if p not in seen:
            seen.add(p)
            ret.append(p)
    return [r if r not in nums else r.replace('#', num_seq(nums[r])) for r in ret]


def configure_logger():
    logger_ = logging.getLogger('pl')
    stm_hdl = logging.StreamHandler()
    logger_.addHandler(stm_hdl)
    logger_.setLevel('INFO')
    return logger_


def get_process_logger():
    logger_ = logging.getLogger('process')
    stm_hdl = logging.StreamHandler()
    local_rank = os.environ.get('LOCAL_RANK')
    fmt = logging.Formatter(f'[rk={local_rank}] %(message)s')
    stm_hdl.setFormatter(fmt)
    logger_.addHandler(stm_hdl)
    if 'DEBUG_PROCESS' in os.environ:
        logger_.setLevel('DEBUG')
    else:
        logger_.setLevel('WARNING')
    return logger_


def filter_warnings():
    datasets.utils.logging.set_verbosity_error()
    datasets.builder.logger.setLevel('ERROR')
    for keyword in [
        'Using the latest cached version of the module',
        'Found cached dataset',
        'NCCL backend in DeepSpeed not yet implemented',
        'does not have many workers which',
        # '\'plutils.args\' found in',
        'Setting ds_accelerator to cuda',
        'LOCAL_RANK.*CUDA_VISIBLE_DEVICES',
        'UserWarning: Positional args are being deprecated',
    ]:
        warnings.filterwarnings('ignore', '.*' + keyword + '.*')


logger = configure_logger()
process_logger = get_process_logger()
