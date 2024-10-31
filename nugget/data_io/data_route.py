import os
from typing import *

from torch.utils import data

from .text_dataset import load_data as load_text_data
from .eli5 import ELI5
from .data_utils import collate_fn
from .segment_text_to_text import SegmentT2T
from .evidence_qa import EvidenceQA
from .lmsumm import LMSumm


def load_t2t_data(
        constructor, batch_size: int, debug: bool, n_val: int, shuffle: bool, test: bool, shuffle_val: bool,
        **data_kwargs
) -> Tuple[data.DataLoader, data.DataLoader]:
    ret = []
    for split in ['train', 'dev']:
        if test:
            split = split.replace('dev', 'test').replace('validation', 'test')
        ds = constructor(
            **data_kwargs, split=split, max_size=n_val if split != 'train' else None,
            shuffle_indices=(split != 'train' and shuffle_val)
        )
        ret.append(data.DataLoader(
            ds, batch_size, shuffle=(split == 'train' and shuffle), num_workers=0 if debug else 1,
            collate_fn=collate_fn, prefetch_factor=None if debug else 32
        ))
    return ret[0], ret[1]


def load_data(data_, **kwargs):
    bn = os.path.basename(data_)
    kwargs['data_path'] = data_
    if data_.startswith('lmsumm:'):
        kwargs['data_path'] = data_[len('lmsumm:'):]
        return load_t2t_data(**kwargs, constructor=LMSumm)
    elif 'eli5' in bn:
        return load_t2t_data(**kwargs, constructor=ELI5)
    elif 'cnn_dailymail' in bn or 'trivia_qa' in bn:
        return load_t2t_data(**kwargs, constructor=SegmentT2T)
    elif 'squad' in bn.lower() or 'wikihow' in bn.lower():
        return load_t2t_data(EvidenceQA, **kwargs)
    else:
        return load_text_data(**kwargs)
