from typing import *
import os
from copy import deepcopy
import math
import json

import torch
from torch.utils import data
from datasets import load_from_disk
from .lazy_data import LazyDataset
from .data_utils import FakeEOS
from .data_utils import collate_fn
from ..utils.types import Objective


class ELI5(LazyDataset):
    def __init__(
            self, data_path: str, split: str, pretrained: str, max_size: Optional[int],
            max_q: int, max_d: int, max_a: int, n_doc: Optional[int], shuffle_indices: bool = False, **_
    ):
        super().__init__(pretrained)
        self.data_path, self.max_size = data_path, max_size or 1e20
        self.max_q, self.max_d, self.max_a, self.n_doc = max_q, max_d, max_a, n_doc
        self._data = load_from_disk(data_path)
        split = split.replace('dev', 'validation')
        if split in ['train', 'validation', 'test']:
            self.split = [f'{split}_{suffix}' for suffix in ['eli5', 'asks', 'askh']]
        else:
            self.split = [split]
        if split.startswith('test') or split.startswith('validation'):
            self.answer_mapping = {k: [(i, 0) for i in range(len(self._data[k]))] for k in self.split}
        else:
            self.answer_mapping = {
                k: json.load(open(os.path.join(data_path, k, 'answer_mapping.json')))
                for k in self.split
            }
        self.split_sizes = [len(self.answer_mapping[sp]) for sp in self.split]
        if shuffle_indices and split in ['validation', 'test'] and max_size is not None:
            self.split_sizes = [math.ceil(ss/sum(self.split_sizes)*max_size) for ss in self.split_sizes]
        self.init()
        self.fake_eos = FakeEOS(self.tokenizer)

    def __len__(self):
        return min(sum(self.split_sizes), self.max_size)

    def idx2example(self, idx: int):
        split_ord, split_idx = 0, idx
        while split_idx >= self.split_sizes[split_ord]:
            split_idx -= self.split_sizes[split_ord]
            split_ord += 1
        split_name = self.split[split_ord]
        exp_idx, answer_idx = self.answer_mapping[split_name][split_idx]
        exp = self._data[split_name][exp_idx]
        ret = deepcopy(exp)
        ret['answer'] = exp['answers']['text'][answer_idx]
        return ret

    def __getitem__(self, idx: int):
        self.init()
        exp = self.idx2example(idx)
        docs = list(filter(lambda st: len(st) > 0, map(lambda st: st.strip(), exp['document'].split('<P>'))))
        if self.n_doc is not None:
            docs = docs[:self.n_doc]
        d = [
            self.tokenizer(doc, truncation=True, add_special_tokens=True, max_length=self.max_d)['input_ids']
            for doc in docs
        ]
        q = self.tokenizer(
            exp['title'].strip()+'\n\n', truncation=True, add_special_tokens=False, max_length=self.max_q
        )['input_ids']
        a = self.tokenizer(
            exp['answer'].strip(), truncation=True, add_special_tokens=False, max_length=self.max_a
        )['input_ids'] + [self.tokenizer.eos_token_id]
        d = self.fake_eos(d)
        return {
            'tgt_input_ids': torch.tensor(q+a), 'src_input_ids': list(map(torch.tensor, d)),
            'skip': len(q), 'objective': Objective.TextContinuation,
            'meta': {k: exp[k] for k in ['q_id', 'answers', 'subreddit', 'title']}
        }

