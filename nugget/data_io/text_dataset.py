from typing import *
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils import data
from datasets import load_dataset

from ..utils.types import Objective, str2obj
from .data_utils import collate_fn, FakeEOS
from .lazy_data import LazyDataset
from ..utils.wiki import iterate_articles


class WikiText(LazyDataset):
    def __init__(
            self, tokenizer_name, max_length: int, split: str, data_length: Optional[int], **_
    ):
        super().__init__(tokenizer_name)
        self.split, self.max_length, self.data_length = split, max_length, data_length
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split.replace('dev', 'validation'))
        self.sections = list()
        for article in iterate_articles(dataset, True):
            self.sections.extend(article)

    def __len__(self):
        if self.data_length is None:
            return len(self.sections)
        return min(len(self.sections), self.data_length)

    def __getitem__(self, idx: int):
        text = ''.join(self.sections[idx]).strip()
        self.init()
        return self.tokenizer(
            text, add_special_tokens=False, max_length=self.max_length, truncation=True
        )['input_ids']


class Corpus(LazyDataset):
    def __init__(
            self, tokenizer_name, dataset_name: str, min_length: int, max_length: int, split: str,
            data_length: Optional[int]
    ):
        super().__init__(tokenizer_name, {'use_fast': False}, dataset_name, None, split, data_length)
        self.min_length = min_length
        self.max_length = max_length

    def __getitem__(self, item) -> List[int]:
        self.init()
        text = self._data[item]['text']
        ret = self.tokenizer(
            text, add_special_tokens=False, max_length=self.max_length, truncation=True,
        )['input_ids']
        # discard short sequences
        if len(ret) < self.min_length:
            return self[(item+1) % len(self._data)]
        return ret


class TextDataset(data.Dataset):
    obj_period = 16 * len(str2obj)

    def __init__(
        self, corpus: Corpus, max_length: int, max_decode: int, obj: str, split_strategy: str
    ):
        # for the text continuation objective, max_decode is the maximum number of tokens in the target side.
        self.corpus: Corpus = corpus
        self.max_length, self.max_decode, self.obj, self.split_strategy = max_length, max_decode, obj, split_strategy
        self.corpus.init()
        self.fake_eos = FakeEOS(self.corpus.tokenizer)

    def __len__(self) -> int:
        if self.obj == 'mix':
            n = len(self.corpus) * len(str2obj)
            return n - n % self.obj_period
        return len(self.corpus)

    def corpus_and_obj(self, idx: int) -> Tuple[List[int], int]:
        if self.obj == 'mix':
            sub_total = self.obj_period // len(str2obj)
            period_cnt = idx // self.obj_period
            obj_idx = idx % self.obj_period // sub_total
            obj = list(str2obj.values())[obj_idx]
            corpus_idx = period_cnt * sub_total + idx % self.obj_period % sub_total
            return self.corpus[corpus_idx], obj
        else:
            return self.corpus[idx], str2obj[self.obj]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # AE: BOS + seq1 + fake_eos -> BOS + seq1 + EOS
        # TC: BOS + seq1 + fake_eos -> seq2
        tokens, obj = self.corpus_and_obj(idx)
        if obj == Objective.TextContinuation:
            # both of inputs and outputs do not contain special tokens
            tokens = tokens[:self.max_length + self.max_decode]
            if self.split_strategy == 'middle':
                tgt_len = self.max_decode if len(tokens) > self.max_decode * 2 else len(tokens) // 2
            else:
                tgt_len = min(self.max_decode, len(tokens))
            inputs, outputs = tokens[:-tgt_len], tokens[-tgt_len:]
            # if len(inputs) == 0:
            #     outputs = [self.corpus.tokenizer.bos_token_id] + outputs
        else:
            # outputs contains eos token
            inputs, outputs = deepcopy(tokens[:self.max_length]), deepcopy(tokens[:self.max_length])
            outputs = [self.corpus.tokenizer.bos_token_id] + outputs + [self.corpus.tokenizer.eos_token_id]
        # add bos and fake_eos
        inputs = self.fake_eos([self.corpus.tokenizer.bos_token_id] + inputs)
        return {
            'src_input_ids': torch.tensor(inputs), 'tgt_input_ids': torch.tensor(outputs), 'objective': obj,
            'meta': {'inputs': np.array(inputs, dtype=np.uint16), 'outputs': np.array(outputs, dtype=np.uint16)}
        }


def load_data(
        pretrained: str, data_path: str, min_length: int, max_length: int, max_decode: int, batch_size: int,
        n_val: int, debug: bool, shuffle: bool, obj: str, test: bool, split_strategy: str, **_
) -> Tuple[data.DataLoader, data.DataLoader]:
    ret = []
    for i, split in enumerate(['train', 'dev' if not test else 'test']):
        if 'wikitext' in data_path.lower():
            corpus = WikiText(pretrained, max_length=max_length, split=split, data_length=[None, n_val][i])
        else:
            if os.path.exists(data_path):
                split = ('train', 'test')[i]
            else:
                split = (f'train[:-{n_val}]', f'train[{-n_val}:]')[i]
            corpus = Corpus(
                tokenizer_name=pretrained, dataset_name=data_path, min_length=min_length,
                max_length=max_length-1+max_decode, split=split, data_length=[None, n_val][i]
            )
        dataset = TextDataset(corpus, max_length, max_decode, obj, split_strategy)
        ret.append(data.DataLoader(
            dataset, batch_size, shuffle=(shuffle and i == 0), collate_fn=collate_fn,
            num_workers=1 if not debug else 0, prefetch_factor=16 if not debug else None,
        ))
    return ret[0], ret[1]
