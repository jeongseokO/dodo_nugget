from typing import *
import math

import torch
from torch.utils import data
from datasets import load_dataset
from transformers import AutoTokenizer

from ..utils.types import Objective


class LMSumm(data.Dataset):
    def __init__(self, data_path, prompt, ratio, max_size, max_length, split, **_):
        super().__init__()
        split = split.replace('dev', 'validation')
        self.data_path, self.ratio, self.max_size, self.max_length = data_path, ratio, max_size, max_length
        if 'cnn' in data_path:
            self.data = load_dataset('cnn_dailymail', '3.0.0', split=split)
        else:
            self.data = load_dataset(data_path, split=split)
        self.tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', legacy=False, use_fast=False)
        if prompt is None:
            self.prompt = '[INST] Summarize the following text with fewer than #NWORD# words: #TEXT# [/INST]\n'
        else:
            self.prompt = prompt
        if 'squad' == data_path:
            seen = set()
            self.indices = []
            for i, exp in enumerate(self.data):
                if exp['context'] in seen:
                    continue
                self.indices.append(i)
                seen.add(exp['context'])
        else:
            self.indices = list(range(len(self.data)))

    def __len__(self):
        if self.max_size is None:
            return len(self.indices)
        return min(len(self.indices), self.max_size)

    def __getitem__(self, idx: int):
        exp = self.data[self.indices[idx]]
        if 'squad' in self.data_path:
            d = exp['context']
            meta = {'id': exp['id'], 'text': exp['context']}
        elif 'cnn' in self.data_path:
            d = exp['article']
            if len(d.split(' ')) > 900:
                d = ' '.join(d.split(' ')[:900])
            meta = {'id': exp['id'], 'text': exp['article'], 'highlights': exp['highlights']}
        else:
            raise NotImplementedError
        n_word = math.ceil(len(d.split()) * self.ratio)
        q_text = self.prompt.replace('#NWORD#', str(n_word)).replace('#TEXT#', d).replace('#N#', '\n')
        q = torch.tensor(self.tok(q_text, max_length=self.max_length, truncation=True)['input_ids'])
        return {
            'tgt_input_ids': q, 'src_input_ids': torch.zeros([1], dtype=torch.int64),
            'skip': len(q), 'objective': Objective.TextContinuation, 'meta': meta
        }
