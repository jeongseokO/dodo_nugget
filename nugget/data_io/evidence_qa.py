from typing import *
import json
from copy import deepcopy
import os
import random

import torch
from torch.utils import data
from datasets import load_from_disk, load_dataset
from transformers import LlamaTokenizer

from .data_utils import FakeEOS
from ..utils.types import Objective
from plutils.common import logger


class EvidenceQA(data.Dataset):
    def __init__(
            self, data_path: str, split: str, pretrained: str, max_size: Optional[int],
            max_q: int, max_d: int, max_a: int, n_doc: Optional[int], inst: bool = False,
            prompt: Optional[str] = None, shuffle_indices: Optional[bool] = False,
            lmsumm_path: Optional[str] = None, doc_prompt: Optional[str] = None,
            split_squad_doc: bool = False, **_
    ):
        if 'wikihow' in data_path.lower():
            self.data_type = 'wikihow'
        elif 'squad' in data_path.lower():
            self.data_type = 'squad'
        else:
            raise NotImplementedError
        self.split_squad_doc = split_squad_doc
        self.max_q, self.max_d, self.max_a, self.n_doc = max_q, max_d, max_a, n_doc
        self.pretrained, self.max_size = pretrained, (max_size if max_size is not None else 999999999999)
        self.inst = inst
        if os.path.exists(data_path):
            self._data = load_from_disk(data_path)[split.replace('dev', 'validation')]
        else:
            self._data = load_dataset(data_path, split=split.replace('dev', 'validation'))
        self.tok = LlamaTokenizer.from_pretrained(pretrained, legacy=False, use_fast=False)
        self.fake_eos = FakeEOS(self.tok)
        if prompt is None:
            prompt = '[INST] Based on the provided document, answer the following question: #QUESTION# [/INST]#N#'
        else:
            assert '#QUESTION#' in prompt
        self.doc_prompt = doc_prompt
        self.prompt = prompt.replace('#N#', '\n').replace('#QUESTION#', '{question}')
        self.data_indices = None
        if shuffle_indices:
            total = list(range(len(self._data)))
            random.seed(9192633710)
            random.shuffle(total)
            self.data_indices = total
        self.lmsumm_map = None
        if lmsumm_path is not None:
            self.lmsumm_map = json.load(open(lmsumm_path))

    def __len__(self):
        return min(self.max_size, len(self._data))

    def __getitem__(self, idx: int):
        if self.data_indices is not None:
            idx = self.data_indices[idx]
        exp = deepcopy(self._data[idx])
        if self.data_type == 'wikihow':
            docs = [json.loads(j_) for j_ in exp.pop('related_document_urls_wayback_snapshots')]
            if self.n_doc is not None:
                docs = docs[:self.n_doc]
            doc_texts = ['\n'.join([doc['parsed_text']['title']] + doc['parsed_text']['passages']) for doc in docs]
            question, answer = exp.pop('question'), exp.pop('answer')
            meta = {'article_id': exp['article_id'], 'question': question, 'answer': answer, 'data': 'wikihow'}
        elif self.data_type == 'squad':
            doc_texts = exp['context']
            if self.split_squad_doc:
                doc_texts = doc_texts.split('\n\n')
            question, answer = exp['question'], random.choice(exp['answers']['text'])
            meta = {'id': exp['id'], 'answers': exp['answers'], 'data': 'squad', 'context': exp['context']}
        else:
            raise NotImplementedError
        if isinstance(doc_texts, list):
            d = list(map(torch.tensor, [
                self.tok(doc, truncation=True, add_special_tokens=True, max_length=self.max_d)['input_ids']
                for doc in doc_texts
            ]))
        else:
            if self.lmsumm_map is not None:
                if doc_texts in self.lmsumm_map:
                    doc_texts = self.lmsumm_map[doc_texts]
                else:
                    logger.warning(f'Doc text not found in lmsumm map!')
            if self.doc_prompt is not None:
                doc_texts = self.doc_prompt.replace('#TEXT#', doc_texts).replace('#N#', '\n')
            d = torch.tensor(
                self.tok(doc_texts, truncation=True, add_special_tokens=True, max_length=self.max_d)['input_ids']
            )
        if self.inst:
            question = self.prompt.format(question=question)
        else:
            question = question + '\n\n'
        if self.data_type == 'squad':
            meta['question'] = question
        q = self.tok(
            question, truncation=True, add_special_tokens=False, max_length=self.max_q
        )['input_ids']
        a = self.tok(
            answer, truncation=True, add_special_tokens=False, max_length=self.max_a
        )['input_ids'] + [self.tok.eos_token_id]
        return {
            'tgt_input_ids': torch.tensor(q+a), 'src_input_ids': d,
            'skip': len(q), 'objective': Objective.TextContinuation, 'meta': meta
        }
