from typing import *
import json
import time
import os

import torch

from .lazy_data import LazyDataset
from .data_utils import SegmentSplit, FakeEOS
from ..utils.types import Objective
from plutils import logger


class SegmentT2T(LazyDataset):
    def __init__(
            self, data_path: str, split: str, pretrained: str, max_size: Optional[int], segment_size,
            max_q: Optional[int], max_a: Optional[int], n_doc: Optional[int], **_
    ):
        if ':' in data_path:
            data_path, data_arg = data_path.split(':')
        else:
            data_arg = None
        super().__init__(
            pretrained, data_name=data_path, data_args=(data_arg,), split=split.replace('dev', 'validation')
        )
        self.init()
        self.fake_eos = FakeEOS(self.tokenizer)
        self.segment_split = SegmentSplit(self.fake_eos, segment_size)
        self.max_q, self.max_a, self.n_doc = max_q, max_a, n_doc
        self.max_size = max_size
        self.idx_map = None
        if self._data_name == 'trivia_qa':
            self.idx_map = self.trivia_qa_map()
        self.trivia_qa_fail = 0

    @property
    def max_d(self):
        return self.n_doc * self.segment_split.segment_size

    def trivia_qa_map(self):
        ret = []
        cache_path = os.path.join(
            os.environ.get('HOME', ''), '.cache', f'triviaqa.{self._data_args[0]}.{self._split}.json'
        )
        if os.path.exists(cache_path):
            return json.load(open(cache_path))
        since = time.time()
        for ie, exp in enumerate(self._data):
            n_search = len(exp['search_results']['search_context'])
            ret.extend([(ie, 'search', j) for j in range(n_search)])
            n_wiki = len(exp['entity_pages']['wiki_context'])
            ret.extend([(ie, 'wiki', j) for j in range(n_wiki)])
        logger.warning(f'Loading idx for trivia QA takes {time.time()-since:.1f}s.')
        json.dump(ret, open(cache_path, 'w'))
        return ret

    def __len__(self):
        if self.idx_map is not None:
            ret = len(self.idx_map)
        else:
            ret = len(self._data)
        if self.max_size is None:
            return ret
        return min(ret, self.max_size)

    def trivia_qa(self, idx):
        i_exp, src, i_context = self.idx_map[idx]
        exp = self._data[i_exp]
        meta = {k: exp[k] for k in ['question', 'question_id', 'answer']}
        meta['source'] = src
        if src == 'search':
            doc = exp['search_results']['search_context'][i_context]
            meta['filename'] = exp['search_results']['filename'][i_context]
        else:
            doc = exp['entity_pages']['wiki_context'][i_context]
            meta['filename'] = exp['entity_pages']['filename'][i_context]
        while '  ' in doc:
            doc = doc.replace('  ', ' ')
        doc = doc.replace('’', '\'')
        # doc_ids = self.fake_eos(doc_ids)
        answer_candidates = [exp['answer']['value'].strip()]
        answer_candidates += [ans.strip() for ans in exp['answer']['normalized_aliases'] + exp['answer']['aliases']]
        answer = None
        for ans in answer_candidates:
            if ans.lower() in doc.lower():
                answer = ans
                break
        if answer is None:
            self.trivia_qa_fail += 1
            answer = answer_candidates[0]
            meta['answer_index'] = 0
        else:
            meta['answer_index'] = doc.lower().index(answer.lower())
        return doc, exp['question'].strip(), answer, meta

    def cnn(self, idx):
        exp = self._data[idx]
        question = 'Summarization: '
        return exp['article'], question, exp['highlights'], {k: exp[k] for k in ['id', 'highlights']}

    def __getitem__(self, idx: int):
        if self._data_name == 'trivia_qa':
            doc, question, answer, meta = self.trivia_qa(idx)
            doc_ids = self.tokenizer(doc, add_special_tokens=False)['input_ids']
            if len(doc_ids) > self.max_d:
                left = max(0, meta['answer_index'] - self.n_doc//2)
                right = min(len(doc_ids), left + self.n_doc)
                left = max(0, right - self.n_doc)
                doc_ids = doc_ids[left:right]
        elif self._data_name == 'cnn_dailymail':
            doc, question, answer, meta = self.cnn(idx)
            doc_ids = self.tokenizer(
                doc, add_special_tokens=False, truncation=True, max_length=self.max_d + self.n_doc,
            )['input_ids']
        else:
            raise NotImplementedError
        d = list(map(torch.tensor, self.segment_split(doc_ids)))
        if self.n_doc is not None and self.n_doc > 0:
            d = d[:self.n_doc]
        q = self.tokenizer(
            question+'\n\n', truncation=True, add_special_tokens=False, max_length=self.max_q
        )['input_ids']
        a = self.tokenizer(
            answer, truncation=True, add_special_tokens=False, max_length=self.max_a
        )['input_ids'] + [self.tokenizer.eos_token_id]
        return {
            'tgt_input_ids': torch.tensor(q+a), 'src_input_ids': d,
            'skip': len(q), 'objective': Objective.TextContinuation, 'meta': meta
        }
