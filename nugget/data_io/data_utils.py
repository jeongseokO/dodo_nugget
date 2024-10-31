from typing import *
import string
from dataclasses import dataclass

import torch


class FakeEOS:
    def __init__(self, tokenizer):
        self.punc = [tokenizer(s, add_special_tokens=False)['input_ids'][-1] for s in string.punctuation]
        self.fake_eos = tokenizer('.', add_special_tokens=False)['input_ids'][-1]
        self.bos_token_id, self.eos_token_id = tokenizer.bos_token_id, tokenizer.eos_token_id

    def __call__(self, strings):
        if isinstance(strings[0], list):
            return [self(s) for s in strings]
        if isinstance(strings[0], int):
            if strings[-1] in self.punc:
                return strings
            return strings + [self.fake_eos]
        raise NotImplementedError


@dataclass()
class SegmentSplit:
    fake_eos: FakeEOS
    segment_size: int
    search: int = 7

    def __call__(self, inputs: List[int]):
        assert len(inputs) > 0
        if inputs[0] == self.fake_eos.bos_token_id:
            inputs = inputs[1:]
        segments = []
        last_split_point = 0
        while True:
            if len(inputs) == last_split_point:
                break
            if len(inputs) - last_split_point < self.segment_size + self.search:
                segments.append(inputs[last_split_point:])
                break
            next_split_point = self.find_split_point(inputs, last_split_point+self.segment_size)
            if next_split_point is None:
                next_split_point = last_split_point + self.segment_size
            segments.append(inputs[last_split_point:next_split_point])
            last_split_point = next_split_point
        segments = [self.fake_eos([self.fake_eos.bos_token_id]+seg) for seg in segments]
        return segments

    def find_split_point(self, inputs: List[int], middle: int) -> Optional[int]:
        for abs_shift in range(self.search):
            for direction in [-1, 1]:
                idx = middle + abs_shift * direction
                if idx < len(inputs) and inputs[idx] in self.fake_eos.punc:
                    return idx


def collate_fn(inputs):
    def pad_seqs(seqs):
        lengths = torch.tensor(list(map(len, seqs)))
        ml = lengths.max()
        mask = torch.arange(ml).unsqueeze(0).expand(len(seqs), -1) < lengths.unsqueeze(1)
        ids = [torch.cat((seq, torch.zeros((ml-len(seq),), dtype=seq.dtype))) for seq in seqs]
        return torch.stack(ids), mask

    ret = dict()
    if inputs[0]['src_input_ids'][0].dim() == 1:
        n_seg = [len(item['src_input_ids']) for item in inputs]
        seg_range = torch.arange(max(n_seg) * len(inputs)).reshape(len(inputs), max(n_seg))
        ret['segment_map'] = torch.cat([seg_range[:ns] for seg_range, ns in zip(seg_range, n_seg)], dim=0)
        ret['n_segment'] = n_seg
        ret['src_input_ids'], ret['src_attention_mask'] = pad_seqs(sum([item['src_input_ids'] for item in inputs], []))
    else:
        ret['src_input_ids'], ret['src_attention_mask'] = pad_seqs([item['src_input_ids'] for item in inputs])
    ret['tgt_input_ids'], ret['tgt_attention_mask'] = pad_seqs([item['tgt_input_ids'] for item in inputs])
    ret['objective'] = torch.tensor([inp['objective'] for inp in inputs])
    if 'skip' in inputs[0]:
        ret['skip'] = torch.tensor([inp['skip'] for inp in inputs])
    if 'meta' in inputs[0]:
        ret['meta'] = [inp['meta'] for inp in inputs]
    return ret
