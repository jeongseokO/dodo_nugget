from transformers.generation.stopping_criteria import StoppingCriteria
from transformers.generation.logits_process import LogitsProcessor
import torch


class NStoppingCriteria(StoppingCriteria):
    def __init__(self, max_new_line: int = 3, min_length: int = 10):
        self.max_new_line = max_new_line
        self.min_length = min_length
        assert max_new_line < min_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        input_ids = input_ids.cpu().tolist()
        if all(2 in line for line in input_ids):
            return True
        if len(input_ids[0]) <= self.min_length:
            return False
        if all(tuple(line[-self.max_new_line:]) == (13,) * self.max_new_line for line in input_ids):
            return True
        return False


class ExtractiveLogitsProcessor(LogitsProcessor):
    def __init__(self, src_input_ids: torch.Tensor, src_attention_mask: torch.Tensor):
        self.src_input_ids, self.src_attention_mask = src_input_ids, src_attention_mask
        self.allow_list = None

    def prepare_allow_list(self, vocab):
        if self.allow_list is not None:
            return self.allow_list
        ran = torch.arange(vocab, device=self.src_input_ids.device).unsqueeze(0).expand(self.src_input_ids.shape[0], -1)
        # shape [bsz, src, vocab]
        src_allow = ran.unsqueeze(1) == self.src_input_ids.unsqueeze(2)
        src_allow[~self.src_attention_mask.unsqueeze(2).expand(-1, -1, vocab)] = False
        mask = src_allow.any(1)
        # period, comma, eos
        for special_id in [29889, 29892, 2]:
            mask[:, special_id] = True
        self.allow_list = mask
        return mask

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # scores shape [bsz, vocab]
        # [bsz, src, tgt]
        allow_list = self.prepare_allow_list(scores.shape[1])
        scores[~allow_list] = torch.finfo(scores.dtype).min
        return scores
