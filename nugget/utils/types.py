from typing import *

from torch import Tensor


class EncoderOut(NamedTuple):
    # Shape (layer, 2 (k+v)) x (bsz, head, nugget, head_dim)
    states: Optional[List[Tuple[Tensor, Tensor]]] = None
    # shape (bsz, nugget)
    mask: Optional[Tensor] = None
    # shape (bsz, nugget)
    logits: Optional[Tensor] = None
    # shape (bsz, nugget)
    indices: Optional[Tensor] = None


class NuggetOut(NamedTuple):
    loss: Tensor
    logits: Tensor
    encoder_out: EncoderOut
    decoder_logits: Tensor
    probs: Optional[Tensor] = None


class SelectedNugget(NamedTuple):
    tokens: List[Union[int, str]]
    nugget_indices: List[int]


class Objective:
    AE = 0
    TextContinuation = 1


str2obj = {'ae': Objective.AE, 'tc': Objective.TextContinuation}
