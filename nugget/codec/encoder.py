from typing import *
import math

import torch
from torch import nn

from ..utils.types import EncoderOut


def scatter_segments(enc_out: EncoderOut, segment_map: torch.Tensor, n_segment: List[int]):
    cat_bsz, n_head, n_token, head_dim = enc_out.states[0][0].shape
    max_seg = max(n_segment)
    bsz = len(n_segment)
    states = []
    expanded_map = segment_map.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, n_head, n_token, head_dim)
    for i_layer, layer_state in enumerate(enc_out.states):
        new_layer_state = tuple()
        for data in layer_state:
            new_state = data.new_zeros((bsz * max_seg, n_head, n_token, head_dim))
            new_state.scatter_(0, expanded_map, data)
            new_state = new_state.reshape(bsz, max_seg, n_head, n_token, head_dim) \
                .permute(0, 2, 1, 3, 4) \
                .reshape(bsz, n_head, max_seg * n_token, head_dim)
            new_layer_state += (new_state,)
        states.append(new_layer_state)

    expanded_map = segment_map.unsqueeze(1).expand(-1, n_token)

    def remap(inputs):
        if inputs is None:
            return None
        ret = inputs.new_zeros((bsz * max_seg, n_token))
        ret.scatter_(0, expanded_map, inputs)
        ret = ret.reshape(bsz, max_seg * n_token)
        return ret

    return EncoderOut(states, remap(enc_out.mask), remap(enc_out.logits), remap(enc_out.indices))


class PassThroughEncoder(torch.nn.Module):
    def __init__(self, **_):
        super().__init__()

    def forward(
            self, lora_model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
            segment_map: Optional[torch.Tensor] = None, n_segment: Optional[List[int]] = None, **_,
    ) -> EncoderOut:
        outs = lora_model(input_ids, attention_mask, use_cache=True)
        enc_out = EncoderOut(outs.past_key_values, attention_mask)
        if segment_map is not None:
            return scatter_segments(enc_out, segment_map, n_segment)
        return enc_out


class NuggetEncoder(torch.nn.Module):
    def __init__(
            self, hidden_size, ratio: float, nugget_layer: int, ind_scorer, freeze_scorer: bool = False,
            auto: bool = False, **_
    ):
        super().__init__()
        self.ratio, self.nugget_layer, self.ind_scorer = ratio, nugget_layer, ind_scorer
        self.freeze_scorer, self.auto = freeze_scorer, auto
        assert not auto or ind_scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def select_nuggets(self, score_out, n_nugget: torch.Tensor, attention_mask: torch.Tensor, n_tok: torch.Tensor):
        """
        1. select nuggets that have top scores
        2. re-order them following the order in the original text
        3. force the last token as nugget
        """
        attention_mask = attention_mask.to(bool)
        max_nugget = n_nugget.max()
        if self.freeze_scorer:
            with torch.no_grad():
                scores_nugget = self.scorer(score_out.hidden_states[self.nugget_layer])
        else:
            scores_nugget = self.scorer(score_out.hidden_states[self.nugget_layer])
        scores_nugget = scores_nugget.squeeze(2)

        scores4nugget = scores_nugget.clone().detach()
        # shape (bsz, tok)
        scores4nugget[~attention_mask] = torch.finfo(scores4nugget.dtype).min
        n_tok_exp = (n_tok-1).unsqueeze(1)
        scores4nugget = scores4nugget.scatter(
            1, n_tok_exp,
            scores4nugget.new_full(n_tok_exp.shape, fill_value=torch.finfo(scores4nugget.dtype).max)
        )
        # shape (bsz, nugget)
        nugget_mask = torch.arange(max_nugget, device=scores4nugget.device).unsqueeze(0) < n_nugget.unsqueeze(1)
        ind = scores4nugget.argsort(1, descending=True)[:, :max_nugget]
        # sort the indices
        ind4sort = ind * nugget_mask + (ind + attention_mask.shape[1] + 1) * (~nugget_mask)
        resort_ind = ind4sort.argsort(1)
        indices_ascending = ind.gather(1, resort_ind)
        scores_gather = scores_nugget.gather(1, indices_ascending)
        return indices_ascending, scores_gather, nugget_mask

    def forward(
        self, lora_model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        segment_map: Optional[torch.Tensor] = None, n_segment: Optional[List[int]] = None,
    ) -> EncoderOut:
        lora_model.set_adapter('decoder' if self.auto else 'encoder')
        outs = lora_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, output_hidden_states=True)

        # NUGGET SELECTION
        # num of nuggets, shape (bsz)
        n_tok = attention_mask.sum(1)
        n_nugget = (n_tok*self.ratio + 0.99).to(torch.int64)
        n_nugget[n_nugget > n_tok] = n_tok[n_nugget > n_tok]

        if self.ind_scorer:
            lora_model.disable_adapter_layers()
            score_out = lora_model(
                inputs_embeds=outs.hidden_states[0], attention_mask=attention_mask,
                n_layer=self.nugget_layer, output_hidden_states=True,
            )
            lora_model.enable_adapter_layers()
        else:
            score_out = outs
        nugget_indices, scores_gather, nugget_mask = self.select_nuggets(score_out, n_nugget, attention_mask, n_tok)
        # shape for past_key_values (layer, 2), (bsz, head, token, head_dim)
        _, n_head, _, head_dim = outs.past_key_values[0][0].shape
        if not self.auto:
            # select all the key value states for all layers of nugget indices
            feats_all_layer = list()
            for i_layer, layer_kv in enumerate(outs.past_key_values):
                feats_all_layer.append(tuple(
                    layer_kv[j].gather(2, nugget_indices.unsqueeze(1).unsqueeze(3).expand(-1, n_head, -1, head_dim))
                    for j in range(2)
                ))
        else:
            feats_all_layer = self.auto_nuggets(lora_model, outs, attention_mask, nugget_indices, nugget_mask)
        enc_out = EncoderOut(tuple(feats_all_layer), nugget_mask, scores_gather, nugget_indices)
        if segment_map is not None:
            return scatter_segments(enc_out, segment_map, n_segment)
        return enc_out

    def auto_nuggets(
            self, lora_model, outs, attention_mask: torch.Tensor,
            nugget_indices: torch.Tensor, nugget_mask: torch.Tensor
    ):
        device = attention_mask.device
        nugget_token_embeds = outs.hidden_states[0].gather(
            1, nugget_indices.unsqueeze(2).expand(-1, -1, outs.hidden_states[0].shape[2])
        )
        bsz, n_token = attention_mask.shape
        n_nugget = nugget_indices.shape[1]
        token_indices = torch.arange(n_token, device=device).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1)
        # shape (bsz, nugget, token)
        attn_mask_left = nugget_indices.unsqueeze(2) > token_indices
        attn_mask_right = torch.eye(n_nugget, device=device).unsqueeze(0)
        attn_cat = torch.cat([attn_mask_left, attn_mask_right], dim=2).unsqueeze(1)
        attn_cat = (1-attn_cat)*torch.finfo(attn_cat.dtype).min
        # attn_mask shape [bsz, 1, tgt, src]. tgt = nugget, src = token + nugget
        token_position_ids = torch.arange(n_token, device=device).unsqueeze(0).view(bsz, n_token)
        nugget_position_ids = token_position_ids.gather(1, nugget_indices)
        lora_model.set_adapter('encoder')
        nugget_out = lora_model(
            inputs_embeds=nugget_token_embeds, attention_mask=attn_cat, past_key_values=outs.past_key_values,
            position_ids=nugget_position_ids
        )
        feats_all_layers = list()
        for layer in nugget_out.past_key_values:
            # shape for past_key_values (layer, 2), (bsz, head, token, head_dim)
            feats_all_layers.append(tuple(layer[j][:, :, -nugget_mask.shape[1]:, :] for j in range(2)))
        return feats_all_layers


class NoneEncoder(torch.nn.Module):
    def __init__(self, **_):
        super().__init__()

    def forward(self, **_) -> EncoderOut:
        return EncoderOut(None, None)


class CompressEncoder(torch.nn.Module):
    def __init__(self, ratio: float, **_):
        super().__init__()
        self.segment = math.ceil(1/ratio)

    def forward(self, lora_model, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_) -> EncoderOut:
        bsz, n_token = input_ids.shape
        pad_length = math.ceil(n_token / self.segment) * self.segment - n_token
        if pad_length > 0:
            input_ids = torch.cat([input_ids, input_ids.new_zeros((bsz, pad_length))], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((bsz, pad_length))], dim=1)
        n_token += pad_length
        n_nugget = n_token // self.segment

        lora_model.set_adapter('decoder')
        outs = lora_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, output_hidden_states=True)
        _, n_head, _, head_dim = outs.past_key_values[0][0].shape

        attention_mask_reshape = attention_mask.reshape(bsz, n_nugget, self.segment)
        nugget_mask = attention_mask_reshape.any(2)
        denom = attention_mask_reshape.sum(2).unsqueeze(1).unsqueeze(3)
        denom[denom == 0] = 1

        past_kv = list()
        for layer in outs.past_key_values:
            past_kv.append(tuple(
                kv.reshape(bsz, n_head, n_nugget, self.segment, head_dim).mean(3) / denom for kv in layer
            ))

        return EncoderOut(tuple(past_kv), nugget_mask)


encoder_types = {
    'llama': PassThroughEncoder,
    'nugget': NuggetEncoder,
    'none': NoneEncoder,
    'compress': CompressEncoder,
}
