from typing import *

import torch
from transformers import LlamaForCausalLM
from peft.peft_model import PeftModelForCausalLM

from ..utils.types import Objective
from .encoder import EncoderOut
from ..my_transformers.gen_utils import NStoppingCriteria, ExtractiveLogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList


class Decoder(torch.nn.Module):
    def __init__(self, hidden_size: int, soft: Optional[int], new_line_stop, extractive: bool):
        super().__init__()
        self.soft = soft
        self.new_line_stop, self.extractive = new_line_stop, extractive
        # soft_emb refers to ae; for compatibility with checkpoints, I won't change its name
        if soft is not None and soft > 0:
            self.soft_emb = torch.nn.Embedding(soft, hidden_size) if soft is not None else None
            self.soft_emb_tc = torch.nn.Embedding(soft, hidden_size) if soft is not None else None
        else:
            self.soft_emb = self.soft_emb_tc = None

    def forward(
            self, lora_model, input_ids, attention_mask, labels: Optional[torch.Tensor] = None,
            encoder_out: EncoderOut = EncoderOut(), objective: Optional[torch.Tensor] = None
    ) -> None:
        # The decoder adapter can be missing
        if 'decoder' in lora_model.base_model.peft_config:
            lora_model.set_adapter('decoder')
        else:
            lora_model.disable_adapter()
        # encoder_out is nugget; input_ids and attention_mask are target seq
        position_ids, soft_prompt, attention_mask = self.prepare_inputs(
            encoder_out, input_ids, attention_mask, objective
        )
        causal_out = lora_model(
            input_ids, attention_mask, past_key_values=encoder_out.states, labels=labels,
            position_ids=position_ids, logits=encoder_out.logits, use_cache=False, soft_prompt=soft_prompt,
        )
        return causal_out

    def prepare_inputs(self, encoder_out: EncoderOut, input_ids, attention_mask, objective):
        # each sequence is (nugget, [soft prompt, optional], target sequence)
        # both nugget and target sequence can be masked for batch_size > 1
        # so in this method we prepare the attention mask and input_ids for these seqs
        # len(attn_mask) = len(nugget) + len(soft_prompt) + len(input_ids)
        # len(position_ids) = len(soft_prompt) + len(input_ids)
        bsz, n_tok = input_ids.shape
        if encoder_out.states is None:
            n_nugget = input_ids.new_zeros([bsz])
        else:
            n_nugget = encoder_out.mask.sum(1)
            attention_mask = torch.cat([encoder_out.mask, attention_mask], dim=1)
        if self.soft_emb is not None:
            # shape (soft, dim)
            ae_soft_prompt = self.soft_emb(torch.arange(self.soft, device=input_ids.device))
            tc_soft_prompt = self.soft_emb(torch.arange(self.soft, device=input_ids.device))
            # shape (bsz, soft, dim)
            soft_prompt = ae_soft_prompt.unsqueeze(0) * (objective == Objective.AE).unsqueeze(1).unsqueeze(1) \
                + tc_soft_prompt.unsqueeze(0) * (objective == Objective.TextContinuation).unsqueeze(1).unsqueeze(1)
            attention_mask = torch.cat([attention_mask.new_ones([bsz, self.soft]), attention_mask], dim=1)
        else:
            soft_prompt = None
        position_ids = torch.arange(n_tok+(self.soft or 0), device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        position_ids = position_ids + n_nugget.unsqueeze(1)
        return position_ids, soft_prompt, attention_mask

    def generate(
            self, lora_model, enc_out: EncoderOut, n_tokens: int, objective, start_tokens, start_attn_mask,
            src_input_ids, src_attention_mask,
    ):
        if 'decoder' in lora_model.base_model.peft_config:
            lora_model.set_adapter('decoder')
        else:
            lora_model.disable_adapter()
        if enc_out.mask is not None:
            bsz, n_nugget = enc_out.mask.shape
        else:
            bsz, n_nugget = objective.shape[0], 0
        if start_tokens is None:
            # assumes that the BOS token is 1 (compatible with LLaMA tokenizer)
            start_tokens = enc_out.mask.new_ones([bsz, 1], dtype=torch.int64)
            start_attn_mask = enc_out.mask.new_ones([bsz, 1], dtype=torch.bool)

        # position_ids (bsz, soft_prompt + start_tokens)
        # soft_prompt (bsz, soft_prompt, dim)
        # attn_mask (bsz, n_nugget + soft_prompt + start_tokens)
        position_ids, soft_prompt, attn_mask = self.prepare_inputs(
            enc_out, start_tokens, start_attn_mask, objective
        )

        base_out = lora_model(
            input_ids=start_tokens, attention_mask=attn_mask,
            past_key_values=enc_out.states, soft_prompt=soft_prompt,
            logits=enc_out.logits, use_cache=True, position_ids=position_ids,
        )
        # shape (bsz, head, n_nugget + soft + start_tokens-1, head_dim)
        past_key_values = [tuple(kv[:, :, :-1] for kv in layer) for layer in base_out.past_key_values]

        if self.extractive:
            logits_processor = LogitsProcessorList([ExtractiveLogitsProcessor(
                torch.cat([src_input_ids, start_tokens], dim=1),
                torch.cat([src_attention_mask, start_attn_mask], dim=1)
            )])
        else:
            logits_processor = None

        generated = lora_model.generate(
            input_ids=start_tokens[:, -1:], attention_mask=attn_mask, past_key_values=past_key_values,
            logits=enc_out.logits, max_new_tokens=n_tokens, output_logits=True,
            stopping_criteria=StoppingCriteriaList([NStoppingCriteria()]) if self.new_line_stop else None,
            logits_processor=logits_processor
        )
        return generated
