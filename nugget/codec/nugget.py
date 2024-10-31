from typing import *
import os

import torch

from .encoder import encoder_types, EncoderOut
from .decoder import Decoder
from ..my_transformers.my_llama import MyLlamaForCausalLM
from ..utils.types import NuggetOut
from plutils.profiler import profile


class Nugget(torch.nn.Module):
    def print_param(self, trainable):
        for n, p in self.named_parameters():
            if p.requires_grad or not trainable:
                print(n, tuple(p.shape))

    def lora_pattern(self, total_layer):
        pat = r'.*layers\.(LAYERS)\.self_attn\.(q_proj|v_proj|o_proj)'
        if self.ind_scorer:
            layers = r'\d+'
        else:
            layers = '|'.join(map(str, range(self.nugget_layer, total_layer)))
        return pat.replace('LAYERS', layers)

    # @profile
    def __init__(
            self, pretrained: str, encoder_type: str, nugget_layer: int, ratio: float,
            lora: int, soft: Optional[int], ind_scorer: bool, freeze_decoder: bool = False,
            freeze_scorer: bool = False, skip_first: Optional[int] = None, new_line_stop=False,
            extractive: Optional[bool] = False, auto: Optional[bool] = False, **_
    ):
        super().__init__()
        self.encoder_type, self.pretrained, self.soft, self.lora = encoder_type, pretrained, soft, lora
        self.nugget_layer, self.ind_scorer = nugget_layer, ind_scorer
        self.freeze_decoder, self.freeze_scorer = freeze_decoder, freeze_scorer
        self.skip_first = skip_first
        if os.environ.get('HOSTNAME', '').lower().startswith('brtx'):
            from ilock import ILock
            with ILock('llama_loading'):
                base_llama = MyLlamaForCausalLM.from_pretrained(pretrained)
        else:
            base_llama = MyLlamaForCausalLM.from_pretrained(pretrained)
        assert lora > 0
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora, lora_alpha=32, lora_dropout=0.1,
            target_modules=self.lora_pattern(len(base_llama.model.layers)),
        )
        if encoder_type == 'nugget':
            lora_llama = get_peft_model(base_llama, peft_config, 'encoder')
            if not freeze_decoder:
                lora_llama.add_adapter('decoder', peft_config)
        else:
            assert not freeze_decoder
            lora_llama = get_peft_model(base_llama, peft_config, 'decoder')
        self.lora_llama = lora_llama
        # lora_llama is passed as arguments to encoder and decoder
        # otherwise deepspeed will save it twice in checkpoints
        self.encoder = encoder_types[encoder_type](
            hidden_size=lora_llama.config.hidden_size, ratio=ratio, nugget_layer=nugget_layer,
            ind_scorer=ind_scorer, freeze_scorer=freeze_scorer, auto=auto
        )
        self.decoder = Decoder(lora_llama.config.hidden_size, soft, new_line_stop, extractive)
        self.trainable_parameters: Optional[Set[str]] = None

    def forward(
            self, src_input_ids: torch.Tensor, src_attention_mask: torch.Tensor,
            tgt_input_ids: torch.Tensor, tgt_attention_mask: torch.Tensor, objective: torch.Tensor,
            segment_map: Optional[torch.Tensor] = None, n_segment: Optional[List[int]] = None,
            skip: Optional[torch.Tensor] = None, return_probs: bool = False,
    ) -> NuggetOut:
        # encoder encodes src sequence into nuggets
        enc_out: EncoderOut = self.encoder(
            lora_model=self.lora_llama, input_ids=src_input_ids, attention_mask=src_attention_mask,
            segment_map=segment_map, n_segment=n_segment,
        )
        # prepare labels for next-token prediction
        labels = tgt_input_ids.clone()
        labels[~tgt_attention_mask] = -100
        if skip is not None:
            skip_mask = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0) < skip.unsqueeze(1)
            labels[skip_mask] = -100
        elif self.skip_first and labels.shape[1] > self.skip_first:
            labels[:, :self.skip_first] = -100

        # decoder takes the nuggets as inputs and decode the target sequence
        causal_out = self.decoder(self.lora_llama, tgt_input_ids, tgt_attention_mask, labels, enc_out, objective)
        probs = None
        if return_probs:
            probs = causal_out.logits.softmax(2)[:, :-1].gather(2, tgt_input_ids[:, 1:].unsqueeze(2)).squeeze(2)
        return NuggetOut(causal_out.loss, causal_out.logits, enc_out, causal_out.logits, probs)

    def generate(
        self, src_input_ids: torch.Tensor, src_attention_mask: torch.Tensor,
        tgt_input_ids: torch.Tensor, tgt_attention_mask: torch.Tensor, objective: torch.Tensor,
        segment_map: Optional[torch.Tensor] = None, n_segment: Optional[List[int]] = None,
        skip: Optional[torch.Tensor] = None, n_tokens: int = 512
    ):
        enc_out: EncoderOut = self.encoder(
            lora_model=self.lora_llama, input_ids=src_input_ids, attention_mask=src_attention_mask,
            segment_map=segment_map, n_segment=n_segment,
        )
        start_tokens = skip_mask = None
        if skip is not None:
            start_tokens = tgt_input_ids[:, :skip.max()]
            skip_mask = torch.arange(start_tokens.shape[1], device=skip.device).unsqueeze(0) < skip.unsqueeze(1)
            start_tokens[~skip_mask] = 0
        return self.decoder.generate(
            self.lora_llama, enc_out, n_tokens, objective, start_tokens, skip_mask,
            src_input_ids, src_attention_mask
        )

    def get_trainable(self):
        if self.trainable_parameters is not None:
            return
        self.trainable_parameters = set()
        for n, p in self.named_parameters():
            if p.requires_grad:
                self.trainable_parameters.add(n)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        if len(args) == 3:
            destination, prefix, keep_vars = args
        ret = super(Nugget, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.get_trainable()
        if isinstance(ret, OrderedDict):
            for k in list(ret):
                if k[len(prefix):] not in self.trainable_parameters:
                    ret.pop(k)
        return ret
