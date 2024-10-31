import os
from typing import *
import re

from lightning.pytorch import LightningModule
import numpy as np
import torch
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

from nugget.codec.nugget import Nugget
from plutils.param import param_to_buffer
from plutils.profiler import profile
from plutils.common import logger
from ..utils.types import str2obj


class NuggetTrainer(LightningModule):
    # @profile
    def __init__(
            self, lr: float, ft_lr: float, warmup: int, gpu_config: str, new_tokens: int,
            load_scorer: Optional[str] = None, perplexity: bool = False, **model_kwargs
    ):
        super().__init__()
        self.lr, self.ft_lr, self.warmup, self.gpu_config = lr, ft_lr, warmup, gpu_config
        self.new_tokens = new_tokens
        self.perplexity = perplexity
        self.nugget = Nugget(**model_kwargs)
        # This is a hack for not saving frozen parameters
        param_to_buffer(self.nugget)
        self.nugget.get_trainable()
        self.load_scorer = load_scorer
        if load_scorer is not None:
            if not os.path.exists(load_scorer):
                logger.error(f'Fail to load {load_scorer}')
            else:
                self.nugget.encoder.scorer.load_state_dict(torch.load(load_scorer))
        self.save_hyperparameters()

    def configure_optimizers(self) -> Any:
        lora, others = [], []
        for name, param in self.trainer.model.named_parameters():
            if not param.requires_grad:
                continue
            (lora if re.findall(r'lora_[AB]', name) else others).append(param)
        param_groups = [{'params': lora, 'lr': self.ft_lr}, {'params': others}]
        param_groups = list(filter(lambda pg: len(pg['params']) > 0, param_groups))
        assert len(param_groups) > 0, "no params to train"
        if self.gpu_config != 'deepspeed_cpu':
            optim = torch.optim.AdamW(
                param_groups, lr=self.lr, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1
            )
        else:
            import deepspeed
            optim = deepspeed.ops.adam.DeepSpeedCPUAdam(param_groups, lr=self.lr)
        if self.warmup <= 0:
            return [optim], []
        sch = {
            'scheduler': get_cosine_schedule_with_warmup(optim, self.warmup, self.warmup*150),
            'interval': 'step', 'frequency': 1,
        }
        return [optim], sch

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        meta = batch.pop('meta', [None]*512)
        if not self.perplexity:
            generated = self.nugget.generate(**batch, n_tokens=self.new_tokens)
            breakpoint()
            return [
                [gen.cpu().numpy().astype(np.uint16), meta_]
                for gen, meta_ in zip(generated, meta)
            ]
        else:
            nugget_out = self.nugget(**batch)
            logits = nugget_out.decoder_logits[:, :-1]
            labels = batch['tgt_input_ids'][:, 1:]
            probs = logits.softmax(2).gather(2, labels.unsqueeze(2)).squeeze(2).cpu()
            probs, labels, tgt_attention_mask = (
                probs.cpu(), labels.cpu(), batch['tgt_attention_mask'].cpu()[:, 1:]
            )
            if 'skip' in batch and batch['skip'] is not None:
                skip = (batch['skip'].cpu()-1).tolist()
            else:
                skip = [None]*999
            ret = []
            for i in range(len(probs)):
                attn_mask = tgt_attention_mask[i]
                ret.append([
                    {'labels': labels[i][attn_mask].numpy(), 'skip': skip[i], 'probs': probs[i][attn_mask].numpy()},
                    meta[i]
                ])
            return ret

    def forward(self, batch: Dict[str, torch.Tensor], is_val: bool):
        batch.pop('meta', None)
        nugget_out = self.nugget(**batch)
        if not is_val:
            self.log('loss', nugget_out.loss, sync_dist=False, prog_bar=True)
        else:
            # This is a hack; we want to show 2 loss curves for ae and tc, but their loss values are mixed
            # but in practice the batch size is always 1; so we use that to classify the type of loss
            loss_type = ''
            if len(batch['objective']) == 1:
                loss_type = list(str2obj)[batch['objective'][0].item()]
            self.log(f'dev_{loss_type}_loss', nugget_out.loss, sync_dist=True, prog_bar=False)
            self.log(f'dev_loss', nugget_out.loss, sync_dist=True, prog_bar=True)
        return nugget_out

    def training_step(self, batch, batch_idx):
        out = self(batch, False)
        return out.loss

    def validation_step(self, batch, batch_idx):
        return self(batch, True)
