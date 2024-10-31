import os
from typing import *
import json

from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.utilities import rank_zero_only

from ..utils.types import NuggetOut, SelectedNugget
from ..utils.plots import plot_nugget_stats


class PeriodicPredict(Callback):
    def __init__(self, period: int, cache: str, n_predict: int, model_name: str):
        self.period, self.cache, self.n_predict = period, cache, n_predict
        self.model_name = model_name
        self.current_idx = 0
        self.saved = None

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.current_idx += 1
        if self.period <= 0 or (self.current_idx % self.period != 1 and self.period != 1):
            self.saved = None
        else:
            self.saved: Optional[List[SelectedNugget]] = list()

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs: NuggetOut, batch, batch_idx, dataloader_idx=0
    ):
        if self.saved is None or outputs.encoder_out.indices is None:
            return
        token_ids, token_masks = batch['src_input_ids'].cpu(), batch['src_attention_mask'].cpu()
        nug_indices, nug_mask = outputs.encoder_out.indices.cpu(), outputs.encoder_out.mask.cpu()
        for ib in range(nug_indices.shape[0]):
            nug_idx = nug_indices[ib][nug_mask[ib]].tolist()
            tokens = token_ids[ib][token_masks[ib]].tolist()
            self.saved.append(SelectedNugget(tokens, nug_idx))

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.saved is not None and len(self.saved) > 0:
            path = os.path.join(self.cache, f'step_{trainer.global_step:06}')
            plot_nugget_stats(self.saved, self.model_name, path, self.n_predict)
            with open(os.path.join(path, 'nuggets.jsonl'), 'w') as fp:
                fp.write('\n'.join(map(json.dumps, self.saved)))
        self.saved = None
