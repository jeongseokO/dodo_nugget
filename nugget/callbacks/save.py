import os
import random
from typing import *
import pickle
import time

from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl


class SavePredict(Callback):
    def __init__(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.save_path = os.path.join(folder, f'{int(time.time())}.{random.randint(0, 99999):05}.pkl')
        self.outputs = list()

    def on_predict_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any,
        batch: Any, batch_idx: int, dataloader_idx: int = 0,
    ) -> None:
        self.outputs.extend(outputs)

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pickle.dump(self.outputs, open(self.save_path, 'wb'))
