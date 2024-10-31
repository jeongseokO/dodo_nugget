from typing import *

from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy, MisconfigurationException


class MyDeepSpeedStrategy(DeepSpeedStrategy):
    def load_checkpoint(self, checkpoint_path) -> Dict[str, Any]:
        if self.load_full_weights and self.zero_stage_3:
            # Broadcast to ensure we load from the rank 0 checkpoint
            # This doesn't have to be the case when using deepspeed sharded checkpointing
            checkpoint_path = self.broadcast(checkpoint_path)
            return super().load_checkpoint(checkpoint_path)

        # Rely on deepspeed to load the checkpoint and necessary information
        assert self.lightning_module is not None

        from lightning.pytorch.trainer.states import TrainerFn

        is_fitting = self.lightning_module.trainer.state.fn == TrainerFn.FITTING
        _, client_state = self.deepspeed_engine.load_checkpoint(
            checkpoint_path,
            # Below is the only difference
            load_module_strict=False,
            load_optimizer_states=is_fitting, load_lr_scheduler_states=False
        )
        if client_state is None:
            raise MisconfigurationException(
                "DeepSpeed was unable to load the checkpoint. Ensure you passed in a DeepSpeed compatible checkpoint "
                "or a single checkpoint file with `Trainer(strategy=DeepSpeedStrategy(load_full_weights=True))`."
            )
        return client_state
