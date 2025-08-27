from typing import Any

from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig

from .nugget import Nugget


class NuggetConfig(PretrainedConfig):
    model_type = "nugget"

    def __init__(
        self,
        pretrained: str,
        encoder_type: str = "nugget",
        nugget_layer: int = 0,
        ratio: float = 0.1,
        lora: int = 16,
        soft: int | None = None,
        ind_scorer: bool = False,
        freeze_decoder: bool = False,
        freeze_scorer: bool = False,
        skip_first: int | None = None,
        new_line_stop: bool = False,
        extractive: bool = False,
        auto: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pretrained = pretrained
        self.encoder_type = encoder_type
        self.nugget_layer = nugget_layer
        self.ratio = ratio
        self.lora = lora
        self.soft = soft
        self.ind_scorer = ind_scorer
        self.freeze_decoder = freeze_decoder
        self.freeze_scorer = freeze_scorer
        self.skip_first = skip_first
        self.new_line_stop = new_line_stop
        self.extractive = extractive
        self.auto = auto
        self.auto_map = {
            "AutoConfig": "hf_nugget.NuggetConfig",
            "AutoModel": "hf_nugget.NuggetForConditionalGeneration",
        }


class NuggetForConditionalGeneration(PreTrainedModel):
    config_class = NuggetConfig
    base_model_prefix = "nugget"

    def __init__(self, config: NuggetConfig) -> None:
        super().__init__(config)
        self.nugget = Nugget(
            pretrained=config.pretrained,
            encoder_type=config.encoder_type,
            nugget_layer=config.nugget_layer,
            ratio=config.ratio,
            lora=config.lora,
            soft=config.soft,
            ind_scorer=config.ind_scorer,
            freeze_decoder=config.freeze_decoder,
            freeze_scorer=config.freeze_scorer,
            skip_first=config.skip_first,
            new_line_stop=config.new_line_stop,
            extractive=config.extractive,
            auto=config.auto,
        )

    def forward(self, **kwargs):
        return self.nugget(**kwargs)

    def generate(self, **kwargs):
        return self.nugget.generate(**kwargs)

    def state_dict(self, *args, **kwargs):  # type: ignore
        return self.nugget.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore
        return self.nugget.load_state_dict(state_dict, strict=strict)


AutoConfig.register("nugget", NuggetConfig)
AutoModel.register(NuggetConfig, NuggetForConditionalGeneration)
