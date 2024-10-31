from typing import *
from dataclasses import dataclass
from lightning.pytorch import loggers as pl_loggers
from argparse import BooleanOptionalAction, ArgumentParser

from .ckpt import process_ckpt
from .gpu import gen_gpu_args, bf16_available


@dataclass()
class Default:
    value: Any = None

    def __repr__(self):
        return str(self.value)


def add_gpu_arguments(p: ArgumentParser, default_strategy: str):
    p.add_argument('--strategy', type=str, default=Default(default_strategy))
    p.add_argument('--precision', type=str, default=Default('bf16-mixed'))
    p.add_argument('--n-gpu', type=int, default=Default(16))
    p.add_argument('--deepspeed', type=int, default=Default())
    p.add_argument('--offload-optim', action=BooleanOptionalAction, default=Default(False))
    p.add_argument('--offload-param', action=BooleanOptionalAction, default=Default(False))


def process_args(args):
    hparams, version = dict(), None
    extras = dict()
    if args.ckpt is not None:
        hparams, log_dir, version, ckpt = process_ckpt(args)
        if not args.resume:
            version = None
        extras['model_path'], extras['resume_from'] = ckpt.model_path, ckpt.top
        if args.o is None:
            extras['predict_path'] = ckpt.predict_path
    if args.o is not None:
        extras['predict_path'] = args.o
    if args.test:
        extras['predict_path'] += '.test'

    # run over all examples fore predict
    if args.action == 'predict' and isinstance(args.n_val, Default):
        args.n_val = 9999999999999
    if args.perplexity and 'predict_path' in extras:
        extras['predict_path'] = extras['predict_path'].replace('predict', 'perplexity')

    for k in set(vars(args)) | set(hparams):
        if not hasattr(args, k):
            setattr(args, k, hparams.get(k))
            continue
        v = getattr(args, k, None)
        if isinstance(v, Default) and k in hparams:
            setattr(args, k, hparams.get(k))
        elif isinstance(v, Default):
            setattr(args, k, getattr(args, k).value)

    extras['gpu_config'], extras['gpu_kwargs'] = gen_gpu_args(
        args.n_gpu, args.precision, args.strategy, args.deepspeed, args.offload_optim, args.offload_param
    )
    if args.action == 'train':
        tensorboard = pl_loggers.TensorBoardLogger(args.cache, name=args.exp, version=version)
        extras['default_root_dir'] = tensorboard.log_dir
    else:
        tensorboard = None
        extras['default_root_dir'] = '/tmp/nugget'
    if args.ckpt is None and tensorboard is not None:
        tensorboard.log_hyperparams(args)

    return tensorboard, extras
