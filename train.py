import argparse
import os
import shutil

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from huggingface_hub import HfApi

from nugget.codec.nugget_trainer import NuggetTrainer
from nugget.codec.hf_nugget import NuggetConfig, NuggetForConditionalGeneration
from nugget.data_io.lmsumm import LMSumm
from nugget.data_io.data_utils import collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hf_repo_id", required=True)
    parser.add_argument("--compress_rate", type=int, default=10)
    parser.add_argument("--output_dir", default="training_output")
    args = parser.parse_args()

    ratio = 1.0 / float(args.compress_rate)

    train_ds = LMSumm("cnn_dailymail", None, ratio, None, 1024, "train")
    val_ds = LMSumm("cnn_dailymail", None, ratio, None, 1024, "validation")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = NuggetTrainer(
        lr=1e-4,
        ft_lr=1e-4,
        warmup=100,
        gpu_config="single",
        new_tokens=32,
        pretrained=args.model_name,
        encoder_type="nugget",
        nugget_layer=0,
        ratio=ratio,
        lora=16,
        soft=None,
        ind_scorer=False,
        freeze_decoder=False,
        freeze_scorer=False,
    )

    ckpt_cb = ModelCheckpoint(monitor="dev_loss", filename="weights", save_top_k=1, mode="min")
    trainer = Trainer(max_epochs=args.epochs, callbacks=[ckpt_cb])
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = ckpt_cb.best_model_path
    if not best_ckpt:
        raise RuntimeError("No checkpoint was created during training")

    trained = NuggetTrainer.load_from_checkpoint(best_ckpt)
    hf_config = NuggetConfig(
        pretrained=args.model_name,
        encoder_type="nugget",
        nugget_layer=0,
        ratio=ratio,
        lora=16,
        soft=None,
        ind_scorer=False,
        freeze_decoder=False,
        freeze_scorer=False,
    )
    hf_model = NuggetForConditionalGeneration(hf_config)
    hf_model.load_state_dict(trained.nugget.state_dict(), strict=False)

    save_dir = os.path.join(args.output_dir, "hf_model")
    os.makedirs(save_dir, exist_ok=True)
    hf_model.save_pretrained(save_dir)
    tok = AutoTokenizer.from_pretrained(args.model_name, legacy=False, use_fast=False)
    tok.save_pretrained(save_dir)
    shutil.copyfile("nugget/codec/hf_nugget.py", os.path.join(save_dir, "hf_nugget.py"))

    api = HfApi()
    api.create_repo(repo_id=args.hf_repo_id, exist_ok=True)
    api.upload_folder(repo_id=args.hf_repo_id, folder_path=save_dir)


if __name__ == "__main__":
    main()
