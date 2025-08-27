# Dodo: Dynamic Contextual Compression for Decoder-only LMs

This repo contains tools for training a summarization model with a Llama 3.1 8B backbone. You can train the model, push it to the Hugging Face Hub and later load it with `AutoModel.from_pretrained`.

## Setup
```
conda create -n dodo python=3.10
conda activate dodo
pip install -r requirements.txt
```

Make sure you have access to the `meta-llama/Llama-3.1-8B-Instruct` weights on Hugging Face.

## Training
```
srun python train.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --epochs 5 \
    --hf_repo_id "jeongseokoh/Llama3.1-8B-dodo" \
    --compress_rate 10
```

## Inference
```
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("jeongseokoh/Llama3.1-8B-dodo", trust_remote_code=True).cuda().eval()
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", legacy=False, use_fast=False)

context = '...'
prompt = '...'
context_tokens = tok(context, add_special_tokens=True, return_tensors='pt')['input_ids'][:, :-1].cuda()
prompt_tokens = tok(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].cuda()
objective = torch.tensor([1], device='cuda')
skip = torch.tensor([prompt_tokens.shape[1]-1], device='cuda')

with torch.inference_mode():
    generated = model.generate(
        src_input_ids=context_tokens, src_attention_mask=context_tokens.new_ones(context_tokens.shape).bool(),
        tgt_input_ids=prompt_tokens, tgt_attention_mask=prompt_tokens.new_ones(prompt_tokens.shape).bool(),
        objective=objective, skip=skip,
    )

out_text = tok.decode(generated[0]).strip()
```
