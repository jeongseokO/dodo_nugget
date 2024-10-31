import torch
import yaml
from nugget.codec.nugget_trainer import NuggetTrainer
from transformers import AutoTokenizer

ckpt_path = '/path/to/ckpt/folder'
pretrained = 'meta-llama/Llama-2-7b-chat-hf'
ckpt_path = '/home/guanghuiqin/dodo_ckpt/ratio20'
hparams = yaml.load(open(ckpt_path+'/hparams.yaml'), yaml.CLoader)

max_new_tokens = 32
nugget_trainer = NuggetTrainer.load_from_checkpoint(
    ckpt_path+'/weights.ckpt', strict=False, new_tokens=max_new_tokens,
)
nugget_trainer = nugget_trainer.cuda().eval()

tok = AutoTokenizer.from_pretrained(pretrained)
context = '''
Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin
Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.
'''.strip()
question = 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'
prompt = '[INST] Based on the provided document, answer the following question: {} [/INST]\n\n '.format(question)

context_tokens = tok(context, add_special_tokens=True, return_tensors='pt')['input_ids'][:, :-1].cuda()
prompt_tokens = tok(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].cuda()
# objective is a flag that guides model to do text continuation instead of autoencoding
objective = torch.tensor([1], device='cuda')
skip = torch.tensor([prompt_tokens.shape[1]-1], device='cuda')

with torch.inference_mode():
    generated = nugget_trainer.nugget.generate(
        src_input_ids=context_tokens, src_attention_mask=context_tokens.new_ones(context_tokens.shape).bool(),
        tgt_input_ids=prompt_tokens, tgt_attention_mask=prompt_tokens.new_ones(prompt_tokens.shape).bool(),
        objective=objective, skip=skip,
    )

out_text = tok.decode(generated[0]).strip()
print(out_text)

