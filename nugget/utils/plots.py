from typing import *

from transformers import AutoTokenizer, LlamaTokenizer
import os
import html

from .types import SelectedNugget


def clean_token(tok_ids, tokenizer):
    tokens = list()
    for ts in tokenizer.convert_ids_to_tokens(tok_ids):
        space = ' ' if ts.startswith('▁') else ''
        tokens.append(space + tokenizer.convert_tokens_to_string([ts]))
    return tokens


def plot_nugget_stats(selects: List[SelectedNugget], model_name: str, path: str, n_example: int):
    klass = LlamaTokenizer if 'llama' in model_name.lower() else AutoTokenizer
    tok = klass.from_pretrained(model_name, use_fast=False, legacy=False)
    select_tokens = list()
    for se in selects:
        select_tokens.append(SelectedNugget(clean_token(se.tokens, tok), se.nugget_indices))
    os.makedirs(path, exist_ok=True)
    gen_highlight(select_tokens[:n_example], path)


def highlight_sent(indices, tokens):
    ret = list()
    colors = ['red', 'green']
    color_idx = 0
    for i, tok in enumerate(tokens):
        if tok in ['<pad>', '<s>']:
            continue
        tok = html.escape(tok)
        space = ' ' if tok.startswith(' ') else ''
        tok = tok.replace(' ', '')
        if i in indices:
            tok = f'<span class="highlight_{colors[color_idx]}">{tok}</span>'
            color_idx = (color_idx + 1) % len(colors)
        tok = space + tok
        ret.append(tok)
    ret_str = ''.join(ret).replace('\n', '&nbsp;')
    return ret_str


def gen_highlight(selects: List[SelectedNugget], save_path: str):
    highlighted = [highlight_sent(set(se.nugget_indices), se.tokens) for se in selects]
    with open(os.path.join(save_path, 'highlight.html'), 'w', encoding='utf-8') as fp:
        fp.write(html_head.replace('BODY_PLACEHOLDER', '<br><hr>'.join(highlighted)))


html_head = '''<!DOCTYPE html>
<html>
<head>
<style>
.highlight {background-color: red;}
.highlight_red {background-color: red;}
.highlight_green {background-color: green;}
</style>
</head>
<body>
BODY_PLACEHOLDER
</body>
</html>'''
