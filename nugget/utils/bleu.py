from argparse import ArgumentParser
import os
import pickle

from transformers import AutoTokenizer, LlamaTokenizer


def main():
    from nltk.translate.bleu_score import corpus_bleu
    import numpy as np
    from matplotlib import pyplot as plt
    p = ArgumentParser()
    p.add_argument('-p', help='Prediction file')
    p.add_argument('-m', help='Model name')
    args = p.parse_args()
    if 'llama' in args.m.lower():
        tok = LlamaTokenizer.from_pretrained(args.m)
    else:
        tok = AutoTokenizer.from_pretrained(args.m)
    predictions = []
    for f in os.listdir(args.p):
        predictions.extend(pickle.load(open(os.path.join(args.p, f), 'rb')))
    bleus = []
    for inputs, mask, outputs in predictions:
        src = inputs[mask]
        assert outputs[0] == 1
        if outputs[0] == 1:
            outputs = outputs[1:]
        tgt = outputs[:mask.shape[0]][mask]
        src, tgt = map(tok.decode, [src, tgt])
        bleus.append([mask.sum(), corpus_bleu([[tgt.split()]], [src.split()])*100])
    bleus = np.array(bleus)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.scatter(bleus[:, 0], bleus[:, 1])
    ax.set_xlabel('Seq Length')
    ax.set_ylabel('BLEU x 100')
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ.get('HOME'), 'scatter.png'), dpi=250)


if __name__ == '__main__':
    main()
