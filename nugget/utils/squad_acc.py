import os
import json
import pickle
import string

from transformers import LlamaTokenizer
from tqdm import tqdm
from plutils.common import logger


class Solution:
    # https://www.tutorialspoint.com/integer-to-english-words-in-python-programming
    less_than_20 = [
        "", "One", "Two", "Three", "Four", "Five", "Six",
        "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
        "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"
    ]
    tens = [
        "", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty",
        "Seventy", "Eighty", "Ninety"
    ]
    thousands = ["", "Thousand", "Million", "Billion"]

    def __call__(self, num):
        if num == 0:
            return "Zero"
        ans = ""
        i = 0
        while num > 0:
            if num % 1000 != 0:
                ans = self.helper(num % 1000) + Solution.thousands[i] + " " + ans
                i += 1
                num //= 1000
            return ans.strip()

    def helper(self, n):
        if n == 0:
            return ""
        elif n < 20:
            return Solution.less_than_20[n] + " "
        elif n < 100:
            return Solution.tens[n//10] + " " + self.helper(n % 10)
        else:
            return Solution.less_than_20[n // 100] + " Hundred " + self.helper(n % 100)


n2w = Solution()
w2n = {n2w(n).lower(): str(n) for n in range(1, 1000)}
w2n['zero'] = str(0)


def normalize_text(t):
    ret = []
    for c in t.lower():
        if c in (string.punctuation + '–' + '-'):
            ret.append(' ')
        else:
            ret.append(c)
    ret = ''.join(ret)
    for w, n in w2n.items():
        ret = ret.replace(w, n)
    for p in string.punctuation:
        ret = ret.strip(p)
    return ret.strip()


def squad_acc(prediction_path, tokenizer_name):
    preds = []
    for fn in os.listdir(prediction_path):
        if fn.endswith('.pkl'):
            preds.extend(pickle.load(open(os.path.join(prediction_path, fn), 'rb')))
    logger.warning(f'Loaded {len(preds)} predictions.')
    n_hit = 0
    tok = LlamaTokenizer.from_pretrained(tokenizer_name)
    for out_ids, meta in tqdm(preds):
        out_text = tok.decode(out_ids, skip_special_tokens=True)
        for ans in meta['answers']['text']:
            if normalize_text(ans) in normalize_text(out_text):
                n_hit += 1
                break
        else:
            # print(out_text.replace('\n', ' ').replace('  ', ' '))
            # print(','.join(meta['answers']['text']))
            # print('-'*10)
            x = 1
            pass
    acc = n_hit / len(preds) * 100
    logger.warning(f'{n_hit}/{len(preds)}: {acc}%')
    with open(os.path.join(prediction_path, 'acc.json'), 'w') as fp:
        json.dump({'acc': acc, 'match': n_hit, 'total': len(preds)}, fp)
