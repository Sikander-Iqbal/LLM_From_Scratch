"""
Phase 1a smoke test: character-level tokenization of tiny-shakespeare.

No BPE tokenizer needed here — every unique character is a token. This is
the fastest possible way to prove the model/train loop works end-to-end
before moving to a real BPE-tokenized corpus.

Produces train.bin, val.bin (uint16 token ids) and meta.pkl (stoi/itos) in
this directory.
"""
import os
import pickle
import urllib.request

import numpy as np

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
HERE = os.path.dirname(__file__)


def main():
    input_path = os.path.join(HERE, 'input.txt')
    if not os.path.exists(input_path):
        print(f"downloading tiny-shakespeare from {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, input_path)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    chars = sorted(set(data))
    vocab_size = len(chars)
    print(f"vocab size: {vocab_size} unique characters")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens, val has {len(val_ids):,} tokens")

    train_ids.tofile(os.path.join(HERE, 'train.bin'))
    val_ids.tofile(os.path.join(HERE, 'val.bin'))

    with open(os.path.join(HERE, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}, f)


if __name__ == '__main__':
    main()
