"""
Phase 1b: tokenize a real corpus with the trained BPE tokenizer for a
GPT-2-small scale pretraining run.

Requires tokenizer/bpe_tokenizer.json to already exist (see
tokenizer/train_tokenizer.py) and the `datasets` library for streaming
the corpus from the Hugging Face Hub.

Usage:
    python data/openwebtext/prepare.py --dataset NeelNanda/pile-10k --num_proc 4
    # or, for a larger/more curated corpus:
    python data/openwebtext/prepare.py --dataset HuggingFaceFW/fineweb-edu \
        --dataset_config sample-10BT --text_field text --num_proc 4
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

HERE = os.path.dirname(__file__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='NeelNanda/pile-10k',
                    help='any HF Hub dataset with a text column (must have real data files, not a legacy loading script)')
    p.add_argument('--dataset_config', type=str, default=None)
    p.add_argument('--text_field', type=str, default='text')
    p.add_argument('--tokenizer_path', type=str,
                    default=os.path.join(HERE, '..', '..', 'tokenizer', 'bpe_tokenizer.json'))
    p.add_argument('--val_fraction', type=float, default=0.0005)
    p.add_argument('--num_proc', type=int, default=4)
    args = p.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eot_id = tokenizer.token_to_id('<|endoftext|>')

    dataset = load_dataset(args.dataset, args.dataset_config, split='train', num_proc=args.num_proc)
    split_dataset = dataset.train_test_split(test_size=args.val_fraction, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = tokenizer.encode(example[args.text_field]).ids
        ids.append(eot_id)
        return {'ids': ids, 'len': len(ids)}

    tokenized = split_dataset.map(
        process, remove_columns=[args.text_field], desc='tokenizing', num_proc=args.num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        out_path = os.path.join(HERE, f'{split}.bin')
        arr = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(arr_len,))

        idx = 0
        for example in dset:
            ids = example['ids']
            arr[idx:idx + len(ids)] = ids
            idx += len(ids)
        arr.flush()
        print(f"wrote {out_path} ({arr_len:,} tokens)")


if __name__ == '__main__':
    main()
