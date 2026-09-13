"""
Pull a text sample from a Hugging Face Hub dataset and dump it to a plain
.txt file, for use as tokenizer training input (see train_tokenizer.py).

Usage:
    python tokenizer/dump_corpus_sample.py --dataset NeelNanda/pile-10k --output tokenizer/corpus_sample.txt
"""
import argparse

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='NeelNanda/pile-10k')
    p.add_argument('--dataset_config', type=str, default=None)
    p.add_argument('--text_field', type=str, default='text')
    p.add_argument('--max_docs', type=int, default=10000)
    p.add_argument('--output', type=str, default='tokenizer/corpus_sample.txt')
    args = p.parse_args()

    dataset = load_dataset(args.dataset, args.dataset_config, split='train', streaming=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            if i >= args.max_docs:
                break
            f.write(example[args.text_field])
            f.write('\n')

    print(f"wrote sample corpus -> {args.output}")


if __name__ == '__main__':
    main()
