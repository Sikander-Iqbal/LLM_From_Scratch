"""
Train a byte-level BPE tokenizer on a text corpus (GPT-2 style).

This is the tokenizer used for the "real corpus" phase (data/openwebtext).
The character-level phase (data/shakespeare_char) doesn't use this at all —
it maps characters to ids directly in data/shakespeare_char/prepare.py.

Usage:
    python tokenizer/train_tokenizer.py --input path/to/corpus.txt --vocab_size 50257 \
        --output tokenizer/bpe_tokenizer.json
"""
import argparse

from tokenizers import ByteLevelBPETokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=True, nargs='+',
                    help='one or more text files to train on')
    p.add_argument('--vocab_size', type=int, default=50257)
    p.add_argument('--min_frequency', type=int, default=2)
    p.add_argument('--output', type=str, default='tokenizer/bpe_tokenizer.json')
    args = p.parse_args()

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=args.input,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=['<|endoftext|>'],
    )
    tokenizer.save(args.output)
    print(f"trained BPE tokenizer (vocab_size={tokenizer.get_vocab_size()}) -> {args.output}")


if __name__ == '__main__':
    main()
