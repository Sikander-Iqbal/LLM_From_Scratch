"""
Generate text from a trained checkpoint.

Usage:
    python sample.py --out_dir checkpoints/shakespeare_char --prompt "ROMEO:" --max_new_tokens 300
"""
import argparse
import os
import pickle

import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--data_dir', type=str, default=None,
                    help='defaults to the data dir implied by out_dir\'s meta.pkl if present')
    p.add_argument('--tokenizer_path', type=str, default='tokenizer/bpe_tokenizer.json',
                    help='used when the checkpoint has no char-level meta.pkl')
    p.add_argument('--prompt', type=str, default='\n')
    p.add_argument('--num_samples', type=int, default=1)
    p.add_argument('--max_new_tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=200)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    torch.manual_seed(args.seed)

    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(args.device)

    # figure out which tokenizer produced this checkpoint's data
    meta_path = os.path.join(args.data_dir or '', 'meta.pkl') if args.data_dir else None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda ids: ''.join(itos[i] for i in ids)
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer_path)
        encode = lambda s: tok.encode(s).ids
        decode = lambda ids: tok.decode(ids)

    start_ids = encode(args.prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    with torch.no_grad():
        for i in range(args.num_samples):
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
            print(decode(y[0].tolist()))
            print('---')


if __name__ == '__main__':
    main()
