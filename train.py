"""
Training loop for the from-scratch GPT.

Works against any data_dir produced by data/*/prepare.py (expects train.bin,
val.bin as uint16 token-id arrays, and optionally meta.pkl for vocab_size).

Examples:
    # smoke test: tiny char-level model on tiny-shakespeare, runs on a laptop GPU in minutes
    python train.py --data_dir data/shakespeare_char --out_dir checkpoints/shakespeare_char \
        --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --batch_size 64 \
        --max_iters 5000 --eval_interval 250

    # GPT-2-small scale on a BPE-tokenized corpus
    python train.py --data_dir data/openwebtext --out_dir checkpoints/gpt2_small \
        --n_layer 12 --n_head 12 --n_embd 768 --block_size 1024 --batch_size 12 \
        --gradient_accumulation_steps 40 --max_iters 600000
"""
import argparse
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
from tqdm.auto import tqdm

from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume'])

    # model
    p.add_argument('--n_layer', type=int, default=12)
    p.add_argument('--n_head', type=int, default=12)
    p.add_argument('--n_embd', type=int, default=768)
    p.add_argument('--block_size', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--bias', action='store_true', default=True)

    # optimization
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--gradient_accumulation_steps', type=int, default=40)
    p.add_argument('--learning_rate', type=float, default=6e-4)
    p.add_argument('--max_iters', type=int, default=600000)
    p.add_argument('--weight_decay', type=float, default=1e-1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--warmup_iters', type=int, default=2000)
    p.add_argument('--lr_decay_iters', type=int, default=600000)
    p.add_argument('--min_lr', type=float, default=6e-5)

    # logging / eval
    p.add_argument('--eval_interval', type=int, default=2000)
    p.add_argument('--eval_iters', type=int, default=200)
    p.add_argument('--log_interval', type=int, default=10)

    # system
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype', type=str, default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'])
    p.add_argument('--compile', action='store_true')
    p.add_argument('--seed', type=int, default=1337)
    return p.parse_args()


def get_batch(split, data_dir, block_size, batch_size, device):
    # re-open the memmap every batch (avoids a slow memory leak from a long-lived mmap, see nanoGPT issue)
    path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, data_dir, block_size, batch_size, device, eval_iters, ctx):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters), desc=f'eval {split}', unit='batch', leave=False):
            x, y = get_batch(split, data_dir, block_size, batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.amp.GradScaler(device=device_type, enabled=(args.dtype == 'float16'))

    # vocab_size: prefer meta.pkl (char-level / custom BPE tokenizer), else GPT-2-style default
    meta_path = os.path.join(args.data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
    else:
        vocab_size = 50304  # nearest multiple of 64 above GPT-2's 50257, for faster matmuls

    model_args = dict(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, dropout=args.dropout, bias=args.bias,
        vocab_size=vocab_size,
    )

    iter_num = 0
    best_val_loss = float('inf')
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')

    if args.init_from == 'resume':
        checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model_args = checkpoint['model_args']
        model = GPT(GPTConfig(**model_args))
        model.load_state_dict(checkpoint['model'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    else:
        model = GPT(GPTConfig(**model_args))

    model.to(args.device)
    print(f"model has {model.num_params() / 1e6:.2f}M non-embedding parameters")

    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )
    if args.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # free memory

    if args.compile:
        model = torch.compile(model)

    x, y = get_batch('train', args.data_dir, args.block_size, args.batch_size, args.device)
    t0 = time.time()

    pbar = tqdm(total=args.max_iters, initial=iter_num, desc='train', unit='it')
    while iter_num <= args.max_iters:
        lr = get_lr(iter_num, args.warmup_iters, args.lr_decay_iters, args.learning_rate, args.min_lr)
        for group in optimizer.param_groups:
            group['lr'] = lr

        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model, args.data_dir, args.block_size, args.batch_size, args.device, args.eval_iters, ctx
            )
            tqdm.write(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }, ckpt_path)
                tqdm.write(f"  saved checkpoint to {ckpt_path}")

        for _ in tqdm(range(args.gradient_accumulation_steps), desc='microstep', unit='step', leave=False):
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.gradient_accumulation_steps
            # prefetch next batch while the GPU is busy with backward
            x, y = get_batch('train', args.data_dir, args.block_size, args.batch_size, args.device)
            scaler.scale(loss).backward()

        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        full_loss = loss.item() * args.gradient_accumulation_steps
        pbar.set_postfix(loss=f"{full_loss:.4f}", lr=f"{lr:.2e}")
        pbar.update(1)

        if iter_num % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tqdm.write(f"iter {iter_num}: loss {full_loss:.4f}, "
                       f"{dt * 1000 / args.log_interval:.1f}ms/iter, lr {lr:.2e}")

        iter_num += 1
    pbar.close()


if __name__ == '__main__':
    main()
