# LLM From Scratch

A GPT-style decoder-only transformer implemented from scratch in PyTorch —
tokenizer, model, and training loop, no Hugging Face Transformers dependency.

Architecture: token + positional embeddings -> N x (causal self-attention +
MLP, pre-norm, residual connections) -> final LayerNorm -> tied output head.
See [model.py](model.py) for the full implementation.

## Setup

This project shares its Python environment with `MSc_Agentic_Cyber_Defence`.
From this directory:

```bash
C:\Users\ifiah\PythonProject\MSc_Agentic_Cyber_Defence\.venv\Scripts\pip install -r requirements.txt
```

Then run everything below with that same interpreter, e.g.:

```bash
C:\Users\ifiah\PythonProject\MSc_Agentic_Cyber_Defence\.venv\Scripts\python train.py ...
```

## Phase 1a — smoke test (character-level, tiny-shakespeare)

Proves the pipeline works end-to-end before spending time on a real corpus.
Runs comfortably on an 8GB laptop GPU in a few minutes.

```bash
python data/shakespeare_char/prepare.py

python train.py --data_dir data/shakespeare_char --out_dir checkpoints/shakespeare_char \
    --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --batch_size 64 \
    --gradient_accumulation_steps 1 --max_iters 5000 --eval_interval 250 \
    --learning_rate 1e-3 --warmup_iters 100 --lr_decay_iters 5000 --min_lr 1e-4 --dropout 0.2

python sample.py --out_dir checkpoints/shakespeare_char --data_dir data/shakespeare_char \
    --prompt "ROMEO:" --max_new_tokens 300
```

## Phase 1b — GPT-2-small scale on a real corpus

Note: on Windows `cmd.exe`, the `\` line continuations below don't work
(that's bash syntax). Either paste each command as one line, or use `^`
instead of `\` for continuation. PowerShell uses a backtick `` ` ``.

1. Get a text sample to train the tokenizer on:

   ```bash
   python tokenizer/dump_corpus_sample.py --dataset NeelNanda/pile-10k --output tokenizer/corpus_sample.txt
   ```

2. Train a BPE tokenizer on it:

   ```bash
   python tokenizer/train_tokenizer.py --input tokenizer/corpus_sample.txt --vocab_size 50257
   ```

3. Tokenize the full corpus into `data/openwebtext/{train,val}.bin`:

   ```bash
   python data/openwebtext/prepare.py --dataset NeelNanda/pile-10k --num_proc 4
   ```

4. Train. On 8GB VRAM, GPT-2-small (125M params) needs a small micro-batch
   with gradient accumulation to reach an effective batch size (one-line
   cmd.exe version):

   ```bash
   python train.py --data_dir data/openwebtext --out_dir checkpoints/gpt2_small --n_layer 12 --n_head 12 --n_embd 768 --block_size 1024 --batch_size 8 --gradient_accumulation_steps 40 --max_iters 600000 --dtype bfloat16
   ```

   Drop `--block_size` / `--batch_size` or add `--compile` if you hit an
   out-of-memory error. Watch `nvidia-smi` on the first few iterations.

5. Sample from it:

   ```bash
   python sample.py --out_dir checkpoints/gpt2_small --prompt "The meaning of life is"
   ```

## Roadmap

- [x] Phase 1 — GPT architecture, BPE + char tokenizers, training loop
- [ ] Phase 2 — code generation: mix code (e.g. The Stack) into the pretraining
      corpus, or fine-tune the Phase 1 checkpoint on it
- [ ] Phase 3 — RAG: FAISS index + retriever on top of this model
- [ ] Phase 4 — scale up (GPT-2-medium/large) once on a machine with more VRAM

## Notes

- `--dtype bfloat16` is used by default (no gradient scaler needed, more
  numerically stable than fp16); switch to `float16` if your GPU predates
  Ampere.
- Checkpoints save only when validation loss improves.
- `--init_from resume` continues training from `checkpoints/<out_dir>/ckpt.pt`.
