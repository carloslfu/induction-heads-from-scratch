"""
The real thing: induction heads emerging from natural text.

The sandbox run (induction_from_scratch.py) trains ON the induction task,
which proves the circuit can be built — but not that it arises unprompted.
This script reproduces the paper's actual methodology:

  - TRAIN on plain text (TinyShakespeare), plain next-token loss.
    Nothing in the data or objective mentions induction. Two token
    granularities over the same corpus (--tok word|char): word-level,
    where a repeated token is informative (the regime of the paper's BPE
    models), and char-level, where it isn't — the negative control.
  - PROBE at eval time only: synthetic repeated-random-token sequences
    (the paper's prefix-matching diagnostic) are run under no_grad every
    EVAL_EVERY steps to ask "has an induction head formed?" The model
    never trains on a probe.
  - MEASURE in-context learning the way the paper does: the ICL score is
    how much better the model predicts late tokens than early tokens in
    held-out text (loss at positions 10..30 minus loss at 200..250).

If Olsson et al. are right, induction heads form abruptly during ordinary
text training, only in the 2-layer model, and the ICL score jumps in the
same window. Nothing here engineers that outcome.

Anti-cheating invariants (each checkable in this file):
  - training batches are raw corpus slices from the first 90% of the text;
    the last 10% is held out for the ICL measurement and never trained on
  - probe sequences are generated from their own fixed RNG and appear
    only inside torch.no_grad()
  - same generic architecture, init, and optimizer as the sandbox run;
    no induction-specific loss terms anywhere

Run:
  python3 induction_on_text.py                        # 2-layer, word-level
  python3 induction_on_text.py --layers 1             # 1-layer control
  python3 induction_on_text.py --tok char             # char-level (null result)
  python3 induction_on_text.py --tok char --layers 1  # char 1-layer control
"""

import argparse
import json
import math
import os
import re
import time

import torch
import torch.nn.functional as F

import induction_from_scratch as g   # reuse forward pass, device, dims


# -----------------------------------------------------------------------------
# Config — text run
# -----------------------------------------------------------------------------
# Word mode needs a corpus the model can't memorize: ~8.5M word-tokens of
# public-domain novels (run data/fetch_gutenberg.sh once to build it).
# Char mode uses the small committed TinyShakespeare. Training word-level
# on TinyShakespeare is the documented memorization failure — reproduce it
# with:  --tok word --corpus data/tinyshakespeare.txt
CORPORA     = {"word": "data/gutenberg.txt",
               "char": "data/tinyshakespeare.txt"}
HELD_FRAC   = 0.10       # last 10% of the corpus: ICL eval only, never trained
SEQ_LEN     = 256        # longer context than the sandbox: room for ICL
BATCH       = 128
WORD_VOCAB  = 4096       # word mode: top 4,095 tokens + <unk> (92% coverage)
N_STEPS     = 10_000
EVAL_EVERY  = 50
LOG_EVERY   = 250
SEED        = 0

PROBE_HALF  = 128        # probe = 128 random tokens repeated twice
PROBE_N     = 64
ICL_EARLY   = (10, 30)   # positions for the "early" loss
ICL_LATE    = (200, 250) # positions for the "late" loss

CHECKPOINT_STEPS = [
    0, 250, 500, 750, 1_000, 1_500, 2_000, 2_500, 3_000,
    4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000,
]


# -----------------------------------------------------------------------------
# Data
#
# Two token granularities over the SAME corpus:
#   char: vocab 65. A repeated character carries almost no information, so
#         induction has nearly nothing to offer — the negative control.
#   word: vocab 2,048 (top words + <unk>). A repeated word ("Romeo") is a
#         strong signal — the regime the paper's BPE models live in.
# -----------------------------------------------------------------------------
def load_corpus(tok="word", path=None):
    text = open(path or CORPORA[tok]).read()
    if tok == "char":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        ids = [stoi[c] for c in text]
        vocab = len(chars)
    else:
        words = re.findall(r"[A-Za-z']+|[^A-Za-z'\s]|\n", text)
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        keep = sorted(counts, key=counts.get, reverse=True)[:WORD_VOCAB - 1]
        stoi = {w: i for i, w in enumerate(keep)}
        unk = WORD_VOCAB - 1
        ids = [stoi.get(w, unk) for w in words]
        vocab = WORD_VOCAB
    data = torch.tensor(ids, dtype=torch.long)
    n_train = int(len(data) * (1 - HELD_FRAC))
    return data[:n_train], data[n_train:], vocab


def text_batch(data, batch, generator=None):
    starts = torch.randint(0, len(data) - SEQ_LEN - 1, (batch,),
                           generator=generator)
    return torch.stack([data[s:s + SEQ_LEN] for s in starts])


def probe_batch(vocab, n=PROBE_N, generator=None):
    """Random tokens repeated twice: [s ; s]. Positions PROBE_HALF+j are
    predictable only by looking up the first copy — the paper's
    prefix-matching diagnostic. Eval-only; the model never trains on this.
    (vocab − 1 keeps word-mode probes free of the <unk> id.)"""
    s = torch.randint(0, max(vocab - 1, 2), (n, PROBE_HALF),
                      generator=generator)
    return torch.cat([s, s], dim=1)


# -----------------------------------------------------------------------------
# Model — same shapes and init style as the sandbox run, text-sized
# -----------------------------------------------------------------------------
def init_params_text(vocab, n_layers, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    d = 1.0 / math.sqrt(g.D_MODEL)

    def param(*shape, std):
        t = torch.randn(*shape, generator=gen) * std
        t = t.to(g.DEVICE)
        t.requires_grad_(True)
        return t

    p = {
        "W_E":   param(vocab, g.D_MODEL, std=d),
        "W_pos": param(SEQ_LEN, g.D_MODEL, std=d),
        "W_U":   param(g.D_MODEL, vocab, std=d),
    }
    for l in range(n_layers):
        p[f"W_Q{l}"] = param(g.N_HEADS, g.D_MODEL, g.D_HEAD, std=d)
        p[f"W_K{l}"] = param(g.N_HEADS, g.D_MODEL, g.D_HEAD, std=d)
        p[f"W_V{l}"] = param(g.N_HEADS, g.D_MODEL, g.D_HEAD, std=d)
        p[f"W_O{l}"] = param(g.N_HEADS, g.D_HEAD, g.D_MODEL,
                             std=1.0 / math.sqrt(g.D_HEAD))
    return p


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def nll_grid(p, x):
    """Per-position next-token NLL. (B, T-1)."""
    logits = g.forward(p, x)
    lp = logits[:, :-1].log_softmax(-1)
    return -lp.gather(-1, x[:, 1:, None])[..., 0]


def icl_metrics(p, x):
    """The paper's in-context learning score: early loss − late loss.
    Positive = the model predicts better with more context."""
    nll = nll_grid(p, x)
    early = nll[:, ICL_EARLY[0]:ICL_EARLY[1]].mean().item()
    late = nll[:, ICL_LATE[0]:ICL_LATE[1]].mean().item()
    return early, late, early - late


def probe_scores(p, x):
    """Prefix-matching diagnostics on repeated-sequence probes.
    prev[l][h]: mean attention to position i−1.
    ind[l][h]:  mean attention from query PROBE_HALF+j to key j+1 — the
                token right after the previous occurrence.
    acc:        next-token accuracy on the second copy."""
    with torch.no_grad():
        logits, attns = g.forward(p, x, want_attn=True)
    ar = torch.arange(PROBE_HALF - 1, device=x.device)
    q, k = PROBE_HALF + ar, ar + 1

    prev, ind = [], []
    for attn in attns:
        d = torch.diagonal(attn, offset=-1, dim1=2, dim2=3)
        prev.append(d.mean(dim=(0, 2)).tolist())
        ind.append([attn[:, h, q, k].mean().item()
                    for h in range(attn.shape[1])])

    pred = logits[:, :-1].argmax(-1)                  # (B, T-1): pred for x[t+1]
    tgt = x[:, 1:]
    acc = (pred[:, q] == tgt[:, q]).float().mean().item()
    return prev, ind, acc


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(n_layers=2, n_steps=N_STEPS, seed=SEED, tok="word", corpus=None):
    torch.manual_seed(seed)
    tag = f"T{n_layers}" + ("w" if tok == "word" else "") \
        + (f"s{seed}" if seed != 0 else "")

    train_data, held_data, vocab = load_corpus(tok, corpus)
    print(f"[{tag}] corpus {len(train_data) + len(held_data):,} "
          f"{tok}-tokens | vocab {vocab} | train {len(train_data):,} | "
          f"held-out {len(held_data):,}")

    p = init_params_text(vocab, n_layers, seed)
    n = sum(t.numel() for t in p.values())
    print(f"[{tag}] layers {n_layers} | params {n:,} | device {g.DEVICE}")

    opt = torch.optim.AdamW(list(p.values()), lr=g.LR, betas=g.BETAS,
                            weight_decay=g.WD)

    # Fixed eval sets, each with its own RNG. Probes and held-out text are
    # never used for gradients.
    ge = torch.Generator().manual_seed(20_000 + seed)
    px = probe_batch(vocab, generator=ge).to(g.DEVICE)
    gh = torch.Generator().manual_seed(30_000 + seed)
    hx = text_batch(held_data, 256, generator=gh).to(g.DEVICE)
    gt = torch.Generator().manual_seed(40_000 + seed)
    tx_eval = text_batch(train_data, 256, generator=gt).to(g.DEVICE)

    log = []
    t0 = time.time()
    for step in range(n_steps + 1):
        if step % EVAL_EVERY == 0 or step == n_steps:
            with torch.no_grad():
                held_early, held_late, icl = icl_metrics(p, hx)
                held_loss = nll_grid(p, hx).mean().item()
                train_eval_loss = nll_grid(p, tx_eval).mean().item()
            prev, ind, probe_acc = probe_scores(p, px)
            log.append({
                "step":       step,
                "train_loss": train_eval_loss,
                "held_loss":  held_loss,
                "icl_early":  held_early,
                "icl_late":   held_late,
                "icl":        icl,
                "prev":       prev,
                "ind":        ind,
                "probe_acc":  probe_acc,
            })
            if step % LOG_EVERY == 0 or step == n_steps:
                best_ind = max(max(r) for r in ind)
                print(f"[{tag}] step {step:5d} | held {held_loss:.3f} | "
                      f"icl {icl:+.3f} | best probe ind {best_ind:.2f} | "
                      f"probe acc {probe_acc:.3f} | {time.time()-t0:6.1f}s")

        if step in CHECKPOINT_STEPS:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in p.items()},
                       f"checkpoints/params_{tag}_{step:06d}.pt")

        if step == n_steps:
            break

        x = text_batch(train_data, BATCH).to(g.DEVICE)
        logits = g.forward(p, x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with open(f"training_log_{tag}.json", "w") as f:
        json.dump(log, f)
    torch.save({k: v.detach().cpu() for k, v in p.items()},
               f"params_{tag}.pt")
    print(f"[{tag}] saved training_log_{tag}.json and params_{tag}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tok", choices=["word", "char"], default="word")
    ap.add_argument("--corpus", default=None,
                    help="override the default corpus for this tokenization")
    args = ap.parse_args()
    train(n_layers=args.layers, n_steps=args.steps, seed=args.seed,
          tok=args.tok, corpus=args.corpus)
