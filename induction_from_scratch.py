"""
Induction heads from scratch — watching in-context learning turn on.

A 2-layer attention-only transformer built from raw torch.Tensors —
no nn.Module, no nn.Linear, no nn.Embedding. Trained on sequences of
random tokens in which one segment repeats at a RANDOM later position.
The only way to predict anything better than chance is to learn the
induction algorithm:

    "find where the current token appeared before, copy what followed"

which requires two attention heads in different layers composing:
  - a previous-token head (layer 0): writes "the token before me was X"
    into each position's residual stream
  - an induction head (layer 1): from the current token A, searches for
    the position whose "before me" note says A — that position holds the
    token that followed A last time — and copies it to the output.

Reproduces the core phenomenon of Olsson et al. 2022 ("In-context
Learning and Induction Heads"): the circuit forms ABRUPTLY during
training (a phase change), and in-context learning turns on at exactly
that moment.

Design notes (each is load-bearing):
  - Attention-only (no MLPs): the setting where the circuit is provable.
  - Repeats at RANDOM offsets: with a fixed offset (e.g. second half =
    first half) a 1-layer model could cheat with a fixed positional-shift
    head and never learn content-based lookup. Random offsets kill that.
  - Data generated fresh every step: memorization is impossible by
    construction; anything the model learns must be an algorithm.
  - N_LAYERS is configurable: run with N_LAYERS=1 for the control that
    reproduces "one layer cannot do induction".

Run:
  python3 induction_from_scratch.py            # 2-layer main run
  python3 induction_from_scratch.py --layers 1 # 1-layer control
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
VOCAB      = 64          # token ids 0..63, uniform random
SEQ_LEN    = 128
SEG_LEN    = 24          # length of the repeated segment
D_MODEL    = 128
N_HEADS    = 4
D_HEAD     = D_MODEL // N_HEADS   # 32
N_LAYERS   = 2           # override with --layers 1 for the control

BATCH      = 256
EVAL_BATCH = 512
LR         = 1e-3
WD         = 0.01
BETAS      = (0.9, 0.98)
N_STEPS    = 6_000
EVAL_EVERY = 25          # induction scores are cheap; log densely
LOG_EVERY  = 250
SEED       = 0

CHECKPOINT_STEPS = [
    0, 100, 250, 500, 750, 1_000, 1_250, 1_500, 1_750, 2_000,
    2_500, 3_000, 3_500, 4_000, 5_000, 6_000,
]

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


def run_tag(n_layers, seed=SEED):
    # seed 0 / default geometry is the canonical run; variants get distinct
    # artifact names (e.g. L2v4096q256 for the vocab-4096 capacity control)
    return (f"L{n_layers}"
            + (f"v{VOCAB}" if VOCAB != 64 else "")
            + (f"q{SEQ_LEN}" if SEQ_LEN != 128 else "")
            + (f"s{seed}" if seed != SEED else ""))


# -----------------------------------------------------------------------------
# Data
#
# Each sequence: SEQ_LEN uniform-random tokens; a SEG_LEN-long segment is
# copied from a random position p1 to a random later position p2.
#
# Predictability structure (targets are next-tokens):
#   - positions p2 .. p2+SEG_LEN-2 : predictable BY INDUCTION ONLY
#     (current token = seg[j], its earlier copy sits at p1+j, and the
#      token after that earlier copy is exactly the next token here)
#   - every other position          : irreducible noise, loss = ln(VOCAB)
#
# So all learnable signal funnels through the induction algorithm.
# -----------------------------------------------------------------------------
def make_batch(batch, generator=None):
    x = torch.randint(0, VOCAB, (batch, SEQ_LEN), generator=generator)
    # p1 in [0, 40], p2 in [p1+SEG_LEN, SEQ_LEN-SEG_LEN] — always disjoint
    p1 = torch.randint(0, 41, (batch,), generator=generator)
    lo = p1 + SEG_LEN
    hi = SEQ_LEN - SEG_LEN
    p2 = lo + (torch.rand(batch, generator=generator) * (hi - lo)).long()

    ar = torch.arange(SEG_LEN)
    src = p1[:, None] + ar[None, :]                    # (B, SEG_LEN)
    dst = p2[:, None] + ar[None, :]
    seg = torch.gather(x, 1, src)
    x.scatter_(1, dst, seg)
    return x, p1, p2


def induction_mask(p1, p2):
    """Boolean mask over POSITIONS (B, SEQ_LEN): True where the *next* token
    is predictable via induction — positions p2 .. p2+SEG_LEN-2."""
    B = p1.shape[0]
    pos = torch.arange(SEQ_LEN)[None, :].expand(B, -1)
    return (pos >= p2[:, None]) & (pos < (p2 + SEG_LEN - 1)[:, None])


# -----------------------------------------------------------------------------
# Parameters — raw tensors, std = 1/sqrt(fan_in)
# -----------------------------------------------------------------------------
def param(*shape, std, g):
    t = torch.randn(*shape, generator=g) * std
    t = t.to(DEVICE)
    t.requires_grad_(True)
    return t


def init_params(n_layers, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    d = 1.0 / math.sqrt(D_MODEL)
    p = {
        "W_E":   param(VOCAB, D_MODEL, std=d, g=g),
        "W_pos": param(SEQ_LEN, D_MODEL, std=d, g=g),
        "W_U":   param(D_MODEL, VOCAB, std=d, g=g),
    }
    for l in range(n_layers):
        p[f"W_Q{l}"] = param(N_HEADS, D_MODEL, D_HEAD, std=d, g=g)
        p[f"W_K{l}"] = param(N_HEADS, D_MODEL, D_HEAD, std=d, g=g)
        p[f"W_V{l}"] = param(N_HEADS, D_MODEL, D_HEAD, std=d, g=g)
        p[f"W_O{l}"] = param(N_HEADS, D_HEAD, D_MODEL, std=1.0 / math.sqrt(D_HEAD), g=g)
    return p


def n_layers_of(p):
    return sum(1 for k in p if k.startswith("W_Q"))


# -----------------------------------------------------------------------------
# Forward pass — attention-only residual stream
# -----------------------------------------------------------------------------
def attn_block(p, l, resid, causal):
    q = torch.einsum("btd,hdk->bthk", resid, p[f"W_Q{l}"])
    k = torch.einsum("btd,hdk->bthk", resid, p[f"W_K{l}"])
    v = torch.einsum("btd,hdk->bthk", resid, p[f"W_V{l}"])
    scores = torch.einsum("bthk,bshk->bhts", q, k) / math.sqrt(D_HEAD)
    scores = scores.masked_fill(causal, float("-inf"))
    attn = scores.softmax(dim=-1)                     # (B, H, T, T)
    z = torch.einsum("bhts,bshk->bthk", attn, v)
    out = torch.einsum("bthk,hkd->btd", z, p[f"W_O{l}"])
    return out, attn


def forward(p, x, want_attn=False):
    B, T = x.shape
    causal = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
    )
    resid = p["W_E"][x] + p["W_pos"][:T]
    attns = []
    for l in range(n_layers_of(p)):
        out, attn = attn_block(p, l, resid, causal)
        resid = resid + out
        if want_attn:
            attns.append(attn)
    logits = resid @ p["W_U"]
    return (logits, attns) if want_attn else logits


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def split_loss(logits, x, ind_mask):
    """Cross-entropy split into induction-predictable positions vs the rest.
    Positions 0..T-2 predict targets 1..T-1."""
    lp = logits[:, :-1].log_softmax(-1)
    tgt = x[:, 1:]
    nll = -lp.gather(-1, tgt[..., None])[..., 0]      # (B, T-1)
    m = ind_mask[:, :-1]                              # position mask, aligned
    ind_loss = nll[m].mean()
    ctl_loss = nll[~m].mean()
    ind_acc = (lp.argmax(-1) == tgt)[m].float().mean()
    return ind_loss, ctl_loss, ind_acc


def head_scores(attns, p1, p2):
    """Per-head diagnostic scores from attention patterns, using the known
    ground-truth structure of the batch.

    prev_score[l,h]: mean attention from position i to i-1 (previous-token
        behavior), averaged over all positions.
    ind_score[l,h]:  mean attention from the induction-predictable query
        positions (p2+j) to their ground-truth induction target (p1+j+1) —
        the position right AFTER the earlier occurrence of the current token.
    """
    B = p1.shape[0]
    ar = torch.arange(SEG_LEN - 1, device=p1.device)
    q_pos = p2[:, None] + ar[None, :]                 # (B, S-1) query: p2+j
    t_pos = p1[:, None] + ar[None, :] + 1             # (B, S-1) target: p1+j+1

    prev, ind = [], []
    for attn in attns:                                # (B, H, T, T) per layer
        H = attn.shape[1]
        d = torch.diagonal(attn, offset=-1, dim1=2, dim2=3)   # (B, H, T-1)
        prev.append(d.mean(dim=(0, 2)).tolist())

        bi = torch.arange(B, device=attn.device)[:, None].expand_as(q_pos)
        layer_ind = []
        for h in range(H):
            a = attn[bi, h, q_pos, t_pos]             # (B, S-1)
            layer_ind.append(a.mean().item())
        ind.append(layer_ind)
    return prev, ind


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def save_checkpoint(p, step, tag, dirpath="checkpoints"):
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, f"params_{tag}_{step:06d}.pt")
    torch.save({k: v.detach().cpu() for k, v in p.items()}, path)


def train(n_layers=N_LAYERS, n_steps=N_STEPS, seed=SEED):
    torch.manual_seed(seed)
    tag = run_tag(n_layers, seed)
    p = init_params(n_layers, seed)
    n = sum(t.numel() for t in p.values())
    print(f"[{tag}] layers {n_layers} | params {n:,} | device {DEVICE}")

    opt = torch.optim.AdamW(list(p.values()), lr=LR, betas=BETAS, weight_decay=WD)

    # fixed eval batch (own generator → same eval data for every run/step)
    ge = torch.Generator().manual_seed(10_000 + seed)
    ex, ep1, ep2 = make_batch(EVAL_BATCH, generator=ge)
    ex, ep1, ep2 = ex.to(DEVICE), ep1.to(DEVICE), ep2.to(DEVICE)
    emask = induction_mask(ep1.cpu(), ep2.cpu()).to(DEVICE)

    log = []
    t0 = time.time()
    for step in range(n_steps + 1):
        if step % EVAL_EVERY == 0 or step == n_steps:
            with torch.no_grad():
                logits, attns = forward(p, ex, want_attn=True)
                ind_loss, ctl_loss, ind_acc = split_loss(logits, ex, emask)
                prev, ind = head_scores(attns, ep1, ep2)
            log.append({
                "step":      step,
                "ind_loss":  ind_loss.item(),
                "ctl_loss":  ctl_loss.item(),
                "ind_acc":   ind_acc.item(),
                "prev":      prev,     # (n_layers, n_heads)
                "ind":       ind,      # (n_layers, n_heads)
            })
            if step % LOG_EVERY == 0 or step == n_steps:
                best_ind = max(max(r) for r in ind)
                best_prev = max(max(r) for r in prev)
                print(
                    f"[{tag}] step {step:5d} | ind_loss {ind_loss.item():.4f} "
                    f"(chance {math.log(VOCAB):.2f}) | ind_acc {ind_acc.item():.3f} | "
                    f"best head: ind {best_ind:.2f} prev {best_prev:.2f} | "
                    f"{time.time()-t0:6.1f}s"
                )

        if step in CHECKPOINT_STEPS:
            save_checkpoint(p, step, tag)

        if step == n_steps:
            break

        x, _, _ = make_batch(BATCH)
        x = x.to(DEVICE)
        logits = forward(p, x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with open(f"training_log_{tag}.json", "w") as f:
        json.dump(log, f)
    torch.save({k: v.detach().cpu() for k, v in p.items()}, f"params_{tag}.pt")
    print(f"[{tag}] saved training_log_{tag}.json and params_{tag}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--seed", type=int, default=SEED)
    # geometry overrides — e.g. the vocab-clock controls (--vocab 512,
    # --vocab 4096) and the pure-signal control at the text run's exact
    # geometry: --vocab 4096 --seq 256 --batch 128 --eval-batch 128
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--seq", type=int, default=SEQ_LEN)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--eval-batch", type=int, default=EVAL_BATCH)
    args = ap.parse_args()
    VOCAB, SEQ_LEN = args.vocab, args.seq
    BATCH, EVAL_BATCH = args.batch, args.eval_batch
    train(n_layers=args.layers, n_steps=args.steps, seed=args.seed)
