"""
Analyze the text runs — do induction heads emerge from natural text at
laptop scale? (Short answer, measured: no — and the reasons are visible.)

Run after induction_on_text.py (word-level 2-layer + 1-layer control, and
the char-level null runs):

  python3 analyze_text.py

  07  the honest experiment: ICL score for word-2L / word-1L / char-2L,
      next to per-head probe induction scores (which stay at the floor)
  08  the half-circuit that does form on text: prev-token stripe present,
      induction band absent

plus a numeric report: milestones, the partial circuit (prev-token head,
copier heads, no K-composition), the in-vivo induction attention on real
text, and the oracle analysis that explains the non-emergence: on this
data, match-and-copy is far less accurate than the model's ordinary
statistical prediction — induction doesn't pay, so it isn't built.
"""

import json
import os

import torch
import matplotlib.pyplot as plt

import induction_from_scratch as g
import induction_on_text as tx
import analyze as an


def load_log(tag):
    path = f"training_log_{tag}.json"
    return json.load(open(path)) if os.path.exists(path) else None


def tok_of(tag):
    return "word" if "w" in tag else "char"


def eval_sets(tok, seed=0):
    """The exact fixed eval batches the training loop used (same seeds)."""
    _, held_data, vocab = tx.load_corpus(tok)
    ge = torch.Generator().manual_seed(20_000 + seed)
    px = tx.probe_batch(vocab, generator=ge).to(g.DEVICE)
    gh = torch.Generator().manual_seed(30_000 + seed)
    hx = tx.text_batch(held_data, 256, generator=gh).to(g.DEVICE)
    return px, hx


# -----------------------------------------------------------------------------
# 07 — Emergence: ICL score and probe induction scores over training
# -----------------------------------------------------------------------------
def plot_emergence(save_to="07_text_emergence.png"):
    log2w = load_log("T2w")
    if log2w is None:
        print("  (no training_log_T2w.json — run induction_on_text.py first)")
        return
    log1w, log2c = load_log("T1w"), load_log("T2")
    steps = [r["step"] for r in log2w]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    axes[0].plot(steps, [r["icl"] for r in log2w], color="#16a34a",
                 label="2-layer, word tokens")
    if log1w:
        axes[0].plot([r["step"] for r in log1w], [r["icl"] for r in log1w],
                     color="#3b82f6", label="1-layer control, word tokens")
    if log2c:
        axes[0].plot([r["step"] for r in log2c], [r["icl"] for r in log2c],
                     color="#94a3b8", lw=1.2,
                     label="2-layer, char tokens (null)")
    axes[0].axhline(0, color="gray", lw=1, ls="--")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("ICL score (early − late NLL, held-out text)")
    axes[0].set_title("In-context learning while training on natural text")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    n_layers = len(log2w[0]["ind"])
    H = len(log2w[0]["ind"][0])
    for h in range(H):
        axes[1].plot(steps, [r["ind"][0][h] for r in log2w], color="#94a3b8",
                     lw=1, label="layer-0 heads" if h == 0 else None)
    for h in range(H):
        axes[1].plot(steps, [r["ind"][n_layers - 1][h] for r in log2w],
                     lw=1.8, label=f"head 1.{h}")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("probe induction score (attn to prev-occurrence+1)")
    axes[1].set_ylim(-0.02, 0.5)
    axes[1].set_title("No induction heads emerge at this scale (word run)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close()
    print(f"  saved {save_to}")


def milestones(tag):
    log = load_log(tag)
    if log is None:
        print(f"  [{tag}] (no log)")
        return None
    steps = [r["step"] for r in log]
    best_ind = [max(r["ind"][-1]) for r in log]
    icl = [r["icl"] for r in log]
    f_ind, f_icl = best_ind[-1], icl[-1]

    def first(series, thresh):
        return next((s for s, v in zip(steps, series) if v > thresh), None)

    t_ind10 = first(best_ind, 0.10)
    t_ind_half = first(best_ind, f_ind / 2)
    t_icl_half = first(icl, f_icl / 2)
    print(f"  [{tag}] final: icl {f_icl:+.3f} | best probe ind {f_ind:.3f} | "
          f"probe acc {log[-1]['probe_acc']:.3f} | "
          f"held {log[-1]['held_loss']:.3f} | train {log[-1]['train_loss']:.3f}")
    print(f"  [{tag}] ind>0.10 @ {t_ind10} | ind>half-final @ {t_ind_half} | "
          f"icl>half-final @ {t_icl_half}")
    return dict(t_ind10=t_ind10, t_ind_half=t_ind_half, t_icl_half=t_icl_half,
                f_ind=f_ind, f_icl=f_icl)


# -----------------------------------------------------------------------------
# 08 — Probe attention patterns of the emergent heads
# -----------------------------------------------------------------------------
def plot_probe_attention(p, prev_head, ind_head, tok="word",
                         save_to="08_text_attention.png"):
    px, _ = eval_sets(tok)
    x = px[:1]
    with torch.no_grad():
        _, attns = g.forward(p, x, want_attn=True)

    pl, ph = (int(v) for v in prev_head.split("."))
    il, ih = (int(v) for v in ind_head.split("."))
    half = tx.PROBE_HALF

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, A, name in (
        (axes[0], attns[pl][0, ph].cpu(),
         f"best prev-token head ({prev_head})"),
        (axes[1], attns[il][0, ih].cpu(),
         f"best induction candidate ({ind_head}) — no band"),
    ):
        ax.imshow(A.numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.axvline(half - 0.5, color="#dc2626", lw=0.8, ls="--")
        ax.axhline(half - 0.5, color="#dc2626", lw=0.8, ls="--")
        ax.set_title(name)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
    fig.suptitle(
        "Probe: 128 random tokens repeated twice (red line = repeat point).\n"
        "After word-level text training: prev-token attention exists (left, "
        "in the last layer — bigram statistics),\nbut no induction band "
        "(right) — the composition was never wired.", y=1.06, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_to, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {save_to}")


# -----------------------------------------------------------------------------
# Why doesn't it emerge? The oracle analysis.
#
# The payoff a single-token induction circuit could earn: at every held-out
# position whose current word already appeared in the window, predict the
# token that followed the most recent occurrence. Compare with what the
# model's ordinary statistics already achieve on those same positions.
# -----------------------------------------------------------------------------
def oracle_analysis(p, tok="word", n=128):
    _, held, _ = tx.load_corpus(tok)
    gh = torch.Generator().manual_seed(30_000)
    hx = tx.text_batch(held, n, generator=gh)
    with torch.no_grad():
        logits = g.forward(p, hx.to(g.DEVICE))
    pred = logits[:, :-1].argmax(-1).cpu()

    match_pos = model_ok = oracle_ok = oracle_only = 0
    for b in range(n):
        seq = hx[b].tolist()
        last = {}
        for t in range(tx.SEQ_LEN - 1):
            w = seq[t]
            if w in last:
                match_pos += 1
                o = seq[last[w] + 1] == seq[t + 1]
                m = pred[b, t].item() == seq[t + 1]
                oracle_ok += o
                model_ok += m
                oracle_only += o and not m
            last[w] = t
    frac = match_pos / (n * (tx.SEQ_LEN - 1))
    oracle = oracle_ok / match_pos
    model = model_ok / match_pos
    margin = oracle_only / (n * (tx.SEQ_LEN - 1))
    print(f"  positions with a prior occurrence of the current word: {frac:.0%}")
    print(f"  match-and-copy oracle accuracy there: {oracle:.3f}")
    print(f"  the model's accuracy on those same positions: {model:.3f}")
    print(f"  unexploited margin (oracle right, model wrong): "
          f"{margin:.1%} of all positions")
    print(f"  -> as a replacement, induction offers {oracle/model:.2f}x the "
          f"statistics the model already has; as an addition, a "
          f"{margin:.1%} margin — thin either way")
    return oracle, model, frac, margin


def in_vivo_induction(p, tok="word", n=64):
    """Mean attention from a repeated word to (previous occurrence + 1) on
    real held-out text — the in-the-wild version of the probe score."""
    _, held, _ = tx.load_corpus(tok)
    gh = torch.Generator().manual_seed(30_000)
    hx = tx.text_batch(held, n, generator=gh).to(g.DEVICE)
    with torch.no_grad():
        _, attns = g.forward(p, hx, want_attn=True)
    n_layers = len(attns)
    H = attns[0].shape[1]
    tot = torch.zeros(n_layers, H)
    count = 0
    for b in range(n):
        seq = hx[b].tolist()
        last = {}
        for t in range(1, tx.SEQ_LEN):
            w = seq[t]
            if w in last and t - last[w] >= 2:
                j = last[w] + 1
                for l in range(n_layers):
                    tot[l] += attns[l][b, :, t, j].cpu()
                count += 1
            last[w] = t
    scores = (tot / count).tolist()
    for l, row in enumerate(scores):
        print(f"  in-vivo induction attention, layer {l}: "
              + "  ".join(f"{l}.{h}: {v:.3f}" for h, v in enumerate(row)))
    return scores


# -----------------------------------------------------------------------------
# The causal test on text: ablate emergent induction heads → ICL collapses
# -----------------------------------------------------------------------------
def text_ablation(p, heads, tok="word"):
    q = {k: v.clone() for k, v in p.items()}
    for l, h in heads:
        q[f"W_O{l}"] = q[f"W_O{l}"].clone()
        q[f"W_O{l}"][h] = 0
    px, hx = eval_sets(tok)
    with torch.no_grad():
        _, _, icl = tx.icl_metrics(q, hx)
    _, _, probe_acc = tx.probe_scores(q, px)
    return icl, probe_acc


def report_circuit(p, tag="T2w"):
    log = load_log(tag)
    last = log[-1]
    n_layers = len(last["ind"])
    H = len(last["ind"][0])
    for l in range(n_layers):
        print(f"  prev-token scores layer {l}: "
              + "  ".join(f"{l}.{h}: {v:.2f}"
                          for h, v in enumerate(last["prev"][l])))
    # best prev-token head anywhere (on text it sits in the LAST layer —
    # bigram statistics — not in layer 0 where a K-composition writer
    # would live); best induction head in the last layer
    pl, ph = max(((l, h) for l in range(n_layers) for h in range(H)),
                 key=lambda lh: last["prev"][lh[0]][lh[1]])
    ih = max(range(H), key=lambda h: last["ind"][n_layers - 1][h])
    prev_head, ind_head = f"{pl}.{ph}", f"{n_layers - 1}.{ih}"
    print(f"  best prev-token head: {prev_head} "
          f"(score {last['prev'][pl][ph]:.2f}) | best induction head: "
          f"{ind_head} (score {last['ind'][n_layers - 1][ih]:.2f})")

    cs = an.copy_scores(p)
    for l, row in enumerate(cs):
        print(f"  copy scores layer {l}: "
              + "  ".join(f"{l}.{h}: {v:+.2f}" for h, v in enumerate(row)))

    # the detector reads layer-0 write weights, so probe the best L0 head
    ph0 = max(range(H), key=lambda h: last["prev"][0][h])
    D = an.same_token_detector(p, f"0.{ph0}", ind_head)
    off = D[~torch.eye(D.shape[0], dtype=torch.bool)]
    print(f"  same-token detector (0.{ph0} → {ind_head}): "
          f"diag {D.diagonal().mean():.2f} vs off {off.mean():.3f}")

    tok = tok_of(tag)
    base_icl, base_acc = text_ablation(p, [], tok)
    ind_heads = [(n_layers - 1, h) for h in range(H)
                 if last["ind"][n_layers - 1][h] > 0.05]
    abl_icl, abl_acc = text_ablation(p, ind_heads, tok)
    print(f"  ablate emergent induction heads "
          f"{[f'{l}.{h}' for l, h in ind_heads]}: "
          f"icl {base_icl:+.3f} -> {abl_icl:+.3f} | "
          f"probe acc {base_acc:.3f} -> {abl_acc:.3f}")
    return prev_head, ind_head


def main():
    print("The honest experiment — text runs:")
    plot_emergence()
    for tag in ("T2w", "T1w", "T2", "T1"):
        milestones(tag)

    if not os.path.exists("params_T2w.pt"):
        print("(no params_T2w.pt — stopping)")
        return
    p = an.load_params("params_T2w.pt")

    print("\nThe partial circuit (word run):")
    prev_head, ind_head = report_circuit(p, "T2w")

    print("\nIn-vivo induction attention (held-out text):")
    in_vivo_induction(p)

    print("\nWhy induction doesn't pay here — the oracle analysis:")
    oracle_analysis(p)

    print("\nProbe attention patterns (the half-circuit):")
    plot_probe_attention(p, prev_head, ind_head)


if __name__ == "__main__":
    main()
