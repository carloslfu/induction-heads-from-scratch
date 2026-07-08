"""
Inspect the trained models — see the induction circuit.

Run after induction_from_scratch.py (both the 2-layer run and, ideally,
the 1-layer control):

  python3 analyze.py

Produces .png plots and prints numeric summaries:
  00  phase change: induction-position loss/accuracy over training (+1L control)
  01  per-head induction & previous-token scores over training
  02  attention patterns of the two circuit heads on one example
  03  circuit wiring from weights alone: which layer-0 head's write is
      read by which layer-1 head's keys (same-token matching strength)
  04  same-token detector: E · W_QK^ind · (W_OV^prev)^T · E^T ≈ diagonal
  05  ablation: zero each head, measure induction accuracy — causal test

Everything runs in seconds on CPU/MPS.
"""

import json
import math
import os

import torch
import matplotlib.pyplot as plt

import induction_from_scratch as g


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def load_params(path, device=None):
    device = device or g.DEVICE
    return {
        k: v.to(device)
        for k, v in torch.load(path, weights_only=True).items()
    }


def load_log(tag):
    path = f"training_log_{tag}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def eval_batch(n=512, seed=10_000):
    ge = torch.Generator().manual_seed(seed)
    x, p1, p2 = g.make_batch(n, generator=ge)
    return x.to(g.DEVICE), p1.to(g.DEVICE), p2.to(g.DEVICE)


# -----------------------------------------------------------------------------
# 00 — The phase change (the headline plot)
# -----------------------------------------------------------------------------
def plot_phase_change(save_to="00_phase_change.png"):
    log2 = load_log("L2")
    log1 = load_log("L1")
    if log2 is None:
        print("  (no training_log_L2.json — train first)")
        return

    steps = [r["step"] for r in log2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].axhline(math.log(g.VOCAB), color="gray", ls="--", lw=1,
                    label=f"chance (ln {g.VOCAB} = {math.log(g.VOCAB):.2f})")
    axes[0].plot(steps, [r["ind_loss"] for r in log2], color="#dc2626",
                 label="2-layer: induction positions")
    axes[0].plot(steps, [r["ctl_loss"] for r in log2], color="#f59e0b",
                 lw=1, label="2-layer: control positions")
    if log1:
        axes[0].plot([r["step"] for r in log1],
                     [r["ind_loss"] for r in log1],
                     color="#3b82f6", label="1-layer control: induction pos.")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss on induction-predictable positions")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, [r["ind_acc"] for r in log2], color="#16a34a",
                 label="2-layer")
    if log1:
        axes[1].plot([r["step"] for r in log1],
                     [r["ind_acc"] for r in log1],
                     color="#3b82f6", label="1-layer control")
    axes[1].axhline(1 / g.VOCAB, color="gray", ls="--", lw=1, label="chance")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Next-token accuracy on induction positions")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close()
    print(f"  saved {save_to}")

    acc = [r["ind_acc"] for r in log2]
    final = acc[-1]
    t10 = next((s for s, a in zip(steps, acc) if a > 0.10), None)
    t50 = next((s for s, a in zip(steps, acc) if a > 0.50), None)
    t90 = next((s for s, a in zip(steps, acc) if a > 0.90), None)
    print(f"  2L induction acc: final {final:.3f} | >10% @ {t10} | "
          f">50% @ {t50} | >90% @ {t90}")
    if log1:
        print(f"  1L induction acc: final {log1[-1]['ind_acc']:.3f}")
    return t10, t50, t90


# -----------------------------------------------------------------------------
# 01 — Per-head score trajectories: watch the circuit assemble
# -----------------------------------------------------------------------------
def plot_head_trajectories(save_to="01_head_scores.png"):
    log2 = load_log("L2")
    if log2 is None:
        return
    steps = [r["step"] for r in log2]
    n_layers = len(log2[0]["prev"])
    H = len(log2[0]["prev"][0])

    fig, axes = plt.subplots(2, n_layers, figsize=(6 * n_layers, 7),
                              squeeze=False)
    for l in range(n_layers):
        for h in range(H):
            axes[0][l].plot(steps, [r["prev"][l][h] for r in log2],
                            label=f"head {l}.{h}")
            axes[1][l].plot(steps, [r["ind"][l][h] for r in log2],
                            label=f"head {l}.{h}")
        axes[0][l].set_title(f"layer {l}: previous-token score")
        axes[1][l].set_title(f"layer {l}: induction score")
        for ax in (axes[0][l], axes[1][l]):
            ax.set_xlabel("step")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    axes[0][0].set_ylabel("mean attn to position i−1")
    axes[1][0].set_ylabel("mean attn to induction target")

    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close()
    print(f"  saved {save_to}")

    last = log2[-1]
    # circuit roles are layer-bound: prev-token head must sit in layer 0
    # (it writes), the induction head in the last layer (it reads)
    bp_h = max(range(H), key=lambda h: last["prev"][0][h])
    bi_h = max(range(H), key=lambda h: last["ind"][n_layers - 1][h])
    bp = (f"0.{bp_h}", last["prev"][0][bp_h])
    bi = (f"{n_layers - 1}.{bi_h}", last["ind"][n_layers - 1][bi_h])
    print(f"  final best prev-token head (layer 0): {bp[0]} (score {bp[1]:.2f})")
    print(f"  final best induction head (layer {n_layers - 1}): "
          f"{bi[0]} (score {bi[1]:.2f})")
    return bp[0], bi[0]


# -----------------------------------------------------------------------------
# 02 — Attention patterns of the circuit heads on one example
# -----------------------------------------------------------------------------
def plot_attention_patterns(p, prev_head, ind_head,
                            save_to="02_attention_patterns.png"):
    x, p1, p2 = eval_batch(n=1, seed=123)
    with torch.no_grad():
        _, attns = g.forward(p, x, want_attn=True)

    pl, ph = (int(v) for v in prev_head.split("."))
    il, ih = (int(v) for v in ind_head.split("."))
    A_prev = attns[pl][0, ph].cpu()
    A_ind = attns[il][0, ih].cpu()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, A, name in ((axes[0], A_prev, f"prev-token head {prev_head}"),
                        (axes[1], A_ind, f"induction head {ind_head}")):
        ax.imshow(A.numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        # mark the two copies of the repeated segment
        s1, s2 = p1[0].item(), p2[0].item()
        for start, color in ((s1, "#16a34a"), (s2, "#dc2626")):
            ax.axvspan(start, start + g.SEG_LEN, alpha=0.10, color=color)
            ax.axhspan(start, start + g.SEG_LEN, alpha=0.10, color=color)

    fig.suptitle(
        "One sequence: segment at green positions repeats at red positions.\n"
        "Prev-token head = subdiagonal stripe. Induction head = off-diagonal "
        "band (red queries → just-after-green keys).",
        y=1.04, fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_to, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {save_to}")


# -----------------------------------------------------------------------------
# 03 — Circuit wiring from weights alone
#
# For each (L0 head h1, L1 head h2) pair, build the token-level bilinear form
# the pair implements:
#   D = E · W_QK^{h2} · (W_OV^{h1})^T · E^T        (VOCAB × VOCAB)
# If the pair implements induction matching, D is diagonal-dominant ("attend
# where the written previous-token equals my current token"). The wiring
# score is the diagonal advantage: mean(diag) − mean(off-diag), in units of
# attention logits. This is sharper than the generic Frobenius composition
# score, which measures subspace overlap without direction.
# -----------------------------------------------------------------------------
def token_match_matrix(p):
    H = g.N_HEADS
    E = p["W_E"]
    eye = torch.eye(g.VOCAB, dtype=torch.bool)
    comp = torch.zeros(H, H)
    for h1 in range(H):
        W_OV1 = p["W_V0"][h1] @ p["W_O0"][h1]          # (D, D) write of L0 head
        for h2 in range(H):
            W_QK2 = p["W_Q1"][h2] @ p["W_K1"][h2].T    # (D, D) match of L1 head
            D = (E @ W_QK2 @ W_OV1.T @ E.T).detach().cpu()
            comp[h2, h1] = D.diagonal().mean() - D[~eye].mean()
    return comp


def plot_k_composition(p, save_to="03_circuit_wiring.png"):
    if n_layers_from(p) < 2:
        return
    comp = token_match_matrix(p)
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(comp.numpy(), cmap="viridis")
    plt.colorbar(label="same-token matching strength (attn logits)")
    plt.xlabel("layer-0 head (writer)")
    plt.ylabel("layer-1 head (reader via keys)")
    plt.title("Wiring from weights alone: which head pairs\nimplement induction matching?")
    plt.xticks(range(g.N_HEADS), [f"0.{h}" for h in range(g.N_HEADS)])
    plt.yticks(range(g.N_HEADS), [f"1.{h}" for h in range(g.N_HEADS)])
    for i in range(g.N_HEADS):
        for j in range(g.N_HEADS):
            plt.text(j, i, f"{comp[i, j]:.1f}", ha="center", va="center",
                     color="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close()
    print(f"  saved {save_to}")
    print("  (rows = L1 readers, cols = L0 writers; high value = this pair")
    print("   implements 'attend where written-prev-token == my token')")
    return comp


def n_layers_from(p):
    return sum(1 for k in p if k.startswith("W_Q"))


# -----------------------------------------------------------------------------
# 04 — The same-token detector
#
# The induction head's attention score between current token A (query) and
# a position annotated "previous token was X" (key) runs through:
#   D = E · W_QK^{ind, layer 1} · (W_OV^{prev, layer 0})^T · E^T   (V × V)
# If the circuit is real, D is diagonal-dominant: score high iff X = A.
# -----------------------------------------------------------------------------
def same_token_detector(p, prev_head, ind_head):
    ph = int(prev_head.split(".")[1])
    ih = int(ind_head.split(".")[1])
    E = p["W_E"]                                       # (V, D)
    W_OV1 = p["W_V0"][ph] @ p["W_O0"][ph]              # (D, D) prev head's write
    W_QK2 = p["W_Q1"][ih] @ p["W_K1"][ih].T            # (D, D) ind head's match
    D = E @ W_QK2 @ W_OV1.T @ E.T                      # query-token × key-written-token
    return D.detach().cpu()


def plot_same_token_detector(p, prev_head, ind_head,
                             save_to="04_same_token_detector.png"):
    if n_layers_from(p) < 2:
        return
    D = same_token_detector(p, prev_head, ind_head)
    diag = D.diagonal()
    off = D[~torch.eye(g.VOCAB, dtype=torch.bool)]
    plt.figure(figsize=(6, 5))
    plt.imshow(D.numpy(), cmap="RdBu_r",
               vmin=-D.abs().max(), vmax=D.abs().max())
    plt.colorbar(label="attention score contribution")
    plt.xlabel("token written by prev-token head (X)")
    plt.ylabel("current query token (A)")
    plt.title(f"Same-token detector via {prev_head} → {ind_head}\n"
              f"diag mean {diag.mean():.2f} vs off-diag {off.mean():.2f}")
    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close()
    print(f"  saved {save_to}")
    print(f"  same-token detector: diagonal mean {diag.mean():.3f}, "
          f"off-diagonal mean {off.mean():.3f}")


# -----------------------------------------------------------------------------
# 05 — Ablations: the causal test
# -----------------------------------------------------------------------------
def ablated_accuracy(p, zero_heads):
    """Induction accuracy with the given (layer, head) pairs' W_O zeroed."""
    q = {k: v.clone() for k, v in p.items()}
    for (l, h) in zero_heads:
        q[f"W_O{l}"] = q[f"W_O{l}"].clone()
        q[f"W_O{l}"][h] = 0
    x, p1, p2 = eval_batch()
    mask = g.induction_mask(p1.cpu(), p2.cpu()).to(g.DEVICE)
    with torch.no_grad():
        logits = g.forward(q, x)
        _, _, acc = g.split_loss(logits, x, mask)
    return acc.item()


def plot_ablations(p, prev_head, ind_head, save_to="05_ablation.png"):
    if n_layers_from(p) < 2:
        return
    pl, ph = (int(v) for v in prev_head.split("."))
    il, ih = (int(v) for v in ind_head.split("."))

    base = ablated_accuracy(p, [])
    others0 = [(0, h) for h in range(g.N_HEADS) if h != ph]
    others1 = [(1, h) for h in range(g.N_HEADS) if h != ih]

    results = {
        "no ablation": base,
        f"− prev head {prev_head}": ablated_accuracy(p, [(pl, ph)]),
        f"− ind head {ind_head}": ablated_accuracy(p, [(il, ih)]),
        f"− other L0 heads ({len(others0)})": ablated_accuracy(p, others0),
        f"− other L1 heads ({len(others1)})": ablated_accuracy(p, others1),
        # sufficiency: zero EVERYTHING except the circuit pair
        "only circuit pair kept": ablated_accuracy(p, others0 + others1),
    }

    plt.figure(figsize=(9, 4))
    names = list(results.keys())
    vals = [results[k] for k in names]
    colors = ["#16a34a", "#dc2626", "#dc2626", "#94a3b8", "#94a3b8", "#3b82f6"]
    plt.bar(names, vals, color=colors)
    plt.axhline(1 / g.VOCAB, color="gray", ls="--", lw=1, label="chance")
    plt.ylabel("induction accuracy")
    plt.title("Ablation: zero a head's output, measure induction accuracy")
    plt.xticks(rotation=12, ha="right", fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close()
    print(f"  saved {save_to}")
    for k, v in results.items():
        print(f"    {k:32s} acc = {v:.3f}")
    return results


# -----------------------------------------------------------------------------
# Theoretical ceiling for a single-token matcher
#
# The model's strategy matches on ONE previous token. With random tokens,
# the current token usually also appears elsewhere by coincidence, and those
# occurrences are followed by random other tokens — irreducible ambiguity.
# Best possible deterministic strategy: predict the majority continuation
# among all earlier occurrences of the current token. This computes that
# strategy's accuracy on the eval distribution.
# -----------------------------------------------------------------------------
def matching_ceiling(n=512):
    # The true source segment always starts at p1 <= 40, so its positions all
    # lie below SRC_LIMIT. A matcher can exploit that spatial prior to discard
    # coincidental matches in later positions — and the trained model does
    # (its prev-token head only annotates the early region). Report both.
    SRC_LIMIT = 40 + g.SEG_LEN
    ge = torch.Generator().manual_seed(10_000)
    x, p1, p2 = g.make_batch(n, generator=ge)
    res = {}
    for name, limit in (("all prior positions", None),
                        (f"positions < {SRC_LIMIT} only", SRC_LIMIT)):
        correct = total = cands = 0
        for b in range(n):
            seq = x[b].tolist()
            for j in range(g.SEG_LEN - 1):
                q = p2[b].item() + j
                a, tgt = seq[q], seq[q + 1]
                hi = q if limit is None else min(q, limit)
                cont = [seq[i + 1] for i in range(hi) if seq[i] == a]
                if cont:
                    counts = {}
                    for c in cont:
                        counts[c] = counts.get(c, 0) + 1
                    pred = max(counts, key=counts.get)
                    correct += int(pred == tgt)
                    cands += len(cont)
                total += 1
        res[name] = correct / total
        print(f"  matcher ceiling ({name}): {correct/total:.3f} "
              f"(avg {cands/total:.1f} candidates/query)")

    # bigram matcher: match TWO consecutive tokens (needs j >= 1)
    correct = total = 0
    for b in range(n):
        seq = x[b].tolist()
        for j in range(g.SEG_LEN - 1):
            q = p2[b].item() + j
            a, tgt = seq[q], seq[q + 1]
            if j >= 1:
                cont = [seq[i + 1] for i in range(1, q)
                        if seq[i] == a and seq[i - 1] == seq[q - 1]]
            else:
                cont = [seq[i + 1] for i in range(q) if seq[i] == a]
            if cont:
                counts = {}
                for c in cont:
                    counts[c] = counts.get(c, 0) + 1
                correct += int(max(counts, key=counts.get) == tgt)
            total += 1
    res["bigram"] = correct / total
    print(f"  matcher ceiling (bigram matching): {correct/total:.3f}")
    return res


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def main():
    print("Phase change:")
    plot_phase_change()
    matching_ceiling()

    print("\nHead score trajectories:")
    heads = plot_head_trajectories()

    if not os.path.exists("params_L2.pt"):
        print("(no params_L2.pt — stopping)")
        return
    p = load_params("params_L2.pt")

    prev_head, ind_head = heads if heads else ("0.0", "1.0")
    print(f"\nCircuit heads: prev={prev_head}, ind={ind_head}")

    print("\nAttention patterns:")
    plot_attention_patterns(p, prev_head, ind_head)

    print("\nK-composition (from weights alone):")
    plot_k_composition(p)

    print("\nSame-token detector:")
    plot_same_token_detector(p, prev_head, ind_head)

    print("\nAblations (causal test):")
    plot_ablations(p, prev_head, ind_head)


if __name__ == "__main__":
    main()
