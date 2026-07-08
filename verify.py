"""
Verify every quantitative claim in the README against the actual artifacts.

Recomputes each number from training_log_*.json and params_L2.pt and
prints PASS/FAIL per claim. Run after training:

  python3 verify.py

Exit code 0 iff all checks pass.
"""

import json
import math
import sys

import torch

import induction_from_scratch as g
import analyze


PASS = True


def check(name, value, lo, hi, fmt="{:.3f}"):
    global PASS
    ok = lo <= value <= hi
    PASS = PASS and ok
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {fmt.format(value)}  (expected [{lo}, {hi}])")
    return ok


def main():
    print("=" * 72)
    print("README claim verification")
    print("=" * 72)

    with open("training_log_L2.json") as f:
        log2 = json.load(f)
    with open("training_log_L1.json") as f:
        log1 = json.load(f)
    steps = [r["step"] for r in log2]
    acc = [r["ind_acc"] for r in log2]

    print("\n-- Phase change (2-layer) --")
    check("final induction accuracy", log2[-1]["ind_acc"], 0.74, 0.77)
    t10 = next(s for s, a in zip(steps, acc) if a > 0.10)
    t50 = next(s for s, a in zip(steps, acc) if a > 0.50)
    t70 = next(s for s, a in zip(steps, acc) if a > 0.70)
    check("step at >10% acc", t10, 2400, 2500, "{:.0f}")
    check("step at >50% acc", t50, 2700, 2750, "{:.0f}")
    check("10%→50% window (steps)", t50 - t10, 200, 350, "{:.0f}")
    check("step at >70% acc", t70, 2950, 3000, "{:.0f}")
    check("starts at chance (1/64)", acc[0], 0.005, 0.03)
    t07 = next(s for s, a in zip(steps, acc) if a > 0.07)
    check("slow crawl: crosses 7% only at ~1,850", t07, 1700, 2000, "{:.0f}")
    a2400 = next(a for s, a in zip(steps, acc) if s == 2400)
    check("pre-jump acc at step 2400 (~9%)", a2400, 0.05, 0.10)
    a3200 = next(a for s, a in zip(steps, acc) if s == 3200)
    check("levels off ~72% at step 3200", a3200, 0.70, 0.74)
    check("control-pos loss final (> ln64 = 4.159)",
          log2[-1]["ctl_loss"], 4.159, 4.60)

    print("\n-- 1-layer control --")
    check("final induction accuracy", log1[-1]["ind_acc"], 0.05, 0.10)
    max_ind_1l = max(max(max(r["ind"])) for r in log1)
    check("max induction head score, entire run", max_ind_1l, 0.0, 0.05)

    print("\n-- Head scores at end of training (2-layer) --")
    last = log2[-1]
    check("prev-token score head 0.0", last["prev"][0][0], 0.35, 0.47)
    check("prev-token score head 0.3", last["prev"][0][3], 0.35, 0.47)
    for h in range(4):
        check(f"induction score head 1.{h}", last["ind"][1][h], 0.38, 0.50)

    print("\n-- Co-formation timing --")
    prev_rise = next(s for s, r in zip(steps, log2)
                     if max(r["prev"][0]) > 0.10)
    ind_rise = next(s for s, r in zip(steps, log2)
                    if max(r["ind"][1]) > 0.10)
    check("prev-head rise (>0.10)", prev_rise, 2300, 2800, "{:.0f}")
    check("ind-head rise (>0.10)", ind_rise, 2300, 2800, "{:.0f}")
    check("rise simultaneity |Δ| steps", abs(prev_rise - ind_rise),
          0, 300, "{:.0f}")

    print("\n-- Weights-level circuit (params_L2.pt) --")
    p = analyze.load_params("params_L2.pt")
    n = sum(t.numel() for t in p.values())
    check("parameter count", n, 163_840, 163_840, "{:.0f}")

    D = analyze.same_token_detector(p, "0.3", "1.1")
    off = D[~torch.eye(g.VOCAB, dtype=torch.bool)]
    check("detector diagonal mean", D.diagonal().mean().item(), 19.0, 21.5)
    check("detector off-diagonal mean", off.mean().item(), -1.0, 0.0)

    comp = analyze.token_match_matrix(p)
    check("wiring: writer 0.0 → readers (min)", comp[:, 0].min().item(), 18, 24)
    check("wiring: writer 0.3 → readers (min)", comp[:, 3].min().item(), 18, 24)
    check("wiring: writer 0.1 (max, ≈none)", comp[:, 1].max().item(), 0, 1.5)
    check("wiring: writer 0.2 (max, weak)", comp[:, 2].max().item(), 3, 8)

    print("\n-- Behavioral eval (fresh forward passes) --")
    base = analyze.ablated_accuracy(p, [])
    check("recomputed induction accuracy == log final",
          abs(base - log2[-1]["ind_acc"]), 0, 0.005, "{:.4f}")
    ab_prev = analyze.ablated_accuracy(p, [(0, 3)])
    ab_ind = analyze.ablated_accuracy(p, [(1, 1)])
    pair_only = analyze.ablated_accuracy(
        p, [(0, h) for h in range(4) if h != 3]
           + [(1, h) for h in range(4) if h != 1])
    check("ablate prev 0.3", ab_prev, 0.53, 0.59)
    check("ablate ind 1.1", ab_ind, 0.60, 0.66)
    o0 = analyze.ablated_accuracy(p, [(0, h) for h in range(4) if h != 3])
    o1 = analyze.ablated_accuracy(p, [(1, h) for h in range(4) if h != 1])
    check("ablate other-3 L0 heads", o0, 0.37, 0.42)
    check("ablate other-3 L1 heads", o1, 0.29, 0.34)
    check("only pair kept", pair_only, 0.13, 0.20)
    check("pair-only vs chance ratio", pair_only / (1 / g.VOCAB), 8, 14,
          "{:.1f}")

    print("\n-- Stripe fade (prev head 0.3 attention to i−1 by region) --")
    x, p1, p2 = analyze.eval_batch()
    with torch.no_grad():
        _, attns = g.forward(p, x, want_attn=True)
    d = torch.diagonal(attns[0][:, 3], offset=-1, dim1=1, dim2=2)  # (B, T-1)
    early = d[:, :63].mean().item()    # query positions 1..63
    late = d[:, 64:].mean().item()     # query positions 65..127
    check("early-region prev attention", early, 0.5, 1.0)
    check("late-region prev attention", late, 0.0, 0.25)
    check("early/late ratio", early / max(late, 1e-9), 3, 100, "{:.1f}")

    print("\n-- Matcher ceilings --")
    res = analyze.matching_ceiling()
    check("single-token matcher", res["all prior positions"], 0.63, 0.66)
    check("bigram matcher", res["bigram"], 0.97, 0.99)
    check("model beats single-token matcher",
          log2[-1]["ind_acc"] - res["all prior positions"], 0.05, 0.20)
    check("position prior doesn't raise the ceiling (|Δ|)",
          abs(res["all prior positions"] - res["positions < 64 only"]),
          0, 0.005, "{:.4f}")

    print("\n-- Open-question probes --")
    d2 = analyze.offset_attention(p, -2)
    check("no two-back head in layer 0 (max attn at i−2)", max(d2), 0, 0.05)
    accs = analyze.depth_accuracy(p)
    check("depth accuracy at j=0", accs[0], 0.69, 0.75)
    rest = sum(accs[1:]) / len(accs[1:])
    check("depth accuracy nearly flat (|j=0 − mean rest|)",
          abs(accs[0] - rest), 0, 0.06)

    print("\n-- Seed-1 replication (training_log_L2s1.json) --")
    with open("training_log_L2s1.json") as f:
        logs1 = json.load(f)
    s_steps = [r["step"] for r in logs1]
    s_acc = [r["ind_acc"] for r in logs1]
    s10 = next(s for s, a in zip(s_steps, s_acc) if a > 0.10)
    s50 = next(s for s, a in zip(s_steps, s_acc) if a > 0.50)
    check("s1: step at >10% acc", s10, 1900, 2000, "{:.0f}")
    check("s1: step at >50% acc", s50, 2250, 2350, "{:.0f}")
    check("s1: 10%→50% window (steps)", s50 - s10, 300, 400, "{:.0f}")
    check("s1: final induction accuracy", logs1[-1]["ind_acc"], 0.74, 0.77)
    s_last = logs1[-1]
    check("s1: all four L1 heads induction (min score)",
          min(s_last["ind"][1]), 0.42, 0.50)
    check("s1: L1 lockstep (max − min score)",
          max(s_last["ind"][1]) - min(s_last["ind"][1]), 0, 0.06)
    prev1 = sorted(s_last["prev"][0], reverse=True)
    check("s1: one dominant writer", prev1[0], 0.40, 0.48)
    check("s1: partial writer #2", prev1[1], 0.15, 0.35)
    check("s1: partial writer #3", prev1[2], 0.15, 0.35)
    check("s1: absent writer", prev1[3], 0.0, 0.12)

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED")
    print("=" * 72)
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
