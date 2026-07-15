"""
Verify every quantitative claim in the README against the actual artifacts.

Recomputes each number from training_log_*.json and params_L2.pt and
prints PASS/FAIL per claim. Run after training:

  python3 verify.py

Exit code 0 iff all checks pass.
"""

import json
import math
import os
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

    print("\n-- Role purity across the whole run --")
    check("max L0 induction score ever (roles never reverse)",
          max(max(r["ind"][0]) for r in log2), 0, 0.05)
    check("max L1 prev-token score ever",
          max(max(r["prev"][1]) for r in log2), 0, 0.06)
    by1 = {r["step"]: r["ind_loss"] for r in log1}
    sep = next(r["step"] for r in log2
               if r["step"] in by1 and by1[r["step"]] - r["ind_loss"] > 0.1)
    check("2L loss first separates from 1L control at the jump itself",
          sep, 2300, 2600, "{:.0f}")

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

    print("\n-- Copying: the OV half of the induction-head definition --")
    cs = analyze.copy_scores(p)
    check("all four L1 heads copy (min diag advantage)", min(cs[1]), 0.55, 0.85)
    check("best ind head 1.1 copy score", cs[1][1], 0.63, 0.78)
    check("no L0 head copies (max |advantage|)",
          max(abs(v) for v in cs[0]), 0, 0.15)
    prof = analyze.induction_offset_profile(p)
    check("attn at the exact target p1+j+1 (min over L1 heads)",
          min(prof[1]), 0.42, 0.50)
    check("attn one position early, p1+j (max)", max(prof[0]), 0, 0.02)
    check("attn one position late, p1+j+2 (max)", max(prof[2]), 0, 0.02)

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

    print("\n-- Text runs: the honest emergence experiment --")

    def tload(tag):
        path = f"training_log_{tag}.json"
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return json.load(fh)

    t2c, t1c = tload("T2"), tload("T1")
    t2w, t1w, t2wt = tload("T2w"), tload("T1w"), tload("T2w_tiny")
    if not all((t2c, t2w, t1w, t2wt)):
        print("  (skipped: text logs missing — run induction_on_text.py)")
    else:
        check("char 2L learns the text (held-out final)",
              t2c[-1]["held_loss"], 1.6, 2.0)
        check("char 2L: no induction head ever (max probe score)",
              max(max(r["ind"][-1]) for r in t2c), 0, 0.02)
        check("char 2L: no in-context learning (|final ICL|)",
              abs(t2c[-1]["icl"]), 0, 0.07)
        check("word-on-TinyShakespeare memorizes: held-out ends above chance",
              t2wt[-1]["held_loss"], 7.63, 9.0)
        check("word-on-TinyShakespeare: train-held gap",
              t2wt[-1]["held_loss"] - t2wt[-1]["train_loss"], 5.0, 8.0)
        check("word-on-TinyShakespeare: ICL goes negative",
              t2wt[-1]["icl"], -0.8, -0.3)
        check("word 2L (novels): healthy held-out loss",
              t2w[-1]["held_loss"], 3.7, 4.0)
        check("word 2L: no memorization (train-held gap)",
              t2w[-1]["held_loss"] - t2w[-1]["train_loss"], 0.0, 0.6)
        check("word 2L: small real ICL", t2w[-1]["icl"], 0.02, 0.10)
        check("word 2L: STILL no induction head (max probe score ever)",
              max(max(r["ind"][-1]) for r in t2w), 0, 0.02)
        check("word 2L: prev-token attention sits in the LAST layer",
              min(t2w[-1]["prev"][1]), 0.18, 0.35)
        check("word 2L: the layer-0 writer role never forms",
              max(t2w[-1]["prev"][0]), 0, 0.12)
        check("word 1L control: same ICL without composition (|Δ| vs 2L)",
              abs(t1w[-1]["icl"] - t2w[-1]["icl"]), 0, 0.05)

        # the vocab-clock controls: pure sandbox signal, only vocab changed
        v512, v4096 = tload("L2v512"), tload("L2v4096")
        if v512 and v4096:
            def t50_of(lg):
                return next((r["step"] for r in lg if r["ind_acc"] > 0.5),
                            None)
            check("vocab clock: 512 forms (final acc)",
                  v512[-1]["ind_acc"], 0.93, 1.0)
            check("vocab clock: 512 phase change ~5.3k",
                  t50_of(v512), 5_100, 5_500, "{:.0f}")
            check("vocab clock: 4096 forms too — a clock, not a wall",
                  v4096[-1]["ind_acc"], 0.90, 1.0)
            check("vocab clock: 4096 phase change ~7.9k",
                  t50_of(v4096), 7_700, 8_000, "{:.0f}")
            check("vocab clock is monotone: 64 < 512 < 4096",
                  1.0 if 2_725 < t50_of(v512) < t50_of(v4096) else 0.0,
                  1, 1, "{:.0f}")
        vq = tload("L2v4096q256")
        if vq:
            check("pure signal at the text run's exact geometry/budget: "
                  "still pre-transition", vq[-1]["ind_acc"], 0, 0.02)
            check("geometry control ends barely below chance (ln 4096)",
                  vq[-1]["ind_loss"], 8.20, 8.318)
        if os.path.exists("params_T2.pt") and \
           os.path.exists("data/tinyshakespeare.txt"):
            import analyze_text as at_c
            p2c = analyze.load_params("params_T2.pt")
            oc, mc, _, _ = at_c.oracle_analysis(p2c, tok="char")
            check("char oracle: match-and-copy accuracy", oc, 0.13, 0.18)
            check("char oracle: statistics beat copying (ratio)",
                  mc / oc, 2.5, 3.5)
        if os.path.exists("params_T2w.pt") and \
           os.path.exists("data/gutenberg.txt"):
            import analyze_text as at
            pw = analyze.load_params("params_T2w.pt")
            comp_w = analyze.token_match_matrix(pw)
            check("word 2L: no K-composition wiring (max, sandbox has ~21)",
                  comp_w.max().item(), -2, 2)
            cs_w = analyze.copy_scores(pw)
            check("word 2L: copier heads exist (max L1 copy score)",
                  max(cs_w[1]), 0.5, 2.0)
            oracle, model_acc, _, margin = at.oracle_analysis(pw)
            check("oracle: match-and-copy accuracy on repeated words",
                  oracle, 0.08, 0.14)
            check("oracle: model's statistics beat induction (ratio)",
                  model_acc / oracle, 2.0, 4.0)
            check("oracle: unexploited margin is thin (frac of all tokens)",
                  margin, 0.015, 0.04)
            vivo = at.in_vivo_induction(pw)
            check("word 2L: no in-vivo induction on real text (max attn)",
                  max(max(r) for r in vivo), 0, 0.05)

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
