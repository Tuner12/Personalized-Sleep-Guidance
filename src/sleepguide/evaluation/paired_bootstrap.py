"""Paired bootstrap confidence intervals over the 300 held-out cases."""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

N_BOOT = 10000
SEED = 20260908


def load_system(paths):
    """Per-seed dicts keyed by case id, one per score file."""
    seeds = []
    for path in paths:
        d = json.load(open(path))
        by_case = {row["case_id"]: row for row in d["per_case"]}
        seeds.append(by_case)
    return seeds


def metric_vectors(by_case, cases):
    """Per-case arrays for the three metrics that have per-case components.

    BERTScore is stored only as a corpus-level per-section mean, so it cannot be
    resampled here and is left out.
    """
    ev, parsed = [], []
    vh, vs, vg = [], [], []
    rg = []
    for cid in cases:
        row = by_case.get(cid)
        if row is None or not row.get("parsed"):
            parsed.append(0.0)
            ev.append(0.0)
            vh.append(0.0)
            vs.append(0.0)
            vg.append(0.0)
            rg.append(0.0)
            continue
        parsed.append(1.0)
        ev.append(row["evidence_f1_case"])
        vh.append(row["value_hits"])
        vs.append(row["value_stated"])
        vg.append(row["value_gold"])
        rg.append(float(np.mean(list(row["rouge_l"].values()))))
    return {
        "parsed": np.array(parsed),
        "evidence_f1_macro": np.array(ev, dtype=float),
        "value_hits": np.array(vh, dtype=float),
        "value_stated": np.array(vs, dtype=float),
        "value_gold": np.array(vg, dtype=float),
        "rouge_l": np.array(rg, dtype=float),
    }


def score(vec, idx):
    """Reproduce the paper's three estimators on a resampled set of case indices."""
    p = vec["parsed"][idx]
    n_parsed = max(p.sum(), 1.0)
    ev = (vec["evidence_f1_macro"][idx] * p).sum() / n_parsed
    hits = vec["value_hits"][idx].sum()
    stated = vec["value_stated"][idx].sum()
    gold = vec["value_gold"][idx].sum()
    val = 2 * hits / max(stated + gold, 1.0)
    rg = (vec["rouge_l"][idx] * p).sum() / n_parsed
    return np.array([ev, val, rg])


METRICS = ["Evidence F1 (macro)", "Value F1", "ROUGE-L"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True,
                    help='JSON: {"systems": {name: [score files, one per seed]}, '
                         '"comparisons": [[a, b, label], ...]}')
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    SYSTEMS = OrderedDict(config["systems"].items())
    PAIRS = [tuple(item) for item in config["comparisons"]]

    # The 300 held-out cases are the same for every system; take the order from the first.
    base = load_system(next(iter(SYSTEMS.values())))[0]
    cases = sorted(base.keys())
    print("cases: %d" % len(cases))

    vecs = {}
    for system, names in SYSTEMS.items():
        seeds = load_system(names)
        for by_case in seeds:
            missing = [c for c in cases if c not in by_case]
            if missing:
                print("  WARNING %s missing %d cases" % (system, len(missing)))
        vecs[system] = [metric_vectors(by_case, cases) for by_case in seeds]

    # Point estimates, for a sanity check against the numbers printed in Table 1.
    print("\n--- point estimates (mean over seeds) ---")
    full = np.arange(len(cases))
    point = {}
    for system in SYSTEMS:
        s = np.mean([score(v, full) for v in vecs[system]], axis=0)
        point[system] = s
        print("%-14s ev=%.4f val=%.4f rouge=%.4f" % (system, s[0], s[1], s[2]))

    rng = np.random.default_rng(SEED)
    n = len(cases)
    case_idx = rng.integers(0, n, size=(N_BOOT, n))

    # One seed draw per replicate per system, reused across every pair that system appears
    # in so the comparisons stay mutually consistent.
    seed_draws = {}
    for system, names in SYSTEMS.items():
        k = len(names)
        seed_draws[system] = rng.integers(0, k, size=(N_BOOT, k)) if k > 1 else None

    boot_scores = {}
    for system in SYSTEMS:
        out = np.empty((N_BOOT, 3))
        per_seed = np.empty((N_BOOT, len(vecs[system]), 3))
        for j, v in enumerate(vecs[system]):
            for b in range(N_BOOT):
                per_seed[b, j] = score(v, case_idx[b])
        if seed_draws[system] is None:
            out = per_seed[:, 0, :]
        else:
            for b in range(N_BOOT):
                out[b] = per_seed[b, seed_draws[system][b], :].mean(axis=0)
        boot_scores[system] = out
        print("bootstrapped %s" % system)

    results = []
    print("\n--- paired bootstrap, %d resamples ---" % N_BOOT)
    for a, b, label in PAIRS:
        print("\n%s" % label)
        for m, name in enumerate(METRICS):
            d = boot_scores[a][:, m] - boot_scores[b][:, m]
            obs = point[a][m] - point[b][m]
            lo, hi = np.percentile(d, [2.5, 97.5])
            # Two-sided bootstrap p: twice the smaller tail mass on the wrong side of 0.
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            p = min(p, 1.0)
            print("  %-20s delta=%+.4f  95%% CI [%+.4f, %+.4f]  p=%s"
                  % (name, obs, lo, hi, ("<0.0001" if p < 1e-4 else "%.4f" % p)))
            results.append({
                "comparison": label, "system_a": a, "system_b": b, "metric": name,
                "delta": round(float(obs), 4), "ci_low": round(float(lo), 4),
                "ci_high": round(float(hi), 4),
                "p": float(p), "n_boot": N_BOOT, "n_cases": n,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_cases": n, "n_boot": N_BOOT, "seed": SEED,
                   "systems": {k: v for k, v in SYSTEMS.items()},
                   "results": results}, f, indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
