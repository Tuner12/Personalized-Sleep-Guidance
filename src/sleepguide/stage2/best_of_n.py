"""Training-free two-statistic Best-of-N observation selection (paper Section 3.3).

Each sampled observation n is parsed into metric-window axes M_n and verdict clauses V_n and
scored by

    s_n = 0.5 * F1(M_n, M_hat) + 0.5 * mean_{n' != n} F1(V_n, V_n')        (paper Eq. 5)

M_hat is the axis set expected from the K=12 most similar training cases: cosine similarity
over a standardized packet feature vector, similarity-softmax votes at temperature 8.0, and an
axis enters M_hat at a weighted vote share of 0.4. The highest-scoring candidate is kept.
Nothing is learned.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from sleepguide.stage2.trajectory.anchors import packet_of, resolve
from sleepguide.stage2.trajectory.card_conditions import condition_holds, gated_cards
from sleepguide.stage2.trajectory.observation_parse import SCALE, parse_obs
from sleepguide.stage2.trajectory.observation_render import PHRASE, _wrapped_delta
from sleepguide.stage2.trajectory.units import unit_for

K, TEMP, SHARE = 12, 8.0, 0.4
METRICS = list(PHRASE)


def f1s(p, g):
    h = len(p & g)
    return 2 * h / (len(p) + len(g)) if h else 0.0


def packet_vec(row):
    p = packet_of(row["input"])
    v = []
    for m in METRICS:
        cu = unit_for(m, "recent_7_center")
        sc = SCALE.get(cu, 60.0)
        rec = resolve(p, m, "insight", "recent_7_center")
        prev = resolve(p, m, "insight", "previous_7_center")
        v += [(rec or 0) / (sc * 10), 1.0 if rec is not None else 0.0]
        v.append(_wrapped_delta(rec, prev, m) / sc
                 if (rec is not None and prev is not None) else 0.0)
    wk = {e.get("metric_id"): e for e in
          (p.get("profile") or {}).get("weekday_weekend_comparisons") or []
          if e.get("reportable")}
    for m in METRICS:
        e = wk.get(m)
        gap = (e or {}).get("weekend_minus_weekday")
        sc = SCALE.get(unit_for(m, "weekend_minus_weekday"), 60.0)
        v += [float(gap) / sc if gap is not None else 0.0, 1.0 if e else 0.0]
    for k in sorted(gated_cards()):
        v.append(1.0 if condition_holds(p, k) else 0.0)
    return v


def axes_and_verdicts(text):
    """The metric-window axes M_n and the canonical verdict clauses V_n of one observation."""
    recs, _, _ = parse_obs(text)
    axes = {(r.axis, r.scope) for r in recs if r.axis != "clause"}
    verdicts = {r.scope[7:] for r in recs if r.axis == "clause"}
    return axes, verdicts


def axes_of(text):
    return axes_and_verdicts(text)[0]


class KnnAxisPicker:
    """Expected axis set M_hat for a case, voted by its K nearest training cases."""

    def __init__(self, train_file):
        rows = [json.loads(l) for l in Path(train_file).read_text().splitlines()
                if l.strip()]
        self.TR = torch.tensor([packet_vec(r) for r in rows])
        self.mu, self.sd = self.TR.mean(0), self.TR.std(0).clamp_min(1e-6)
        self.TRn = torch.nn.functional.normalize((self.TR - self.mu) / self.sd, dim=1)
        self.contract_axes = []
        for row in rows:
            obs = row["target"].split("<OBSERVATION>")[1].split("</OBSERVATION>")[0]
            self.contract_axes.append(axes_of(obs))

    def expected(self, test_row):
        v = torch.tensor([packet_vec(test_row)])
        vn = torch.nn.functional.normalize((v - self.mu) / self.sd, dim=1)
        sims = (vn @ self.TRn.T)[0]
        vals, idxs = torch.topk(sims, K)
        wts = torch.softmax(vals * TEMP, dim=0).tolist()
        votes = {}
        for wt, j in zip(wts, idxs.tolist()):
            for a in self.contract_axes[j]:
                votes[a] = votes.get(a, 0) + wt
        return {a for a, w in votes.items() if w >= SHARE}


def _verdict_f1(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return f1s(a, b)


def select(candidates: list[str], expected_axes: set) -> int:
    """Index of the candidate with the highest score s_n (paper Eq. 5)."""
    n = len(candidates)
    parsed = [axes_and_verdicts(text) for text in candidates]
    scores = [
        0.5 * (f1s(parsed[i][0], expected_axes) if expected_axes else 0.0)
        + 0.5 * statistics.mean(_verdict_f1(parsed[i][1], parsed[j][1])
                                for j in range(n) if j != i)
        for i in range(n)
    ]
    return max(range(n), key=lambda i: scores[i])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-file", type=Path, required=True)
    ap.add_argument("--test-file", type=Path, required=True)
    ap.add_argument("--samples", nargs="+", required=True,
                    help="output prefixes of sample_observations.py (files <prefix>_k{i}.json)")
    ap.add_argument("--per-prefix", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    candidates: dict[str, list[str]] = {}
    for prefix in args.samples:
        for k in range(args.per_prefix):
            draw = json.loads(Path(f"{prefix}_k{k}.json").read_text(encoding="utf-8"))
            for case_id, text in draw.items():
                candidates.setdefault(case_id, []).append(text)
    rows = {json.loads(line)["case_id"]: json.loads(line)
            for line in args.test_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    picker = KnnAxisPicker(args.train_file)
    picked, index = {}, {}
    for case_id, texts in candidates.items():
        best = select(texts, picker.expected(rows[case_id]))
        picked[case_id], index[case_id] = texts[best], best
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")
    args.out.with_name(args.out.stem + "_index.json").write_text(json.dumps(index))
    print(f"picked {len(picked)} cases -> {args.out}")


if __name__ == "__main__":
    main()
