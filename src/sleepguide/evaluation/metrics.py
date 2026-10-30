"""Score generated guidance against the Stage-1 reference (paper Section 4.1, Metrics).

    Evidence F1   evidence invoked in the four sections (regex signatures per card) versus
                  the Stage-1 selection, macro-averaged over cases
    Value F1      numeric values stated in the sections versus Stage-1 certified anchors
    ROUGE-L       per section, averaged
    BERTScore     roberta-large, layer 17, per section, averaged

`per_case` in the output holds every case's scores for paired comparisons.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sleepguide.core.contracts import LongitudinalInsight, SleepProfile  # noqa: E402
from sleepguide.evaluation.dossiers import load_readable_evidence_bank  # noqa: E402
from sleepguide.evaluation.value_index import (  # noqa: E402
    CanonicalValue,
    canonical_value_index,
    numeric_mentions,
    supporting_values,
)
from sleepguide.paths import EVIDENCE_BANK

SECTIONS = ("section_1", "section_2", "section_3", "section_4")

# Which Evidence a generation actually invokes, read off the text. Only signatures whose
# recall and precision were both validated against Stage 1's own certified output are
# used, so the metric measures the model rather than the detector. Validated on 1200
# training cases: caffeine 0.982/1.000, wake-detection 0.928/0.975, sleep-extension
# 0.914/0.895, efficiency-85 0.986/0.709. The two duration-reference cards share one
# surface form ("7 hours") and are merged, since text cannot separate them.
_EVIDENCE_SIGNATURES = {
    "caffeine-timing-meta": r"caffeine|coffee|energy drink|\balcohol\b",
    "wake-detection-unreliable": r"recorded awake time may not|may not precisely represent",
    "behavioral-sleep-extension": (
        r"more (?:time|opportunit|room) for sleep|sleep[- ]extension|"
        r"create more sleep opportunit|earlier sleep start"
    ),
    "sleep-efficiency-quality-reference": r"\b85\s*%",
    # Precision 0.983, recall 0.922 on Stage 1's own text. A card is left without a
    # signature when no pattern separates it from generic advice (best candidate F1 < 0.8).
    "healthy-sleep-natural-variation": (
        r"night[- ]to[- ]night variation|natural variation|"
        r"variation is (?:normal|substantial|expected)"
    ),
    # The reference is usually cited in words ("below the seven-hour lower reference"), so
    # spelled-out numbers count, but only with a reference word in the same sentence, so that
    # a description such as "slept seven hours" is not read as a citation (precision 0.931,
    # recall 0.949).
    "duration-reference": (
        r"(?:\bseven|\b7)[\s-]?hours?\b(?!\s*\d)"
        r"(?=[^.]{0,60}(?:reference|recommend|consensus|lower|guideline|below|above))"
        r"|(?:reference|recommend|consensus|guideline)[^.]{0,60}(?:\bseven|\b7)[\s-]?hours?\b"
    ),
}
_DURATION_CARDS = {
    "adult-sleep-duration-reference-18-64",
    "older-adult-sleep-duration-reference-65-plus",
}
# Stage-1 forces this card into every selection, so it carries no per-case information.
_FORCED = {"consumer-tracker-minimum-nights"}
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for index, other in enumerate(b):
            current.append(
                previous[index] + 1 if token == other else max(current[index], previous[index + 1])
            )
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    a, b = _tokens(prediction), _tokens(reference)
    common = _lcs(a, b)
    if not common:
        return 0.0
    precision, recall = common / len(a), common / len(b)
    return 2 * precision * recall / (precision + recall)


def _parse_sections(text: str) -> dict[str, str] | None:
    """Recover the four Sections from a generation, ignoring any extra fields around them.

    Every system is scored on the same four keys, so the comparison never depends on format.
    """

    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    sections = {key: payload.get(key) for key in SECTIONS}
    if not all(isinstance(value, str) and value.strip() for value in sections.values()):
        return None
    return sections


# Detector table in the uniform shape {name: (compiled pattern, the keys it stands for)}.
# `--signatures` loads the table for the current Evidence Bank (evidence_bank/scoring/).
_SIGNATURES: dict[str, tuple[str, frozenset[str]]] = {
    name: (pattern,
           frozenset(_DURATION_CARDS) if name == "duration-reference" else frozenset({name}))
    for name, pattern in _EVIDENCE_SIGNATURES.items()
}


def load_signatures(path: Path) -> None:
    """Replace the detector table from a fitted file: {name: {pattern, keys}}."""
    global _SIGNATURES
    payload = json.loads(path.read_text(encoding="utf-8"))
    _SIGNATURES = {name: (entry["pattern"], frozenset(entry["keys"]))
                   for name, entry in payload.items()}


def _evidence_invoked(sections: dict[str, str]) -> set[str]:
    text = " ".join(sections[section] for section in SECTIONS)
    return {name for name, (pattern, _) in _SIGNATURES.items()
            if re.search(pattern, text, re.I)}


def _evidence_gold(auxiliary: dict) -> set[str]:
    selected = {item["dossier_id"] for item in auxiliary.get("evidence_selection", [])}
    return {name for name, (_, keys) in _SIGNATURES.items() if selected & keys}


def _committed_selection(text: str) -> set[str] | None:
    """Read back the Evidence selection a generation committed to, if it emitted one."""

    start = text.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    selection = payload.get("evidence_selection")
    if not isinstance(selection, list):
        return None
    # A selection item is either a bare identifier or {"dossier_id": ..., "why": ...}.
    return {
        str(item["dossier_id"]) if isinstance(item, dict) and "dossier_id" in item else str(item)
        for item in selection
    }


def _set_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    overlap = len(predicted & gold)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _committed_anchor_keys(text: str) -> set[str]:
    """Return the committed anchors as ``origin.metric.statistic`` identity keys.

    ``anchor_recall`` says whether the prose states the right numbers and
    ``commitment_kept`` says whether it states the numbers it promised, but neither
    says whether the promise itself was right. Without that the two ways of failing --
    committing to the wrong statistic, and committing correctly then ignoring it -- are
    indistinguishable, and they call for opposite fixes.
    """

    start = text.find("{")
    if start < 0:
        return set()
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return set()
    keys: set[str] = set()
    for section in ("section_2", "section_3", "section_4"):
        for item in payload.get(f"{section}_anchors", []) or []:
            head = str(item).split("=")[0].strip()
            if head:
                keys.add(f"{section}|{head}")
    return keys


def _committed_anchor_values(text: str) -> list[tuple[float, str]]:
    """Read back the ``origin.metric.statistic=value unit`` commitments a generation made."""

    start = text.find("{")
    if start < 0:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return []
    out: list[tuple[float, str]] = []
    for section in ("section_2", "section_3", "section_4"):
        for item in payload.get(f"{section}_anchors", []) or []:
            match = re.search(r"=\s*(-?[\d.]+)\s*(\S+)\s*$", str(item))
            if match:
                out.append((float(match.group(1)), match.group(2)))
    return out


def _value_stated(text: str, value: float, unit: str) -> bool:
    kind = {"clock_minute": "clock", "min": "duration", "%": "percent"}.get(unit, "bare")
    for _, _, mention, mention_kind in numeric_mentions(text):
        if mention_kind != kind:
            continue
        if kind == "clock" and abs((round(mention) % 1440) - (round(value) % 1440)) <= 1:
            return True
        if kind == "duration" and abs(round(mention) - round(abs(value))) <= 1:
            return True
        if kind not in {"clock", "duration"} and abs(mention - abs(value)) <= 0.5:
            return True
    return False


def _stated_values(sections: dict[str, str]) -> set[tuple[float, str]]:
    """The distinct numeric values a four-section text actually states.

    Durations and clock times are rounded to the minute because the reader text writes
    `6 hours 38 minutes` for 398.5; percentages to one decimal. Comparing sets of these
    against the certified answer's own set is symmetric, so neither omitting nor padding
    numbers can improve the score.
    """

    out: set[tuple[float, str]] = set()
    for text in sections.values():
        for _, _, value, kind in numeric_mentions(text):
            out.add((round(value) if kind in {"duration", "clock"} else round(value, 1), kind))
    return out


def _reference_values(bank) -> list[CanonicalValue]:
    return [
        CanonicalValue(
            metric_id=value.metric_id,
            origin="evidence_reference",
            statistic=dossier.dossier_id,
            value=float(value.value),
            unit=value.unit,
        )
        for dossier in bank.dossiers
        for value in dossier.reference_values
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decoded", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True,
                        help="Stage-1 output directory (one case_*/stage1_run.json per case)")
    parser.add_argument(
        "--evidence-bank", type=Path, default=EVIDENCE_BANK / "scoring/reference_dossiers.json"
    )
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument(
        "--signatures", type=Path, default=EVIDENCE_BANK / "scoring/evidence_signatures.json",
        help="per-card regex signatures used to detect which evidence the text invokes")
    args = parser.parse_args(argv)
    if args.signatures:
        global _SIGNATURES
        _SIGNATURES = json.loads(args.signatures.read_text(encoding="utf-8"))
        covered = {k for entry in _SIGNATURES.values() for k in entry["keys"]}
        print(f"evidence signatures: {len(_SIGNATURES)} detectors over {len(covered)} keys",
              file=sys.stderr)
    if args.signatures:
        load_signatures(args.signatures)

    references = _reference_values(load_readable_evidence_bank(args.evidence_bank))
    auxiliary = {
        record["case_id"]: record["auxiliary"]
        for record in (
            json.loads(line)
            for line in args.test_file.read_text(encoding="utf-8").splitlines()
            if line
        )
    }

    rows = [
        json.loads(line)
        for line in args.decoded.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = _score(rows, args, auxiliary, references)
    # The same shard scored with each case's text swapped for its neighbour's. Every
    # number above is only interpretable against this: `numeric_support` in particular
    # sits near 0.6 on text written for a different person.
    shifted = [
        {**row, "generated": rows[(index + 1) % len(rows)]["generated"]}
        for index, row in enumerate(rows)
    ]
    summary["chance_floor"] = {
        key: value
        for key, value in _score(shifted, args, auxiliary, references).items()
        if key in {"numeric_support", "anchor_recall", "value_precision", "value_recall",
                   "value_f1", "rouge_l", "rouge_l_mean"}
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _bertscore(pairs: list[tuple[str, str]], model: str) -> float | None:
    """Rescaled BERTScore F1 for one Section across the shard.

    Rescaled with the package baseline (roberta-large, layer 17): unrescaled scores of any
    two fluent passages about the same record sit above 0.85 and barely separate systems.
    """

    usable = [(c, g) for c, g in pairs if c.strip() and g.strip()]
    if not usable:
        return None
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(
        [c for c, _ in usable],
        [g for _, g in usable],
        model_type=model,
        num_layers=17,
        rescale_with_baseline=True,
        lang="en",
        verbose=False,
        batch_size=16,
    )
    return round(float(f1.mean()), 4)


def _constant_policy(gold_sets: list[set[str]], forced: set[str]) -> tuple[float, float]:
    """Best constant selection, computed from the data.

    The constant is chosen on the same cases it is scored against, which is the strongest
    form of the baseline and therefore the conservative comparison.
    """

    counts: Counter = Counter()
    for gold in gold_sets:
        counts.update(gold)
    ranked = [name for name, _ in counts.most_common(6)]

    def best(pool: list[str], golds: list[set[str]]) -> float:
        top = 0.0
        for size in (1, 2, 3):
            for combo in itertools.combinations(pool, size):
                candidate = set(combo)
                mean = statistics.mean(_set_f1(candidate, gold) for gold in golds)
                top = max(top, mean)
        return round(top, 4)

    free_gold = [gold - forced for gold in gold_sets]
    free_counts: Counter = Counter()
    for gold in free_gold:
        free_counts.update(gold)
    free_ranked = [name for name, _ in free_counts.most_common(6)]
    return best(ranked, gold_sets), best(free_ranked, free_gold)


def _forced_cards(gold_sets: list[set[str]]) -> set[str]:
    """Cards Stage 1 puts in (almost) every case, which carry no per-case information."""

    if not gold_sets:
        return set()
    counts: Counter = Counter()
    for gold in gold_sets:
        counts.update(gold)
    return {name for name, n in counts.items() if n >= 0.99 * len(gold_sets)}


def _selection_summary(parsed: list[dict], gold_sets: list[set] | None = None) -> dict:
    """Report whether an emitted Evidence selection is case-specific or a constant policy.

    ``selection_f1`` is reported with and without cards that appear in nearly every case,
    together with the best constant policy for each.
    """

    selections = [item["selection"] for item in parsed if item.get("selection") is not None]
    if not selections:
        return {}
    scores = [item["selection_f1"] for item in parsed if item.get("selection_f1") is not None]
    free = [
        item["selection_f1_free"]
        for item in parsed
        if item.get("selection_f1_free") is not None
    ]
    forced = _forced_cards(gold_sets or [])
    constant, constant_free = (_constant_policy(gold_sets, forced)
                               if gold_sets else (None, None))
    return {
        "selection_f1": round(statistics.mean(scores), 4),
        "constant_policy_f1": constant,
        "selection_f1_free": round(statistics.mean(free), 4) if free else None,
        "constant_policy_f1_free": constant_free,
        "forced_cards": sorted(forced),
        "selection_distinct_sets": len({tuple(item) for item in selections}),
        "selection_mean_size": round(statistics.mean(len(item) for item in selections), 2),
        "selection_cases": len(selections),
    }


def _run_path(batch_root: Path, case_id: str) -> Path:
    """Locate a case's Stage-1 run file (`stage1_run.json`) under `--batch-root`."""

    for name in ("stage1_run.json",):
        candidate = batch_root / case_id / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no Stage-1 run file for {case_id} under {batch_root}")


def _score(rows, args, auxiliary, references) -> dict:
    per_case: list[dict] = []
    gold_sets: list[set[str]] = []
    bert_pairs: dict[str, list[tuple[str, str]]] = {section: [] for section in SECTIONS}
    parse_failures = 0
    for row in rows:
        case_id = row["case_id"]
        generated = _parse_sections(row["generated"])
        target = _parse_sections(row["reference"])
        if target is None:
            raise ValueError(f"{case_id}: reference is not a four-section object")
        if generated is None:
            parse_failures += 1
            per_case.append({"case_id": case_id, "parsed": False})
            continue

        run = json.loads(_run_path(args.batch_root, case_id).read_text(encoding="utf-8"))
        index = canonical_value_index(
            SleepProfile.model_validate(run["profile"]),
            LongitudinalInsight.model_validate(run["longitudinal_insight"]),
        ) + references

        supported = unsupported = 0
        unsupported_examples: list[str] = []
        for section in SECTIONS:
            for start, end, value, kind in numeric_mentions(generated[section]):
                if supporting_values(value, kind, index):
                    supported += 1
                else:
                    unsupported += 1
                    if len(unsupported_examples) < 3:
                        unsupported_examples.append(generated[section][start:end])

        anchors = auxiliary.get(case_id, {}).get("certified_anchors", [])
        wanted = {(round(a["value"], 3), a["unit"]) for a in anchors}
        whole = "\n".join(generated[section] for section in SECTIONS)
        recalled = sum(1 for value, unit in wanted if _value_stated(whole, value, unit))

        # `anchor_recall` alone rewards stating more numbers, so a system that dumps
        # statistics scores well without being more faithful. The symmetric comparison
        # below closes that: it takes the set of values each text actually states and
        # scores the generation against the certified answer's set, so saying too little
        # costs recall and saying too much costs precision. Stage 1 self-scores 1.0
        # and the mismatched-reference control gives the floor, as for every other metric.
        stated = _stated_values(generated)
        gold_stated = _stated_values(target)

        committed = _committed_anchor_values(row["generated"])
        kept = sum(1 for value, unit in committed if _value_stated(whole, value, unit))

        # Decompose the commitment: did the model promise the statistics Stage 1 used
        # (commit precision/recall), and separately, did the prose honour the promise?
        promised = _committed_anchor_keys(row["generated"])
        wanted_keys = {
            f"{a['section']}|{a['origin']}.{a['metric_id']}.{a['statistic']}" for a in anchors
        }

        # The number of distinct selections separates a case-specific selection from a
        # constant one, which a mean F1 alone does not.
        selection = _committed_selection(row["generated"])
        if selection is None and row.get("evidence_selection") is not None:
            # Two-phase generation records its identifiers in `evidence_selection`; use them
            # for `selection_f1` against Stage 1's selection.
            selection = {
                item["dossier_id"] if isinstance(item, dict) else item
                for item in row["evidence_selection"]
            }
        gold = {
            item["dossier_id"]
            for item in auxiliary.get(case_id, {}).get("evidence_selection", [])
        }
        gold_sets.append(gold)

        for section in SECTIONS:
            bert_pairs[section].append((generated[section], target[section]))
        per_case.append(
            {
                "case_id": case_id,
                "parsed": True,
                "selection": sorted(selection) if selection is not None else None,
                "selection_f1": _set_f1(selection, gold) if selection is not None else None,
                "selection_f1_free": (
                    _set_f1(selection - _FORCED, gold - _FORCED) if selection is not None else None
                ),
                "evidence_hits": len(_evidence_invoked(generated) & _evidence_gold(
                    auxiliary.get(case_id, {})
                )),
                "evidence_invoked": len(_evidence_invoked(generated)),
                "evidence_gold": len(_evidence_gold(auxiliary.get(case_id, {}))),
                # Micro-F1 pools hits over the whole shard, so a case that cites eight
                # identifiers weighs eight times one that cites one. The per-case F1 and the
                # exact-match rate below weigh every case equally, which is what a
                # multi-label evidence-selection table normally reports.
                "evidence_f1_case": _set_f1(
                    _evidence_invoked(generated), _evidence_gold(auxiliary.get(case_id, {}))
                ),
                "evidence_em": float(
                    _evidence_invoked(generated) == _evidence_gold(auxiliary.get(case_id, {}))
                    and bool(_evidence_gold(auxiliary.get(case_id, {})))
                ),
                "numeric_supported": supported,
                "numeric_unsupported": unsupported,
                "unsupported_examples": unsupported_examples,
                "anchors_wanted": len(wanted),
                "anchors_recalled": recalled,
                "value_hits": len(stated & gold_stated),
                "value_stated": len(stated),
                "value_gold": len(gold_stated),
                "anchors_committed": len(committed),
                "commitments_kept": kept,
                "commit_hits": len(promised & wanted_keys),
                "commit_promised": len(promised),
                "commit_wanted": len(wanted_keys),
                "rouge_l": {
                    section: rouge_l(generated[section], target[section]) for section in SECTIONS
                },
            }
        )

    bertscore: dict[str, float | None] = {}
    if not getattr(args, "no_bertscore", False):
        for section in SECTIONS:
            bertscore[section] = _bertscore(bert_pairs[section], args.bertscore_model)

    parsed = [item for item in per_case if item["parsed"]]
    totals = Counter()
    for item in parsed:
        for key in (
            "numeric_supported",
            "numeric_unsupported",
            "anchors_wanted",
            "anchors_recalled",
            "value_hits",
            "value_stated",
            "value_gold",
            "evidence_hits",
            "evidence_invoked",
            "evidence_gold",
            "anchors_committed",
            "commitments_kept",
            "commit_hits",
            "commit_promised",
            "commit_wanted",
        ):
            totals[key] += item[key]
    by_section = defaultdict(list)
    for item in parsed:
        for section, score in item["rouge_l"].items():
            by_section[section].append(score)

    return {
        "decoded": args.decoded.name,
        "cases": len(per_case),
        "parsed": len(parsed),
        "parse_failure_rate": round(parse_failures / max(len(per_case), 1), 4),
        "numeric_support": round(
            totals["numeric_supported"]
            / max(totals["numeric_supported"] + totals["numeric_unsupported"], 1),
            4,
        ),
        "numbers_per_case": round(
            (totals["numeric_supported"] + totals["numeric_unsupported"]) / max(len(parsed), 1), 2
        ),
        "anchor_recall": round(totals["anchors_recalled"] / max(totals["anchors_wanted"], 1), 4),
        # Evidence fidelity, measured on the text alone so it applies to every system,
        # including Guidance-only SFT, which emits no identifiers.
        "evidence_f1": round(
            2 * totals["evidence_hits"]
            / max(totals["evidence_invoked"] + totals["evidence_gold"], 1),
            4,
        ),
        "evidence_precision": round(
            totals["evidence_hits"] / max(totals["evidence_invoked"], 1), 4
        ),
        "evidence_recall": round(totals["evidence_hits"] / max(totals["evidence_gold"], 1), 4),
        "evidence_f1_macro": round(
            sum(item["evidence_f1_case"] for item in parsed) / max(len(parsed), 1), 4
        ),
        "evidence_em": round(
            sum(item["evidence_em"] for item in parsed) / max(len(parsed), 1), 4
        ),
        "value_precision": round(totals["value_hits"] / max(totals["value_stated"], 1), 4),
        "value_recall": round(totals["value_hits"] / max(totals["value_gold"], 1), 4),
        "value_f1": round(
            2 * totals["value_hits"] / max(totals["value_stated"] + totals["value_gold"], 1), 4
        ),
        "commitment_kept": (
            round(totals["commitments_kept"] / totals["anchors_committed"], 4)
            if totals["anchors_committed"]
            else None
        ),
        # The two links in the chain, reported separately so a regression can be
        # attributed: choosing the right statistics, then actually stating them.
        "commit_precision": (
            round(totals["commit_hits"] / totals["commit_promised"], 4)
            if totals["commit_promised"] else None
        ),
        "commit_recall": (
            round(totals["commit_hits"] / totals["commit_wanted"], 4)
            if totals["commit_promised"] else None
        ),
        **_selection_summary(parsed, gold_sets),
        "rouge_l": {
            section: round(statistics.mean(scores), 4) for section, scores in by_section.items()
        },
        "rouge_l_mean": round(
            statistics.mean([statistics.mean(item["rouge_l"].values()) for item in parsed]), 4
        )
        if parsed
        else None,
        "bertscore": bertscore,
        "bertscore_mean": (
            round(statistics.mean([v for v in bertscore.values() if v is not None]), 4)
            if any(v is not None for v in bertscore.values())
            else None
        ),
        # Per-case scores, for paired bootstrap comparisons between systems.
        "per_case": per_case,
    }


if __name__ == "__main__":
    main()
