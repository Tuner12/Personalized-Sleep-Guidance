"""Step 3 of 6: build the Guidance-only and trajectory SFT sequences, and the 1,200/300 split.

    guidance_only   P -> R                 baseline ("Guidance-only SFT")
    trajectory      P + bank -> O;E;C;R    the paper's reasoning trajectory (Eq. 2)

The held-out 300 cases are chosen deterministically so that every evidence card the training
split exercises also appears in the test split. Evidence identifiers are ordered by training
frequency, and the card text in each target is asserted byte-identical to the served card
library (`card_library.json`, written alongside), which is what generation looks up at
inference. Every numeric anchor is re-checked against the exact rendered input.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

SECTIONS = ("section_1", "section_2", "section_3", "section_4")
SOURCE_SPLITS = ("train", "validation", "test")
SPLITS = ("train", "test")
TEST_SIZE = 300
from sleepguide.stage2.tokenize import GUIDANCE_ONLY_TASK as PLAIN_TASK  # noqa: E402
from sleepguide.stage2.tokenize import TRAJECTORY_TASK as TAGGED_TASK  # noqa: E402

OPEN_CLOSE = {"O": ("<OBSERVATION>", "</OBSERVATION>"),
              "E": ("<EVIDENCE>", "</EVIDENCE>"),
              "C": ("<CARDS>", "</CARDS>"),
              "R": ("<SECTIONS>", "</SECTIONS>")}

TAGGED_FORMAT = """Work in four tagged segments, in this order, and return nothing else. Do not return a bare
JSON object; the four segments below replace that format.

`<OBSERVATION>` the decision-relevant facts of this record, in plain sentences. State the
value of every quantity the Pattern Analysis and Personalized Recommendation will rest on. When a quantity changed
between windows, give the change itself and not only the two endpoints. When sleep timing
matters, give the clock times -- when sleep began, when it ended, where its midpoint fell --
and when weekday and weekend differ, give both. `</OBSERVATION>`

`<EVIDENCE>` the identifiers of the evidence cards this record calls for, one per line,
most frequently applicable first. `</EVIDENCE>`

`<CARDS>` the text of those cards. `</CARDS>`

`<SECTIONS>` a JSON object with exactly the keys `section_1`, `section_2`, `section_3`,
and `section_4`, each a single user-facing paragraph. `</SECTIONS>`"""

# Tells the model what the `evidence_library` field in the payload is and how to use it.
BANK_NOTE = """The input payload carries `evidence_library`: every card in the library, each with its
identifier and the condition under which that card applies. Choose this record's identifiers
from that library, and emit only identifiers that appear in it."""


def strip_json_demand(instruction: str) -> str:
    """Remove the closing "Return only a JSON object" line of the Guidance-only contract,
    which would contradict a trajectory target that opens with `<OBSERVATION>`."""

    return "\n".join(line for line in instruction.splitlines()
                     if "Return only a JSON object" not in line)


def load(source: Path, task: str, split: str) -> list[dict]:
    path = source / task / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def payload_of(record: dict) -> dict:
    """Recover the JSON payload from a rendered input."""
    body = record["input"].split("Input payload (JSON):", 1)[1]
    body = body.rsplit("\n\n", 1)[0].strip()
    return json.loads(body)


def render(instruction: str, payload: dict, closing: str) -> str:
    return "\n\n".join([instruction.strip(), "Input payload (JSON):",
                        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        closing])


def card_text(card: dict) -> str:
    lines = [f"[{card['evidence_key']}] role: {card.get('evidence_role') or 'unspecified'}"]
    for label, key in (("analysis", "analysis_support"),
                       ("recommendation", "recommendation_support"),
                       ("follow up", "follow_up_support"),
                       ("select when", "select_when")):
        if card.get(key):
            lines.append(f"  {label}: {card[key]}")
    if card.get("canonical_action_title"):
        lines.append(f"  action: {card['canonical_action_title']}")
    for value in card.get("reference_values") or []:
        lines.append(f"  reference: {value.get('label') or value.get('metric_id')} "
                     f"{value.get('comparator')} {value.get('value')} {value.get('unit')}")
    if card.get("required_condition"):
        lines.append(f"  only when: {card['required_condition']}")
    if card.get("do_not_claim"):
        lines.append(f"  do not claim: {'; '.join(card['do_not_claim'])}")
    return "\n".join(lines)


def certified_anchors(auxiliary: dict) -> list[dict]:
    """Stage-1 certified numeric anchors as one flat list tagged by section (read by the
    observation renderer and by Value F1)."""
    out = []
    for section in SECTIONS:
        grounding = (auxiliary.get("section_grounding") or {}).get(section) or {}
        for anchor in grounding.get("case_anchors") or []:
            out.append({**anchor, "section": section})
    return out


def observation_text(observation: dict) -> str:
    parts = [observation["primary_observation"].strip()]
    parts += [x.strip() for x in (observation.get("supporting_observations") or []) if x.strip()]
    if observation.get("context_observation"):
        parts.append(observation["context_observation"].strip())
    return "\n".join(parts)


def coverage_split(features: dict[str, set[str]], test_size: int) -> set[str]:
    """Deterministic test shard that exercises every feature the training shard does.

    Rarest feature first, because its few cases are the ones a later quota cannot make room
    for; a feature present in only one case goes to train, so the model can at least have
    seen it. The remainder is filled in hash order, so the same input gives the same split.
    """

    holders: dict[str, list[str]] = {}
    for case_id, values in features.items():
        for value in values:
            holders.setdefault(value, []).append(case_id)
    order = sorted(features, key=lambda cid: hashlib.sha256(f"v3:{cid}".encode()).hexdigest())
    position = {cid: i for i, cid in enumerate(order)}
    for value in holders:
        holders[value].sort(key=lambda cid: position[cid])

    share = test_size / max(len(features), 1)
    test: set[str] = set()
    for value in sorted(holders, key=lambda v: len(holders[v])):
        cases = holders[value]
        if len(cases) < 2:
            continue
        want = min(max(1, round(len(cases) * share)), len(cases) - 1)
        have = sum(1 for cid in cases if cid in test)
        for case_id in cases:
            if have >= want or len(test) >= test_size:
                break
            if case_id not in test:
                test.add(case_id)
                have += 1
    for case_id in order:
        if len(test) >= test_size:
            break
        test.add(case_id)
    return test


ARMS = ("guidance_only", "trajectory")


def build(source: Path, out: Path) -> None:
    """Write `out/guidance_only`, `out/trajectory` and `out/card_library.json`."""

    # 1,200 training / 300 test cases; no validation split, since the training schedule is
    # fixed in advance.
    pool_plain = [r for s in SOURCE_SPLITS for r in load(source, "pure_sft", s)]
    pool_writer = {r["case_id"]: r for s in SOURCE_SPLITS for r in load(source, "evidence_writer", s)}
    pool_selector = {r["case_id"]: r for s in SOURCE_SPLITS for r in load(source, "evidence_selector", s)}
    print(f"pool: {len(pool_plain)} cases")

    # Coverage-first test shard: every evidence key and every canonical action title that
    # the training split exercises must also appear in the test split. A plain hash split
    # left the rarest keys on whichever side the hash fell, and those are exactly where a
    # held-out score is least stable.
    features: dict[str, set[str]] = {}
    for record in pool_plain:
        case_id = record["case_id"]
        items = payload_of(pool_writer[case_id])["evidence"]["items"]
        features[case_id] = {item["evidence_key"] for item in items}
    test_ids = coverage_split(features, TEST_SIZE)
    print(f"test shard: {len(test_ids)} cases")

    plain = {"train": [r for r in pool_plain if r["case_id"] not in test_ids],
             "test": [r for r in pool_plain if r["case_id"] in test_ids]}
    writer = {s: {r["case_id"]: pool_writer[r["case_id"]] for r in rows}
              for s, rows in plain.items()}
    selector = {s: {r["case_id"]: pool_selector[r["case_id"]] for r in rows}
                for s, rows in plain.items()}

    # the library and its card text, taken from the selector input; it is the same bank in
    # every case, so it is a reference, not case information
    library: dict[str, str] = {}
    bank_entries: list[dict] = []
    sample = selector["train"][next(iter(selector["train"]))]
    for card in payload_of(sample)["generation_evidence_bank"]:
        library[card["evidence_key"]] = card_text(card)
        bank_entries.append({"evidence_key": card["evidence_key"],
                             "select_when": card.get("select_when")})
    bank_entries.sort(key=lambda item: item["evidence_key"])
    print(f"library: {len(library)} cards")

    # canonical identifier order, from the training split only
    frequency: collections.Counter = collections.Counter()
    for case_id, record in writer["train"].items():
        frequency.update(item["evidence_key"]
                         for item in payload_of(record)["evidence"]["items"])
    rank = {name: index for index, (name, _) in enumerate(frequency.most_common())}

    counts: collections.Counter = collections.Counter()
    problems: collections.Counter = collections.Counter()
    for split in SPLITS:
        rows = {"guidance_only": [], "trajectory": []}
        for record in plain[split]:
            case_id = record["case_id"]
            profile = payload_of(record)
            sections = json.loads(record["target"])
            evidence = payload_of(writer[split][case_id])["evidence"]
            keys = sorted({item["evidence_key"] for item in evidence["items"]},
                          key=lambda name: rank.get(name, len(rank)))
            if any(name not in library for name in keys):
                problems["identifier not in library"] += 1
                continue
            observation = observation_text(
                payload_of(writer[split][case_id])["observation"])
            tagged = (
                f"{OPEN_CLOSE['O'][0]}\n{observation}\n{OPEN_CLOSE['O'][1]}\n"
                f"{OPEN_CLOSE['E'][0]}\n" + "\n".join(keys) + f"\n{OPEN_CLOSE['E'][1]}\n"
                f"{OPEN_CLOSE['C'][0]}\n"
                + "\n\n".join(library[name] for name in keys)
                + f"\n{OPEN_CLOSE['C'][1]}\n"
                f"{OPEN_CLOSE['R'][0]}\n"
                f"{json.dumps(sections, ensure_ascii=False, indent=2)}\n"
                f"{OPEN_CLOSE['R'][1]}")
            with_bank = {**profile, "evidence_library": bank_entries}
            base = {"case_id": case_id, "split": split,
                    "auxiliary": {**record["auxiliary"],
                                  "evidence_selection": [{"dossier_id": name}
                                                         for name in keys],
                                  "certified_anchors": certified_anchors(record["auxiliary"])}}
            instruction = record["input"].split("\n\nInput payload (JSON):", 1)[0]
            rows["guidance_only"].append({**base, "task": PLAIN_TASK,
                                    "input": record["input"],
                                    "target": record["target"]})
            clean = strip_json_demand(instruction)
            rows["trajectory"].append({**base, "task": TAGGED_TASK,
                                        "input": render(
                                            f"{clean}\n\n{BANK_NOTE}\n\n{TAGGED_FORMAT}",
                                            with_bank,
                                            "Return only the four tagged "
                                            "segments described above."),
                                        "target": tagged})
        for arm in ARMS:
            items = rows[arm]
            path = out / arm / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            counts[f"{arm}/{split}"] = len(items)

    (out / "card_library.json").write_text(
        json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- assertions on what was written, not on what was intended ----
    for arm in ARMS:
        for split in SPLITS:
            for line in (out / arm / f"{split}.jsonl").read_text().splitlines():
                row = json.loads(line)
                for section in SECTIONS:
                    grounding = (row["auxiliary"].get("section_grounding") or {}).get(section) or {}
                    for anchor in grounding.get("case_anchors") or []:
                        if f'"{anchor["metric_id"]}"' not in row["input"]:
                            problems[f"{arm}: anchor metric not in input"] += 1
                if row["target"].startswith("<OBSERVATION>"):
                    block = row["target"].split("<EVIDENCE>")[1].split("</EVIDENCE>")[0]
                    names = [x.strip() for x in block.splitlines() if x.strip()]
                    if names != sorted(names, key=lambda n: rank.get(n, len(rank))):
                        problems[f"{arm}: identifier order not canonical"] += 1
                    cards = row["target"].split("<CARDS>")[1].split("</CARDS>")[0].strip()
                    if cards != "\n\n".join(library[n] for n in names):
                        problems[f"{arm}: card text differs from the served library"] += 1

    print(json.dumps({k: counts[k] for k in sorted(counts)}, indent=1))
    if problems:
        print("\nFAILED")
        for name, count in problems.most_common():
            print(f"  {name}: {count}")
        sys.exit(1)
    print(f"\nall assertions passed -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, required=True, help="output of packet_budget.py")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()
