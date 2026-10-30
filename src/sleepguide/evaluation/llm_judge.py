"""LLM-as-a-Judge: Claude Opus 5 on the 16-item Generation Quality rubric (paper Table 1).

The judge sees, for each case, the person's measured data (the reference observation), the
Stage-1 reference guidance and its evidence (card identifier and claim), and the candidate's
four sections; it is blind to which system wrote the candidate. Cases are judged in batches of
12, single vote. A case's score is the mean of its 16 items; a system's score is the mean over
its cases.

    python -m sleepguide.evaluation.llm_judge --generated outputs/generations/<run>.jsonl \
        --test-file outputs/sft/trajectory/test.jsonl --out outputs/judge/<run>
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

from sleepguide.paths import PROMPTS
from sleepguide.stage2.tokenize import TRAJECTORY_TASK, load_records

MODEL = "claude-opus-5"
BATCH = 12
KEYS = ["s1_selection", "s1_consistency", "s1_clarity",
        "s2_pattern", "s2_role", "s2_consistency",
        "s3_action_fidelity", "s3_actionability", "s3_role",
        "s4_followup_fidelity", "s4_consistency", "s4_role",
        "evidence_selectivity", "proportionality", "safety_scope", "readability"]
SECTION = {1: "Section 1 - Summary", 2: "Section 2 - Pattern Analysis",
           3: "Section 3 - Personalized Recommendation", 4: "Section 4 - Follow-up"}


def sections_of(text: str) -> dict[int, str] | None:
    try:
        d = json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
    except Exception:
        return None
    if not all(f"section_{i}" in d for i in (1, 2, 3, 4)):
        return None
    return {i: str(d.get(f"section_{i}", "")).strip() for i in (1, 2, 3, 4)}


def load_references(test_file: Path, card_library: Path) -> dict[str, dict]:
    """case_id -> measured data, reference sections and reference evidence (id, claim)."""
    claim = {}
    for key, body in json.loads(card_library.read_text(encoding="utf-8")).items():
        c = body.split("claim:", 1)[1].split("applies when")[0].strip() if "claim:" in body else body
        claim[key] = " ".join(c.split())[:160]
    refs = {}
    for r in load_records(test_file, task=TRAJECTORY_TASK):
        t = r["target"]
        stats = t.split("<OBSERVATION>")[1].split("</OBSERVATION>")[0].strip()
        evidence = [x.strip() for x in t.split("<EVIDENCE>")[1].split("</EVIDENCE>")[0].splitlines()
                    if x.strip()]
        sections = sections_of(t.split("<SECTIONS>")[1].split("</SECTIONS>")[0]) or {}
        refs[r["case_id"]] = {"stats": stats, "sections": sections,
                              "evidence": [(e, claim.get(e, "")) for e in evidence]}
    return refs


def case_block(index: int, candidate: dict[int, str], ref: dict) -> str:
    stats = ref["stats"] or "(no canonical statistics available)"
    reference = "\n".join(f"  [Reference S{i}] {ref['sections'].get(i, '')}" for i in (1, 2, 3, 4))
    evidence = "\n".join(f"  - {cid}: {cl}" for cid, cl in ref["evidence"]) or "  (none)"
    body = "\n".join(f"--- Candidate Section {i} ({SECTION[i]}) ---\n{candidate[i]}\n"
                     for i in (1, 2, 3, 4))
    return (f"=== CASE {index} ===\n"
            f"-- MEASURED SLEEP DATA (verified ground truth) --\n{stats}\n\n"
            f"-- CERTIFIED REFERENCE ANSWER (the standard for this person) --\n{reference}\n\n"
            f"-- CERTIFIED EVIDENCE (what the standard deems applicable) --\n{evidence}\n\n"
            f"-- CANDIDATE RESPONSE --\n{body}\n")


def build_prompt(blocks: list[str]) -> str:
    template = (PROMPTS / "judge/generation_quality.md").read_text(encoding="utf-8")
    schema = "[{" + '"case":1, ' + ", ".join(f'"{k}":_' for k in KEYS) + "}, ...]"
    return (template.replace("{n}", str(len(blocks))).replace("{schema}", schema)
            .replace("{cases}", "\n".join(blocks)))


def judge(prompt: str, expected: int) -> list[dict] | None:
    for _ in range(2):
        try:
            raw = subprocess.run(["claude", "--print", "--model", MODEL, "--output-format", "json",
                                  prompt], capture_output=True, text=True, timeout=420)
            out = json.loads(raw.stdout)["result"]
            scores = json.JSONDecoder().raw_decode(out[out.index("["):])[0]
            if (isinstance(scores, list) and len(scores) == expected and
                    all(all(isinstance(o.get(k), int) for k in KEYS) for o in scores)):
                return sorted(scores, key=lambda o: o.get("case", 0))
        except Exception:
            pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generated", type=Path, required=True)
    ap.add_argument("--test-file", type=Path, default=Path("outputs/sft/trajectory/test.jsonl"))
    ap.add_argument("--card-library", type=Path, default=Path("outputs/sft/card_library.json"))
    ap.add_argument("--out", type=Path, required=True, help="directory, one JSON per case")
    args = ap.parse_args()

    refs = load_references(args.test_file, args.card_library)
    args.out.mkdir(parents=True, exist_ok=True)
    todo = []
    for line in args.generated.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = args.out / f"{row['case_id']}.json"
        if path.exists():
            continue
        candidate = sections_of(row.get("generated", "") or "")
        if candidate is None or row["case_id"] not in refs:
            path.write_text(json.dumps({k: None for k in KEYS}))  # unparsable output
            continue
        todo.append((row["case_id"], candidate))

    for offset in range(0, len(todo), BATCH):
        chunk = todo[offset:offset + BATCH]
        prompt = build_prompt([case_block(i + 1, cand, refs[cid])
                               for i, (cid, cand) in enumerate(chunk)])
        scores = judge(prompt, len(chunk))
        if scores is None:
            print(f"judge failed for {len(chunk)} cases starting at {chunk[0][0]}")
            continue
        for (cid, _), item in zip(chunk, scores):
            (args.out / f"{cid}.json").write_text(json.dumps({k: item[k] for k in KEYS}))

    per_case = []
    for path in sorted(args.out.glob("case_*.json")):
        item = json.loads(path.read_text())
        if all(item.get(k) is not None for k in KEYS):
            per_case.append(statistics.mean(item[k] for k in KEYS))
    if per_case:
        print(f"LLM-as-a-Judge: {statistics.mean(per_case):.3f} over {len(per_case)} cases")


if __name__ == "__main__":
    main()
