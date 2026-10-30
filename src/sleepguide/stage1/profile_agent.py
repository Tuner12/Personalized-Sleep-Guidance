"""Profile Agent: build the profile packet P for any wearable dataset (paper Section 3.1).

The agent receives `prompts/stage1/profile_agent.md`, the metric table
(`prompts/stage1/profile_metrics.json`), the JSON Schema of P and the dataset location. It writes
and runs its own code, producing `<case_id>.json` per case; each file is then validated
against the contracts that the other agents and Stage 2 rely on.

    python scripts/build_profiles.py --dataset <path> --notes <field documentation> --out outputs/profiles
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from pydantic import BaseModel, ValidationError

from sleepguide.core.contracts import LongitudinalInsight, SleepProfile
from sleepguide.paths import PROMPTS


class ProfilePacket(BaseModel):
    profile: SleepProfile
    longitudinal_insight: LongitudinalInsight


def build_request(dataset: Path, out: Path, notes: str | None) -> str:
    instructions = (PROMPTS / "stage1/profile_agent.md").read_text(encoding="utf-8")
    metrics = (PROMPTS / "stage1/profile_metrics.json").read_text(encoding="utf-8")
    schema = json.dumps(ProfilePacket.model_json_schema(), indent=1)
    return "\n\n".join([
        instructions.strip(),
        f"## Dataset\n\nLocation: `{dataset}`\n\n{notes.strip() if notes else '(no documentation supplied)'}",
        f"## Output directory\n\n`{out}`",
        f"## Metric table\n\n```json\n{metrics.strip()}\n```",
        f"## JSON Schema of each output file\n\n```json\n{schema}\n```",
    ])


def validate(out: Path) -> tuple[int, dict[str, str]]:
    """Validate every packet; return the number valid and the errors by case."""
    valid, errors = 0, {}
    for path in sorted(out.glob("*.json")):
        try:
            packet = ProfilePacket.model_validate_json(path.read_text(encoding="utf-8"))
            if packet.profile.case_id != path.stem or packet.longitudinal_insight.case_id != path.stem:
                raise ValueError("case_id does not match the file name")
            valid += 1
        except (ValidationError, ValueError) as error:
            errors[path.stem] = str(error).splitlines()[0]
    return valid, errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--notes", type=Path, default=None, help="documentation of the dataset's fields")
    ap.add_argument("--out", type=Path, default=Path("outputs/profiles"))
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    if not args.validate_only:
        args.out.mkdir(parents=True, exist_ok=True)
        notes = args.notes.read_text(encoding="utf-8") if args.notes else None
        subprocess.run(["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write",
                        "-m", args.model, "-c", 'model_reasoning_effort="medium"',
                        build_request(args.dataset.resolve(), args.out.resolve(), notes)],
                       check=True)
    valid, errors = validate(args.out)
    print(f"{valid} valid profile packets in {args.out}; {len(errors)} invalid")
    for case_id, message in list(errors.items())[:20]:
        print(f"  {case_id}: {message}")


if __name__ == "__main__":
    main()
