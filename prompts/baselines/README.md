# Baseline prompt

GPT-5.6-Sol, GPT Health and OpenEvidence receive one prompt per case, assembled in this order:

| Part | Source |
|---|---|
| Health-assistant instruction (GPT Health only) | `prompts/baselines/health_assistant_instruction.md` |
| Writing contract for the four sections | `prompts/shared/guidance_contract.md` |
| Task: write the four sections from P (without its final "Return only a JSON object" line) | `prompts/stage2/guidance_only.md` |
| Reader-language rules | `prompts/shared/reader_language_contract.md` |
| Note on the evidence library, and the tagged output segments | `BANK_NOTE`, `TAGGED_FORMAT` in `src/sleepguide/stage2/trajectory/sequences.py` (observation line from `canonical_observation.py`) |
| Card thresholds the record crosses | `src/sleepguide/stage2/trajectory/threshold_flags.py` |
| Record summary P with the 22-card menu (identifier and applicability condition) | the case's data |
| Full text of all 22 evidence cards | `evidence_bank/card_library.json` |
| Output format | `prompts/baselines/output_format.md` |

Every part except the first and the last two is the input our trajectory model reads (each
case's `input` in `outputs/sft/trajectory/test.jsonl`, written by `scripts/build_sft_data.py`).
The writing contract and reader-language rules are the same files the Composer Agent uses;
`prompts/stage1/composer_agent.md` adds the Stage-1 grounding and revision rules.

To print the complete prompt for one case:

```bash
python -m sleepguide.baselines.llm_prompting --system gpt_health --show-prompt [CASE_ID]
```
