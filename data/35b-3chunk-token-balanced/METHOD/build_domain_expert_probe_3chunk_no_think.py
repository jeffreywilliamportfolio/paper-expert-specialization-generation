#!/usr/bin/env python3
"""Build 3 long no-think prompts by concatenating the 60-domain probe by slot."""

from __future__ import annotations

import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = THIS_DIR.parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"

SOURCE_TSV = (
    EXPERIMENT_DIR.parent / "qwen-huahua-6cond-moe-manips" / "prompts" / "domain_expert_probe_60_no_think.tsv"
)
ROWS_JSON = PROMPTS_DIR / "domain_expert_probe_3chunk_prompts.json"
OUTPUT_TSV = PROMPTS_DIR / "domain_expert_probe_3chunk_no_think.tsv"

USER_PREFIX = "<|im_start|>user\\n"
ASSISTANT_SUFFIX = "<|im_end|>\\n<|im_start|>assistant\\n</think>\\n\\n"

CHUNKS = (
    ("A", "01"),
    ("B", "02"),
    ("C", "03"),
)

PADDING_COUNTS = {
    "A": 0,
    "B": 73,
    "C": 44,
}

PADDING_UNIT = " ."


def extract_prompt_body(raw_prompt: str) -> str:
    prefix = USER_PREFIX
    suffix = ASSISTANT_SUFFIX
    if not raw_prompt.startswith(prefix) or not raw_prompt.endswith(suffix):
        raise ValueError("Unexpected TSV prompt template")
    return raw_prompt[len(prefix) : -len(suffix)]


def parse_source_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in SOURCE_TSV.read_text().splitlines():
        if not line.strip():
            continue
        prompt_id, raw_prompt = line.split("\t", 1)
        prompt_stem, slot = prompt_id.rsplit("_", 1)
        domain_prefix, domain_name = prompt_stem.split("_", 1)
        rows.append(
            {
                "id": prompt_id,
                "domain_index": domain_prefix,
                "domain": domain_name,
                "slot": slot,
                "prompt": extract_prompt_body(raw_prompt),
            }
        )
    return rows


def main() -> None:
    source_rows = parse_source_rows()

    prompts = []
    tsv_lines = []
    for chunk_label, slot in CHUNKS:
        chunk_rows = [row for row in source_rows if row["slot"] == slot]
        if len(chunk_rows) != 20:
            raise ValueError(f"Expected 20 rows for slot {slot}, found {len(chunk_rows)}")

        combined_prompt = "\\n".join(row["prompt"] for row in chunk_rows)
        padding = PADDING_UNIT * PADDING_COUNTS[chunk_label]
        if padding:
            combined_prompt = f"{combined_prompt}{padding}"
        prompt_id = f"domain_expert_probe_20{chunk_label}_chunk"
        row = {
            "id": prompt_id,
            "chunk": chunk_label,
            "slot": slot,
            "n_domains": len(chunk_rows),
            "padding_unit": PADDING_UNIT,
            "padding_count": PADDING_COUNTS[chunk_label],
            "source_ids": [item["id"] for item in chunk_rows],
            "source_domains": [item["domain"] for item in chunk_rows],
            "prompt": combined_prompt,
        }
        prompts.append(row)
        tsv_lines.append(f"{prompt_id}\t{USER_PREFIX}{combined_prompt}{ASSISTANT_SUFFIX}")

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    ROWS_JSON.write_text(json.dumps(prompts, indent=2) + "\n")
    OUTPUT_TSV.write_text("\n".join(tsv_lines) + "\n")
    print(f"Wrote {len(prompts)} chunk rows to {ROWS_JSON}")
    print(f"Wrote {len(tsv_lines)} prompts to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
