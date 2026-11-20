from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

# Allow running directly via `python src/adapters/trizbench_loader.py`
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schemas import RawCaseRecord

CPC_SECTIONS = {
    "A": "Human Necessities",
    "B": "Operations & Transport",
    "C": "Chemistry & Metallurgy",
    "D": "Textiles & Paper",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering",
    "G": "Physics",
    "H": "Electricity",
    "Y": "Emerging Technologies",
}


import re

def parse_cpc_domain(cpc_raw: str) -> str:
    """Extract the first valid CPC subclass (e.g. 'G02B') from raw or list-formatted strings."""
    if not cpc_raw:
        return "Unknown"
    match = re.search(r"[A-HY]\d{2}[A-Z]", cpc_raw.upper())
    if not match:
        return "Unknown"
    subclass = match.group(0)
    section_name = CPC_SECTIONS.get(subclass[0], "Other")
    return f"{subclass} ({section_name})"


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip()


def load_trizbench_patents(
    csv_path: str | Path = "data/raw/trizbench/data/patents_text_classical.csv",
    min_before_chars: int = 100,
    min_after_chars: int = 100,
    max_text_chars: int = 12000,
) -> tuple[list[RawCaseRecord], dict[str, int]]:
    """
    Load TRIZBENCH classical patents into unified RawCaseRecord objects.
    Uses (field + background) as before_text and (abstract + summary + claim1) as after_text
    to prevent solution leakage into Pass 1.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"TRIZBENCH CSV not found at: {path}")

    # Patents can contain large text fields
    csv.field_size_limit(sys.maxsize)

    records: list[RawCaseRecord] = []
    skipped = Counter()

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patent_id = _clean(row.get("patent_id"))
            if not patent_id:
                skipped["missing_patent_id"] += 1
                continue

            field_txt = _clean(row.get("field"))
            background_txt = _clean(row.get("background"))
            abstract_txt = _clean(row.get("abstract"))
            summary_txt = _clean(row.get("summary"))
            claim1_txt = _clean(row.get("claim1"))

            before_parts = []
            if field_txt:
                before_parts.append(f"FIELD: {field_txt}")
            if background_txt:
                before_parts.append(f"BACKGROUND: {background_txt}")
            before_text = "\n\n".join(before_parts)[:max_text_chars]

            after_parts = []
            if abstract_txt:
                after_parts.append(f"ABSTRACT: {abstract_txt}")
            if summary_txt:
                after_parts.append(f"SUMMARY: {summary_txt}")
            if claim1_txt:
                after_parts.append(f"CLAIM 1: {claim1_txt}")
            after_text = "\n\n".join(after_parts)[:max_text_chars]

            if len(before_text) < min_before_chars:
                skipped["short_or_missing_before_text"] += 1
                continue
            if len(after_text) < min_after_chars:
                skipped["short_or_missing_after_text"] += 1
                continue

            records.append(
                RawCaseRecord(
                    case_id=f"trizbench_{patent_id}",
                    source="trizbench",
                    domain=parse_cpc_domain(row.get("cpc", "")),
                    before_text=before_text,
                    after_text=after_text,
                )
            )

    return records, dict(skipped)


if __name__ == "__main__":
    records, skipped = load_trizbench_patents()
    print(f"Loaded records: {len(records)} | Skipped: {skipped}")

    if records:
        avg_before = sum(len(r.before_text) for r in records) // len(records)
        avg_after = sum(len(r.after_text) for r in records) // len(records)
        domains = Counter(r.domain for r in records).most_common(5)

        print(f"Avg length -> before_text: {avg_before} chars | after_text: {avg_after} chars")
        print(f"Top 5 domains: {domains}")
        print(f"\nSample case_id: {records[0].case_id} | domain: {records[0].domain}")
        print(f"Before preview: {records[0].before_text[:250]}...")
        print(f"After preview:  {records[0].after_text[:250]}...")