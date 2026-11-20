from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterator

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adapters.trizbench_loader import CPC_SECTIONS
from schemas import RawCaseRecord

csv.field_size_limit(sys.maxsize)

DEFAULT_CPC_SUBCLASSES = [
    "F16F",  # Springs, shock absorbers, vibration damping
    "F16H",  # Gearing & mechanical transmissions
    "F16K",  # Valves, taps, cocks
    "F28D",  # Heat-exchange apparatus
    "F04B",  # Positive-displacement pumps
    "B25J",  # Robotic manipulators
    "B60G",  # Vehicle suspension arrangements
    "H01M",  # Batteries & electrochemical cells
    "A61B",  # Surgical & medical instruments
    "B01D",  # Separation & filtration devices
]


def _format_domain(subclass: str) -> str:
    sc = subclass.strip().upper()[:4]
    section_name = CPC_SECTIONS.get(sc[:1], "Other")
    return f"{sc} ({section_name})"


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip()


def _find_zip(raw_dir: Path, patterns: list[str], required: bool = True) -> Path | None:
    for pat in patterns:
        matches = sorted(raw_dir.glob(pat))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find any file matching {patterns} in {raw_dir}")
    return None


def _iter_tsv_zip(zip_path: Path) -> Iterator[dict[str, str]]:
    """Stream rows from the first .tsv file inside a zip archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m for m in zf.namelist()
            if not m.endswith("/") and not Path(m).name.startswith(".")
        ]
        if not members:
            raise ValueError(f"No data file found inside {zip_path}")
        with zf.open(members[0], "r") as raw_f:
            text_f = io.TextIOWrapper(raw_f, encoding="utf-8", errors="replace", newline="")
            reader = csv.DictReader(text_f, delimiter="\t")
            for row in reader:
                yield row


def _first_key(row: dict[str, str], candidates: list[str]) -> str:
    for key in candidates:
        val = row.get(key)
        if val:
            return val.strip()
    return ""


def _is_claim_one(row: dict[str, str]) -> bool:
    num = (row.get("claim_number") or "").strip().lstrip("0")
    if num:
        return num == "1"
    seq = (row.get("claim_sequence") or "").strip()
    return seq == "0"


def build_uspto_pairs_from_bulk(
    raw_dir: str | Path = "data/raw/uspto",
    output_path: str | Path = "data/raw/uspto_pairs.jsonl",
    subclasses: list[str] | None = None,
    pairs_per_subclass: int = 80,
    min_text_chars: int = 200,
    max_text_chars: int = 10000,
) -> list[RawCaseRecord]:
    """
    Memory-efficient streaming extractor over local PatentsView bulk .zip files.
    Outputs unified RawCaseRecord JSONL where Patent B cites Patent A in the same CPC subclass.
    """
    raw_path = Path(raw_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    patent_zip = _find_zip(raw_path, ["g_patent.tsv.zip", "*g_patent.zip"])
    abstract_zip = _find_zip(raw_path, ["*g_patent_abstract*.zip"], required=False)
    cpc_zip = _find_zip(raw_path, ["*g_cpc_current*.zip", "*g_cpc_at_issue*.zip"])
    cit_zip = _find_zip(raw_path, ["*g_us_patent_citation*.zip"])
    claim_zips = sorted(raw_path.glob("*g_claims*.zip"))
    if not claim_zips:
        raise FileNotFoundError(f"No g_claims_*.zip files found in {raw_path}")

    target_subclasses = {s.strip().upper()[:4] for s in (subclasses or DEFAULT_CPC_SUBCLASSES)}

    print(f"Using files in {raw_path}:")
    print(f"  Patents:   {patent_zip.name}")
    print(f"  Abstracts: {abstract_zip.name if abstract_zip else 'NOT FOUND (will use Title + Claim 1)'}")
    print(f"  CPC:       {cpc_zip.name}")
    print(f"  Citations: {cit_zip.name}")
    print(f"  Claims:    {[p.name for p in claim_zips]}")

    # Pass 1: Collect patent_ids that have a Claim 1 in our downloaded claims files
    print("\n[Pass 1/5] Indexing patent IDs with Claim 1...")
    claim_pids: set[str] = set()
    for cz in claim_zips:
        for row in _iter_tsv_zip(cz):
            if _is_claim_one(row):
                pid = (row.get("patent_id") or "").strip()
                if pid:
                    claim_pids.add(pid)
    print(f"  Found {len(claim_pids):,} patents with Claim 1.")

    # Pass 2: Filter to patents belonging to our target CPC subclasses
    print("[Pass 2/5] Filtering to target CPC subclasses...")
    pid_to_subclasses: dict[str, set[str]] = {}
    for row in _iter_tsv_zip(cpc_zip):
        pid = (row.get("patent_id") or "").strip()
        if pid not in claim_pids:
            continue
        sc = _first_key(row, ["cpc_subclass", "cpc_subclass_id"]).upper()[:4]
        if sc in target_subclasses:
            pid_to_subclasses.setdefault(pid, set()).add(sc)
    print(f"  Matched {len(pid_to_subclasses):,} patents across {len(target_subclasses)} subclasses.")

    # Pass 3: Scan citations for same-CPC (Patent A <- Patent B) pairs
    target_candidates = int(pairs_per_subclass * 1.5)
    print(f"[Pass 3/5] Scanning citations (target {pairs_per_subclass} pairs/subclass)...")
    pairs_by_sc: dict[str, list[tuple[str, str]]] = {sc: [] for sc in target_subclasses}
    seen_b: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    for row in _iter_tsv_zip(cit_zip):
        pid_b = (row.get("patent_id") or "").strip()
        if pid_b not in pid_to_subclasses or pid_b in seen_b:
            continue
        pid_a = (row.get("citation_patent_id") or "").strip()
        if pid_a not in pid_to_subclasses or pid_a == pid_b:
            continue

        shared = pid_to_subclasses[pid_b] & pid_to_subclasses[pid_a]
        if not shared:
            continue

        for sc in sorted(shared):
            if len(pairs_by_sc[sc]) < target_candidates and (pid_a, pid_b) not in seen_pairs:
                pairs_by_sc[sc].append((pid_a, pid_b))
                seen_b.add(pid_b)
                seen_pairs.add((pid_a, pid_b))
                break

        if all(len(v) >= target_candidates for v in pairs_by_sc.values()):
            break

    total_cand = sum(len(v) for v in pairs_by_sc.values())
    print(f"  Found {total_cand} candidate same-CPC pairs:")
    for sc in sorted(pairs_by_sc):
        print(f"    {sc}: {len(pairs_by_sc[sc])}")

    if total_cand == 0:
        print("  No pairs found.")
        return []

    needed_pids = {pid for pairs in pairs_by_sc.values() for pair in pairs for pid in pair}

    # Pass 4: Fetch abstracts (from g_patent_abstract.tsv.zip) and/or titles (from g_patent.tsv.zip)
    abstracts: dict[str, str] = {}
    titles: dict[str, str] = {}

    if abstract_zip:
        print(f"[Pass 4/5] Fetching abstracts for {len(needed_pids):,} patents...")
        for row in _iter_tsv_zip(abstract_zip):
            pid = (row.get("patent_id") or "").strip()
            if pid in needed_pids:
                abs_txt = _clean(row.get("patent_abstract"))
                if abs_txt:
                    abstracts[pid] = abs_txt
                if len(abstracts) == len(needed_pids):
                    break
        print(f"  Retrieved {len(abstracts):,} abstracts.")
    else:
        print(f"[Pass 4/5] Fetching titles (fallback) for {len(needed_pids):,} patents...")
        for row in _iter_tsv_zip(patent_zip):
            pid = (row.get("patent_id") or "").strip()
            if pid in needed_pids:
                t_txt = _clean(row.get("patent_title"))
                if t_txt:
                    titles[pid] = t_txt
                if len(titles) == len(needed_pids):
                    break
        print(f"  Retrieved {len(titles):,} titles.")

    # Pass 5: Fetch Claim 1 text for needed_pids only
    print(f"[Pass 5/5] Fetching Claim 1 text for {len(needed_pids):,} patents...")
    claims: dict[str, str] = {}
    for cz in claim_zips:
        for row in _iter_tsv_zip(cz):
            pid = (row.get("patent_id") or "").strip()
            if pid in needed_pids and pid not in claims and _is_claim_one(row):
                txt = _clean(row.get("claim_text"))
                if txt:
                    claims[pid] = txt
        if len(claims) == len(needed_pids):
            break
    print(f"  Retrieved {len(claims):,} Claim 1 texts.")

    def _compose(pid: str) -> str:
        parts = []
        if pid in abstracts:
            parts.append(f"ABSTRACT: {abstracts[pid]}")
        elif pid in titles:
            parts.append(f"TITLE: {titles[pid]}")
        if pid in claims:
            parts.append(f"CLAIM 1: {claims[pid]}")
        return "\n\n".join(parts)[:max_text_chars]

    # Assemble & write JSONL
    records: list[RawCaseRecord] = []
    with out.open("w", encoding="utf-8") as f:
        for sc in sorted(pairs_by_sc):
            kept = 0
            for pid_a, pid_b in pairs_by_sc[sc]:
                if kept >= pairs_per_subclass:
                    break
                if pid_a not in claims or pid_b not in claims:
                    continue
                if abstract_zip and (pid_a not in abstracts or pid_b not in abstracts):
                    continue

                before_text = _compose(pid_a)
                after_text = _compose(pid_b)
                if len(before_text) < min_text_chars or len(after_text) < min_text_chars:
                    continue

                rec = RawCaseRecord(
                    case_id=f"uspto_{pid_a}_to_{pid_b}",
                    source="uspto",
                    domain=_format_domain(sc),
                    before_text=before_text,
                    after_text=after_text,
                )
                f.write(rec.model_dump_json() + "\n")
                records.append(rec)
                kept += 1

    print(f"\nWrote {len(records)} clean USPTO evolution pairs to {out}")
    return records


def load_uspto_pairs(
    jsonl_path: str | Path = "data/raw/uspto_pairs.jsonl",
) -> tuple[list[RawCaseRecord], dict[str, int]]:
    """Load pre-built USPTO evolution pairs into unified RawCaseRecord objects."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"USPTO cache not found at {path}. Run `python src/adapters/uspto_loader.py --build` first."
        )

    records: list[RawCaseRecord] = []
    skipped = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(RawCaseRecord.model_validate_json(line))
            except Exception:
                skipped["invalid_record"] += 1

    return records, dict(skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or inspect USPTO patent evolution pairs from bulk zips.")
    parser.add_argument("--build", action="store_true", help="Stream bulk .zip files and build uspto_pairs.jsonl.")
    parser.add_argument("--raw-dir", type=str, default="data/raw/uspto", help="Directory containing g_*.zip files.")
    parser.add_argument("--pairs-per-subclass", type=int, default=80, help="Target pairs per CPC subclass.")
    parser.add_argument("--subclasses", nargs="+", default=None, help="Optional CPC subclasses list.")
    args = parser.parse_args()

    if args.build:
        build_uspto_pairs_from_bulk(
            raw_dir=args.raw_dir,
            subclasses=args.subclasses,
            pairs_per_subclass=args.pairs_per_subclass,
        )

    records, skipped = load_uspto_pairs()
    print(f"Total cached USPTO records: {len(records)} | Skipped: {skipped}")
    if records:
        print("Domain breakdown:", Counter(r.domain for r in records).most_common())
        print(f"Sample: {records[0].case_id} | {records[0].domain}")