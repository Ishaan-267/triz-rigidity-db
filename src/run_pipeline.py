from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adapters.trizbench_loader import load_trizbench_patents
from adapters.uspto_loader import load_uspto_pairs
from db_manager import DBManager
from extractor import TRIZExtractor
from schemas import RawCaseRecord
from validator import TRIZValidator


def _append_rejected(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_rejected_case_ids(path: Path) -> set[str]:
    """Return case_ids that failed validation (excludes transient API errors so they retry)."""
    rejected: set[str] = set()
    if not path.exists():
        return rejected
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if "reasons" in row and row.get("case_id"):
                    rejected.add(row["case_id"])
            except json.JSONDecodeError:
                continue
    return rejected


def _interleave_by_domain(records: list[RawCaseRecord]) -> list[RawCaseRecord]:
    """Round-robin interleave records across domains so partial batches stay balanced."""
    buckets: dict[str, deque[RawCaseRecord]] = defaultdict(deque)
    for rec in records:
        buckets[rec.domain].append(rec)

    domain_order = sorted(buckets.keys())
    interleaved: list[RawCaseRecord] = []
    while any(buckets[d] for d in domain_order):
        for d in domain_order:
            if buckets[d]:
                interleaved.append(buckets[d].popleft())
    return interleaved


async def process_one(
    record: RawCaseRecord,
    extractor: TRIZExtractor,
    validator: TRIZValidator,
    db: DBManager,
    rejected_log: Path,
    use_cache: bool,
) -> tuple[str, str]:
    """Process a single record end-to-end. Returns (status, detail)."""
    try:
        # 1. Pass 1: Extract Function Model Graph from before_text only
        raw_graph = await extractor.extract_pass1(record, use_cache=use_cache)

        # 2. Canonicalize synonymous nodes BEFORE Pass 2
        canon_graph, merges = validator.canonicalize_graph(raw_graph)

        # 3. Pass 2: Label canonical nodes against after_text
        labels = await extractor.extract_pass2(record, canon_graph, use_cache=use_cache)

        # 4. Validate topology, label variance, confidence, and evidence quotes
        val = validator.validate_extraction(record, canon_graph, labels)
        if not val.is_valid:
            _append_rejected(
                rejected_log,
                {
                    "case_id": record.case_id,
                    "source": record.source,
                    "domain": record.domain,
                    "merges": merges,
                    "reasons": val.reasons,
                },
            )
            return "rejected", "; ".join(val.reasons)

        # 5. Idempotent SQLite insert
        inserted = db.insert_graph(record, canon_graph, labels)
        rigid = sum(l.rigidity_label for l in labels.labels)
        flex = len(labels.labels) - rigid
        status = "inserted" if inserted else "skipped_existing"
        return (
            status,
            f"{len(canon_graph.nodes)}N/{len(canon_graph.edges)}E "
            f"(R:{rigid}/F:{flex}, merges:{len(merges)})",
        )

    except Exception as exc:
        _append_rejected(
            rejected_log,
            {
                "case_id": record.case_id,
                "source": record.source,
                "domain": record.domain,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return "error", f"{type(exc).__name__}: {exc}"


async def run_pipeline(
    source: str,
    limit: int | None,
    concurrency: int,
    db_path: str,
    use_cache: bool,
    retry_rejected: bool,
) -> None:
    t0 = time.time()
    db = DBManager(db_path)
    extractor = TRIZExtractor(max_concurrency=concurrency)
    validator = TRIZValidator()

    # Pre-load sentence-transformer once before async batch starts
    validator._get_embedder()

    records: list[RawCaseRecord] = []
    load_skipped: dict[str, int] = {}

    if source in ("trizbench", "all"):
        tb_recs, tb_skip = load_trizbench_patents()
        records.extend(tb_recs)
        load_skipped.update({f"tb_{k}": v for k, v in tb_skip.items()})

    if source in ("uspto", "all"):
        us_recs, us_skip = load_uspto_pairs()
        records.extend(us_recs)
        load_skipped.update({f"uspto_{k}": v for k, v in us_skip.items()})

    # Round-robin across domains before slicing with --limit
    records = _interleave_by_domain(records)

    rejected_log = Path("data/rejected_cases.jsonl")
    already_done = db.get_processed_case_ids()
    already_rejected = set() if retry_rejected else _load_rejected_case_ids(rejected_log)

    pending = [
        r for r in records
        if r.case_id not in already_done and r.case_id not in already_rejected
    ]

    if limit is not None:
        pending = pending[:limit]

    print(
        f"Source={source} | Loaded {len(records)} records (skipped at load: {load_skipped}) | "
        f"In DB: {len(already_done)} | Skipped rejected: {len(already_rejected)} | "
        f"To process now: {len(pending)}"
    )
    if not pending:
        print("Nothing new to process.")
        stats = db.generate_summary_stats()
        print(f"Current DB totals: {stats['totals']} | Balance: {stats['label_balance']}")
        return

    domain_preview = Counter(r.domain.split(" ")[0] for r in pending)
    print(f"Batch domain spread: {dict(sorted(domain_preview.items()))}")

    counts = Counter()

    async def _worker(idx: int, rec: RawCaseRecord) -> None:
        status, detail = await process_one(
            rec, extractor, validator, db, rejected_log, use_cache
        )
        counts[status] += 1
        print(
            f"[{idx}/{len(pending)}] [{rec.domain.split(' ')[0]}] "
            f"{rec.case_id:<28} -> {status.upper():<10} | {detail}"
        )

    await asyncio.gather(*(_worker(i + 1, rec) for i, rec in enumerate(pending)))

    # Refresh exports & dataset card
    jsonl_path = db.export_jsonl()
    stats = db.generate_summary_stats()
    elapsed = time.time() - t0

    print("\n--- Run Complete ---")
    print(f"Elapsed: {elapsed:.1f}s | Run counts: {dict(counts)}")
    print(f"DB totals: {stats['totals']}")
    print(f"Label balance: {stats['label_balance']}")
    print(f"Exported JSONL: {jsonl_path}")
    if counts["rejected"] or counts["error"]:
        print(f"Inspect rejected/error details in: {rejected_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TRIZ Rigidity DB extraction pipeline.")
    parser.add_argument(
        "--source",
        choices=["trizbench", "uspto", "all"],
        default="trizbench",
        help="Which ingestion adapter(s) to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max new records to process.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent LLM calls (keep 1 for Gemini free tier).",
    )
    parser.add_argument("--db", type=str, default="data/triz_rigidity.db", help="SQLite DB path.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached Pass 1/2 JSON files.")
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="Re-evaluate previously rejected cases (uses disk cache if available).",
    )
    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            source=args.source,
            limit=args.limit,
            concurrency=args.concurrency,
            db_path=args.db,
            use_cache=not args.no_cache,
            retry_rejected=args.retry_rejected,
        )
    )


if __name__ == "__main__":
    main()