from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL export not found at {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stratified_sample(
    records: list[dict[str, Any]], n_graphs: int, seed: int
) -> list[dict[str, Any]]:
    """Sample n_graphs proportionally across CPC subclasses."""
    rng = random.Random(seed)
    if n_graphs >= len(records):
        sampled = list(records)
        rng.shuffle(sampled)
        return sampled

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        cpc = (rec.get("domain") or "Unknown").split(" ")[0]
        buckets[cpc].append(rec)

    for cpc in buckets:
        rng.shuffle(buckets[cpc])

    total = len(records)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    # Proportional allocation per stratum
    for cpc in sorted(buckets.keys()):
        items = buckets[cpc]
        quota = max(1, round(n_graphs * len(items) / total))
        quota = min(quota, len(items))
        selected.extend(items[:quota])
        leftovers.extend(items[quota:])

    # Adjust to exact n_graphs
    if len(selected) > n_graphs:
        rng.shuffle(selected)
        selected = selected[:n_graphs]
    elif len(selected) < n_graphs:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n_graphs - len(selected)])

    return selected


def export_audit_csv(
    jsonl_path: Path,
    output_csv: Path,
    n_graphs: int = 50,
    seed: int = 42,
    max_after_chars: int = 1500,
) -> None:
    records = _load_jsonl(jsonl_path)
    sampled = _stratified_sample(records, n_graphs=n_graphs, seed=seed)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "source",
        "domain",
        "node_id",
        "label_name",
        "component_type",
        "role_description",
        "llm_rigidity_label",
        "llm_modification_type",
        "llm_confidence",
        "evidence_quote",
        "after_text_excerpt",
        "human_node_valid",      # Fill: 1 (valid component) or 0 (hallucinated/junk)
        "human_rigidity_label",  # Fill: 0 (Flexible/Modified) or 1 (Rigid/Untouched)
        "notes",                 # Optional reviewer notes
    ]

    total_rows = 0
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in sampled:
            after_excerpt = (rec.get("after_text") or "").replace("\n", " ")[:max_after_chars]
            for node in rec.get("nodes", []):
                writer.writerow(
                    {
                        "case_id": rec.get("case_id", ""),
                        "source": rec.get("source", ""),
                        "domain": rec.get("domain", ""),
                        "node_id": node.get("node_id", ""),
                        "label_name": node.get("label_name", ""),
                        "component_type": node.get("component_type", ""),
                        "role_description": node.get("role_description", ""),
                        "llm_rigidity_label": node.get("rigidity_label", ""),
                        "llm_modification_type": node.get("modification_type", ""),
                        "llm_confidence": f"{float(node.get('confidence_score', 0.0)):.2f}",
                        "evidence_quote": node.get("evidence_quote", ""),
                        "after_text_excerpt": after_excerpt,
                        "human_node_valid": "",
                        "human_rigidity_label": "",
                        "notes": "",
                    }
                )
                total_rows += 1

    print(f"Exported {len(sampled)} graphs ({total_rows} nodes) -> {output_csv}")
    print("Fill in `human_node_valid` (1/0) and `human_rigidity_label` (0/1), then run:")
    print(f"  python src/audit_tool.py score --csv {output_csv}")


def _cohens_kappa(y_true: list[int], y_pred: list[int]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    p_o = agree / n

    p_true_1 = sum(y_true) / n
    p_pred_1 = sum(y_pred) / n
    p_e = (p_true_1 * p_pred_1) + ((1.0 - p_true_1) * (1.0 - p_pred_1))

    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _compute_metrics(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Compute accuracy, Cohen's Kappa, and confusion counts for (human, llm) pairs."""
    if not pairs:
        return {"n": 0, "accuracy": None, "cohens_kappa": None}
    y_human = [p[0] for p in pairs]
    y_llm = [p[1] for p in pairs]
    acc = sum(1 for h, l in pairs if h == l) / len(pairs)
    kappa = _cohens_kappa(y_human, y_llm)
    return {
        "n": len(pairs),
        "accuracy": round(acc, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion": {
            "both_flexible_0": sum(1 for h, l in pairs if h == 0 and l == 0),
            "both_rigid_1": sum(1 for h, l in pairs if h == 1 and l == 1),
            "human_0_llm_1": sum(1 for h, l in pairs if h == 0 and l == 1),
            "human_1_llm_0": sum(1 for h, l in pairs if h == 1 and l == 0),
        },
    }


def score_audit_csv(
    csv_path: Path, report_path: Path, min_conf: float = 0.70
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found at {csv_path}")

    valid_flags: list[int] = []
    all_pairs: list[tuple[int, int]] = []
    masked_pairs: list[tuple[int, int]] = []
    buckets: dict[str, list[tuple[int, int]]] = {
        "<0.70": [],
        "0.70-0.84": [],
        ">=0.85": [],
    }
    audited_cases: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw_valid = (row.get("human_node_valid") or "").strip()
            raw_human = (row.get("human_rigidity_label") or "").strip()

            if raw_valid in ("0", "1"):
                valid_flags.append(int(raw_valid))
                audited_cases.add(row["case_id"])

            if raw_human in ("0", "1"):
                h_lbl = int(raw_human)
                l_lbl = int(row["llm_rigidity_label"])
                conf = float(row["llm_confidence"])
                audited_cases.add(row["case_id"])

                all_pairs.append((h_lbl, l_lbl))
                if conf >= min_conf:
                    masked_pairs.append((h_lbl, l_lbl))

                if conf < 0.70:
                    buckets["<0.70"].append((h_lbl, l_lbl))
                elif conf < 0.85:
                    buckets["0.70-0.84"].append((h_lbl, l_lbl))
                else:
                    buckets[">=0.85"].append((h_lbl, l_lbl))

    extraction_precision = (
        round(sum(valid_flags) / len(valid_flags), 4) if valid_flags else None
    )

    report = {
        "audit_csv": str(csv_path),
        "audited_graphs": len(audited_cases),
        "extraction_precision": {
            "n_nodes": len(valid_flags),
            "precision": extraction_precision,
        },
        "all_nodes": _compute_metrics(all_pairs),
        f"high_conf_masked (>={min_conf:.2f})": _compute_metrics(masked_pairs),
        "confidence_calibration": {
            k: _compute_metrics(v) for k, v in buckets.items()
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Human Audit Report ---")
    print(f"Audited graphs: {report['audited_graphs']}")
    if extraction_precision is not None:
        print(f"Extraction Precision: {extraction_precision * 100:.2f}% ({sum(valid_flags)}/{len(valid_flags)} nodes)")
    else:
        print("Extraction Precision: N/A (no `human_node_valid` entries filled)")

    all_m = report["all_nodes"]
    mask_m = report[f"high_conf_masked (>={min_conf:.2f})"]
    if all_m["n"] > 0:
        print(f"All Nodes        (n={all_m['n']}): Acc={all_m['accuracy']*100:.2f}% | Cohen's Kappa={all_m['cohens_kappa']:.4f}")
        if mask_m["n"] > 0:
            print(f"Masked (>={min_conf:.2f}) (n={mask_m['n']}): Acc={mask_m['accuracy']*100:.2f}% | Cohen's Kappa={mask_m['cohens_kappa']:.4f}")
        print("\nConfidence Calibration:")
        for bname, bm in report["confidence_calibration"].items():
            if bm["n"] > 0:
                print(f"  conf {bname:<10} (n={bm['n']:<3}): Acc={bm['accuracy']*100:.2f}% | Kappa={bm['cohens_kappa']:.4f}")
    else:
        print("No `human_rigidity_label` entries filled yet.")
    print(f"\nSaved JSON report -> {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human audit sampling & scoring tool for TRIZ Rigidity DB.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="Export stratified random sample to CSV.")
    p_exp.add_argument("--jsonl", type=Path, default=Path("data/exports/triz_rigidity.jsonl"))
    p_exp.add_argument("--out", type=Path, default=Path("data/audit/audit_sample.csv"))
    p_exp.add_argument("--n-graphs", type=int, default=50)
    p_exp.add_argument("--seed", type=int, default=42)

    p_sc = sub.add_parser("score", help="Score a completed audit CSV.")
    p_sc.add_argument("--csv", type=Path, default=Path("data/audit/audit_sample.csv"))
    p_sc.add_argument("--report", type=Path, default=Path("data/audit/audit_report.json"))
    p_sc.add_argument("--min-conf", type=float, default=0.70)

    args = parser.parse_args()
    if args.cmd == "export":
        export_audit_csv(args.jsonl, args.out, n_graphs=args.n_graphs, seed=args.seed)
    elif args.cmd == "score":
        score_audit_csv(args.csv, args.report, min_conf=args.min_conf)


if __name__ == "__main__":
    main()