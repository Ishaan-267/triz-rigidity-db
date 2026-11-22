from __future__ import annotations

import argparse
import csv
import json
import random
import textwrap
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


def _load_case_ids_from_csv(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Exclude CSV not found at {path}")
    ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = (row.get("case_id") or "").strip()
            if cid:
                ids.add(cid)
    return ids


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

    for cpc in sorted(buckets.keys()):
        items = buckets[cpc]
        quota = max(1, round(n_graphs * len(items) / total))
        quota = min(quota, len(items))
        selected.extend(items[:quota])
        leftovers.extend(items[quota:])

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
    n_graphs: int = 25,
    seed: int = 42,
    max_text_chars: int = 3000,
    exclude_csv: Path | None = None,
) -> None:
    records = _load_jsonl(jsonl_path)
    if exclude_csv is not None:
        excluded = _load_case_ids_from_csv(exclude_csv)
        records = [r for r in records if r.get("case_id") not in excluded]
        print(f"Excluded {len(excluded)} existing case_ids from {exclude_csv} ({len(records)} remaining).")

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
        "before_text_excerpt",
        "after_text_excerpt",
        "human_node_valid",      # 1 (valid component) or 0 (hallucinated/junk)
        "human_rigidity_label",  # 0 (Flexible/Modified) or 1 (Rigid/Untouched)
        "notes",
    ]

    total_rows = 0
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in sampled:
            before_excerpt = " ".join((rec.get("before_text") or "").split())[:max_text_chars]
            after_excerpt = " ".join((rec.get("after_text") or "").split())[:max_text_chars]
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
                        "before_text_excerpt": before_excerpt,
                        "after_text_excerpt": after_excerpt,
                        "human_node_valid": "",
                        "human_rigidity_label": "",
                        "notes": "",
                    }
                )
                total_rows += 1

    print(f"Exported {len(sampled)} graphs ({total_rows} nodes) -> {output_csv}")
    print("Start interactive labeling:")
    print(f"  python src/audit_tool.py label --csv {output_csv}")


def _save_csv(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_row_done(row: dict[str, str]) -> bool:
    v = (row.get("human_node_valid") or "").strip()
    h = (row.get("human_rigidity_label") or "").strip()
    return v == "0" or (v == "1" and h in ("0", "1"))


def _format_patent_text(text: str, width: int = 88) -> str:
    if not text:
        return "(Not available — re-run `export` to include before_text)"
    # Split ABSTRACT and CLAIM 1 onto separate paragraphs for readability
    formatted = text.replace(" CLAIM 1:", "\n\nCLAIM 1:")
    paragraphs = [p.strip() for p in formatted.split("\n\n") if p.strip()]
    return "\n\n".join(textwrap.fill(p, width=width) for p in paragraphs)


def _print_patent_context(
    c_num: int, total_cases: int, cid: str, first: dict[str, str], n_nodes: int
) -> None:
    print("\n" + "=" * 88)
    print(
        f"PATENT [{c_num}/{total_cases}]  {cid}  |  {first['domain']}  |  {n_nodes} nodes"
    )
    print("=" * 88)
    print("BEFORE TEXT (Original Patent / Baseline):")
    print(_format_patent_text(first.get("before_text_excerpt", "")))
    print("-" * 88)
    print("AFTER TEXT (Improved Patent / Solution):")
    print(_format_patent_text(first.get("after_text_excerpt", "")))
    print("-" * 88)


def interactive_label_csv(csv_path: Path, show_llm: bool = False) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Audit CSV not found at {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    cases_order: list[str] = []
    case_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        cid = row["case_id"]
        if cid not in case_indices:
            cases_order.append(cid)
        case_indices[cid].append(idx)

    total_cases = len(cases_order)
    done_cases = sum(
        1 for cid in cases_order if all(_is_row_done(rows[i]) for i in case_indices[cid])
    )
    print(f"\nLoaded {csv_path} | Progress: {done_cases}/{total_cases} patents completed.")
    print("Rule of thumb: Did AFTER modify, replace, reshape, or remove this part?")
    print("  [0]=Flexible (changed in AFTER)  [1]=Rigid (untouched / external constraint)  [j]=Junk")
    print("  [r]=Re-read texts  [n]=Add note  [b]=Back 1 node  [s]=Skip node  [q]=Save & quit")

    try:
        for c_num, cid in enumerate(cases_order, start=1):
            idxs = case_indices[cid]
            if all(_is_row_done(rows[i]) for i in idxs):
                continue

            first = rows[idxs[0]]
            _print_patent_context(c_num, total_cases, cid, first, len(idxs))

            pos = 0
            while pos < len(idxs):
                row = rows[idxs[pos]]
                if _is_row_done(row):
                    pos += 1
                    continue

                print(
                    f"\n  Node {pos + 1}/{len(idxs)}: {row['label_name']} "
                    f"[{row['component_type']}]"
                )
                print(
                    textwrap.fill(
                        f"Role: {row['role_description']}",
                        width=88,
                        initial_indent="    ",
                        subsequent_indent="    ",
                    )
                )
                if show_llm:
                    print(
                        f"    LLM -> label={row['llm_rigidity_label']} "
                        f"({row['llm_modification_type']}, conf={row['llm_confidence']})"
                    )
                    print(
                        textwrap.fill(
                            f"Quote: {row['evidence_quote']}",
                            width=88,
                            initial_indent="    ",
                            subsequent_indent="    ",
                        )
                    )

                while True:
                    ans = input(
                        "  [0=Flex/Changed | 1=Rigid/Untouched | j=Junk | r=Re-read | n=Note | b=Back | s=Skip | q=Quit]: "
                    ).strip().lower()
                    if ans == "0":
                        row["human_node_valid"] = "1"
                        row["human_rigidity_label"] = "0"
                        pos += 1
                        break
                    elif ans == "1":
                        row["human_node_valid"] = "1"
                        row["human_rigidity_label"] = "1"
                        pos += 1
                        break
                    elif ans == "j":
                        row["human_node_valid"] = "0"
                        row["human_rigidity_label"] = ""
                        pos += 1
                        break
                    elif ans == "r":
                        _print_patent_context(c_num, total_cases, cid, first, len(idxs))
                        print(
                            f"\n  Node {pos + 1}/{len(idxs)}: {row['label_name']} "
                            f"[{row['component_type']}]"
                        )
                        print(f"    Role: {row['role_description']}")
                        continue
                    elif ans == "n":
                        note = input("    Note: ").strip()
                        row["notes"] = note
                        continue
                    elif ans == "b":
                        if pos > 0:
                            pos -= 1
                            rows[idxs[pos]]["human_node_valid"] = ""
                            rows[idxs[pos]]["human_rigidity_label"] = ""
                            break
                        else:
                            print("    Already at first node of this patent.")
                    elif ans == "s":
                        pos += 1
                        break
                    elif ans == "q":
                        _save_csv(csv_path, fieldnames, rows)
                        print(f"\nProgress saved -> {csv_path}. Resume anytime with the same command.")
                        return
                    else:
                        print("    Invalid key. Use 0, 1, j, r, n, b, s, or q.")

            _save_csv(csv_path, fieldnames, rows)
            print(f"\n[Saved {cid}]")

    except KeyboardInterrupt:
        _save_csv(csv_path, fieldnames, rows)
        print(f"\n\nInterrupted — progress saved -> {csv_path}.")
        return

    print(f"\nAll patents processed! Run scoring with:\n  python src/audit_tool.py score --csv {csv_path}")


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


def _bootstrap_ci(
    values: list[Any], metric_fn, n_boot: int = 2000, seed: int = 42
) -> list[float] | None:
    n = len(values)
    if n < 2:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        boot = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(metric_fn(boot))
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot)]
    return [round(lo, 4), round(hi, 4)]


def _compute_metrics(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Compute accuracy, Cohen's Kappa, 95% bootstrap CIs, and confusion counts."""
    if not pairs:
        return {
            "n": 0,
            "accuracy": None,
            "accuracy_ci95": None,
            "cohens_kappa": None,
            "kappa_ci95": None,
        }

    acc_fn = lambda ps: sum(1 for h, l in ps if h == l) / len(ps)
    kap_fn = lambda ps: _cohens_kappa([p[0] for p in ps], [p[1] for p in ps])

    acc = acc_fn(pairs)
    kappa = kap_fn(pairs)
    return {
        "n": len(pairs),
        "accuracy": round(acc, 4),
        "accuracy_ci95": _bootstrap_ci(pairs, acc_fn),
        "cohens_kappa": round(kappa, 4),
        "kappa_ci95": _bootstrap_ci(pairs, kap_fn),
        "confusion": {
            "both_flexible_0": sum(1 for h, l in pairs if h == 0 and l == 0),
            "both_rigid_1": sum(1 for h, l in pairs if h == 1 and l == 1),
            "human_0_llm_1": sum(1 for h, l in pairs if h == 0 and l == 1),
            "human_1_llm_0": sum(1 for h, l in pairs if h == 1 and l == 0),
        },
    }


def _fmt_ci(ci: list[float] | None, pct: bool = False) -> str:
    if not ci:
        return ""
    if pct:
        return f" [95% CI: {ci[0]*100:.1f}%-{ci[1]*100:.1f}%]"
    return f" [95% CI: {ci[0]:.3f}-{ci[1]:.3f}]"


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

    prec_fn = lambda xs: sum(xs) / len(xs)
    extraction_precision = round(prec_fn(valid_flags), 4) if valid_flags else None
    extraction_ci95 = _bootstrap_ci(valid_flags, prec_fn) if valid_flags else None

    report = {
        "audit_csv": str(csv_path),
        "audited_graphs": len(audited_cases),
        "audited_case_ids": sorted(audited_cases),
        "extraction_precision": {
            "n_nodes": len(valid_flags),
            "precision": extraction_precision,
            "precision_ci95": extraction_ci95,
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
        print(
            f"Extraction Precision: {extraction_precision * 100:.2f}% "
            f"({sum(valid_flags)}/{len(valid_flags)} nodes){_fmt_ci(extraction_ci95, pct=True)}"
        )
    else:
        print("Extraction Precision: N/A (no `human_node_valid` entries filled)")

    all_m = report["all_nodes"]
    mask_m = report[f"high_conf_masked (>={min_conf:.2f})"]
    if all_m["n"] > 0:
        print(
            f"All Nodes        (n={all_m['n']}): "
            f"Acc={all_m['accuracy']*100:.2f}%{_fmt_ci(all_m['accuracy_ci95'], pct=True)} | "
            f"Kappa={all_m['cohens_kappa']:.4f}{_fmt_ci(all_m['kappa_ci95'])}"
        )
        if mask_m["n"] > 0:
            print(
                f"Masked (>={min_conf:.2f}) (n={mask_m['n']}): "
                f"Acc={mask_m['accuracy']*100:.2f}%{_fmt_ci(mask_m['accuracy_ci95'], pct=True)} | "
                f"Kappa={mask_m['cohens_kappa']:.4f}{_fmt_ci(mask_m['kappa_ci95'])}"
            )
        print("\nConfidence Calibration:")
        for bname, bm in report["confidence_calibration"].items():
            if bm["n"] > 0:
                print(
                    f"  conf {bname:<10} (n={bm['n']:<3}): "
                    f"Acc={bm['accuracy']*100:.2f}% | "
                    f"Kappa={bm['cohens_kappa']:.4f}{_fmt_ci(bm['kappa_ci95'])}"
                )
    else:
        print("No `human_rigidity_label` entries filled yet.")
    print(f"\nSaved JSON report -> {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human audit sampling, interactive labeling & scoring tool."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="Export stratified random sample to CSV.")
    p_exp.add_argument("--jsonl", type=Path, default=Path("data/exports/triz_rigidity.jsonl"))
    p_exp.add_argument("--out", type=Path, default=Path("data/audit/audit_sample.csv"))
    p_exp.add_argument("--n-graphs", type=int, default=25)
    p_exp.add_argument("--seed", type=int, default=42)
    p_exp.add_argument(
        "--exclude-csv",
        type=Path,
        default=None,
        help="Optional existing audit CSV whose case_ids should be excluded.",
    )

    p_lbl = sub.add_parser("label", help="Interactively label an exported audit CSV patent-by-patent.")
    p_lbl.add_argument("--csv", type=Path, default=Path("data/audit/audit_sample.csv"))
    p_lbl.add_argument(
        "--show-llm",
        action="store_true",
        help="Show LLM label/quote while labeling (omit for blind thesis audit).",
    )

    p_sc = sub.add_parser("score", help="Score a completed audit CSV with 95%% bootstrap CIs.")
    p_sc.add_argument("--csv", type=Path, default=Path("data/audit/audit_sample.csv"))
    p_sc.add_argument("--report", type=Path, default=Path("data/audit/audit_report.json"))
    p_sc.add_argument("--min-conf", type=float, default=0.70)

    args = parser.parse_args()
    if args.cmd == "export":
        export_audit_csv(
            args.jsonl,
            args.out,
            n_graphs=args.n_graphs,
            seed=args.seed,
            exclude_csv=args.exclude_csv,
        )
    elif args.cmd == "label":
        interactive_label_csv(args.csv, show_llm=args.show_llm)
    elif args.cmd == "score":
        score_audit_csv(args.csv, args.report, min_conf=args.min_conf)


if __name__ == "__main__":
    main()