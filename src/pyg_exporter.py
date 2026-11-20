from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

COMPONENT_TYPES = ["System_Component", "Target_Object", "Super_System"]
INTERACTION_QUALITIES = ["Useful_Normal", "Insufficient", "Excessive", "Harmful"]
FUNCTION_CATEGORIES = ["Productive", "Supporting", "Transport", "Corrective"]
PROBLEMATIC_QUALITIES = {"Harmful", "Excessive", "Insufficient"}


def _one_hot(value: str, vocab: list[str]) -> list[float]:
    vec = [0.0] * len(vocab)
    if value in vocab:
        vec[vocab.index(value)] = 1.0
    return vec


def _extract_node_fields(node: dict[str, Any], label_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Handle both flat (merged) and nested (separate labels list) JSONL node formats."""
    nid = node["node_id"]
    lbl = label_lookup.get(nid, {})
    return {
        "node_id": nid,
        "label_name": node.get("label_name", nid),
        "component_type": node.get("component_type", "System_Component"),
        "is_standard_or_regulated": bool(node.get("is_standard_or_regulated", False)),
        "role_description": node.get("role_description", ""),
        "rigidity_label": int(node.get("rigidity_label", lbl.get("rigidity_label", 1))),
        "confidence_score": float(node.get("confidence_score", lbl.get("confidence_score", 0.0))),
    }


def load_jsonl_graphs(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL export not found at {jsonl_path}")
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def export_pyg_dataset(
    jsonl_path: str | Path = "data/exports/triz_rigidity.jsonl",
    output_path: str | Path = "data/exports/triz_rigidity_pyg_v1.pt",
    embedding_model: str = "all-MiniLM-L6-v2",
    min_confidence_mask: float = 0.70,
    version: str = "v1",
) -> Path:
    records = load_jsonl_graphs(Path(jsonl_path))
    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    print(f"Loaded {len(records)} graphs from {jsonl_path}")
    embedder = SentenceTransformer(embedding_model)

    # Normalize record structure & collect all texts for fast single-pass batch encoding
    parsed_graphs: list[dict[str, Any]] = []
    all_node_texts: list[str] = []
    all_edge_texts: list[str] = []

    for rec in records:
        raw_nodes = rec.get("nodes") or rec.get("graph", {}).get("nodes") or []
        raw_edges = rec.get("edges") or rec.get("graph", {}).get("edges") or []
        raw_labels = rec.get("labels") or []
        label_lookup = {l["node_id"]: l for l in raw_labels if "node_id" in l}

        nodes = [_extract_node_fields(n, label_lookup) for n in raw_nodes]
        node_idx = {n["node_id"]: i for i, n in enumerate(nodes)}

        edges = [
            e for e in raw_edges
            if (e.get("source_node_id") or e.get("source")) in node_idx
            and (e.get("target_node_id") or e.get("target")) in node_idx
        ]
        if not nodes or not edges:
            continue

        node_start = len(all_node_texts)
        for n in nodes:
            all_node_texts.append(
                f"{n['label_name']} ({n['component_type']}): {n['role_description']}"
            )

        edge_start = len(all_edge_texts)
        for e in edges:
            all_edge_texts.append(e.get("action_verb", "interacts with"))

        parsed_graphs.append(
            {
                "case_id": rec.get("case_id", ""),
                "source": rec.get("source", "unknown"),
                "domain": rec.get("domain", "Unknown"),
                "nodes": nodes,
                "edges": edges,
                "node_idx": node_idx,
                "node_slice": (node_start, len(all_node_texts)),
                "edge_slice": (edge_start, len(all_edge_texts)),
            }
        )

    print(
        f"Encoding {len(all_node_texts)} node descriptions and "
        f"{len(all_edge_texts)} edge verbs with {embedding_model}..."
    )
    node_embs = embedder.encode(
        all_node_texts, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False
    ).cpu().float()
    edge_embs = embedder.encode(
        all_edge_texts, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False
    ).cpu().float()

    emb_dim = int(node_embs.shape[1])
    pyg_graphs: list[Data] = []
    domain_counts = Counter()
    total_nodes = 0
    total_masked_nodes = 0
    total_flex = 0
    total_rigid = 0

    for g in parsed_graphs:
        nodes = g["nodes"]
        edges = g["edges"]
        node_idx = g["node_idx"]
        num_nodes = len(nodes)
        denom = max(1, num_nodes - 1)

        # Compute leakage-free BEFORE structural features per node
        in_deg = [0] * num_nodes
        out_deg = [0] * num_nodes
        inc_total = [0] * num_nodes
        inc_problem = [0] * num_nodes

        src_list: list[int] = []
        tgt_list: list[int] = []
        edge_cat_feats: list[list[float]] = []

        for e in edges:
            s = node_idx[e.get("source_node_id") or e["source"]]
            t = node_idx[e.get("target_node_id") or e["target"]]
            qual = e.get("interaction_quality", "Useful_Normal")
            cat = e.get("function_category", "Productive")

            src_list.append(s)
            tgt_list.append(t)
            out_deg[s] += 1
            in_deg[t] += 1
            inc_total[s] += 1
            inc_total[t] += 1
            if qual in PROBLEMATIC_QUALITIES:
                inc_problem[s] += 1
                inc_problem[t] += 1

            edge_cat_feats.append(
                _one_hot(qual, INTERACTION_QUALITIES) + _one_hot(cat, FUNCTION_CATEGORIES)
            )

        node_struct_feats: list[list[float]] = []
        y_list: list[int] = []
        conf_list: list[float] = []

        for i, n in enumerate(nodes):
            prob_ratio = inc_problem[i] / inc_total[i] if inc_total[i] > 0 else 0.0
            node_struct_feats.append(
                _one_hot(n["component_type"], COMPONENT_TYPES)
                + [1.0 if n["is_standard_or_regulated"] else 0.0]
                + [in_deg[i] / denom, out_deg[i] / denom, prob_ratio]
            )
            y_list.append(n["rigidity_label"])
            conf_list.append(n["confidence_score"])

        ns, ne = g["node_slice"]
        es, ee = g["edge_slice"]

        x = torch.cat(
            [node_embs[ns:ne], torch.tensor(node_struct_feats, dtype=torch.float32)], dim=1
        )
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        edge_attr = torch.cat(
            [edge_embs[es:ee], torch.tensor(edge_cat_feats, dtype=torch.float32)], dim=1
        )
        y = torch.tensor(y_list, dtype=torch.long)
        confidence = torch.tensor(conf_list, dtype=torch.float32)
        label_mask = confidence >= min_confidence_mask

        cpc_subclass = g["domain"].split(" ")[0] if g["domain"] else "UNK"
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            confidence=confidence,
            label_mask=label_mask,
            case_id=g["case_id"],
            source=g["source"],
            domain=g["domain"],
            cpc_subclass=cpc_subclass,
            node_ids=[n["node_id"] for n in nodes],
        )
        pyg_graphs.append(data)

        domain_counts[cpc_subclass] += 1
        total_nodes += num_nodes
        total_masked_nodes += int(label_mask.sum().item())
        total_rigid += int((y == 1).sum().item())
        total_flex += int((y == 0).sum().item())

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model,
        "min_confidence_mask": min_confidence_mask,
        "feature_spec": {
            "node_dim": int(pyg_graphs[0].x.shape[1]),
            "node_layout": f"{emb_dim} MiniLM + 3 component_type + 1 is_regulated + 3 topology (in_deg, out_deg, problem_edge_ratio)",
            "edge_dim": int(pyg_graphs[0].edge_attr.shape[1]),
            "edge_layout": f"{emb_dim} MiniLM action_verb + 4 interaction_quality + 4 function_category",
        },
        "stats": {
            "num_graphs": len(pyg_graphs),
            "num_nodes": total_nodes,
            "num_masked_train_nodes": total_masked_nodes,
            "flexible_0_count": total_flex,
            "rigid_1_count": total_rigid,
            "domain_counts": dict(sorted(domain_counts.items())),
        },
        "graphs": pyg_graphs,
    }
    torch.save(payload, out_path)
    print(f"Saved PyG dataset -> {out_path}")
    return out_path


def verify_export(pt_path: Path) -> None:
    """Reload the .pt artifact and verify tensor integrity."""
    bundle = torch.load(pt_path, weights_only=False)
    graphs: list[Data] = bundle["graphs"]

    for g in graphs:
        assert not torch.isnan(g.x).any() and not torch.isinf(g.x).any(), f"NaN/Inf in x: {g.case_id}"
        assert not torch.isnan(g.edge_attr).any(), f"NaN in edge_attr: {g.case_id}"
        assert g.edge_index.max().item() < g.num_nodes, f"Edge index OOB: {g.case_id}"
        assert g.y.shape[0] == g.num_nodes == g.label_mask.shape[0]

    sample = graphs[0]
    print("\n--- Verification Passed ---")
    print(f"Graphs: {bundle['stats']['num_graphs']} | Nodes: {bundle['stats']['num_nodes']} "
          f"(high-conf masked: {bundle['stats']['num_masked_train_nodes']})")
    print(f"Node feature dim: {bundle['feature_spec']['node_dim']} ({bundle['feature_spec']['node_layout']})")
    print(f"Edge feature dim: {bundle['feature_spec']['edge_dim']} ({bundle['feature_spec']['edge_layout']})")
    print(f"Sample graph: {sample.case_id} | {sample.domain} | x={tuple(sample.x.shape)} "
          f"| edge_index={tuple(sample.edge_index.shape)} | edge_attr={tuple(sample.edge_attr.shape)}")
    print(f"CPC spread: {bundle['stats']['domain_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TRIZ Rigidity DB to PyTorch Geometric (.pt).")
    parser.add_argument("--jsonl", type=str, default="data/exports/triz_rigidity.jsonl")
    parser.add_argument("--out", type=str, default="data/exports/triz_rigidity_pyg_v1.pt")
    parser.add_argument("--min-conf", type=float, default=0.70, help="Threshold for label_mask.")
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()

    out_path = export_pyg_dataset(
        jsonl_path=args.jsonl,
        output_path=args.out,
        min_confidence_mask=args.min_conf,
        version=args.version,
    )
    verify_export(out_path)


if __name__ == "__main__":
    main()