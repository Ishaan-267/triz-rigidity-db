from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schemas import RawCaseRecord, RigidityLabelOutput, TRIZEdge, TRIZGraph, TRIZNode

# Pairs of tokens that indicate distinct physical components even with high lexical overlap
OPPOSING_TOKEN_PAIRS = [
    {"inlet", "outlet"},
    {"input", "output"},
    {"upper", "lower"},
    {"top", "bottom"},
    {"inner", "outer"},
    {"internal", "external"},
    {"primary", "secondary"},
    {"first", "second"},
    {"front", "rear"},
    {"left", "right"},
    {"hot", "cold"},
    {"high", "low"},
    {"positive", "negative"},
]


@dataclass
class ValidationResult:
    is_valid: bool
    reasons: list[str] = field(default_factory=list)
    low_confidence_nodes: list[str] = field(default_factory=list)
    ungrounded_quote_nodes: list[str] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _has_opposing_tokens(name_a: str, name_b: str) -> bool:
    tokens_a = set(_normalize_text(name_a).split())
    tokens_b = set(_normalize_text(name_b).split())
    for pair in OPPOSING_TOKEN_PAIRS:
        in_a = tokens_a & pair
        in_b = tokens_b & pair
        if in_a and in_b and in_a != in_b:
            return True
    return False


class TRIZValidator:
    """Handles entity canonicalization (between Pass 1 & 2) and post-Pass 2 validation."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        fuzzy_threshold: float = 85.0,
        cosine_threshold: float = 0.86,
        min_nodes: int = 4,
        min_confidence: float = 0.7,
        quote_match_threshold: float = 88.0,
    ) -> None:
        self.embedding_model_name = embedding_model
        self._embedder = None
        self.fuzzy_threshold = fuzzy_threshold
        self.cosine_threshold = cosine_threshold
        self.min_nodes = min_nodes
        self.min_confidence = min_confidence
        self.quote_match_threshold = quote_match_threshold

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def canonicalize_graph(self, graph: TRIZGraph) -> tuple[TRIZGraph, dict[str, str]]:
        """
        Deduplicate synonymous nodes using fuzzy string matching + embedding cosine similarity.
        Returns (canonical_graph, merge_map) where merge_map maps dropped_id -> kept_id.
        """
        nodes = graph.nodes
        n = len(nodes)
        if n <= 1:
            return graph, {}

        embedder = self._get_embedder()
        texts = [f"{node.label_name}: {node.role_description}" for node in nodes]
        embeddings = embedder.encode(texts, normalize_embeddings=True)
        cos_matrix = np.matmul(embeddings, embeddings.T)

        # Union-Find to group synonymous nodes
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(n):
            for j in range(i + 1, n):
                if _has_opposing_tokens(nodes[i].label_name, nodes[j].label_name):
                    continue

                fuzzy_score = fuzz.token_sort_ratio(
                    nodes[i].label_name.lower(), nodes[j].label_name.lower()
                )
                cos_score = float(cos_matrix[i, j])

                # Merge if strong lexical + semantic match, or near-identical embeddings
                if (
                    fuzzy_score >= self.fuzzy_threshold and cos_score >= 0.70
                ) or (
                    cos_score >= self.cosine_threshold and fuzzy_score >= 60.0
                ):
                    union(i, j)

        # Group indices by root
        clusters: dict[int, list[int]] = {}
        for idx in range(n):
            clusters.setdefault(find(idx), []).append(idx)

        merge_map: dict[str, str] = {}
        canonical_nodes: list[TRIZNode] = []

        for indices in clusters.values():
            # Pick the node with the shortest (cleanest) node_id as canonical representative
            rep_idx = min(indices, key=lambda idx: len(nodes[idx].node_id))
            rep = nodes[rep_idx]

            is_regulated = any(nodes[idx].is_standard_or_regulated for idx in indices)
            canonical_nodes.append(
                TRIZNode(
                    node_id=rep.node_id,
                    label_name=rep.label_name,
                    component_type=rep.component_type,
                    is_standard_or_regulated=is_regulated,
                    role_description=rep.role_description,
                )
            )
            for idx in indices:
                if idx != rep_idx:
                    merge_map[nodes[idx].node_id] = rep.node_id

        # Remap edges, drop self-loops created by merging, and deduplicate
        seen_edges: set[tuple[str, str, str]] = set()
        canonical_edges: list[TRIZEdge] = []
        for edge in graph.edges:
            src = merge_map.get(edge.source, edge.source)
            tgt = merge_map.get(edge.target, edge.target)
            if src == tgt:
                continue
            key = (src, tgt, edge.action_verb.strip().lower())
            if key in seen_edges:
                continue
            seen_edges.add(key)
            canonical_edges.append(
                TRIZEdge(
                    source=src,
                    target=tgt,
                    action_verb=edge.action_verb,
                    interaction_quality=edge.interaction_quality,
                    function_category=edge.function_category,
                )
            )

        return TRIZGraph(nodes=canonical_nodes, edges=canonical_edges), merge_map

    def _quote_is_grounded(self, quote: str, after_text: str) -> bool:
        norm_after = _normalize_text(after_text)
        # Handle ellipses where LLM joins two verbatim clauses with "..."
        segments = [s.strip() for s in re.split(r"\.\.\.|…", quote) if len(s.strip()) >= 3]
        if not segments:
            return False
        for seg in segments:
            norm_seg = _normalize_text(seg)
            if not norm_seg:
                continue
            if norm_seg not in norm_after and fuzz.partial_ratio(norm_seg, norm_after) < self.quote_match_threshold:
                return False
        return True

    def validate_extraction(
        self,
        record: RawCaseRecord,
        graph: TRIZGraph,
        label_output: RigidityLabelOutput,
    ) -> ValidationResult:
        """
        Validate graph topology, label variance, confidence threshold, and evidence grounding.
        """
        reasons: list[str] = []
        low_conf_nodes: list[str] = []
        ungrounded_nodes: list[str] = []

        # 1. Minimum node count
        if len(graph.nodes) < self.min_nodes:
            reasons.append(f"too_few_nodes ({len(graph.nodes)} < {self.min_nodes})")

        # 2. Disconnected singletons (0 edges)
        connected_ids = set()
        for e in graph.edges:
            connected_ids.add(e.source)
            connected_ids.add(e.target)
        singletons = [n.node_id for n in graph.nodes if n.node_id not in connected_ids]
        if singletons:
            reasons.append(f"disconnected_singletons ({singletons})")

        # 3. Node <-> Label alignment
        graph_ids = {n.node_id for n in graph.nodes}
        label_ids = {lbl.node_id for lbl in label_output.labels}
        if graph_ids != label_ids:
            reasons.append(
                f"node_label_mismatch (missing={graph_ids - label_ids}, extra={label_ids - graph_ids})"
            )

        # 4. Non-degenerate label variance (not all 0 or all 1)
        unique_labels = {lbl.rigidity_label for lbl in label_output.labels}
        if len(unique_labels) < 2:
            reasons.append(f"zero_label_variance (all labels = {unique_labels})")

        # 5. Confidence threshold (graph-level) & evidence quote grounding
        for lbl in label_output.labels:
            if lbl.confidence_score < self.min_confidence:
                low_conf_nodes.append(f"{lbl.node_id}({lbl.confidence_score:.2f})")
            if not self._quote_is_grounded(lbl.evidence_quote, record.after_text):
                ungrounded_nodes.append(lbl.node_id)

        avg_conf = (
            sum(lbl.confidence_score for lbl in label_output.labels) / len(label_output.labels)
            if label_output.labels
            else 0.0
        )
        low_conf_ratio = len(low_conf_nodes) / len(label_output.labels) if label_output.labels else 1.0

        if avg_conf < 0.60 or low_conf_ratio > 0.65:
            reasons.append(
                f"low_graph_confidence (avg={avg_conf:.2f}, low_nodes={len(low_conf_nodes)}/{len(label_output.labels)})"
            )
        if ungrounded_nodes:
            reasons.append(f"ungrounded_evidence_quotes ({ungrounded_nodes})")

        return ValidationResult(
            is_valid=len(reasons) == 0,
            reasons=reasons,
            low_confidence_nodes=low_conf_nodes,
            ungrounded_quote_nodes=ungrounded_nodes,
        )


if __name__ == "__main__":
    # Smoke test directly against the cached trizbench_US4966257 extraction (no API calls)
    from adapters.trizbench_loader import load_trizbench_patents

    records, _ = load_trizbench_patents()
    sample = records[0]

    p1_path = Path("data/cache") / f"{sample.case_id}_pass1.json"
    p2_path = Path("data/cache") / f"{sample.case_id}_pass2.json"

    if not (p1_path.exists() and p2_path.exists()):
        print("Cache files not found. Run `python src/extractor.py` first.")
        sys.exit(1)

    graph = TRIZGraph.model_validate_json(p1_path.read_text(encoding="utf-8"))
    labels = RigidityLabelOutput.model_validate_json(p2_path.read_text(encoding="utf-8"))

    validator = TRIZValidator()
    canon_graph, merges = validator.canonicalize_graph(graph)
    print(f"Canonicalization: {len(graph.nodes)} -> {len(canon_graph.nodes)} nodes | merges={merges}")

    result = validator.validate_extraction(sample, graph, labels)
    print(f"Validation valid={result.is_valid} | reasons={result.reasons}")