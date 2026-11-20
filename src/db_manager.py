from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from schemas import RawCaseRecord, RigidityLabelOutput, TRIZGraph


class DBManager:
    """Manages the SQLite database, resumable inserts, exports, and dataset stats."""

    def __init__(self, db_path: str | Path = "data/triz_rigidity.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_schema(self) -> None:
        """Create the 3 normalized tables (graphs, nodes, edges) if they don't exist."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS graphs (
                    case_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL CHECK (source IN ('trizbench', 'uspto')),
                    domain TEXT NOT NULL,
                    before_text TEXT NOT NULL,
                    after_text TEXT NOT NULL,
                    node_count INTEGER NOT NULL,
                    edge_count INTEGER NOT NULL,
                    rigid_ratio REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    case_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    label_name TEXT NOT NULL,
                    component_type TEXT NOT NULL CHECK (
                        component_type IN ('System_Component', 'Super_System', 'Target_Object')
                    ),
                    is_standard_or_regulated INTEGER NOT NULL CHECK (is_standard_or_regulated IN (0, 1)),
                    role_description TEXT NOT NULL,
                    rigidity_label INTEGER NOT NULL CHECK (rigidity_label IN (0, 1)),
                    modification_type TEXT NOT NULL CHECK (
                        modification_type IN ('Untouched', 'Trimmed', 'Replaced', 'Modified', 'Merged')
                    ),
                    evidence_quote TEXT NOT NULL,
                    confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
                    PRIMARY KEY (case_id, node_id),
                    FOREIGN KEY (case_id) REFERENCES graphs(case_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS edges (
                    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    action_verb TEXT NOT NULL,
                    interaction_quality TEXT NOT NULL CHECK (
                        interaction_quality IN ('Useful_Normal', 'Insufficient', 'Excessive', 'Harmful')
                    ),
                    function_category TEXT NOT NULL CHECK (
                        function_category IN ('Productive', 'Supporting', 'Transport', 'Corrective')
                    ),
                    FOREIGN KEY (case_id) REFERENCES graphs(case_id) ON DELETE CASCADE,
                    FOREIGN KEY (case_id, source_node_id) REFERENCES nodes(case_id, node_id) ON DELETE CASCADE,
                    FOREIGN KEY (case_id, target_node_id) REFERENCES nodes(case_id, node_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_graphs_domain ON graphs(domain);
                CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(rigidity_label);
                CREATE INDEX IF NOT EXISTS idx_edges_case ON edges(case_id);
                """
            )

    def get_processed_case_ids(self) -> set[str]:
        """Return all case_ids already stored in SQLite for resumable execution."""
        with self._connect() as conn:
            rows = conn.execute("SELECT case_id FROM graphs").fetchall()
            return {row["case_id"] for row in rows}

    def case_exists(self, case_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM graphs WHERE case_id = ? LIMIT 1", (case_id,)
            ).fetchone()
            return row is not None

    def insert_graph(
        self,
        record: RawCaseRecord,
        graph: TRIZGraph,
        label_output: RigidityLabelOutput,
        overwrite: bool = False,
    ) -> bool:
        """
        Idempotently insert a validated graph + labels inside a single transaction.
        Returns True if inserted, False if skipped (already exists and overwrite=False).
        """
        label_map = {lbl.node_id: lbl for lbl in label_output.labels}
        graph_node_ids = {n.node_id for n in graph.nodes}

        if graph_node_ids != set(label_map.keys()):
            missing = graph_node_ids - set(label_map.keys())
            extra = set(label_map.keys()) - graph_node_ids
            raise ValueError(
                f"[{record.case_id}] Node/Label mismatch. Missing: {missing}, Extra: {extra}"
            )

        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM graphs WHERE case_id = ? LIMIT 1", (record.case_id,)
            ).fetchone()

            if exists:
                if not overwrite:
                    return False
                conn.execute("DELETE FROM graphs WHERE case_id = ?", (record.case_id,))

            rigid_count = sum(lbl.rigidity_label for lbl in label_output.labels)
            rigid_ratio = round(rigid_count / len(graph.nodes), 4)
            created_at = datetime.now(timezone.utc).isoformat()

            conn.execute(
                """
                INSERT INTO graphs (
                    case_id, source, domain, before_text, after_text,
                    node_count, edge_count, rigid_ratio, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.case_id,
                    record.source,
                    record.domain,
                    record.before_text,
                    record.after_text,
                    len(graph.nodes),
                    len(graph.edges),
                    rigid_ratio,
                    created_at,
                ),
            )

            conn.executemany(
                """
                INSERT INTO nodes (
                    case_id, node_id, label_name, component_type,
                    is_standard_or_regulated, role_description,
                    rigidity_label, modification_type, evidence_quote, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.case_id,
                        n.node_id,
                        n.label_name,
                        n.component_type,
                        int(n.is_standard_or_regulated),
                        n.role_description,
                        label_map[n.node_id].rigidity_label,
                        label_map[n.node_id].modification_type,
                        label_map[n.node_id].evidence_quote,
                        label_map[n.node_id].confidence_score,
                    )
                    for n in graph.nodes
                ],
            )

            conn.executemany(
                """
                INSERT INTO edges (
                    case_id, source_node_id, target_node_id,
                    action_verb, interaction_quality, function_category
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.case_id,
                        e.source,
                        e.target,
                        e.action_verb,
                        e.interaction_quality,
                        e.function_category,
                    )
                    for e in graph.edges
                ],
            )
            return True

    def export_jsonl(self, output_path: str | Path = "data/exports/triz_rigidity.jsonl") -> Path:
        """Export one complete nested graph record per line in JSONL format."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn, out.open("w", encoding="utf-8") as f:
            graphs = conn.execute("SELECT * FROM graphs ORDER BY case_id").fetchall()
            for g in graphs:
                case_id = g["case_id"]
                nodes = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM nodes WHERE case_id = ? ORDER BY node_id", (case_id,)
                    ).fetchall()
                ]
                for n in nodes:
                    n["is_standard_or_regulated"] = bool(n["is_standard_or_regulated"])

                edges = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM edges WHERE case_id = ? ORDER BY edge_id", (case_id,)
                    ).fetchall()
                ]
                payload = {**dict(g), "nodes": nodes, "edges": edges}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return out

    def export_parquet(self, output_dir: str | Path = "data/exports/parquet") -> Path:
        """Export normalized graphs, nodes, and edges tables to Parquet snapshots."""
        import pandas as pd  # Requires: pip install pandas pyarrow

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            for table in ("graphs", "nodes", "edges"):
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                df.to_parquet(out_dir / f"{table}.parquet", index=False)
        return out_dir

    def generate_summary_stats(
        self, output_path: str | Path = "data/exports/summary_stats.json"
    ) -> dict[str, Any]:
        """Compute dataset card metrics (counts, histograms, domains, label balance)."""
        with self._connect() as conn:
            graphs = conn.execute("SELECT * FROM graphs").fetchall()
            nodes = conn.execute(
                "SELECT rigidity_label, modification_type, component_type, confidence_score FROM nodes"
            ).fetchall()
            edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        total_graphs = len(graphs)
        total_nodes = len(nodes)

        domain_counts = Counter(g["domain"] for g in graphs)
        source_counts = Counter(g["source"] for g in graphs)
        node_hist = Counter(g["node_count"] for g in graphs)
        edge_hist = Counter(g["edge_count"] for g in graphs)

        label_counts = Counter(n["rigidity_label"] for n in nodes)
        mod_counts = Counter(n["modification_type"] for n in nodes)
        comp_counts = Counter(n["component_type"] for n in nodes)

        rigid_pct = round((label_counts[1] / total_nodes) * 100, 2) if total_nodes else 0.0
        flexible_pct = round((label_counts[0] / total_nodes) * 100, 2) if total_nodes else 0.0
        avg_conf = (
            round(sum(n["confidence_score"] for n in nodes) / total_nodes, 4)
            if total_nodes
            else 0.0
        )

        stats: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "totals": {
                "graphs": total_graphs,
                "nodes": total_nodes,
                "edges": edge_count,
                "avg_nodes_per_graph": round(total_nodes / total_graphs, 2) if total_graphs else 0.0,
                "avg_edges_per_graph": round(edge_count / total_graphs, 2) if total_graphs else 0.0,
                "avg_confidence_score": avg_conf,
            },
            "label_balance": {
                "flexible_0_count": label_counts[0],
                "rigid_1_count": label_counts[1],
                "flexible_0_pct": flexible_pct,
                "rigid_1_pct": rigid_pct,
            },
            "modification_type_breakdown": dict(mod_counts),
            "component_type_breakdown": dict(comp_counts),
            "source_breakdown": dict(source_counts),
            "domain_breakdown": dict(domain_counts.most_common()),
            "node_count_histogram": dict(sorted(node_hist.items())),
            "edge_count_histogram": dict(sorted(edge_hist.items())),
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return stats


if __name__ == "__main__":
    # Quick smoke test using a temporary DB (doesn't touch your real DB)
    import tempfile
    from schemas import NodeRigidityLabel, TRIZEdge, TRIZNode

    with tempfile.TemporaryDirectory() as tmpdir:
        db = DBManager(Path(tmpdir) / "test.db")

        sample_record = RawCaseRecord(
            case_id="smoke_test_001",
            source="trizbench",
            domain="Mechanical",
            before_text="Steel pipe transports hot steam and corrodes inner valve.",
            after_text="Replaced steel pipe with ceramic-lined pipe; steam flow unchanged.",
        )
        sample_graph = TRIZGraph(
            nodes=[
                TRIZNode(
                    node_id="steel_pipe",
                    label_name="Steel Pipe",
                    component_type="System_Component",
                    is_standard_or_regulated=False,
                    role_description="Transports hot steam through the system.",
                ),
                TRIZNode(
                    node_id="hot_steam",
                    label_name="Hot Steam",
                    component_type="Target_Object",
                    is_standard_or_regulated=False,
                    role_description="Working fluid being transported.",
                ),
            ],
            edges=[
                TRIZEdge(
                    source="steel_pipe",
                    target="hot_steam",
                    action_verb="transports",
                    interaction_quality="Useful_Normal",
                    function_category="Transport",
                )
            ],
        )
        sample_labels = RigidityLabelOutput(
            labels=[
                NodeRigidityLabel(
                    node_id="steel_pipe",
                    rigidity_label=0,
                    modification_type="Replaced",
                    evidence_quote="Replaced steel pipe with ceramic-lined pipe",
                    confidence_score=0.95,
                ),
                NodeRigidityLabel(
                    node_id="hot_steam",
                    rigidity_label=1,
                    modification_type="Untouched",
                    evidence_quote="steam flow unchanged",
                    confidence_score=0.92,
                ),
            ]
        )

        first = db.insert_graph(sample_record, sample_graph, sample_labels)
        second = db.insert_graph(sample_record, sample_graph, sample_labels)
        stats = db.generate_summary_stats(Path(tmpdir) / "stats.json")
        db.export_jsonl(Path(tmpdir) / "out.jsonl")

        print(f"First insert: {first} | Duplicate skipped: {not second}")
        print(f"Stats totals: {stats['totals']} | Balance: {stats['label_balance']}")