from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schemas import RawCaseRecord, RigidityLabelOutput, TRIZGraph

load_dotenv()

T = TypeVar("T", bound=BaseModel)

PROVIDER_DEFAULTS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash-lite",
    },
    "openai": {
        "base_url": None,
        "model": "gpt-4o-mini",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:14b",
    },
}

PASS1_SYSTEM_PROMPT = """You are an expert TRIZ engineering analyst.
Extract a directed TRIZ Function Model Graph G = (V, E) describing the engineering system BEFORE the inventive solution, using ONLY the provided prior-art / problem text.

Rules for Nodes (V):
- Extract 6 to 20 distinct physical components, target objects, and super-system elements.
- `node_id`: lowercase snake_case slug (e.g., "piston_rod", "guiding_sleeve").
- `component_type`:
  - "System_Component": internal parts of the engineered device/system.
  - "Target_Object": the main workpiece, fluid, user, or object being acted upon.
  - "Super_System": external environment, power sources, ambient heat/moisture, adjacent systems.
- `is_standard_or_regulated`: true ONLY if the text indicates the component is governed by standards, safety regulations, or external fixed specs; otherwise false.
- `role_description`: concise 1-sentence summary of what the component does in the BEFORE system.

Rules for Edges (E):
- Represent directed Subject-Action-Object (SAO) physical interactions.
- Every `source` and `target` MUST exactly match a `node_id` in `nodes`.
- `action_verb`: short active verb phrase (e.g., "heats", "abrades", "supports", "seals").
- `interaction_quality`: "Useful_Normal", "Insufficient", "Excessive", or "Harmful".
- `function_category`: "Productive", "Supporting", "Transport", or "Corrective".
- Ensure every node participates in at least one edge (no disconnected singletons).
"""

PASS2_SYSTEM_PROMPT = """You are an expert TRIZ engineering analyst labeling component rigidity.
You are given:
1. A list of components (nodes) from the BEFORE system.
2. The AFTER text describing the inventive solution.

For EVERY `node_id` in the input list (do not skip or invent any node_id), assign:
- `rigidity_label`:
  - 0 (Flexible / Modifiable): The component was trimmed, removed, replaced, merged, or structurally altered in the AFTER solution.
  - 1 (Rigid / Untouchable): The component survived untouched in the AFTER solution, or was explicitly treated as a fixed constraint.
- `modification_type`:
  - If `rigidity_label` == 1 -> MUST be "Untouched".
  - If `rigidity_label` == 0 -> MUST be one of "Trimmed", "Replaced", "Modified", "Merged".
- `evidence_quote`: An EXACT verbatim substring copied from the AFTER text justifying the label (never fabricate quotes). If the component is not explicitly mentioned in AFTER text, quote the closest relevant phrase and assign `confidence_score` < 0.65.
- `confidence_score`: float in [0.0, 1.0]. Use >= 0.80 only when the AFTER text clearly supports the label.
"""


class TRIZExtractor:
    """Async/batched two-pass LLM extractor with disk caching, RPM pacing, and retries."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "data/cache",
        max_concurrency: int = 5,
        rpm_limit: int = 12,
        
    ) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        if self.provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

        defaults = PROVIDER_DEFAULTS[self.provider]
        self.model = model or os.getenv("LLM_MODEL") or defaults["model"]
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or defaults["base_url"]
        self.api_key = api_key or os.getenv("LLM_API_KEY") or (
            "ollama" if self.provider == "ollama" else None
        )

        if not self.api_key:
            raise ValueError(f"Missing LLM_API_KEY in environment for provider '{self.provider}'.")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=0,
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrency)

        # Rate limiter (default 12 RPM = 1 call every 5s to stay under Gemini free-tier 15 RPM)
        self.min_interval = 60.0 / rpm_limit if rpm_limit > 0 else 0.0
        self._rate_lock = asyncio.Lock()
        self._last_request_ts = 0.0

    def _cache_path(self, case_id: str, step: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in case_id)
        return self.cache_dir / f"{safe_id}_{step}.json"

    async def _wait_for_rate_limit(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._rate_lock:
            now = time.monotonic()
            wait_s = self.min_interval - (now - self._last_request_ts)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last_request_ts = time.monotonic()

    @retry(
        retry=retry_if_exception_type(
            (
                RateLimitError,
                InternalServerError,
                APITimeoutError,
                APIConnectionError,
                ValidationError,
                ValueError,
            )
        ),
        wait=wait_exponential(multiplier=2, min=4, max=45),
        stop=stop_after_attempt(4),
        before_sleep=lambda rs: print(
            f"  [retry {rs.attempt_number}] {type(rs.outcome.exception()).__name__}: {rs.outcome.exception()}"
        ),
        reraise=True,
    )
    async def _call_structured(
        self, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        kwargs = {}
        if self.provider == "gemini" and "lite" not in self.model:
            kwargs["reasoning_effort"] = "low"

        async with self.semaphore:
            await self._wait_for_rate_limit()
            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                **kwargs,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("LLM returned empty or refused structured output.")
            return parsed

    async def extract_pass1(
        self, record: RawCaseRecord, use_cache: bool = True
    ) -> TRIZGraph:
        """Pass 1: Extract TRIZ Function Model Graph from before_text ONLY."""
        cache_file = self._cache_path(record.case_id, "pass1")
        if use_cache and cache_file.exists():
            return TRIZGraph.model_validate_json(cache_file.read_text(encoding="utf-8"))

        user_prompt = (
            f"CASE ID: {record.case_id}\n"
            f"DOMAIN: {record.domain}\n\n"
            f"BEFORE SYSTEM TEXT:\n{record.before_text}"
        )
        graph = await self._call_structured(PASS1_SYSTEM_PROMPT, user_prompt, TRIZGraph)

        cache_file.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
        return graph

    async def extract_pass2(
        self, record: RawCaseRecord, graph: TRIZGraph, use_cache: bool = True
    ) -> RigidityLabelOutput:
        """Pass 2: Label every node in graph against after_text."""
        cache_file = self._cache_path(record.case_id, "pass2")
        if use_cache and cache_file.exists():
            return RigidityLabelOutput.model_validate_json(
                cache_file.read_text(encoding="utf-8")
            )

        nodes_payload = [
            {
                "node_id": n.node_id,
                "label_name": n.label_name,
                "component_type": n.component_type,
                "role_description": n.role_description,
            }
            for n in graph.nodes
        ]
        user_prompt = (
            f"CASE ID: {record.case_id}\n\n"
            f"BEFORE SYSTEM NODES (label ALL of these):\n"
            f"{json.dumps(nodes_payload, indent=2)}\n\n"
            f"AFTER SOLUTION TEXT:\n{record.after_text}"
        )
        labels = await self._call_structured(
            PASS2_SYSTEM_PROMPT, user_prompt, RigidityLabelOutput
        )

        # Verify all node_ids from Pass 1 are present; if not, trigger retry
        expected_ids = {n.node_id for n in graph.nodes}
        returned_ids = {lbl.node_id for lbl in labels.labels}
        if expected_ids != returned_ids:
            raise ValueError(
                f"Pass 2 node mismatch. Missing: {expected_ids - returned_ids}, "
                f"Extra: {returned_ids - expected_ids}"
            )

        cache_file.write_text(labels.model_dump_json(indent=2), encoding="utf-8")
        return labels