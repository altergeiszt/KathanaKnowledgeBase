#!/usr/bin/env python3
"""
extraction_benchmark.py — measure real LLM entity-extraction cost on YOUR hardware.

The graph store (Neo4j) and the orchestration library (LlamaIndex / neo4j-graphrag)
are NOT the bottleneck for KG construction — the per-chunk LLM call is. This script
measures that call directly against your local Ollama model, removing both libraries
as confounders, so the number reflects what either framework would actually cost.

It answers three questions with real seconds on your 4080 + qwen2.5:14b:
  1. How long does entity/relation extraction take per chunk?
  2. How much does SCHEMA-CONSTRAINED extraction (neo4j-graphrag style) save over
     OPEN-ENDED extraction (generic LlamaIndex style)?
  3. How much does GLEANING (a 2nd refinement pass) cost — the thing that caused the
     runaway chunk?

Then it extrapolates to full-corpus build time at 80k / 300k / 1M chunks.

USAGE
-----
    # make sure Ollama is running and the model is pulled:
    #   ollama serve         (usually already running as a service)
    #   ollama pull qwen2.5:14b
    python extraction_benchmark.py --checkpoint lightrag_data/chunks_checkpoint.json
    python extraction_benchmark.py --checkpoint ... --sample 60 --model qwen2.5:14b
    python extraction_benchmark.py --checkpoint ... --concurrency 4   # optional 2nd pass

Only dependency beyond the stdlib is `requests` (pip install requests). No GPU code,
no LlamaIndex, no Neo4j — just timed HTTP calls to Ollama's /api/generate.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("This script needs `requests`. Install with: pip install requests")


# ---------------------------------------------------------------------------
# Prompts: OPEN-ENDED vs SCHEMA-CONSTRAINED
# ---------------------------------------------------------------------------
# The schema-constrained prompt mirrors how neo4j-graphrag grounds extraction with
# predefined node/relation types. The expectation (to be measured, not assumed) is
# that constraining the output produces fewer generated tokens per chunk and thus
# less GPU time. The open-ended prompt mirrors a generic LlamaIndex SimpleLLMPathExtractor.

OPEN_PROMPT = """You are an information extraction system. From the text below, extract \
entities and the relationships between them. Return ONLY valid JSON, no prose, in this form:
{{"entities": [{{"name": "...", "type": "..."}}], \
"relations": [{{"source": "...", "target": "...", "type": "..."}}]}}

Text:
{text}
"""

SCHEMA_PROMPT = """You are an information extraction system. Extract ONLY entities and \
relationships that match the allowed schema. Ignore anything outside it.

Allowed entity types: Concept, Technology, Person, Method, Term
Allowed relation types: RELATES_TO, IS_A, USES, DEFINED_AS, PART_OF

Return ONLY valid JSON, no prose, in this form:
{{"entities": [{{"name": "...", "type": "<one allowed entity type>"}}], \
"relations": [{{"source": "...", "target": "...", "type": "<one allowed relation type>"}}]}}

Text:
{text}
"""

# A minimal gleaning follow-up: ask the model if it missed anything. This is the
# second LLM call per chunk that gleaning adds — measured so you can see its cost.
GLEAN_PROMPT = """You previously extracted this from the text:
{prior}

Re-read the text and add ONLY entities or relationships you missed. Same JSON format. \
If nothing was missed, return {{"entities": [], "relations": []}}.

Text:
{text}
"""


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------
@dataclass
class CallResult:
    seconds: float
    eval_count: int          # tokens generated (from Ollama)
    prompt_eval_count: int   # prompt tokens processed
    ok: bool
    n_entities: int = 0
    n_relations: int = 0
    raw: str = ""


def ollama_generate(host: str, model: str, prompt: str, timeout: float) -> CallResult:
    """One timed, non-streaming call to Ollama /api/generate."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0}},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return CallResult(time.perf_counter() - t0, 0, 0, ok=False, raw=str(exc))

    elapsed = time.perf_counter() - t0
    text = data.get("response", "")
    n_ent, n_rel = _count_extractions(text)
    return CallResult(
        seconds=elapsed,
        eval_count=data.get("eval_count", 0),
        prompt_eval_count=data.get("prompt_eval_count", 0),
        ok=True,
        n_entities=n_ent,
        n_relations=n_rel,
        raw=text,
    )


def _count_extractions(text: str) -> tuple[int, int]:
    """Best-effort parse of the model's JSON to count entities/relations."""
    try:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return 0, 0
        obj = json.loads(text[start:end + 1])
        return len(obj.get("entities", []) or []), len(obj.get("relations", []) or [])
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def stratified_sample(chunks: list[dict], n: int, seed: int) -> list[dict]:
    """
    Sample proportionally across (content_type, has_code) strata so the estimate
    isn't skewed by chunk type. Skips trivially short chunks (headers/artifacts).
    """
    rng = random.Random(seed)
    usable = [c for c in chunks if len(c.get("text", "")) >= 200]

    strata: dict[tuple, list[dict]] = {}
    for c in usable:
        key = (c.get("content_type", "?"), bool((c.get("metadata") or {}).get("has_code")))
        strata.setdefault(key, []).append(c)

    total = len(usable)
    sample: list[dict] = []
    for key, items in strata.items():
        take = max(1, round(n * len(items) / total))
        rng.shuffle(items)
        sample.extend(items[:take])
    rng.shuffle(sample)
    return sample[:n]


# ---------------------------------------------------------------------------
# Benchmark conditions
# ---------------------------------------------------------------------------
@dataclass
class ConditionStats:
    name: str
    per_chunk_s: list[float] = field(default_factory=list)
    gen_tokens: list[int] = field(default_factory=list)
    entities: list[int] = field(default_factory=list)
    relations: list[int] = field(default_factory=list)
    failures: int = 0

    def summary(self) -> dict[str, Any]:
        s = sorted(self.per_chunk_s)
        if not s:
            return {"name": self.name, "note": "no successful calls"}
        return {
            "name": self.name,
            "n": len(s),
            "failures": self.failures,
            "median_s": round(statistics.median(s), 2),
            "mean_s": round(statistics.mean(s), 2),
            "p90_s": round(s[int(len(s) * 0.9) - 1], 2),
            "min_s": round(s[0], 2),
            "max_s": round(s[-1], 2),
            "median_gen_tokens": int(statistics.median(self.gen_tokens)) if self.gen_tokens else 0,
            "median_entities": int(statistics.median(self.entities)) if self.entities else 0,
            "median_relations": int(statistics.median(self.relations)) if self.relations else 0,
        }


def run_condition(name: str, prompt_tmpl: str, sample: list[dict],
                  host: str, model: str, timeout: float,
                  glean: bool = False) -> ConditionStats:
    stats = ConditionStats(name=name)
    n = len(sample)
    for i, chunk in enumerate(sample, 1):
        text = chunk["text"]
        r = ollama_generate(host, model, prompt_tmpl.format(text=text), timeout)
        total_s = r.seconds
        gen_tok = r.eval_count
        ent, rel = r.n_entities, r.n_relations

        if not r.ok:
            stats.failures += 1
            print(f"  [{name}] {i}/{n}  FAILED: {r.raw[:80]}")
            continue

        if glean:
            g = ollama_generate(host, model,
                                GLEAN_PROMPT.format(prior=r.raw[:1200], text=text), timeout)
            if g.ok:
                total_s += g.seconds
                gen_tok += g.eval_count
                ent += g.n_entities
                rel += g.n_relations

        stats.per_chunk_s.append(total_s)
        stats.gen_tokens.append(gen_tok)
        stats.entities.append(ent)
        stats.relations.append(rel)
        print(f"  [{name}] {i}/{n}  {total_s:5.1f}s  "
              f"{gen_tok:4d} tok  {ent:2d} ent  {rel:2d} rel")
    return stats


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------
def extrapolate(median_s: float, concurrency_factor: float = 1.0) -> dict[str, str]:
    """Full-corpus build time at each scale scenario, given median per-chunk seconds."""
    scenarios = {"80k (conservative)": 80_000,
                 "300k (mid)": 300_000,
                 "1M (text-interpretation)": 1_000_000}
    out = {}
    for label, n in scenarios.items():
        secs = n * median_s / concurrency_factor
        out[label] = _human_time(secs)
    return out


def _human_time(secs: float) -> str:
    h = secs / 3600
    if h < 48:
        return f"{h:.1f} hours"
    return f"{h/24:.1f} days"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark LLM entity-extraction cost on local Ollama.")
    ap.add_argument("--checkpoint", required=True, help="Path to chunks_checkpoint.json")
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--sample", type=int, default=40, help="Chunks to test per condition")
    ap.add_argument("--timeout", type=float, default=180.0, help="Per-call timeout (s)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="extraction_benchmark_results.json")
    args = ap.parse_args()

    with open(args.checkpoint, encoding="utf-8") as f:
        chunks = json.load(f)
    sample = stratified_sample(chunks, args.sample, args.seed)
    print(f"Loaded {len(chunks)} chunks; testing {len(sample)} sampled chunks "
          f"against {args.model} at {args.host}\n")

    # Warm-up so model-load time doesn't pollute the first measurement.
    print("Warming up model (loading into VRAM)...")
    w = ollama_generate(args.host, args.model, "Reply with OK.", args.timeout)
    if not w.ok:
        sys.exit(f"Could not reach Ollama / model not available: {w.raw}\n"
                 f"Check `ollama serve` is running and `ollama pull {args.model}` is done.")
    print(f"Warm-up OK ({w.seconds:.1f}s).\n")

    conditions = [
        ("open_single",   OPEN_PROMPT,   False),
        ("schema_single", SCHEMA_PROMPT, False),
        ("open_gleaned",  OPEN_PROMPT,   True),
    ]

    results = {}
    for name, tmpl, glean in conditions:
        print(f"--- Condition: {name} ---")
        stats = run_condition(name, tmpl, sample, args.host, args.model, args.timeout, glean)
        results[name] = stats.summary()
        print()

    # Report
    print("=" * 68)
    print("RESULTS (per-chunk, serial / concurrency=1)")
    print("=" * 68)
    for name, s in results.items():
        if "median_s" not in s:
            print(f"{name}: {s.get('note')}")
            continue
        print(f"\n{name}:")
        print(f"  median {s['median_s']}s  mean {s['mean_s']}s  "
              f"p90 {s['p90_s']}s  (range {s['min_s']}–{s['max_s']}s)")
        print(f"  median {s['median_gen_tokens']} tokens, "
              f"{s['median_entities']} entities, {s['median_relations']} relations "
              f"({s['failures']} failures)")

    # Efficiency delta: schema vs open
    if "median_s" in results.get("open_single", {}) and "median_s" in results.get("schema_single", {}):
        o, sc = results["open_single"]["median_s"], results["schema_single"]["median_s"]
        if o > 0:
            print(f"\nSchema-constrained vs open-ended: "
                  f"{(1 - sc/o)*100:+.0f}% time per chunk "
                  f"({'faster' if sc < o else 'slower'})")
    if "median_s" in results.get("open_single", {}) and "median_s" in results.get("open_gleaned", {}):
        o, g = results["open_single"]["median_s"], results["open_gleaned"]["median_s"]
        if o > 0:
            print(f"Gleaning (2nd pass) adds: {(g/o - 1)*100:+.0f}% time per chunk")

    # Extrapolation on the cheapest realistic config (schema_single if available)
    base = results.get("schema_single") or results.get("open_single")
    if base and "median_s" in base:
        print("\n" + "=" * 68)
        print(f"FULL-CORPUS BUILD TIME  (basis: {base['name']}, {base['median_s']}s/chunk, serial)")
        print("=" * 68)
        for label, t in extrapolate(base["median_s"]).items():
            print(f"  {label:28s} {t}")
        print("\nNote: serial estimate. A single GPU serializes generation, so raising")
        print("insertion concurrency mostly queues rather than parallelizes — treat these")
        print("as the realistic floor, and the hybrid design (extract over notes + a curated")
        print("slice, not the whole corpus) is what actually collapses these numbers.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
