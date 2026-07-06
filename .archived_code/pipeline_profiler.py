"""
GraphRAG Pipeline Profiler
---------------------------
Drop-in profiler for your ingestion pipeline (docling -> chunking -> embedding ->
LightRAG/Qwen2.5 extraction -> SurrealDB write). Wrap each stage of your pipeline
in the `stage()` context manager, and it will:

  1. Time each stage (wall clock)
  2. Sample GPU utilization + VRAM in the background via nvidia-smi (polling,
     no extra deps required) while a stage runs
  3. Track disk read bytes/time for I/O-heavy stages (optional, via psutil if present)
  4. Dump a report at the end classifying each stage as GPU-bound, I/O-bound,
     or CPU-bound, plus an overall verdict for the whole run

Usage in your pipeline code:

    from pipeline_profiler import Profiler

    profiler = Profiler(out_dir="./profile_results")

    with profiler.stage("parse_docling", book="some_book.epub"):
        doc = docling_parse(path)

    with profiler.stage("chunk_semantic"):
        chunks = splitter.chunks(doc)

    with profiler.stage("embed", n_chunks=len(chunks)):
        vectors = embedder.encode(chunks)

    with profiler.stage("extract_entities_qwen", n_chunks=len(chunks)):
        entities = lightrag_extract(chunks)  # your Ollama/Qwen2.5:14B call

    with profiler.stage("write_surrealdb"):
        db.insert(entities, vectors)

    profiler.report()  # prints + writes JSON/CSV summary

Requirements:
    - nvidia-smi on PATH (standard with NVIDIA drivers on Windows/Linux)
    - Optional: `pip install psutil` for disk I/O counters (falls back gracefully
      if not installed)

Nothing here talks to Ollama, SurrealDB, or LightRAG directly — it's fully
pipeline-agnostic. You wrap whatever calls you already have.
"""

from __future__ import annotations

import contextlib
import csv
import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# GPU sampling (background thread, polls nvidia-smi every `interval` seconds)
# ---------------------------------------------------------------------------

class _GpuSampler:
    """Polls nvidia-smi in a background thread while active."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self._samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = shutil.which("nvidia-smi") is not None

    def _poll_once(self) -> dict | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode != 0:
                return None
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            return {
                "gpu_util_pct": float(parts[0]),
                "mem_util_pct": float(parts[1]),
                "mem_used_mb": float(parts[2]),
                "mem_total_mb": float(parts[3]),
                "power_draw_w": float(parts[4]) if parts[4] not in ("N/A", "[N/A]") else None,
                "temp_c": float(parts[5]) if parts[5] not in ("N/A", "[N/A]") else None,
                "t": time.time(),
            }
        except Exception:
            return None

    def _run(self):
        while not self._stop_event.is_set():
            sample = self._poll_once()
            if sample:
                self._samples.append(sample)
            time.sleep(self.interval)

    def start(self):
        if not self._available:
            return
        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if not self._available:
            return {"available": False}
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self._samples:
            return {"available": True, "samples": 0}
        return {
            "available": True,
            "samples": len(self._samples),
            "gpu_util_avg_pct": round(mean(s["gpu_util_pct"] for s in self._samples), 1),
            "gpu_util_max_pct": round(max(s["gpu_util_pct"] for s in self._samples), 1),
            "mem_used_avg_mb": round(mean(s["mem_used_mb"] for s in self._samples), 1),
            "mem_used_max_mb": round(max(s["mem_used_mb"] for s in self._samples), 1),
            "power_draw_avg_w": (
                round(mean(v for s in self._samples if (v := s["power_draw_w"]) is not None), 1)
                if any(s["power_draw_w"] is not None for s in self._samples)
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Disk I/O sampling (optional, via psutil)
# ---------------------------------------------------------------------------

def _disk_counters() -> dict | None:
    if not _HAS_PSUTIL:
        return None
    try:
        c = psutil.disk_io_counters()
        if c is None:
            return None
        return {"read_bytes": c.read_bytes, "write_bytes": c.write_bytes,
                "read_time_ms": c.read_time, "write_time_ms": c.write_time}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stage record + classification
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    name: str
    wall_seconds: float
    gpu: dict
    disk_delta: dict | None
    meta: dict = field(default_factory=dict)
    verdict: str = ""

    def classify(self):
        gpu_avg = self.gpu.get("gpu_util_avg_pct") if self.gpu.get("available") else None
        if gpu_avg is None:
            self.verdict = "unknown (no GPU data)"
            return
        if gpu_avg >= 60:
            self.verdict = f"GPU-bound (avg util {gpu_avg}%)"
        elif gpu_avg >= 20:
            self.verdict = f"mixed (avg util {gpu_avg}%) — check disk/CPU too"
        else:
            # low GPU util: likely I/O or CPU bound (parsing, disk reads, etc.)
            if self.disk_delta and self.disk_delta.get("read_time_ms", 0) > 0.3 * self.wall_seconds * 1000:
                self.verdict = f"I/O-bound (avg GPU util only {gpu_avg}%, heavy disk read time)"
            else:
                self.verdict = f"CPU/I/O-bound (avg GPU util only {gpu_avg}%)"


class Profiler:
    def __init__(self, out_dir: str = "./profile_results", gpu_poll_interval: float = 0.25):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_poll_interval = gpu_poll_interval
        self.results: list[StageResult] = []

    @contextlib.contextmanager
    def stage(self, name: str, **meta):
        sampler = _GpuSampler(interval=self.gpu_poll_interval)
        disk_before = _disk_counters()
        sampler.start()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            gpu_summary = sampler.stop()
            disk_after = _disk_counters()
            disk_delta = None
            if disk_before and disk_after:
                disk_delta = {
                    k: disk_after[k] - disk_before[k] for k in disk_before
                }
            result = StageResult(
                name=name,
                wall_seconds=round(elapsed, 3),
                gpu=gpu_summary,
                disk_delta=disk_delta,
                meta=meta,
            )
            result.classify()
            self.results.append(result)
            print(f"[profiler] {name}: {result.wall_seconds}s | {result.verdict}")

    def report(self, print_summary: bool = True):
        total_time = sum(r.wall_seconds for r in self.results)

        if print_summary:
            print("\n" + "=" * 70)
            print("PIPELINE PROFILE SUMMARY")
            print("=" * 70)
            for r in self.results:
                pct = (r.wall_seconds / total_time * 100) if total_time else 0
                print(f"  {r.name:30s} {r.wall_seconds:8.2f}s  ({pct:5.1f}%)  {r.verdict}")
            print("-" * 70)
            print(f"  {'TOTAL':30s} {total_time:8.2f}s")
            print("=" * 70)

            # Overall verdict
            gpu_bound_time = sum(r.wall_seconds for r in self.results if "GPU-bound" in r.verdict)
            io_bound_time = sum(r.wall_seconds for r in self.results if "I/O-bound" in r.verdict)
            if total_time > 0:
                gpu_pct = gpu_bound_time / total_time * 100
                io_pct = io_bound_time / total_time * 100
                print(f"\nOverall: {gpu_pct:.0f}% of wall time in GPU-bound stages, "
                      f"{io_pct:.0f}% in I/O-bound stages.")
                if gpu_pct > io_pct * 2:
                    print("-> Pipeline is predominantly COMPUTE-bound (GPU). "
                          "Cloud GPU rental for extraction is where money would help, "
                          "if anywhere. Faster local disk / networking will not move the needle much.")
                elif io_pct > gpu_pct * 2:
                    print("-> Pipeline is predominantly I/O-bound. "
                          "Cloud GPU spend would NOT help — look at local disk "
                          "(NVMe vs HDD), parallel file reads, or caching parsed docs instead.")
                else:
                    print("-> Mixed bottleneck. Check the per-stage breakdown above: "
                          "optimize whichever named stage dominates wall time.")

        # Write JSON
        json_path = self.out_dir / "profile_report.json"
        with open(json_path, "w") as f:
            json.dump(
                [
                    {
                        "name": r.name,
                        "wall_seconds": r.wall_seconds,
                        "gpu": r.gpu,
                        "disk_delta": r.disk_delta,
                        "meta": r.meta,
                        "verdict": r.verdict,
                    }
                    for r in self.results
                ],
                f,
                indent=2,
            )

        # Write CSV
        csv_path = self.out_dir / "profile_report.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "wall_seconds", "gpu_util_avg_pct", "mem_used_avg_mb", "verdict", "meta"])
            for r in self.results:
                writer.writerow([
                    r.name,
                    r.wall_seconds,
                    r.gpu.get("gpu_util_avg_pct", ""),
                    r.gpu.get("mem_used_avg_mb", ""),
                    r.verdict,
                    json.dumps(r.meta),
                ])

        print(f"\nWrote: {json_path}")
        print(f"Wrote: {csv_path}")
        return json_path, csv_path
