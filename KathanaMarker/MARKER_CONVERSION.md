# Marker Parser Conversion — Findings & Deferred Plan

**Status: DEFERRED (not adopted for MVP).** Everything needed to adopt Marker later is
captured here plus in two companion files: [`ingest_marker.py`](ingest_marker.py) (the
full converted pipeline) and [`requirements-marker.txt`](requirements-marker.txt) (its
verified dependency set). The production pipeline stays on **docling**.

- **Branch:** `marker-parser` (off `markerTest`)
- **Date of investigation:** 2026-07-10
- **Author context:** GraphRAG book-ingestion pipeline, single-GPU (RTX 4080 Super, 16GB)

---

## 1. Why we looked at Marker — and why we're deferring

The ingest router (`_route_chunker` in `ingest.py`) decides per book whether to run the
`software`, `math`, or `selfhelp` chunker based on content-density signals
(`classify.route_signals`). Two extractor evals fed this decision:

- **docling vs marker** (`compare_marker_docling.py`): docling emits **0–1 `$$` math blocks
  per book**, so its document math-signal collapses below `_MATH_DENSITY_FLOOR (1.0)`. A pure-
  math book (*Mathematics For Machine Learning*, no code) therefore **misroutes to `selfhelp`**
  instead of `math`. Marker emits **416–1853 `$$` blocks/book** and routes it correctly.
  Marker is the only engine that captures equations as LaTeX.
- **pymupdf4llm** (`compare_pymupdf.py`): the lightweight/fast alternative — **strictly worse**.
  Zero math extraction (same misroute as docling) *plus* a code-space-stripping bug
  (`frommatplotlib.pyplotimportsubplots`) that mangles code and misroutes math+code books.
  Rejected.

**Cost:** Marker runs **~8–9× slower** than docling (ISLP 989s vs 115s; ~16 min/book vs ~2)
and is GPU-bound (~5GB VRAM/worker).

**Decision:** For an MVP the LaTeX-math gain does not justify 8–9× ingest time across the
~70-book corpus. Math-heavy books will be handled by a **dedicated pipeline post-MVP**, so the
MVP stays on docling. This document preserves the conversion so it can be picked up cleanly.

> Full evidence lives in `marker_eval_out/report.txt` (docling/marker) and
> `marker_eval_out/report_pymupdf.txt` (pymupdf4llm), plus the cached `*.md` extractions.

---

## 2. Dependency conflict analysis (the important part)

Earlier belief (see the `marker-mineru-pillow-deadlock` note) was that `marker-pdf` **cannot**
coexist with the production venv — a double deadlock:

1. **Pillow:** marker/surya need `Pillow<11`; `mineru` needs `Pillow>=11`.
2. **click:** marker needs `click>=8.2`; `typer` needs `click<8.2`.

**What we discovered:** that deadlock was an artifact of a **stale venv**, not the intended
dependency set. The LightRAG→LlamaIndex migration already removed `raganything`/`mineru`/
`gradio` from `requirements.txt`, and **no code imports them** — they were just leftover
`.dist-info` in `.venv` (`mineru-2.1.11`, `raganything-1.3.1`, `gradio-5.50.0`). Those packages
were the *sole* sources of both conflict legs:

| Constraint | Came from | Still needed? |
|---|---|---|
| `pillow>=11` | `mineru` | No — mineru removed from the stack |
| `click<8.2` (via `typer`) | `gradio→typer` **and** `docling→typer (>=0.12.5,<0.16.0)` | No — gradio removed; docling being replaced |

`huggingface_hub` requires `typer` only as an `mcp` **extra** (not installed), so once docling
and the old stack are gone, **nothing hard-requires `typer`** → `click` is free to satisfy
marker's `>=8.2`, and **nothing requires `pillow>=11`** → free for marker's `<11`.

### Verified resolution (isolated dry-run, 2026-07-10)

Built a clean venv and ran `pip install -r requirements-marker.txt --dry-run --report`.
**129 packages, zero conflicts.** Key resolved versions:

| package | resolved | note |
|---|---|---|
| `marker-pdf` | 1.10.2 | + `surya-ocr` 0.17.1, `pdftext` 0.6.3 |
| `pillow` | **10.4.0** | satisfies marker `<11` |
| `click` | **8.4.2** | satisfies marker `>=8.2` |
| `torch` | **2.12.1+cu130** | **unchanged** — no re-download, GPU build preserved |
| `typer`, `docling`, `mineru`, `gradio`, `raganything` | **absent** | old stack fully gone |

This matches the pre-existing `.venv-marker-test` (marker-pdf 1.10.2 + torch 2.12.1+cu130),
so the runtime is already known-good, not just the resolver.

---

## 3. Environment (how to build the Marker venv when adopting)

Marker must live in **its own venv**, never mixed into the docling production `.venv`. Approved
approach was a **surgical swap** (avoids re-downloading the ~2.5GB torch+cu130 build):

```bash
# In the target venv (a fresh one, or a copy of the prod venv):
python -m pip uninstall -y docling docling-core mineru mineru-vl-utils raganything gradio gradio_pdf
python -m pip install -r requirements-marker.txt   # pulls marker-pdf, downgrades pillow→10.4, bumps click→8.4
python -c "import marker; import torch; print(torch.cuda.is_available())"   # expect True
```

Rollback: `pip uninstall -y marker-pdf surya-ocr && pip install docling==2.31.0`.
The reference `.venv-marker-test` (docling+marker coexisting, pillow 10.x) is left untouched.

---

## 4. Code changes (all in `ingest_marker.py`, diffed against `ingest.py`)

Only the **PDF extractor and its concurrency** change. Routing, chunking (`chunk_math`,
`chunk_software`, `chunk_prose`), dedup, and insertion are **untouched** — they consume markdown
generically, and `chunk_math`'s `_FORMULA_RE` already matches `$$…$$`/`$…$`/`\begin{}…\end{}`,
which is exactly what Marker emits. That's why the swap is small.

**a. Imports** — replace docling with Marker:
```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
```

**b. Per-process cached model dict** (new) — the model dict is ~5GB VRAM / tens of seconds to
build, so it must be loaded **once per worker process**, never per PDF:
```python
_MARKER_MODELS = None
def _marker_models():
    global _MARKER_MODELS
    if _MARKER_MODELS is None:
        _MARKER_MODELS = create_model_dict()
    return _MARKER_MODELS
```

**c. `extract_pdf` rewrite:**
```python
def extract_pdf(path: Path) -> str:
    converter = PdfConverter(artifact_dict=_marker_models())
    rendered = converter(str(path))
    out = text_from_rendered(rendered)
    return out[0] if isinstance(out, tuple) else out   # (text, ext, images) across versions
```

**d. Concurrency (GPU-bound now, not CPU):**
- `PipelineConfig.extraction_workers` default **8 → 2**, and the `EXTRACTION_WORKERS` env default
  **"8" → "2"** (2 workers ≈ 10GB VRAM, safe headroom on the 16GB card).
- New `Pool` initializer warms the model dict once per worker (predictable startup, fail-fast on
  VRAM exhaustion instead of silently dropping documents mid-run):
  ```python
  def _init_extraction_worker() -> None:
      _marker_models()
  # ...
  Pool(processes=config.extraction_workers, initializer=_init_extraction_worker)
  ```

> Optional future tweak: pass `config={"extract_images": False}` to `PdfConverter` to skip image
> work (we only keep text). Marker also logs/tqdm per document — may want to quiet it under the
> rich progress bar.

The full, exact change is the diff `ingest.py → ingest_marker.py` (regenerate with
`diff -u ingest.py ingest_marker.py`).

---

## 5. Verification checklist (when adopted)

1. **Routing smoke test** (the regression this fixes): `extract_pdf("Mathematics For Machine
   Learning.pdf")` output contains `$$` blocks, and `_route_chunker(raw, path) == "math"`
   (docling returned `"selfhelp"`). Cross-check *Mathematics of Machine Learning* → `"software"`.
2. **Small pipeline dry run** (2–3 books, scratch `working_dir`, `EXTRACTION_WORKERS=2`): no VRAM
   OOM, chunks produced, math chunks carry LaTeX. Watch `nvidia-smi` for ≤~10GB and 2 model loads.
3. **Chunk inspection:** existing checkpoint/inspection tooling — confirm math books yield
   LaTeX-bearing chunks routed through `chunk_math`.

---

## 6. To adopt later

1. Replace `ingest.py` with `ingest_marker.py`'s contents (or `git mv`).
2. Replace `requirements.txt` with `requirements-marker.txt`.
3. Build the Marker venv (§3) and run the verification checklist (§5).
4. Re-ingest the corpus (GPU-hours; sequential, ~16 min/book).

## 7. Open items / future work
- **Dedicated math-heavy pipeline** (the reason Marker is deferred) — decide its scope and which
  books route to it vs. the MVP docling pipeline.
- Consider Marker only for a *subset* (math-labeled books) rather than the whole corpus, to cap
  the 8–9× cost while still fixing math routing where it matters.
- `.archived_code/docling_to_content_list.py` also imports docling but is inactive — ignore.
