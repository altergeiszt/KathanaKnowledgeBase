"""
classify.py — chunk-level content-type classification, two tiers.

Tier 1 (BOOK_LABELS): the 18 books you hand-labeled in curated-slice.md. Trusted
as-is, exact title match, no scoring. These are also your full-graph-extraction
curated slice (§11 of Migration_LlamaIndex.md) — a book's presence in this dict
does double duty as both its content_type override AND its extraction-tier
membership. Keep it that way rather than maintaining two lists that can drift.

Tier 2 (classify_chunk): everything else — the ~50 unread books, embeddings-only
in the hybrid design. Cheap regex/keyword density scoring per CHUNK, not per book,
because your own labels proved a single book can legitimately carry multiple types
(e.g. "Stats and Calculus Workshop with Python" -> {MATH, PYTHON}). A book-level
label would repeat the exact folder-name mistake classify() made before. Returns a
frozenset — every type clearing threshold, not a forced single winner — so a mixed
chunk (prose intro + LaTeX proof) comes back as {FOUNDATION, MATH}, not a coin flip.

No LLM call. Acceptable here specifically because Tier 2 is embeddings-only: a wrong
or fuzzy tag costs a mistagged metadata filter, not a corrupted extraction.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path


class ContentType(Enum):
    SOFTWARE = "software"
    PYTHON = "python"
    MATH = "math"
    DISCRETE_MATH = "discrete_math"
    ML = "ml"
    ALGORITHMS = "algorithms"
    DATA = "data"
    FOUNDATION = "foundation"
    ARCHITECTURE = "architecture"


# ---------------------------------------------------------------------------
# Tier 1 — hand-labeled, from curated-slice.md. Exact book title -> label set.
# Presence here ALSO means: full kg_extractor graph extraction (§11), not just
# this content_type override.
# ---------------------------------------------------------------------------
BOOK_LABELS: dict[str, frozenset[ContentType]] = {
    "The Data Science Design Manual": frozenset({ContentType.FOUNDATION}),
    "Data Science From Scratch": frozenset({ContentType.FOUNDATION}),
    "Python Programming With Design Patterns": frozenset({ContentType.SOFTWARE, ContentType.PYTHON}),
    "Effective Python": frozenset({ContentType.SOFTWARE, ContentType.PYTHON}),
    "Clean Code": frozenset({ContentType.SOFTWARE, ContentType.FOUNDATION}),
    "Code Complete": frozenset({ContentType.SOFTWARE, ContentType.FOUNDATION}),
    "Fundamentals of Software Architecture": frozenset({ContentType.ARCHITECTURE}),
    "Software Architecture The Hard Parts": frozenset({ContentType.ARCHITECTURE}),
    "A Mathematical Introduction to Data Science": frozenset({ContentType.MATH}),
    "Stats and Calculus Workshop with Python": frozenset({ContentType.MATH, ContentType.PYTHON}),
    "Mathematics of Machine Learning": frozenset({ContentType.ML, ContentType.MATH}),
    "An Introduction to Statistical Learning with Applications in Python": frozenset({ContentType.ML, ContentType.MATH}),
    "The Elements of Statistical Learning": frozenset({ContentType.ML, ContentType.MATH}),
    "An Introduction to the Analysis of Algorithms": frozenset({ContentType.ALGORITHMS}),
    "Mathematical Structures for Computer Science": frozenset(
        {ContentType.DISCRETE_MATH, ContentType.ALGORITHMS, ContentType.DATA}
    ),
    "Learning SQL": frozenset({ContentType.DATA}),
    "SQL for Data Scientists": frozenset({ContentType.DATA}),
    "Python Machine Learning": frozenset({ContentType.ML, ContentType.PYTHON}),
    "Machine Learning Algorithms": frozenset({ContentType.ALGORITHMS, ContentType.ML}),
}

CURATED_SLICE: frozenset[str] = frozenset(BOOK_LABELS)  # = full-graph-extraction tier


# ---------------------------------------------------------------------------
# Tier 2 — density-scored heuristics, per chunk
# ---------------------------------------------------------------------------

# Each pattern is checked against the chunk text; SCORE_WEIGHT counts matches,
# normalized by chunk length so short and long chunks are comparable.
_CODE_FENCE = re.compile(r"```(.*?)```", re.S)
_CODE_TOKEN_IN_FENCE = re.compile(
    r"\b(def|class|return|import|function|void|public|private)\b"
    r"|\w+\s*\([^)]*\)\s*[:{]"     # function-call/def shape: name(args): or name(args){
    r"|;\s*$"                      # statement-terminating semicolon at line end
)


def _fenced_code_block_count(text: str) -> int:
    """
    Count fenced blocks that actually look like code, not just any ``` fence.
    Docling wraps equation/matrix layouts in ``` during PDF->markdown conversion
    too (grid-formatted systems of equations read as "preformatted" by the
    converter) — a bare fence is not sufficient signal on this corpus. Require
    a genuine code-shaped construct inside the fence.

    NOTE: an earlier version of this check accepted a bare brace/semicolon as
    signal, which false-positived on math set-builder notation like "{ 0 }"
    (the trivial subspace) inside an equation fence — braces are exactly as
    common in set notation as in code. Tightened to require a function-call/def
    shape or a genuinely statement-terminating semicolon.
    """
    return sum(
        1 for body in _CODE_FENCE.findall(text)
        if _CODE_TOKEN_IN_FENCE.search(body, re.M)
    )


_PATTERNS: dict[ContentType, list[re.Pattern]] = {
    ContentType.SOFTWARE: [
        re.compile(r"\bdef\s+\w+\s*\("),
        re.compile(r"\b(function|return)\s*\w*\s*[({]"),
        re.compile(r"^\s*(import|from)\s+\w+", re.M),
        re.compile(r"[;]{1}\s*$", re.M),                     # trailing semicolons (code lines)
        re.compile(r"\b(void|public|private|static)\s+\w+\("),
    ],
    ContentType.PYTHON: [
        re.compile(r"\bdef\s+\w+\("),
        re.compile(r"^\s*(import|from)\s+\w+", re.M),
        re.compile(r"\bself\.\w+"),
        re.compile(r"\bprint\("),
    ],
    ContentType.MATH: [
        re.compile(r"\$\$?[^$\n]{2,}\$\$?"),                  # inline/display LaTeX, non-trivial
        re.compile(r"\\(begin\{|frac\{|sum_|int_|partial |nabla)"),
        re.compile(r"\b(theorem|lemma|corollary)\b", re.I),
        re.compile(r"\bproof\b.{0,20}\bQ\.?E\.?D\.?", re.I | re.S),
        re.compile(r"[∑∫∂∇≤≥≠±√]"),
    ],
    ContentType.DISCRETE_MATH: [
        re.compile(r"\bvertex\b|\bvertices\b|\badjacency\b", re.I),
        re.compile(r"\bgraph\s+(traversal|theory|coloring)\b", re.I),
        re.compile(r"\bpermutations?\s+and\s+combinations?\b", re.I),
        re.compile(r"\brecurrence\s+relation\b", re.I),
        re.compile(r"\bbinary\s+tree\b|\bB-tree\b", re.I),
    ],
    ContentType.ML: [
        re.compile(r"\b(neural network|gradient descent|loss function|training set|epoch)\b", re.I),
        re.compile(r"\b(sklearn|tensorflow|pytorch)\b", re.I),
        re.compile(r"\bnumpy\.|pandas\.|torch\.|nn\.", re.I),
        re.compile(r"\b(classif(y|ier|ication)|regression model|overfit(ting)?)\b", re.I),
    ],
    ContentType.ALGORITHMS: [
        re.compile(r"\btime complexity\b|\bspace complexity\b", re.I),
        re.compile(r"\bO\(\s*(n|log|1|n\^?\d|n log n)\s*\)"),   # O(n), O(log n), O(n^2)...
        re.compile(r"\b(sorting algorithm|search algorithm|recursive algorithm)\b", re.I),
        re.compile(r"\b(binary search|quicksort|mergesort|dijkstra)\b", re.I),
    ],
    ContentType.DATA: [
        re.compile(r"\bSELECT\b.{0,100}\bFROM\b", re.I | re.S),
        re.compile(r"\bJOIN\b|\bINSERT INTO\b|\bCREATE TABLE\b", re.I),
        re.compile(r"\bprimary key\b|\bforeign key\b|\bdatabase schema\b", re.I),
    ],
    ContentType.ARCHITECTURE: [
        re.compile(r"\bmicroservices?\b|\bmonolith(ic)?\b", re.I),
        re.compile(r"\bsoftware architecture\b|\bdesign pattern\b", re.I),
        re.compile(r"\bdependency injection\b|\bevent-driven architecture\b", re.I),
    ],
}

# Minimum matches-per-1000-chars to count as "present". Raised from the initial
# guess after the first self-check run showed common-word false positives
# (e.g. "tree" as a gardening metaphor tripping discrete_math) — patterns above
# were also tightened to require CS-specific phrases/structures rather than
# single generic words. Re-validate with __main__ after any further pattern edit.
_THRESHOLD_PER_1000_CHARS = 2.5

# If nothing clears threshold, fall back to this rather than an empty set —
# most textbook prose that doesn't match any specific signal is still expository.
_DEFAULT = frozenset({ContentType.FOUNDATION})


def _score(text: str, patterns: list[re.Pattern]) -> float:
    n = sum(len(p.findall(text)) for p in patterns)
    return n / (len(text) / 1000) if text else 0.0


def classify_chunk(text: str) -> frozenset[ContentType]:
    """Tier 2: density-score a single chunk's text against each type's patterns."""
    hits = {
        ctype for ctype, patterns in _PATTERNS.items()
        if _score(text, patterns) >= _THRESHOLD_PER_1000_CHARS
    }
    # Fenced blocks with genuine code tokens inside are strong software signal,
    # checked separately from _PATTERNS since it needs to inspect fence CONTENTS,
    # not just count matches in the raw text (see _fenced_code_block_count).
    if _fenced_code_block_count(text) >= 1:
        hits.add(ContentType.SOFTWARE)
    return frozenset(hits) if hits else _DEFAULT


def route_signals(text: str) -> dict[str, float]:
    """Raw content signals for DOCUMENT-level chunker routing (ingest.py §2.1).

    classify_chunk()'s frozenset output can't drive whole-document routing: its
    threshold (_THRESHOLD_PER_1000_CHARS) is calibrated for individual chunks, and
    those signals dilute badly over a whole book (measured book-level densities run
    ~0.02–2.1, never near 2.5). This returns the underlying densities + genuine-code
    fence count so the caller can apply its own document-scaled thresholds instead of
    inheriting the per-chunk one.
    """
    return {
        "math": _score(text, _PATTERNS[ContentType.MATH]),
        "software": _score(text, _PATTERNS[ContentType.SOFTWARE]),
        "python": _score(text, _PATTERNS[ContentType.PYTHON]),
        "fences": float(_fenced_code_block_count(text)),
    }


def classify(text: str, source_path: str) -> frozenset[ContentType]:
    """
    Public entry point: Tier 1 (BOOK_LABELS) if the source title matches exactly,
    else Tier 2 (classify_chunk) on the chunk's own text.
    """
    title = Path(source_path).stem
    if title in BOOK_LABELS:
        return BOOK_LABELS[title]
    return classify_chunk(text)


def is_curated(source_path: str) -> bool:
    """True if this book is in the full-graph-extraction tier (§11)."""
    return Path(source_path).stem in CURATED_SLICE


# ---------------------------------------------------------------------------
# Self-check against real chunk data, if a checkpoint is available. Not a unit
# test suite — a sanity pass to look at BEFORE trusting the thresholds above.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    from collections import Counter

    path = sys.argv[1] if len(sys.argv) > 1 else "chunks_checkpoint.json"
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)

    tier_counts = Counter()
    type_counts = Counter()
    examples: dict[frozenset, list[str]] = {}

    for c in chunks:
        title = Path(c["source_path"]).stem
        tier = "tier1_book_labels" if title in BOOK_LABELS else "tier2_auto"
        tier_counts[tier] += 1

        labels = classify(c["text"], c["source_path"])
        type_counts[labels] += 1
        examples.setdefault(labels, []).append(c["text"][:120].replace("\n", " "))

    print(f"n={len(chunks)}  tiers: {dict(tier_counts)}\n")
    print("label-set distribution (top 15):")
    for labels, n in type_counts.most_common(15):
        names = "+".join(t.value for t in labels) or "(none)"
        print(f"  {n:>5}  {names}")

    print("\none example per label-set (first 5 sets):")
    for labels, exs in list(examples.items())[:5]:
        names = "+".join(t.value for t in labels) or "(none)"
        print(f"\n  [{names}]")
        print(f"    {exs[0]!r}")
