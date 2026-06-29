"""
patch_lightrag.py

Registers the SurrealDB storage adapter in LightRAG 1.5.4's storage factory.
Run once after `pip install lightrag-hku`, or bake into the Docker build.
Safe to re-run — checks for existing entries before writing.

What this does:
  1. Copies surrealdb_impl.py into lightrag/kg/ so it's importable
  2. Adds four entries to the STORAGES dict in lightrag/kg/__init__.py
  3. Adds four entries to STORAGE_IMPLEMENTATIONS (validation registry)
  4. Adds four entries to STORAGE_ENV_REQUIREMENTS (env-var check registry)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Class name -> relative module path (matches STORAGES dict convention)
STORAGES_ENTRIES = {
    "SurrealDBKVStorage":        ".kg.surrealdb_impl",
    "SurrealDBVectorStorage":    ".kg.surrealdb_impl",
    "SurrealDBGraphStorage":     ".kg.surrealdb_impl",
    "SurrealDBDocStatusStorage": ".kg.surrealdb_impl",
}

# Entries to add to each STORAGE_IMPLEMENTATIONS bucket
IMPL_ENTRIES = {
    "KV_STORAGE":        "SurrealDBKVStorage",
    "VECTOR_STORAGE":    "SurrealDBVectorStorage",
    "GRAPH_STORAGE":     "SurrealDBGraphStorage",
    "DOC_STATUS_STORAGE":"SurrealDBDocStatusStorage",
}

# No special env vars required (connection params read from .env via python-dotenv)
ENV_REQUIREMENTS = {name: [] for name in STORAGES_ENTRIES}


def find_lightrag_kg_init() -> Path:
    spec = importlib.util.find_spec("lightrag")
    if spec is None or spec.origin is None:
        print("ERROR: lightrag is not installed. Run: pip install lightrag-hku")
        sys.exit(1)
    kg_init = Path(spec.origin).parent / "kg" / "__init__.py"
    if not kg_init.exists():
        print(f"ERROR: Could not find {kg_init}")
        sys.exit(1)
    return kg_init


def find_surrealdb_impl_src() -> Path:
    here = Path(__file__).parent
    impl = here / "surrealdb_impl.py"
    if not impl.exists():
        print(f"ERROR: surrealdb_impl.py not found at {impl}")
        sys.exit(1)
    return impl


def copy_impl(kg_init: Path, impl_src: Path) -> None:
    dest = kg_init.parent / "surrealdb_impl.py"
    dest.write_text(impl_src.read_text())
    print(f"  Copied surrealdb_impl.py → {dest}")


def patch_storages_dict(content: str) -> str:
    """Add entries to the STORAGES = { ... } dict."""
    already = [name for name in STORAGES_ENTRIES if f'"{name}"' in content]
    to_add = {k: v for k, v in STORAGES_ENTRIES.items() if k not in already}
    if not to_add:
        print("  STORAGES dict: all entries already present.")
        return content

    # Find the closing brace of STORAGES = { ... }
    start = content.find("STORAGES = {")
    if start == -1:
        print("  WARNING: STORAGES dict not found — appending at end of file.")
        lines = [f'    "{k}": "{v}",' for k, v in to_add.items()]
        return content + "\n# SurrealDB (added by patch_lightrag.py)\n" + "\n".join(lines) + "\n"

    # Find the matching closing brace
    depth, i = 0, start
    while i < len(content):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1

    insert_pos = i  # position of closing }
    injection = "\n    # SurrealDB adapter (added by patch_lightrag.py)\n"
    for k, v in to_add.items():
        injection += f'    "{k}": "{v}",\n'
        print(f"    + STORAGES[\"{k}\"] = \"{v}\"")

    return content[:insert_pos] + injection + content[insert_pos:]


def patch_storage_implementations(content: str) -> str:
    """Add class names to each bucket's 'implementations' list."""
    import re
    patched = content
    for bucket, class_name in IMPL_ENTRIES.items():
        if class_name in patched:
            print(f"  STORAGE_IMPLEMENTATIONS[{bucket}]: {class_name} already present.")
            continue
        # Find the implementations list for this bucket
        pattern = rf'("{bucket}".*?"implementations":\s*\[)(.*?)(\])'
        def inserter(m, cn=class_name):
            return m.group(1) + m.group(2) + f'            "{cn}",\n        ' + m.group(3)
        new, count = re.subn(pattern, inserter, patched, flags=re.DOTALL)
        if count:
            patched = new
            print(f"    + STORAGE_IMPLEMENTATIONS[\"{bucket}\"][\"implementations\"] += [\"{class_name}\"]")
        else:
            print(f"  WARNING: could not find bucket {bucket} in STORAGE_IMPLEMENTATIONS")
    return patched


def patch_env_requirements(content: str) -> str:
    """Add entries to STORAGE_ENV_REQUIREMENTS dict."""
    to_add = {k: v for k, v in ENV_REQUIREMENTS.items() if f'"{k}"' not in content}
    if not to_add:
        print("  STORAGE_ENV_REQUIREMENTS: all entries already present.")
        return content

    start = content.find("STORAGE_ENV_REQUIREMENTS")
    if start == -1:
        print("  WARNING: STORAGE_ENV_REQUIREMENTS not found — skipping.")
        return content

    # Find its closing brace
    depth, i = 0, start
    while i < len(content):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1

    injection = "\n    # SurrealDB adapter (added by patch_lightrag.py)\n"
    for k in to_add:
        injection += f'    "{k}": [],\n'
        print(f"    + STORAGE_ENV_REQUIREMENTS[\"{k}\"] = []")

    return content[:i] + injection + content[i:]


def main() -> None:
    print("=== Patching lightrag-hku 1.5.4 to register SurrealDB adapter ===")
    kg_init  = find_lightrag_kg_init()
    impl_src = find_surrealdb_impl_src()
    print(f"  LightRAG kg init: {kg_init}")

    copy_impl(kg_init, impl_src)

    content = kg_init.read_text()
    content = patch_storages_dict(content)
    content = patch_storage_implementations(content)
    content = patch_env_requirements(content)
    kg_init.write_text(content)

    print("=== Patch complete ===")


if __name__ == "__main__":
    main()
