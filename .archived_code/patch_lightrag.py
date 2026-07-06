"""
patch_lightrag.py

Registers the SurrealDB storage adapter in LightRAG 1.3.9 (pinned in requirements.txt).
Safe to re-run — all operations are idempotent.

Three things it does:
  1. Copies surrealdb_impl.py into lightrag/kg/ so it's importable
  2. Appends SurrealDB class names to lightrag/kg/__init__.py by direct
     string manipulation (avoids fragile regex on nested dicts)
  3. Adds entries to STORAGES, STORAGE_IMPLEMENTATIONS, and
     STORAGE_ENV_REQUIREMENTS
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SURREAL_CLASSES = {
    "KV_STORAGE":         "SurrealDBKVStorage",
    "VECTOR_STORAGE":     "SurrealDBVectorStorage",
    "GRAPH_STORAGE":      "SurrealDBGraphStorage",
    "DOC_STATUS_STORAGE": "SurrealDBDocStatusStorage",
}

MODULE_PATH = ".kg.surrealdb_impl"


def find_kg_init() -> Path:
    spec = importlib.util.find_spec("lightrag")
    if spec is None:
        print("ERROR: lightrag not installed. Run: pip install lightrag-hku")
        sys.exit(1)
    kg_init = Path(spec.origin).parent / "kg" / "__init__.py"
    if not kg_init.exists():
        print(f"ERROR: {kg_init} not found")
        sys.exit(1)
    return kg_init


def copy_impl(kg_init: Path) -> None:
    src = Path(__file__).parent / "surrealdb_impl.py"
    if not src.exists():
        print(f"ERROR: surrealdb_impl.py not found next to patch_lightrag.py")
        sys.exit(1)
    dest = kg_init.parent / "surrealdb_impl.py"
    dest.write_text(src.read_text())
    print(f"  Copied surrealdb_impl.py → {dest}")


def patch_file(kg_init: Path) -> None:
    content = kg_init.read_text()
    original = content

    # ── 1. STORAGES dict ────────────────────────────────────────────────
    # Add entries before the closing brace of STORAGES = { ... }
    # Strategy: find `}\n\n\ndef verify` which is the end of STORAGES
    for class_name in SURREAL_CLASSES.values():
        if f'"{class_name}"' in content:
            print(f"  STORAGES: {class_name} already present")
            continue
        new_entry = f'    "{class_name}": "{MODULE_PATH}",\n'
        # Insert before the closing brace of STORAGES
        marker = '}\n\n\ndef verify_storage_implementation'
        if marker in content:
            content = content.replace(marker, new_entry + marker, 1)
            print(f"  STORAGES: added {class_name}")
        else:
            print(f"  WARNING: could not find STORAGES closing brace for {class_name}")

    # ── 2. STORAGE_IMPLEMENTATIONS buckets ──────────────────────────────
    # Each bucket looks like:
    #   "KV_STORAGE": {
    #       "implementations": [
    #           "JsonKVStorage",
    #           ...
    #       ],
    # Strategy: find the implementations list for each bucket and append
    for bucket, class_name in SURREAL_CLASSES.items():
        if f'"{class_name}"' in content:
            # already added in STORAGES step or previous run
            # but check specifically in the implementations list
            bucket_start = content.find(f'"{bucket}"')
            if bucket_start != -1:
                bucket_end = content.find('"required_methods"', bucket_start)
                bucket_section = content[bucket_start:bucket_end]
                if class_name in bucket_section:
                    print(f"  STORAGE_IMPLEMENTATIONS[{bucket}]: {class_name} already present")
                    continue

        # Find the implementations list for this bucket and insert before its closing ]
        bucket_start = content.find(f'"{bucket}"')
        if bucket_start == -1:
            print(f"  WARNING: bucket {bucket} not found in STORAGE_IMPLEMENTATIONS")
            continue

        # Find the closing ] of the implementations list within this bucket
        impl_start = content.find('"implementations": [', bucket_start)
        if impl_start == -1:
            print(f"  WARNING: implementations list not found for {bucket}")
            continue

        bracket_open  = content.find('[', impl_start)
        bracket_close = content.find(']', bracket_open)

        new_entry = f'\n            "{class_name}",'
        content = content[:bracket_close] + new_entry + content[bracket_close:]
        print(f"  STORAGE_IMPLEMENTATIONS[{bucket}]: added {class_name}")

    # ── 3. STORAGE_ENV_REQUIREMENTS dict ───────────────────────────────
    # Add empty env requirement entries before the closing brace
    for class_name in SURREAL_CLASSES.values():
        env_entry = f'    "{class_name}": [],'
        if env_entry.strip() in content or f'"{class_name}": []' in content:
            print(f"  STORAGE_ENV_REQUIREMENTS: {class_name} already present")
            continue
        # Insert before the last } of STORAGE_ENV_REQUIREMENTS
        # Find STORAGE_ENV_REQUIREMENTS dict end (before STORAGES = {)
        env_start = content.find("STORAGE_ENV_REQUIREMENTS")
        storages_start = content.find("STORAGES = {")
        env_section = content[env_start:storages_start]
        env_close = env_section.rfind("}")
        insert_pos = env_start + env_close
        new_line = f'\n    "{class_name}": [],  # SurrealDB (added by patch_lightrag.py)\n'
        content = content[:insert_pos] + new_line + content[insert_pos:]
        print(f"  STORAGE_ENV_REQUIREMENTS: added {class_name}")

    if content != original:
        kg_init.write_text(content)
        print(f"  Wrote patched {kg_init}")
    else:
        print("  No changes needed — already fully patched")


def verify(kg_init: Path) -> None:
    """Quick sanity check — import the module and confirm classes are registered."""
    import importlib
    try:
        # Force reload since we just modified the file
        import lightrag.kg as kg_module
        importlib.reload(kg_module)
        for class_name in SURREAL_CLASSES.values():
            assert class_name in kg_module.STORAGES, \
                f"{class_name} missing from STORAGES"
        for bucket, class_name in SURREAL_CLASSES.items():
            implementations = kg_module.STORAGE_IMPLEMENTATIONS[bucket]["implementations"]
            assert class_name in implementations, \
                f"{class_name} missing from STORAGE_IMPLEMENTATIONS[{bucket}]"
        print("  Verification passed — all classes registered correctly")
    except Exception as e:
        print(f"  WARNING: verification failed: {e}")
        print("  The patch may still work at runtime — check docker logs on startup")


def main() -> None:
    print("=== Patching lightrag-hku for SurrealDB adapter ===")
    kg_init = find_kg_init()
    print(f"  Target: {kg_init}")
    copy_impl(kg_init)
    patch_file(kg_init)
    verify(kg_init)
    print("=== Done ===")


if __name__ == "__main__":
    main()