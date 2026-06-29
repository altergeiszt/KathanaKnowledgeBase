"""
patch_lightrag.py

Registers the SurrealDB storage adapter in LightRAG's storage factory.
Run once after `pip install lightrag-hku`, or bake into the Docker build.

Safe to re-run — checks for existing entries before writing.
"""

import importlib.util
import sys
from pathlib import Path


REGISTRATIONS = {
    "SurrealDBKVStorage":        "lightrag.kg.surrealdb_impl.SurrealDBKVStorage",
    "SurrealDBVectorStorage":    "lightrag.kg.surrealdb_impl.SurrealDBVectorStorage",
    "SurrealDBGraphStorage":     "lightrag.kg.surrealdb_impl.SurrealDBGraphStorage",
    "SurrealDBDocStatusStorage": "lightrag.kg.surrealdb_impl.SurrealDBDocStatusStorage",
}


def find_lightrag_kg_init() -> Path:
    """Locate lightrag/kg/__init__.py in the active Python environment."""
    spec = importlib.util.find_spec("lightrag")
    if spec is None or spec.origin is None:
        print("ERROR: lightrag is not installed. Run: pip install lightrag-hku")
        sys.exit(1)
    lightrag_root = Path(spec.origin).parent
    kg_init = lightrag_root / "kg" / "__init__.py"
    if not kg_init.exists():
        print(f"ERROR: Could not find {kg_init}")
        print("Your version of lightrag-hku may have a different structure.")
        sys.exit(1)
    return kg_init


def find_surrealdb_impl_src() -> Path:
    """Find surrealdb_impl.py relative to this script."""
    here = Path(__file__).parent
    impl = here / "surrealdb_impl.py"
    if not impl.exists():
        print(f"ERROR: surrealdb_impl.py not found at {impl}")
        sys.exit(1)
    return impl


def copy_impl_to_lightrag(kg_init: Path, impl_src: Path) -> None:
    """Copy surrealdb_impl.py into lightrag/kg/ so it's importable."""
    dest = kg_init.parent / "surrealdb_impl.py"
    dest.write_text(impl_src.read_text())
    print(f"  Copied surrealdb_impl.py → {dest}")


def patch_init(kg_init: Path) -> None:
    """
    Insert STORAGE_IMPLEMENTATIONS entries into lightrag/kg/__init__.py.
    Finds the STORAGE_IMPLEMENTATIONS dict and appends missing entries
    directly after the last existing entry in that block.
    """
    original = kg_init.read_text()
    lines = original.splitlines()

    # Find lines already registering something so we know the dict name
    insert_after = -1
    for i, line in enumerate(lines):
        if "STORAGE_IMPLEMENTATIONS" in line and "=" in line:
            insert_after = i

    if insert_after == -1:
        # Dict not found — append at end of file as a fallback
        print("  WARNING: STORAGE_IMPLEMENTATIONS dict not found in __init__.py")
        print("  Appending registrations at end of file.")
        new_lines = lines + ["", "# SurrealDB adapter registrations (added by patch_lightrag.py)"]
        insert_after = len(new_lines) - 1
    else:
        new_lines = lines[:]

    already_registered = {k for k, v in REGISTRATIONS.items() if any(k in l for l in lines)}
    to_add = {k: v for k, v in REGISTRATIONS.items() if k not in already_registered}

    if not to_add:
        print("  All SurrealDB entries already present — nothing to patch.")
        return

    injection = []
    for key, value in to_add.items():
        injection.append(f'STORAGE_IMPLEMENTATIONS["{key}"] = "{value}"')

    # Insert after the last found STORAGE_IMPLEMENTATIONS line
    patched = new_lines[:insert_after + 1] + injection + new_lines[insert_after + 1:]
    kg_init.write_text("\n".join(patched) + "\n")
    print(f"  Patched {kg_init}")
    for line in injection:
        print(f"    + {line}")


def main() -> None:
    print("=== Patching lightrag-hku to register SurrealDB adapter ===")
    kg_init  = find_lightrag_kg_init()
    impl_src = find_surrealdb_impl_src()
    print(f"  LightRAG kg init: {kg_init}")
    copy_impl_to_lightrag(kg_init, impl_src)
    patch_init(kg_init)
    print("=== Patch complete ===")


if __name__ == "__main__":
    main()
