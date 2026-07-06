import hashlib
from pathlib import Path

def stable_doc_id(pdf_path: str, library_root: str) -> str:
    """
    Stable across re-runs: based on the book's path within your library,
    not on the extracted content. A book at the same relative path
    always gets the same doc_id, even if extraction output shifts.
    """
    relative_path = str(Path(pdf_path).resolve().relative_to(Path(library_root).resolve()))
    digest = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
    return f"doc-{digest}"

doc_id = stable_doc_id(pdf_path, library_root="./library")

existing = await rag.lightrag.doc_status.get_by_id(doc_id)
if existing:
    # Book was already ingested — remove old entities/chunks/graph edges
    # before reinserting, or you'll get duplicates alongside the update.
    await rag.lightrag.adelete_by_doc_id(doc_id)

await rag.insert_content_list(
    content_list=content_list,
    file_path=pdf_path,
    doc_id=doc_id,
)