"""
docling -> RAG-Anything content_list converter.

Scaffold only: field names on docling's DocumentItem / TableItem / PictureItem
should be checked against your installed docling version, and against how
your existing pipeline currently tags [CODE] blocks and skips formula
regions. Swap the TODO-marked spots for your real logic.

Target schema (RAG-Anything insert_content_list):
    {"type": "text", "text": ..., "text_level": int, "page_idx": int}
    {"type": "image", "img_path": ..., "image_caption": [...], "page_idx": int}
    {"type": "table", "table_body": ..., "table_caption": [...], "page_idx": int}
    {"type": "equation", "latex": ..., "equation_caption": ..., "page_idx": int}
"""

from pathlib import Path
from docling.document_converter import DocumentConverter


def convert_docling_to_content_list(
    pdf_path: str,
    image_output_dir: str,
    content_category: str,  # "software_dev" | "math" | "self_help"
) -> list[dict]:
    """
    Run docling on a PDF and emit a RAG-Anything content_list.

    content_category drives the per-type rules your architecture doc
    already defines (code-block tagging for dev books, formula handling
    for math books).
    """
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    image_dir = Path(image_output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    content_list: list[dict] = []
    image_counter = 0

    for item, _level in doc.iterate_items():
        # docling page numbers are 1-based via item.prov; content_list wants 0-based
        page_idx = (item.prov[0].page_no - 1) if item.prov else 0
        label = getattr(item, "label", "").lower()

        # --- Headings / prose text -------------------------------------
        if label in ("title", "section_header", "text", "paragraph"):
            text = item.text.strip()
            if not text:
                continue

            # TODO: reproduce your existing [CODE] detection here.
            # If this block is code, either:
            #   (a) skip it and attach as metadata on the prior text item, or
            #   (b) emit it as a custom_type so it's still retrievable
            if content_category == "software_dev" and _looks_like_code(text):
                content_list.append({
                    "type": "custom_type",
                    "content": text,
                    "content_type": "code",
                    "page_idx": page_idx,
                })
                continue

            content_list.append({
                "type": "text",
                "text": text,
                "text_level": 1 if label in ("title", "section_header") else 0,
                "page_idx": page_idx,
            })

        # --- Tables ------------------------------------------------------
        elif label == "table":
            table_md = item.export_to_markdown(doc)  # TODO: confirm API on your docling version
            caption = getattr(item, "caption_text", None)
            content_list.append({
                "type": "table",
                "table_body": table_md,
                "table_caption": [caption] if caption else [],
                "page_idx": page_idx,
            })

        # --- Pictures / figures -------------------------------------------
        elif label in ("picture", "figure"):
            image_counter += 1
            img_path = image_dir / f"{Path(pdf_path).stem}_p{page_idx}_{image_counter}.png"
            pil_image = item.get_image(doc)  # TODO: confirm this returns a PIL image
            if pil_image is not None:
                pil_image.save(img_path)
                caption = getattr(item, "caption_text", None)
                content_list.append({
                    "type": "image",
                    "img_path": str(img_path.resolve()),
                    "image_caption": [caption] if caption else [],
                    "page_idx": page_idx,
                })

        # --- Equations -----------------------------------------------------
        elif label in ("formula", "equation"):
            # Currently your pipeline skips these for math books.
            # This branch reverses that: keep the LaTeX instead of dropping it.
            if content_category == "math":
                latex = getattr(item, "text", "").strip()  # TODO: confirm formula text field
                if latex:
                    content_list.append({
                        "type": "equation",
                        "latex": latex,
                        "equation_caption": None,
                        "page_idx": page_idx,
                    })

        # else: unhandled label, e.g. list items, footnotes -- extend as needed

    return content_list


def _looks_like_code(text: str) -> bool:
    """
    Placeholder for your existing [CODE] tagging heuristic.
    Replace with whatever rule your current extraction stage already uses.
    """
    code_signals = ("def ", "class ", "import ", "{", "};", "=>")
    return any(signal in text for signal in code_signals)


# --- Usage sketch --------------------------------------------------------
if __name__ == "__main__":
    content_list = convert_docling_to_content_list(
        pdf_path="example_dev_book.pdf",
        image_output_dir="./extracted_images",
        content_category="software_dev",
    )
    print(f"Extracted {len(content_list)} content_list items")

    # await rag.insert_content_list(
    #     content_list=content_list,
    #     file_path="example_dev_book.pdf",
    # )
