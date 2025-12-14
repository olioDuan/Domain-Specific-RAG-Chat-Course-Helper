# loader.py

from pathlib import Path
from typing import List, Optional
import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


import config


def parse_lecture_id(pdf_path: Path) -> str:
    """
    Normalize lecture filenames like 'Lecture1.pdf', 'Lecture2.pdf', 'Lec1_Intro.pdf' to 'Lec X'.
    """
    name = pdf_path.stem  # filename without extension

    # first try matching 'LectureX'
    m = re.search(r"[Ll]ecture\s*(\d+)", name)
    if not m:
        # then try matching 'LecX'
        m = re.search(r"[Ll]ec\s*(\d+)", name)

    if m:
        return f"Lec {m.group(1)}"

    # if parsing fails, fall back to the original filename
    return name



def is_outline_page(lines: list[str]) -> bool:
    """
    Heuristic to detect an outline page:
    - if the first few lines contain 'Outline'
    """
    for line in lines[:5]:
        if "outline" in line.lower():
            return True
    return False


def clean_page_text(raw_text: str) -> str:
    """
    Clean text for a single page:
    - remove empty lines
    - remove headers/footers (NYU TANDON SCHOOL OF ENGINEERING, standalone page numbers, etc.)
    - drop outline pages
    """
    lines = [l.strip() for l in raw_text.splitlines()]

    cleaned_lines: list[str] = []
    for line in lines:
        if not line:
            continue

        upper = line.upper()

        # remove school header
        if "NYU TANDON SCHOOL OF ENGINEERING" in upper:
            continue

        # remove plain page numbers / "Page x of y"
        if re.match(r"^(\d+|Page\s+\d+(\s+of\s+\d+)?)$", line, re.IGNORECASE):
            continue

        # remove very short logo-like strings
        if len(line) <= 2 and not line.isalpha():
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def load_all_slides(slides_dir: Optional[Path] = None) -> List[Document]:
    """
    Load all lecture PDFs from the Slides/ directory, treat each page as a Document,
    and attach lecture/page metadata.
    """
    if slides_dir is None:
        slides_dir = config.SLIDES_DIR

    docs: List[Document] = []

    pdf_files = sorted(Path(slides_dir).glob("*.pdf"))
    for pdf_path in pdf_files:
        lecture_id = parse_lecture_id(pdf_path)
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()  # each page is a Document

        for page_doc in pages:
            cleaned = clean_page_text(page_doc.page_content)
            if not cleaned:
                continue

            meta = dict(page_doc.metadata or {})
            # prefer page number from metadata, default to 0 if missing
            page_num = meta.get("page", meta.get("page_number", 0))

            meta.update(
                {
                    "source": pdf_path.name,
                    "lecture": lecture_id,
                    "page": page_num,
                }
            )

            docs.append(
                Document(
                    page_content=cleaned,
                    metadata=meta,
                )
            )

    return docs