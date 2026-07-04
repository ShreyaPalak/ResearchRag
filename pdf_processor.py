import re
from pypdf import PdfReader


def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    """Extract and clean text from each page of a PDF.

    Returns a list of {'page': page_num, 'text': cleaned_text} dicts,
    skipping pages with no extractable text (e.g. scanned images).
    """
    reader = PdfReader(pdf_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text()
        if not raw or not raw.strip():
            continue
        cleaned = re.sub(r'[ \t]+', ' ', raw)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        pages.append({'page': page_num, 'text': cleaned})
    return pages
