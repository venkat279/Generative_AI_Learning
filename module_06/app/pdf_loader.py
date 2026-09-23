"""Document ingestion and parsing - loading PDFs.

One concrete example of the roadmap's "Document Ingestion and Parsing"
topic: extracting plain text from a PDF file with pypdf (a pure-Python,
open-source PDF library - no PyMuPDF/poppler system dependency needed).
The other formats the roadmap lists (Word, HTML, CSV/Excel, code files,
OCR) aren't exercised here - one practical example per topic.
"""

from pathlib import Path

from pypdf import PdfReader


def load_pdf(path: Path) -> str:
    """Extract and concatenate the text of every page in a PDF."""
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() for page in reader.pages)
