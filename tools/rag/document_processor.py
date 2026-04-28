"""Parses PDF and DOCX resumes into plain text chunks for RAG indexing."""
from pathlib import Path
from typing import Generator


def parse_pdf(path: str | Path) -> str:
    """Extract text from a PDF file."""
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(str(path))
    except Exception:
        pass
    # Fallback to PyPDF2
    import PyPDF2
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def parse_docx(path: str | Path) -> str:
    """Extract text from a DOCX file."""
    import docx
    doc = docx.Document(str(path))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def parse_txt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def parse_resume(path: str | Path) -> str:
    """Auto-detect format and extract resume text."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    elif suffix in (".docx", ".doc"):
        return parse_docx(path)
    elif suffix in (".txt", ".md"):
        return parse_txt(path)
    else:
        raise ValueError(f"Unsupported resume format: {suffix}")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


class DocumentProcessor:
    """High-level document processing for resumes and job descriptions."""

    def process_resume(self, path: str | Path) -> tuple[str, list[str]]:
        """Returns (full_text, chunks)."""
        text = parse_resume(path)
        chunks = chunk_text(text)
        return text, chunks

    def process_job_description(self, text: str) -> list[str]:
        return chunk_text(text, chunk_size=256, overlap=32)

    def process_past_application(self, text: str) -> list[str]:
        return chunk_text(text, chunk_size=256, overlap=32)
