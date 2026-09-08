from pathlib import Path
from io import BytesIO
import hashlib

import pypdfium2 as pdfium  # BSD-licensed PDF renderer (replaces AGPL PyMuPDF)
from PIL import Image


def pdf_page_count(pdf_path: str | Path) -> int:
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        return len(pdf)


def pdf_to_images(pdf_path: str | Path, dpi: int = 100) -> list[Image.Image]:
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        images = []
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            bitmap = page.render(scale=dpi / 72)
            img = bitmap.to_pil()
            img.load()
            images.append(img)
        return images


def pdf_to_base64_jpeg(pdf_path: str | Path, dpi: int = 100, quality: int = 75) -> list[str]:
    """Convert PDF pages to base64-encoded JPEG strings.

    Each page is rendered at the requested DPI and re-encoded as JPEG
    through PIL so the output size stays small for LLM upload.
    """
    import base64
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        results = []
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            bitmap = page.render(scale=dpi / 72)
            img = bitmap.to_pil()
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            results.append(b64)
        return results


def image_to_base64_jpeg(img: Image.Image, quality: int = 75) -> str:
    import base64
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
