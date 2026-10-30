from pathlib import Path
from io import BytesIO
import hashlib

import pypdfium2 as pdfium  # BSD-licensed PDF renderer (replaces AGPL PyMuPDF)
from PIL import Image

# ``page.render(scale=dpi/72)`` is relative to the PAGE BOX, not to the scan
# inside it. A scanner that shrink-wraps its output writes a tiny MediaBox (e.g.
# 210x280 pt = 2.9x3.9 in) around a high-resolution embedded JPEG (1500x2000 px,
# ~514 effective DPI). Rendering such a page at "200 DPI" yields only 583x778 px
# and silently throws away ~70% of the linear detail — small digits become
# unreadable and the model misreads cells (it read the reference column's first
# number as the measurement). A normal A4/Letter page has a ~595x842 pt box, so
# the same 200 DPI already renders ~1650x2360 px and is unaffected.
#
# So: when a page box is anomalously small, render at the embedded scan's NATIVE
# resolution (the most real detail that exists) instead of the page-box DPI.
_SMALL_PAGE_LONG_PT = 500.0   # ~7 in: below this the box is shrink-wrapped, not a real page
_MAX_BOOSTED_LONG_PX = 2400.0  # payload ceiling so a 600-DPI scan can't explode the base64


def pdf_page_count(pdf_path: str | Path) -> int:
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        return len(pdf)


def _native_image_dpi(page) -> float:
    """Effective DPI of the highest-resolution image embedded in a page.

    ``get_px_size()`` is the scan's real pixel size; ``get_bounds()`` is the page
    region (in points) it is drawn into. The limiting axis gives the DPI the
    scanner actually captured. Returns 0.0 for vector/text-only pages or if the
    object walk fails — such pages keep the plain DPI rendering.
    """
    best = 0.0
    try:
        for obj in page.get_objects():
            if isinstance(obj, pdfium.PdfImage):
                px_w, px_h = obj.get_px_size()
                bounds = obj.get_bounds()
                w_pt, h_pt = bounds[2] - bounds[0], bounds[3] - bounds[1]
                if w_pt > 0 and h_pt > 0:
                    best = max(best, min(px_w / (w_pt / 72), px_h / (h_pt / 72)))
    except Exception:
        return 0.0
    return best


def _render_scale(page, dpi: int) -> float:
    """Render scale for one page: ``dpi/72``, except that a shrink-wrapped page
    box is scaled up to the native resolution of its embedded scan (capped), so
    the scan's real detail reaches the model instead of being downsampled away.
    """
    scale = dpi / 72
    page_w_pt, page_h_pt = page.get_size()
    long_pt = max(page_w_pt, page_h_pt)
    if long_pt <= 0 or long_pt >= _SMALL_PAGE_LONG_PT:
        return scale  # a normal page: the DPI number means what it says
    native_dpi = _native_image_dpi(page)
    if native_dpi <= dpi:
        return scale  # nothing to gain — the DPI request already meets/exceeds the scan
    target_px = min(native_dpi / 72 * long_pt, _MAX_BOOSTED_LONG_PX)
    return max(scale, target_px / long_pt)


def pdf_to_images(pdf_path: str | Path, dpi: int = 100) -> list[Image.Image]:
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        images = []
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            bitmap = page.render(scale=_render_scale(page, dpi))
            img = bitmap.to_pil()
            img.load()
            images.append(img)
        return images


def pdf_to_base64_jpeg(pdf_path: str | Path, dpi: int = 100, quality: int = 75) -> list[str]:
    """Convert PDF pages to base64-encoded JPEG strings.

    Each page is rendered at the requested DPI (raised to the embedded scan's
    native resolution for shrink-wrapped page boxes — see ``_render_scale``) and
    re-encoded as JPEG through PIL so the output size stays small for LLM upload.
    """
    import base64
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        results = []
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            bitmap = page.render(scale=_render_scale(page, dpi))
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
