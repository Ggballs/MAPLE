from __future__ import annotations

from pathlib import Path


def export_screenshots(pdf_path: Path, output_dir: Path, scale: float = 2.0) -> int:
    """Render one PDF into page_XXXX.png files.

    PyMuPDF is preferred when installed. pypdfium2 is used as a lightweight
    fallback so the public experiment code can run without legacy preprocessors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import fitz  # type: ignore

        doc = fitz.open(pdf_path)
        try:
            for index, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                pix.save(output_dir / f"page_{index:04d}.png")
            return len(doc)
        finally:
            doc.close()
    except ModuleNotFoundError:
        pass

    try:
        import pypdfium2 as pdfium  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Neither PyMuPDF (fitz) nor pypdfium2 is installed") from exc

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for index, page in enumerate(pdf, start=1):
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            try:
                image.save(output_dir / f"page_{index:04d}.png")
            finally:
                image.close()
        return len(pdf)
    finally:
        pdf.close()
