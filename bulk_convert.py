#!/usr/bin/env python3
"""
bulk_convert.py — batch-convert documents, spreadsheets, emails, and scanned
images in a source directory into Markdown files in an output directory.

    .docx / .xlsx / .pptx / .pdf / .eml / .msg  -> converted via markitdown
    .png / .jpg / .jpeg / .tiff                 -> OCR'd via pytesseract

Design notes (security):
  - Runs fully offline: no network calls are made by this script, and
    MarkItDown is instantiated with no LLM/API client, so file contents
    (which may be privileged or confidential) never leave the machine.
  - Original files are always opened read-only and are never modified or
    deleted.
  - Symlinks are skipped (files and directories) to avoid escaping the
    source tree. Every output path is re-validated to stay inside the
    output directory before anything is written.
  - Oversized files are skipped rather than loaded, and Pillow's
    decompression-bomb guard is left enabled, to bound memory/CPU use
    against malformed or hostile input files.
  - Output .md files and directories are created with restrictive
    permissions (0600 / 0700) since extracted text may be confidential.
  - A failure on any single file is caught and logged; the batch
    continues so one bad file can't stop the run.
"""

import argparse
import logging
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from markitdown import MarkItDown
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

DOC_EXTENSIONS = {".docx", ".xlsx", ".pptx", ".pdf", ".eml", ".msg"}
OOXML_EXTENSIONS = {".docx", ".xlsx", ".pptx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff"}
SUPPORTED_EXTENSIONS = DOC_EXTENSIONS | IMAGE_EXTENSIONS

DEFAULT_MAX_FILE_SIZE_MB = 200
MAX_PDF_PAGES = 500
MAX_PDF_PAGE_POINTS = 5000  # ~69in; guards against a hostile oversized page
PDF_OCR_DPI = 300

log = logging.getLogger("bulk_convert")


class ConversionError(Exception):
    pass


def is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def build_output_path(source_root: Path, output_root: Path, source_file: Path) -> Path:
    relative = source_file.relative_to(source_root)
    # Rebuild from sanitized components only (defends against any component
    # that isn't a plain name, e.g. "..", even though a real filesystem walk
    # shouldn't produce one).
    safe_relative = Path(*[Path(part).name for part in relative.parts])
    output_path = (output_root / safe_relative).with_suffix(".md")
    if not is_within_directory(output_path, output_root):
        raise ConversionError(f"refusing to write outside output directory: {output_path}")
    return output_path


def markdown_header(title: str, relative_source: Path, kind: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"# {title}\n\n"
        f"*Converted from `{relative_source.as_posix()}` on {timestamp} ({kind})*\n\n"
        "---\n\n"
    )


def is_blank(text: str) -> bool:
    # markitdown occasionally returns the literal string "None" when an underlying
    # converter (e.g. mammoth on a malformed .docx) fails internally without raising.
    return text.strip().lower() in ("", "none")


def convert_document(source_file: Path, converter: MarkItDown) -> str:
    suffix = source_file.suffix.lower()
    try:
        with open(source_file, "rb") as fh:
            result = converter.convert_stream(fh, file_extension=suffix)
        return result.text_content or ""
    except (KeyError, zipfile.BadZipFile) as exc:
        if suffix in OOXML_EXTENSIONS:
            raise ConversionError(
                f"not a valid modern Office file ({exc}) - likely an old .doc/.xls/.ppt "
                f"file saved with a .{suffix.lstrip('.')} extension, or a corrupted file; "
                "re-save it from Word/Excel/PowerPoint (or export as PDF) and retry"
            ) from exc
        raise


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    return ImageOps.autocontrast(gray)


def ocr_image(source_file: Path, lang: str) -> str:
    import pytesseract

    with open(source_file, "rb") as fh:
        try:
            image = Image.open(fh)
            image.load()
        except Image.DecompressionBombError as exc:
            raise ConversionError(f"image exceeds safe pixel-count limit: {exc}") from exc
        except UnidentifiedImageError as exc:
            raise ConversionError(f"not a readable image file: {exc}") from exc

        with image:
            frames = list(ImageSequence.Iterator(image)) if getattr(image, "n_frames", 1) > 1 else [image]
            pages = []
            for index, frame in enumerate(frames, start=1):
                text = pytesseract.image_to_string(preprocess_for_ocr(frame), lang=lang).strip()
                pages.append(f"--- Page {index} ---\n\n{text}" if len(frames) > 1 else text)
    return "\n\n".join(pages)


def ocr_pdf(source_file: Path, lang: str) -> str:
    import pytesseract

    pages = []
    with pymupdf.open(source_file) as doc:
        if doc.page_count > MAX_PDF_PAGES:
            raise ConversionError(f"PDF has {doc.page_count} pages, exceeds safety limit of {MAX_PDF_PAGES}")
        for index, page in enumerate(doc, start=1):
            if page.rect.width > MAX_PDF_PAGE_POINTS or page.rect.height > MAX_PDF_PAGE_POINTS:
                pages.append(f"--- Page {index} ---\n\n*[page too large to OCR safely, skipped]*")
                continue
            pix = page.get_pixmap(dpi=PDF_OCR_DPI)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text = pytesseract.image_to_string(preprocess_for_ocr(image), lang=lang).strip()
            pages.append(f"--- Page {index} ---\n\n{text}" if doc.page_count > 1 else text)
    return "\n\n".join(pages)


def process_file(
    source_file: Path,
    source_root: Path,
    output_root: Path,
    converter: MarkItDown,
    lang: str,
) -> bool:
    """Returns True if the output has no meaningful extracted text (still written, but flagged)."""
    relative = source_file.relative_to(source_root)
    suffix = source_file.suffix.lower()

    if suffix in DOC_EXTENSIONS:
        body = convert_document(source_file, converter)
        used_ocr_fallback = False
        if suffix == ".pdf" and is_blank(body):
            ocr_text = ocr_pdf(source_file, lang)
            if not is_blank(ocr_text):
                body = ocr_text
                used_ocr_fallback = True
        empty = is_blank(body)
        kind = "document, OCR fallback" if used_ocr_fallback else "document"
        header = markdown_header(source_file.name, relative, kind)
        if empty:
            note = "even after OCR " if used_ocr_fallback else ""
            body = f"*No extractable text found {note}in this file.*\n"
    elif suffix in IMAGE_EXTENSIONS:
        body = ocr_image(source_file, lang)
        empty = is_blank(body)
        header = markdown_header(source_file.name, relative, "OCR")
        body = (
            f"```text\n{body}\n```\n"
            if not empty
            else "*No extractable text found by OCR (image may contain no legible text).*\n"
        )
    else:
        raise ConversionError(f"unsupported extension: {suffix}")

    output_path = build_output_path(source_root, output_root, source_file)
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_path.write_text(header + body, encoding="utf-8")
    output_path.chmod(0o600)
    return empty


def iter_candidate_files(source_root: Path, recursive: bool):
    walker = source_root.rglob("*") if recursive else source_root.iterdir()
    for path in sorted(walker):
        if any(part.startswith(".") for part in path.relative_to(source_root).parts):
            continue
        if path.is_symlink():
            log.warning("skip (symlink, not followed): %s", path.relative_to(source_root))
            continue
        if not path.is_file():
            continue
        yield path


def run(source_dir: Path, output_dir: Path, recursive: bool, max_size_bytes: int, lang: str) -> int:
    if shutil.which("tesseract") is None:
        log.error("tesseract binary not found on PATH; install it first (e.g. `sudo apt-get install tesseract-ocr`)")
        return 1

    source_root = source_dir.resolve()
    output_root = output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    converter = MarkItDown()  # no LLM client / API credentials: conversion stays local

    converted, skipped, failed, empty_text = 0, 0, [], []

    for source_file in iter_candidate_files(source_root, recursive):
        relative = source_file.relative_to(source_root)
        suffix = source_file.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        try:
            size = source_file.stat().st_size
        except OSError as exc:
            log.warning("skip (cannot stat %s): %s", relative, exc)
            skipped += 1
            continue

        if size > max_size_bytes:
            log.warning("skip (%.1f MB exceeds limit): %s", size / (1024 * 1024), relative)
            skipped += 1
            continue

        try:
            was_empty = process_file(source_file, source_root, output_root, converter, lang)
            if was_empty:
                log.warning("no extractable text: %s", relative)
                empty_text.append(str(relative))
            else:
                log.info("converted: %s", relative)
            converted += 1
        except Exception as exc:  # noqa: BLE001 - keep the batch going on any single-file failure
            log.error("FAILED: %s (%s: %s)", relative, type(exc).__name__, exc)
            failed.append(str(relative))

    log.info(
        "done: %d converted (%d with no extractable text), %d skipped, %d failed",
        converted,
        len(empty_text),
        skipped,
        len(failed),
    )
    if empty_text:
        log.info("files with no extractable text (likely scans/photos with no legible text - originals untouched):")
        for name in empty_text:
            log.info("  - %s", name)
    if failed:
        log.info("files that could not be converted at all (originals untouched):")
        for name in failed:
            log.info("  - %s", name)
    return 0 if not failed else 2


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path, help="directory containing files to convert")
    parser.add_argument("output_dir", type=Path, help="directory to write converted .md files into")
    parser.add_argument("--no-recursive", action="store_true", help="only scan the top level of source_dir")
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=DEFAULT_MAX_FILE_SIZE_MB,
        help=f"skip files larger than this (default: {DEFAULT_MAX_FILE_SIZE_MB} MB)",
    )
    parser.add_argument("--lang", default="eng", help="Tesseract language code(s), e.g. 'eng' or 'eng+fra' (default: eng)")
    parser.add_argument("--log-file", type=Path, help="also write the run log to this file")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", handlers=handlers)
    # pdfminer (used internally for PDF text extraction) logs benign warnings about
    # malformed font metadata in some PDFs (e.g. "Could not get FontBBox..."); these
    # don't affect extraction and only add noise, so keep pdfminer to errors only.
    logging.getLogger("pdfminer").setLevel(logging.ERROR)

    if not args.source_dir.is_dir():
        log.error("source directory does not exist or is not a directory: %s", args.source_dir)
        return 1

    return run(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,
        max_size_bytes=int(args.max_file_size_mb * 1024 * 1024),
        lang=args.lang,
    )


if __name__ == "__main__":
    sys.exit(main())
