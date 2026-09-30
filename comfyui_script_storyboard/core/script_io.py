"""Script loading: .txt / .fountain / .md / .fdx / .pdf / .docx -> normalized plain text.

Page breaks are preserved as form-feed characters (``\\f``) where the source
format knows about pages (PDF), so the parser can attribute scenes to pages.
"""

from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET

SUPPORTED_EXTENSIONS = (".txt", ".fountain", ".spmd", ".md", ".fdx", ".pdf", ".docx")

LINES_PER_PAGE = 55  # industry rule of thumb for screenplay pages
MAX_PAGES = 1000


class ScriptTooLongError(ValueError):
    pass


def load_script(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = _load_pdf(path)
    elif ext == ".fdx":
        text = _load_fdx(path)
    elif ext == ".docx":
        text = _load_docx(path)
    else:
        with open(path, "rb") as f:
            raw = f.read()
        text = _decode(raw)
    return normalize_text(text)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", "    ").replace(" ", " ")
    # Smart quotes/dashes -> ascii so regexes stay simple.
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("—", "--").replace("–", "-"))
    return text


def estimate_pages(text: str) -> int:
    if "\f" in text:
        return text.count("\f") + 1
    return max(1, -(-text.count("\n") // LINES_PER_PAGE))


def check_length(text: str, max_pages: int = MAX_PAGES) -> int:
    pages = estimate_pages(text)
    if pages > max_pages:
        raise ScriptTooLongError(
            f"Script is ~{pages} pages; the limit is {max_pages}. "
            "Split it or raise max_pages on the loader node.")
    return pages


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and "\x00" in text:
                continue
            return text
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError("PDF scripts need `pypdf` (pip install pypdf)") from e
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            t = page.extract_text(extraction_mode="layout") or ""
        except TypeError:  # older pypdf
            t = page.extract_text() or ""
        lines = t.split("\n")
        # Drop bare page numbers / "CONTINUED" furniture.
        lines = [ln for ln in lines
                 if not re.fullmatch(r"\s*\d+\.?\s*", ln)
                 and not re.fullmatch(r"\s*\(?(CONTINUED|MORE)\)?:?\s*", ln, re.I)]
        pages.append(_dedent_pdf_page(lines))
    return "\f".join(pages)


def _dedent_pdf_page(lines):
    """PDF layout text keeps screenplay indentation. Blank lines between
    blocks are often lost, so re-insert them when indentation changes."""
    out = []
    prev_indent = None
    for ln in lines:
        if not ln.strip():
            out.append("")
            prev_indent = None
            continue
        indent = len(ln) - len(ln.lstrip())
        if prev_indent is not None and abs(indent - prev_indent) >= 8:
            stripped = ln.strip()
            is_paren = stripped.startswith("(")
            if not is_paren and (indent < prev_indent or _looks_like_cue(stripped)):
                out.append("")
        out.append(ln.strip())
        prev_indent = indent
    return "\n".join(out)


def _looks_like_cue(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9 .'\-]+(\s*\([A-Z.' ]+\))*", s)) and len(s) < 40


_FDX_PREFIX = {
    "Scene Heading": "", "Action": "", "Character": "", "Dialogue": "",
    "Parenthetical": "", "Transition": "> ", "Shot": "", "General": "",
}


def _load_fdx(path: str) -> str:
    tree = ET.parse(path)
    out = []
    for para in tree.iter("Paragraph"):
        ptype = para.get("Type", "Action")
        text = "".join(t.text or "" for t in para.iter("Text")).strip()
        if not text:
            continue
        if ptype == "Scene Heading":
            if not re.match(r"(?i)(INT|EXT|I/E|EST)", text):
                text = "." + text
            out += ["", text.upper(), ""]
        elif ptype == "Character":
            out += ["", "@" + text.upper() if not text.isupper() else text]
        elif ptype == "Parenthetical":
            out.append(text if text.startswith("(") else f"({text})")
        elif ptype == "Dialogue":
            out.append(text)
        elif ptype == "Transition":
            out += ["", "> " + text.upper(), ""]
        else:
            out += ["", text, ""]
    return "\n".join(out)


def _load_docx(path: str) -> str:
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    out = []
    for p in root.iter(ns + "p"):
        parts = []
        for node in p.iter():
            if node.tag == ns + "t" and node.text:
                parts.append(node.text)
            elif node.tag == ns + "br" and node.get(ns + "type") == "page":
                parts.append("\f")
            elif node.tag == ns + "tab":
                parts.append(" ")
        out.append("".join(parts))
    # docx paragraphs rarely have blank separators between blocks
    text = "\n".join(out)
    return re.sub(r"\n(?=(?:INT|EXT|I/E)[. ])", "\n\n", text)
