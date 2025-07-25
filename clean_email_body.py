# gmail_body_cleaner_v3.py  –  trim at “> On Fri, …” header

import re
import unicodedata
from base64 import urlsafe_b64decode
from bs4 import BeautifulSoup

# ───── whitespace normalisation ───────────────────────────────
_WS_XLAT = {
    0x00A0: " ",
    0x2007: " ",
    0x202F: " ",  # NBSP family → space
    0x200B: "",
    0x200C: "",
    0x200D: "",
    0xFEFF: "",  # zero‑width → nothing
}
_LONG_WS = re.compile(r"[ \t]{2,}")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_WS_XLAT)
    text = _LONG_WS.sub(" ", text)
    return "\n".join(ln.rstrip() for ln in text.splitlines()).strip()


# ─── 0.  make _norm fail‑safe ──────────────────────────────────────────
_WS_XLAT = {
    0x00A0: " ",
    0x2007: " ",
    0x202F: " ",
    0x200B: "",
    0x200C: "",
    0x200D: "",
    0xFEFF: "",
}
_LONG_WS = re.compile(r"[ \t]{2,}")


def _norm(text: str | None) -> str:
    """Unicode‑normalise & tidy whitespace; safe if *text* is None."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).translate(_WS_XLAT)
    text = _LONG_WS.sub(" ", text)
    return "\n".join(ln.rstrip() for ln in text.splitlines()).strip()


# ── regex that matches ONE LINE of the form “On <anything> wrote[:.]” ──
_HEADER_RE = re.compile(
    r"""
        >?\s*On\s+                              # ‘> On ’ (quote optional)
        (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+    # weekday
        [^\n]*?                                 # rest of that line (lazy)
        (?:\n[^\n]*?)*?                         # maybe one extra line (lazy)
        \bwrote[:.]?                            # wrote:  / wrote.
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _cut_from_header_down(text: str | None) -> str:
    """
    Delete everything from the first Gmail / Outlook reply‑header downward.
    Works for:
      • header all on one line
      • header split onto the next line
      • header embedded inside your own sentence
      • quoted headers (starting with ‘>’)
    Always returns *some* string (never None).
    """
    if not text:
        return ""

    m = _HEADER_RE.search(text)
    if m:
        return text[: m.start()].rstrip()

    return text


# ───── plain‑text cleaner ─────────────────────────────────────
def _clean_plain(txt: str) -> str:
    return _norm(_cut_from_header_down(txt))


# ───── HTML cleaner ───────────────────────────────────────────
def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # remove Gmail/Yahoo stitched thread blocks outright
    for n in soup.select(
        "div.gmail_quote, div.gmail_extra," "table.gmail_quote, div.yahoo_quoted"
    ):
        n.decompose()

    # convert blockquotes to real “> ” lines
    for bq in soup.find_all("blockquote"):
        quoted = bq.get_text("\n", strip=True)
        bq.replace_with("\n".join("> " + ln for ln in quoted.splitlines()))

    plain = soup.get_text("\n", strip=True)
    return _norm(_cut_from_header_down(plain))


# ───── public helper – unchanged signature ────────────────────
def get_message_body(message: dict) -> str:
    """
    Extract only the author’s fresh text from a Gmail message, cutting off
    at the first line that looks like ‘> On Fri, Jul 18, 2025 …’.
    """
    for part in message["payload"].get("parts", [message["payload"]]):
        mime = part["mimeType"]
        data = part["body"].get("data")
        if not data:
            continue

        decoded = urlsafe_b64decode(data).decode("utf-8", "ignore")

        if mime == "text/plain":
            return _clean_plain(decoded)
        if mime == "text/html":
            return _clean_html(decoded)

    return ""  # no recognised part
