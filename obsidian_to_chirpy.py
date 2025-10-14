#!/usr/bin/env python3
import re
import sys
import os
from pathlib import Path
from typing import List, Tuple

# --- Config -----------------------------------------------------------------

# Map Obsidian callout types -> Chirpy box classes
TYPE_MAP = {
    # math family
    "definition":"box-definition","lemma":"box-lemma","proposition":"box-proposition",
    "theorem":"box-theorem","example":"box-example","corollary":"box-corollary",
    "remark":"box-remark","proof":"box-proof","principle":"box-principle","axiom":"box-axiom",
    "postulate":"box-postulate","conjecture":"box-conjecture","claim":"box-claim",
    "notation":"box-notation","algorithm":"box-algorithm","problem":"box-problem",
    "exercise":"box-exercise","solution":"box-solution","assumption":"box-assumption",
    "convention":"box-convention","fact":"box-fact",
    # standard + aliases
    "note":"box-info","abstract":"box-info","summary":"box-info","tldr":"box-info","todo":"box-info",
    "info":"box-info",
    "tip":"box-tip","hint":"box-tip","important":"box-tip","success":"box-tip","check":"box-tip","done":"box-tip",
    "warning":"box-warning","caution":"box-warning","attention":"box-warning",
    "danger":"box-danger","error":"box-danger","bug":"box-danger","failure":"box-danger","fail":"box-danger","missing":"box-danger",
    "quote":"box-info","cite":"box-info",
    "question":"box-info","help":"box-info","faq":"box-info",  # treated as collapsible-open by default (see below)
}

# Types that should default to "open" if they are collapsible and no +/− is given
DEFAULT_OPEN_TYPES = {"question","help","faq"}

# --- Helpers ----------------------------------------------------------------

CODE_FENCE_RE = re.compile(r"(^```.*?$)(.*?)(^```$)", re.M | re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
HTML_BLOCK_RE = re.compile(r"(?is)<(blockquote|details)(\s[^>]*)?>.*?</\1>")

def _protect_regions(text: str) -> Tuple[str, List[str]]:
    r"""
    Replace code fences, inline code, and existing HTML blockquote/details with
    sentinels so math/callout transforms don't touch them.
    """
    buckets = []

    def stash(m):
        buckets.append(m.group(0))
        return f"@@PROTECT{len(buckets)-1}@@"

    # Order: code fences, HTML blocks, inline code
    text = CODE_FENCE_RE.sub(stash, text)
    text = HTML_BLOCK_RE.sub(stash, text)
    text = INLINE_CODE_RE.sub(stash, text)
    return text, buckets

def _restore_regions(text: str, buckets: List[str]) -> str:
    for i, payload in enumerate(buckets):
        text = text.replace(f"@@PROTECT{i}@@", payload)
    return text

# --- Math conversion ---------------------------------------------------------

def convert_block_math(text: str) -> str:
    """Convert Obsidian $$...$$ blocks to Chirpy Kramdown \\[...\\] syntax."""
    def repl_multiline(m):
        inner = m.group(1).strip("\n")
        return r"\\[\n" + inner + r"\n\\]"
    text = re.sub(r"(?m)^\s*\$\$\s*\n([\s\S]*?)\n\s*\$\$\s*$", repl_multiline, text)

    def repl_singleline(m):
        return r"\\[" + m.group(1) + r"\\]"
    text = re.sub(r"(?<!\$)\$\$(?!\$)\s*([^\n]+?)\s*(?<!\$)\$\$(?!\$)", repl_singleline, text)
    return text

def convert_inline_math(text: str) -> str:
    """Convert Obsidian $...$ inline math to Chirpy Kramdown \\(...\\) syntax."""
    pattern = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)(?<!\$)\$(?!\$)")
    return pattern.sub(lambda m: r"\\(" + m.group(1) + r"\\)", text)

# --- Callout parsing ---------------------------------------------------------

CALLOUT_START_RE = re.compile(
    r'^\s*>\s*\[\!(?P<typ>[A-Za-z0-9_-]+)\](?P<state>[\+\-])?\s*(?P<title>.*)?$'
)

def parse_callout_block(lines: List[str], i: int) -> Tuple[str, int]:
    """
    Given lines and an index at a callout start, consume contiguous '>' lines
    that belong to the same blockquote and return converted HTML plus new index.
    """
    m = CALLOUT_START_RE.match(lines[i])
    if not m:
        return None, i

    ctype = m.group("typ").lower()
    state = m.group("state")  # '+' open, '-' closed, None unknown
    title = (m.group("title") or "").strip()

    # Gather this callout’s content: subsequent lines that continue the same '>' block
    content_lines = []
    i += 1
    while i < len(lines):
        if lines[i].lstrip().startswith(">"):
            # Strip the leading '>' and one optional space
            content_lines.append(re.sub(r'^\s*>\s?', '', lines[i]))
            i += 1
        else:
            break

    # Map to class
    box_class = TYPE_MAP.get(ctype, "box-info")

    # Decide if collapsible: we’ll treat a block as collapsible iff state is given,
    # or for question/help/faq (Obsidian defaults them open).
    is_collapsible = (state is not None) or (ctype in DEFAULT_OPEN_TYPES)
    is_open = (state == "+") or (ctype in DEFAULT_OPEN_TYPES)

    # Title handling: Obsidian allows optional title; for math boxes you also support auto-titles,
    # but when we render HTML we’ll include a title div if a title string is provided.
    body = "\n".join(content_lines).strip()
    title_div = f'\n<div class="title" markdown="1">\n{title}\n</div>\n' if title else "\n"

    if is_collapsible:
        open_attr = " open" if is_open else ""
        html = (
            f'<details class="details-block {box_class}"{open_attr} markdown="1">\n'
            f'<summary markdown="1">\n{title or ctype.capitalize() + "."}\n</summary>\n'
            f'{body}\n'
            f'</details>\n'
        )
    else:
        html = (
            f'<blockquote class="{box_class}" markdown="1">'
            f'{title_div}'
            f'{body}\n'
            f'</blockquote>\n'
        )

    return html, i

def convert_callouts(md: str) -> str:
    lines = md.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if CALLOUT_START_RE.match(line):
            html, i2 = parse_callout_block(lines, i)
            if html is not None:
                out.append(html)
                i = i2
                continue
        out.append(line)
        i += 1
    return "\n".join(out)

# --- Whole-file transform ----------------------------------------------------

def transform_markdown(src: str) -> str:
    # 1) protect code/HTML regions
    protected, buckets = _protect_regions(src)
    # 2) callouts first (works on blockquote lines)
    protected = convert_callouts(protected)
    # 3) math: display then inline
    protected = convert_block_math(protected)
    protected = convert_inline_math(protected)
    # 4) restore protected regions
    return _restore_regions(protected, buckets)

# --- CLI --------------------------------------------------------------------

def process_path(path: Path, out_dir: Path = None):
    if path.is_dir():
        for p in sorted(path.rglob("*.md")):
            rel = p.name if out_dir is None else p.with_suffix(".md").name
            dst = (out_dir / rel) if out_dir else None
            txt = p.read_text(encoding="utf-8")
            converted = transform_markdown(txt)
            if dst:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(converted, encoding="utf-8")
                print(f"[OK] {p} -> {dst}", file=sys.stderr)
            else:
                print(converted)
    else:
        txt = path.read_text(encoding="utf-8")
        converted = transform_markdown(txt)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / path.with_suffix(".md").name
            dst.write_text(converted, encoding="utf-8")
            print(f"[OK] {path} -> {dst}", file=sys.stderr)
        else:
            print(converted)

def main():
    if len(sys.argv) < 2:
        print("Usage: obsidian_to_chirpy.py <input.md | folder> [--out DIR]", file=sys.stderr)
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_dir = None
    if "--out" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--out")+1])
    process_path(in_path, out_dir)

if __name__ == "__main__":
    main()
