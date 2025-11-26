#!/usr/bin/env python3
import re
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional

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
    "question":"box-info","help":"box-info","faq":"box-info",
}

# Types that should default to "open" if they are collapsible and no +/− is given
# Note: In standard Obsidian, these aren't collapsible by default, but this preserves your logic.
DEFAULT_OPEN_TYPES = {"question","help","faq"}

# --- Helpers ----------------------------------------------------------------

# Allow whitespace before backticks for indented blocks
CODE_FENCE_RE = re.compile(r"(^\s*```.*?$)(.*?)(^\s*```$)", re.M | re.S)
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

    # Order matters: Blocks first, then inline
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
    # Multiline: $$ \n ... \n $$
    def repl_multiline(m):
        inner = m.group(1).strip("\n")
        return r"\\[" + "\n" + inner + "\n" + r"\\]"
    text = re.sub(r"(?m)^\s*\$\$\s*\n([\s\S]*?)\n\s*\$\$\s*$", repl_multiline, text)

    # Single line block: $$ ... $$
    def repl_singleline(m):
        return r"\\[ " + m.group(1).strip() + r" \\]"
    text = re.sub(r"(?<!\$)\$\$(?!\$)\s*([^\n]+?)\s*(?<!\$)\$\$(?!\$)", repl_singleline, text)
    return text

def convert_inline_math(text: str) -> str:
    """Convert Obsidian $...$ inline math to Chirpy Kramdown \\(...\\) syntax."""
    # Look for $...$ that isn't $$...$$ and isn't escaped.
    # We use [^$\n] to ensure we don't match across newlines or grab $$
    pattern = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^$\n]+?)(?<!\\)(?<!\$)\$(?!\$)")
    return pattern.sub(lambda m: r"\\(" + m.group(1) + r"\\)", text)

# --- Callout parsing ---------------------------------------------------------

CALLOUT_START_RE = re.compile(
    r'^\s*>\s*\[\!(?P<typ>[A-Za-z0-9_-]+)\](?P<state>[\+\-])?\s*(?P<title>.*)?$'
)

def parse_callout_block(lines: List[str], i: int) -> Tuple[Optional[str], int]:
    """
    Given lines and an index at a callout start, consume contiguous '>' lines,
    recursively process the content, and return converted HTML + new index.
    """
    m = CALLOUT_START_RE.match(lines[i])
    if not m:
        return None, i

    ctype = m.group("typ").lower()
    state = m.group("state")  # '+' open, '-' closed, None unknown
    title = (m.group("title") or "").strip()

    # Gather content: strip ONE level of '>'
    content_lines = []
    i += 1
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith(">"):
            # Determine how much indent was used before the '>'
            # Standard markdown usually requires a space after '>', but Obsidian is flexible.
            # We remove the first '>' and the first space if present.
            content_without_marker = re.sub(r'^\s*>\s?', '', line, count=1)
            content_lines.append(content_without_marker)
            i += 1
        else:
            break

    # Recursively process the body (handles nested callouts)
    raw_body = "\n".join(content_lines)
    processed_body = convert_callouts(raw_body)

    # If the body ended with a newline, preserve it for markdown block separation
    if raw_body.strip():
        processed_body = "\n" + processed_body.strip() + "\n"
    else:
        processed_body = "\n"

    # Map to class
    box_class = TYPE_MAP.get(ctype, "box-info")

    # Logic: If +/- is provided, it is a <details>.
    # If not, checks DEFAULT_OPEN_TYPES. If not there, it's a standard blockquote box.
    is_collapsible = (state is not None) or (ctype in DEFAULT_OPEN_TYPES)
    is_open = (state == "+") or (ctype in DEFAULT_OPEN_TYPES)

    title_div = f'\n<div class="title" markdown="1">{title}</div>' if title else ""

    # Note: indenting HTML blocks inside the string isn't strictly necessary
    # but helps read debug output.
    if is_collapsible:
        open_attr = " open" if is_open else ""
        # Default title if none provided for collapsible
        summary_text = title if title else ctype.capitalize()
        html = (
            f'<details class="{box_class}"{open_attr} markdown="1">\n'
            f'<summary markdown="1">{summary_text}</summary>\n'
            f'{processed_body}'
            f'</details>'
        )
    else:
        html = (
            f'<blockquote class="{box_class}" markdown="1">'
            f'{title_div}'
            f'{processed_body}'
            f'</blockquote>'
        )

    return html, i

def convert_callouts(md: str) -> str:
    lines = md.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Only check for callout start if line starts with '>'
        if line.lstrip().startswith(">") and CALLOUT_START_RE.match(line):
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
    # 1) Protect code/HTML regions (so we don't process math/callouts inside code blocks)
    protected, buckets = _protect_regions(src)

    # 2) Callouts (Recursive)
    # This must happen before math if we want math inside callouts to work safely,
    # though usually math and callouts are orthogonal.
    protected = convert_callouts(protected)

    # 3) Math: display then inline
    protected = convert_block_math(protected)
    protected = convert_inline_math(protected)

    # 4) Restore protected regions
    return _restore_regions(protected, buckets)

# --- CLI --------------------------------------------------------------------

def process_path(in_path: Path, out_dir: Optional[Path] = None, root_input: Optional[Path] = None):
    """
    in_path: The specific file or folder being processed.
    out_dir: The root output directory.
    root_input: The root of the input scan (used for calculating relative paths).
    """
    if in_path.is_dir():
        # If recursively scanning, ensure we know the root for relative paths
        current_root = root_input if root_input else in_path
        for p in sorted(in_path.rglob("*.md")):
            process_path(p, out_dir, current_root)
    else:
        try:
            txt = in_path.read_text(encoding="utf-8")
            converted = transform_markdown(txt)

            if out_dir:
                # Calculate relative path to preserve folder structure
                if root_input:
                    rel_path = in_path.relative_to(root_input)
                else:
                    rel_path = in_path.name

                dst = out_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(converted, encoding="utf-8")
                print(f"[OK] {in_path} -> {dst}", file=sys.stderr)
            else:
                # Stdout mode
                print(converted)
        except Exception as e:
            print(f"[ERR] Failed to process {in_path}: {e}", file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        print("Usage: obsidian_to_chirpy.py <input.md | folder> [--out DIR]", file=sys.stderr)
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_dir = None

    if "--out" in sys.argv:
        try:
            idx = sys.argv.index("--out")
            out_dir = Path(sys.argv[idx+1])
        except IndexError:
            print("Error: --out specified but no directory provided.", file=sys.stderr)
            sys.exit(1)

    process_path(in_path, out_dir, in_path if in_path.is_dir() else None)

if __name__ == "__main__":
    main()
