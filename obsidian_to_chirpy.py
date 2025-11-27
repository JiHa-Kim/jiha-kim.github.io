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
DEFAULT_OPEN_TYPES = {"question","help","faq"}

# Environments where we should NOT apply auto-formatting (glyph replacements)
SKIP_FORMATTING_ENVS = {
    "align", "align*", "equation", "equation*", "gather", "gather*",
    "multline", "multline*", "split", "cases", "dcases", "array",
    "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
    "aligned", "alignedat", "gathered", "subequations", "flalign",
    "flalign*", "eqnarray"
}

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

def cleanup_latex_syntax(content: str) -> str:
    """
    Applies specific glyph replacements to clean up 'lazy' LaTeX.
    Ported from fix-math.sh.
    """
    # 1. ... -> \dots (with space)
    content = content.replace("...", r"\dots ")

    # 2. \| or || -> \Vert (with space)
    content = content.replace(r"\|", r"\Vert ")
    content = content.replace("||", r"\Vert ")

    # 3. | -> \vert (using negative lookbehind to ensure we don't match \| or \Vert)
    #    matches a pipe that is NOT preceded by a backslash
    content = re.sub(r'(?<!\\)\|', r'\\vert ', content)

    # 4. * -> \ast (with space)
    content = content.replace("*", r"\ast ")

    # 5. ~ -> \sim (with space)
    content = content.replace("~", r"\sim ")

    return content

def convert_block_math(text: str) -> str:
    """
    Normalize Obsidian $$...$$ blocks into Kramdown-safe MathJax blocks.

    1. Finds $$...$$ sequences.
    2. Checks if they contain specific environments (align, etc).
    3. If NOT, applies cleanup_latex_syntax (replacing *, |, etc).
    4. Wraps the result in a <div markdown="0"> with \[ ... \] delimiters
       so Kramdown does not touch the TeX and MathJax can render it.
    """

    pattern = re.compile(
        r"(?m)(?:^([\t ]*))?(?<!\\)(?<!\$)\$\$(?!\$)([\s\S]+?)(?<!\\)(?<!\$)\$\$(?!\$)"
    )

    def repl(m):
        indent = m.group(1) or ""
        content = m.group(2)

        # Check if this block contains an environment that should be skipped
        has_env = False
        for env in SKIP_FORMATTING_ENVS:
            if f"\\begin{{{env}}}" in content:
                has_env = True
                break

        # If no complex environment is found, apply the cleanup logic
        if not has_env:
            content = cleanup_latex_syntax(content)

        content = content.strip()

        # Build the inner TeX: \[ ... \]
        inner = "\\[\n" + content + "\n\\]"

        # Wrap in an HTML block that disables Markdown parsing
        return (
            f"\n{indent}<div class=\"math-block\" markdown=\"0\">\n"
            f"{inner}\n"
            f"{indent}</div>\n"
        )

    return pattern.sub(repl, text)

def convert_inline_math(text: str) -> str:
    """Convert Obsidian $...$ inline math to Kramdown-friendly MathJax syntax.

    We wrap the \\(...\\) in <span markdown="0"> so Kramdown does not
    interpret underscores, asterisks, etc, before MathJax sees them.
    """
    # Look for $...$ that isn't $$...$$ and isn't escaped.
    pattern = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^$\n]+?)(?<!\\)(?<!\$)\$(?!\$)")

    def repl(m):
        # Apply cleanup to inline math as well (fixes |x| -> \vert x \vert etc.)
        content = cleanup_latex_syntax(m.group(1))

        # Build the MathJax inline form \(...\)
        inner = r"\(" + content + r"\)"

        # Wrap in a span that disables Markdown parsing inside
        # so Kramdown will not turn _ into <em>, etc.
        return f'<span class="math-inline" markdown="0">{inner}</span>'

    return pattern.sub(repl, text)

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
    state = m.group("state")
    title = (m.group("title") or "").strip()

    content_lines = []
    i += 1

    in_code_fence = False
    in_math_block = False

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        is_quoted = stripped.startswith(">")

        if is_quoted:
            content_part = re.sub(r'^\s*>\s?', '', line, count=1)
        else:
            content_part = line

        keep_line = False

        if is_quoted:
            keep_line = True
        elif in_code_fence or in_math_block:
            keep_line = True
        elif not line.strip():
            break
        else:
            break

        if keep_line:
            content_lines.append(content_part)

            if re.match(r'^\s*```', content_part):
                if not in_math_block:
                    in_code_fence = not in_code_fence

            if re.match(r'^\s*\$\$\s*$', content_part):
                if not in_code_fence:
                    in_math_block = not in_math_block

            i += 1
        else:
            break

    raw_body = "\n".join(content_lines)
    processed_body = convert_callouts(raw_body)

    if raw_body.strip():
        processed_body = "\n" + processed_body.strip() + "\n"
    else:
        processed_body = "\n"

    box_class = TYPE_MAP.get(ctype, "box-info")
    is_collapsible = (state is not None) or (ctype in DEFAULT_OPEN_TYPES)
    is_open = (state == "+") or (ctype in DEFAULT_OPEN_TYPES)

    if title:
        title_html = (
            f'<div class="title" markdown="1">\n'
            f'{title}\n'
            f'</div>'
        )
    else:
        title_html = ""

    if is_collapsible:
        open_attr = " open" if is_open else ""
        summary_text = title if title else ctype.capitalize()
        summary_html = (
            f'<summary markdown="1">\n'
            f'{summary_text}\n'
            f'</summary>'
        )
        html = (
            f'<details class="{box_class}"{open_attr} markdown="1">\n'
            f'{summary_html}'
            f'{processed_body}'
            f'</details>'
        )
    else:
        inner_content = f"{title_html}{processed_body}" if title_html else processed_body
        html = (
            f'<blockquote class="{box_class}" markdown="1">\n'
            f'{inner_content}'
            f'</blockquote>'
        )

    return html, i

def convert_callouts(md: str) -> str:
    lines = md.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
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
    protected, buckets = _protect_regions(src)
    protected = convert_callouts(protected)
    protected = convert_block_math(protected)
    protected = convert_inline_math(protected)
    return _restore_regions(protected, buckets)

# --- CLI --------------------------------------------------------------------

def process_path(in_path: Path, out_dir: Optional[Path] = None, root_input: Optional[Path] = None):
    if in_path.is_dir():
        current_root = root_input if root_input else in_path
        for p in sorted(in_path.rglob("*.md")):
            process_path(p, out_dir, current_root)
    else:
        try:
            txt = in_path.read_text(encoding="utf-8")
            converted = transform_markdown(txt)

            if out_dir:
                if root_input:
                    rel_path = in_path.relative_to(root_input)
                else:
                    rel_path = in_path.name

                dst = out_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(converted, encoding="utf-8")
                print(f"[OK] {in_path} -> {dst}", file=sys.stderr)
            else:
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
