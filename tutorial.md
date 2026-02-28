# Tutorial

Writing notes on syntax etc. in the Chirpy theme for Jekyll so I don't forget
and can reference later

## Images in folders: relative paths

To use relative paths for images and include them in the same folder as the 
content markdown file, include the following in front-matter.
The `_plugins/auto_media_subpath.rb` plugin prepends the absolute path to the
current file, so we use the following syntax for
generic collections other than post, e.g. series/crash-courses:

```
my-series/
├── image.png
└── text.md
```

For posts specifically, we need to use `YYYY-MM-DD-title.md` as the filename:

```
my-post/
├── 2023-02-20-title.md
└── image.png
```

Then, reference the path without prefix, as the `media_subpath` gets prepended:

```markdown
GOOD:
![alt text](image.png)
BAD:
![alt text](./image.png)
```

## Citations with Jekyll Scholar

For citations in posts or other collections, use the `{% cite %}` and `{% bibliography %}` tags. 
It is recommended to use per-post bibliography files to keep things organized.

### Syntax

- **Inline citation**: `{% cite KEY --file path/to/file.bib %}`
- **Bibliography list**: `{% bibliography --file path/to/file.bib %}`

The `--file` path should be relative to the `_bibliography` directory.

Example:
If your bib file is at `_bibliography/posts/my-post/ref.bib`:

```markdown
As shown in {% cite smith2023 --file posts/my-post/ref.bib %}, ...

## References
{% bibliography --file posts/my-post/ref.bib %}
```

## Preprocessing with `preprocess.py`

If you are drafting notes in Obsidian (or elsewhere) and want to convert them to Chirpy-compatible Markdown (handling callouts, MathJax normalization, etc.), use the `preprocess.py` script.

### Usage

Run the script providing the input file or directory:

```bash
python3 preprocess.py path/to/your/post.md > path/to/your/post.md.tmp && mv path/to/your/post.md.tmp path/to/your/post.md
```

Note: The script currently outputs to stdout, so redirection is necessary to overwrite the existing file.

### What it does

- **Callouts**: Converts Obsidian-style `> [!info]` callouts to Chirpy `box-info` blockquotes/details.
- **MathJax**: Wraps block math (`$$...$$` or `\[...\]`) and inline math (`$...$` or `\(...\)`) in `<div class="math-block">` and `<span class="math-inline">` respectively, disabling Kramdown's markdown processing inside them to avoid conflicts.
- **LaTeX Cleanup**: Automatically replaces certain "lazy" LaTeX glyphs (like `...` with `\dots`, `|` with `\vert`, etc.) when they are not inside complex environments.

