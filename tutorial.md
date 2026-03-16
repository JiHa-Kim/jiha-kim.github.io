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

## Preprocessing with `obsidian_preprocess.rb`

You can draft notes in Obsidian using standard callouts and math syntax. A Jekyll plugin (`_plugins/obsidian_preprocess.rb`) automatically converts them to Chirpy-compatible Markdown during the build process, both locally and in CI.

### Usage

No manual action is required. Simply run the standard Jekyll commands:

```bash
bundle exec jekyll s  # Local preview
bundle exec jekyll b  # Build site
```

The plugin ensures that your source files stay in their original format.

### What it does

- **Callouts**: Converts Obsidian-style `> [!info]` callouts to Chirpy `box-info` blockquotes/details.
- **MathJax**: Wraps block math (`$$...$$` or `\[...\]`) and inline math (`$...$` or `\(...\)`) in `<div class="math-block">` and `<span class="math-inline">` respectively, disabling Kramdown's markdown processing inside them to avoid conflicts.
- **LaTeX Cleanup**: Automatically replaces certain "lazy" LaTeX glyphs (like `...` with `\dots`, `|` with `\vert$, etc.) when they are not inside complex environments.

