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