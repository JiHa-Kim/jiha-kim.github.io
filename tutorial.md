# Tutorial

Writing notes on syntax etc. in the Chirpy theme for Jekyll so I don't forget
and can reference later

## Images in folders: relative paths

To use relative paths for images and include them in the same folder as the 
content markdown file, include the following in front-matter:

```yaml
media_subpath: ../
```

e.g.:

```
test/
├── image.png
└── text.md
```

Then, reference the path without prefix:

```markdown
GOOD:
![alt text](image.png)
BAD:
![alt text](./image.png)
```