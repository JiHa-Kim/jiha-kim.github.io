# Development Guide

This guide explains how to manage dependencies and perform common development tasks for this Jekyll site.

## Dependency Management

### Ruby (Jekyll)
The site's main dependencies are managed using [Bundler](https://bundler.io/).

#### Installing Dependencies
To install the required gems, run:
```bash
bundle install
```

#### Bumping Dependencies
To update all gems to their latest allowed versions (within the constraints of the `Gemfile`):
```bash
bundle update
```

To update a specific gem:
```bash
bundle update <gem-name>
```

After updating, verify the site still builds and tests pass:
```bash
bash tools/test.sh
```

### Python scripts
The Python scripts in this repository (e.g., `obsidian_to_chirpy.py`) currently only use the Python standard library and do not require external dependencies.

## Verification
Always run the local test suite before pushing changes:
```bash
bash tools/test.sh
```
