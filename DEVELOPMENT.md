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

## Local Preview (Obsidian-style)

This site natively supports Obsidian-style math and callouts. When you run the Jekyll server locally, a custom plugin handles the conversion automatically.

To preview your changes:

```bash
bundle exec jekyll s
```

This ensures that your files in `collections/` remain in their original Obsidian format while being rendered correctly in the browser.

## Local CI Replication

To replicate the GitHub Actions workflow locally (build + test), use the provided test script:

```bash
bash tools/test.sh
```

This script performs the following steps:
1.  **Cleanup**: Removes existing `_site` directory.
2.  **Build**: Runs `bundle exec jekyll build` with `JEKYLL_ENV=production`.
3.  **Verify**: Runs `bundle exec htmlproofer` to catch broken links and HTML errors.

> [!TIP]
> If you have [act](https://github.com/nektos/act) and Docker installed, you can run the exact `.yml` workflow:
> ```bash
> act
> ```

## Verification
Always run the local test suite before pushing changes:
```bash
bash tools/test.sh
```
