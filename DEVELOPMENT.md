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

## Build Optimization & Profiling

To speed up local development and identify bottlenecks, use the following techniques:

### Incremental Builds
For faster local previews, enable incremental builds. Jekyll will only re-generate files that have changed:
```bash
bundle exec jekyll s --incremental
```

### Profiling
To see which pages or liquid templates are taking the most time to render:
```bash
bundle exec jekyll build --profile
```
The output will show a table of the slowest files, helping you identify complex math or large collections that may need optimization.

### Persistent Caching
The custom `ObsidianPreprocess` plugin uses a disk-backed cache in `.jekyll-cache/`. This cache is shared between builds. If you ever need to force a full re-process of all files, you can clear the cache:
```bash
rm -rf .jekyll-cache
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
