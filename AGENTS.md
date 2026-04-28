# LLM Instructions for j-kim.link

This file contains instructions for AI agents when creating or modifying content for this blog.

## General Guidelines
- I am using the Chirpy theme in Jekyll with a custom pre-processor plugin (`_plugins/obsidian_preprocess.rb`).
- **Automation**: The pre-processor runs automatically during the Jekyll build process. You should commit files in the **Obsidian-style math and callout format** (see below); Jekyll will handle the conversion to the processed HTML format required by the theme.
- Maintain a consistent tone and style across posts.

## Metadata
- Up to 2 levels of categories (e.g., `- Machine Learning`, `- Mathematical Optimization`).
- Use Title Case for tags and categories.

## Math Syntax (Preferred)
Use Obsidian-style math. The pre-processor handles conversion and protects it from Kramdown.
- **Inline**: `$E = mc^2$`
- **Block**:
  ```markdown
  $$
  \frac{d}{dx} e^x = e^x
  $$
  ```
- **Automated Replacements**: Use `*` for multiplication, `|` for absolute value, and `...` for ellipses; the pre-processor converts them to `\ast`, `\vert`, and `\dots`.
- Avoid literal `|` for pipes; use `\vert` or `\Vert` if you want to be explicit, but the pre-processor handles simple ones.

## Callouts (Preferred)
Use Obsidian-style callouts. They are automatically converted to styled boxes or collapsible details.
- **Standard**:
  ```markdown
  > [!info] Title
  > Body content with **Markdown** and $math$.
  ```
- **Collapsible (default closed)**:
  ```markdown
  > [!example]- Title
  > Body content.
  ```
- **Collapsible (default open)**:
  ```markdown
  > [!question]+ Title
  > Body content.
  ```

## Available Box Types (for `[!type]`)
- **Math/Logic**: definition, lemma, proposition, theorem, example, corollary, remark, proof, principle, axiom, postulate, conjecture, claim, notation, algorithm, problem, exercise, solution, assumption, convention, fact.
- **Standard**: info, note, abstract, summary, tldr, todo, tip, hint, important, success, check, done, question, help, faq, warning, caution, attention, danger, error, bug, failure, fail, missing, quote, cite.

## Manual HTML (Only if needed)
If you must use HTML (e.g., complex nesting not supported by Obsidian callouts), always include `markdown="1"` in the opening tag.
- **Example**: `<blockquote class="box-definition" markdown="1">...</blockquote>`

## Interactive Widgets
- Use the existing Interactive Widget Framework (`iaw_skeleton.html`, `iaw__figure-card`, `iaw__canvas-shell`, `iaw__metric-pill`, sliders, status items) unless there is a concrete reason not to.
- Keep widget visuals clean and consistent with prior posts: neutral surfaces, borders, compact cards, semantic colors, and readable labels.
- Do not add decorative glow effects, arbitrary radial gradients, neon styling, or one-off visual treatments that are not part of the established widget language.
- Prefer data visualization clarity over ornament: encode meaning with position, line style, opacity, and the shared semantic color tokens.
- Add reusable layout, legend, metric, canvas, and footer patterns to `_sass/widgets.scss` instead of redefining them inside individual widget includes.
- Scope unavoidable widget-specific CSS to the widget id and keep it minimal; only add styles that the shared IAW classes do not already provide.

## Sources and References
Please do not modify sources, references, or further reading without an explicit request.
