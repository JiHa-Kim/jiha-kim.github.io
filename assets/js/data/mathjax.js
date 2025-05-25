---
layout: compress
# WARNING: Don't use '//' to comment out code, use '{% comment %}' and '{% endcomment %}' instead.
---

{%- comment -%}
  See: <https://docs.mathjax.org/en/latest/options/input/tex.html#tex-options>
{%- endcomment -%}

MathJax = {
  loader: {
    load: ['ui/lazy']              // load the lazy-typesetting extension
  },
  options: {
    lazyMargin: '200px',           // start typesetting 200px before in-view
    lazyAlwaysTypeset: null       // only typeset when intersecting
  },
  tex: {
    {%- comment -%} start/end delimiter pairs for in-line math {%- endcomment -%}
    inlineMath: [
      ['$', '$'],
      ['\\(', '\\)']
    ],
    {%- comment -%} start/end delimiter pairs for display math {%- endcomment -%}
    displayMath: [
      ['$$', '$$'],
      ['\\[', '\\]']
    ],
    {%- comment -%} equation numbering {%- endcomment -%}
    tags: 'ams'
  }
};