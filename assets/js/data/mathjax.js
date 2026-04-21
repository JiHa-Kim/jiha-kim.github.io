---
---
MathJax = {
  loader: {
    // Add '[tex]/cancel' and '[tex]/color' to the load array
    load: ['ui/lazy', '[tex]/cancel', '[tex]/color'],
  },
  options: {
    lazyMargin: '200px',
    lazyAlwaysTypeset: null,
    ignoreHtmlClass: 'tex2jax_ignore|mathjax_ignore',
    processHtmlClass: 'tex2jax_process|mathjax_process',
  },
  tex: {
    inlineMath: [
      ['$', '$'],
      ['\\(', '\\)'],
    ],
    displayMath: [
      ['$$', '$$'],
      ['\\[', '\\]'],
    ],
    tags: 'ams',
    packages: { '[+]': ['cancel', 'color', 'ams'] },
  },
};
