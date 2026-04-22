---
---
window.MathJax = {
  startup: {
    typeset: false,
  },
  loader: {
    load: ['[tex]/cancel', '[tex]/color'],
  },
  options: {
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
