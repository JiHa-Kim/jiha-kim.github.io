MathJax = {
  loader: {
    // Add '[tex]/cancel' and '[tex]/color' to the load array
    load: ['ui/lazy', '[tex]/cancel', '[tex]/color'],
  },
  options: {
    lazyMargin: '200px',
    lazyAlwaysTypeset: null,
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
    packages: { '[+]': ['cancel', 'color'] },
  },
};
