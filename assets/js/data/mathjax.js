MathJax = {
  loader: {
    load: ['ui/lazy'], // load the lazy-typesetting extension
  },
  options: {
    lazyMargin: '200px', // start typesetting 200px before in-view
    lazyAlwaysTypeset: null, // only typeset when intersecting
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
  },
};
