(() => {
  const MATH_SELECTOR = '.math-inline, .math-block';
  const VIEWPORT_MARGIN = 400;
  const MAX_TYPES_PER_PASS = 24;
  const READY_POLL_MS = 100;
  const MAX_READY_POLLS = 200;

  let installed = false;
  let checkScheduled = false;
  let typesetInFlight = false;
  let rerunRequested = false;

  function hasRenderedMath(node) {
    return !!node.querySelector('mjx-container');
  }

  function isNearViewport(node) {
    const rect = node.getBoundingClientRect();
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;

    return (
      rect.bottom >= -VIEWPORT_MARGIN &&
      rect.top <= viewportHeight + VIEWPORT_MARGIN &&
      rect.right >= -VIEWPORT_MARGIN &&
      rect.left <= viewportWidth + VIEWPORT_MARGIN
    );
  }

  function getPendingMathRoots() {
    const pending = [];

    for (const node of document.querySelectorAll(MATH_SELECTOR)) {
      if (hasRenderedMath(node) || !isNearViewport(node)) {
        continue;
      }

      pending.push(node);

      if (pending.length >= MAX_TYPES_PER_PASS) {
        break;
      }
    }

    return pending;
  }

  function runTypesetPass() {
    if (document.visibilityState === 'hidden') {
      return;
    }

    if (!window.MathJax || typeof window.MathJax.typesetPromise !== 'function') {
      return;
    }

    if (typesetInFlight) {
      rerunRequested = true;
      return;
    }

    const pendingRoots = getPendingMathRoots();

    if (!pendingRoots.length) {
      return;
    }

    typesetInFlight = true;

    window.MathJax.typesetPromise(pendingRoots)
      .catch(() => {})
      .finally(() => {
        typesetInFlight = false;

        if (rerunRequested) {
          rerunRequested = false;
          scheduleCheck();
          return;
        }

        if (getPendingMathRoots().length) {
          scheduleCheck();
        }
      });
  }

  function scheduleCheck() {
    if (checkScheduled) {
      return;
    }

    checkScheduled = true;

    window.requestAnimationFrame(() => {
      checkScheduled = false;
      runTypesetPass();
    });
  }

  function handleVisibilityChange() {
    if (document.visibilityState === 'visible') {
      scheduleCheck();
    }
  }

  function installFallback() {
    if (installed) {
      return;
    }

    installed = true;

    window.addEventListener('scroll', scheduleCheck, { passive: true });
    window.addEventListener('resize', scheduleCheck, { passive: true });
    window.addEventListener('focus', scheduleCheck);
    window.addEventListener('pageshow', scheduleCheck);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    scheduleCheck();
    window.setTimeout(scheduleCheck, 250);
    window.setTimeout(scheduleCheck, 1000);
  }

  function waitForMathJax(remainingPolls = MAX_READY_POLLS) {
    if (window.MathJax?.startup?.promise) {
      window.MathJax.startup.promise.then(installFallback).catch(() => {});
      return;
    }

    if (remainingPolls <= 0) {
      return;
    }

    window.setTimeout(() => waitForMathJax(remainingPolls - 1), READY_POLL_MS);
  }

  waitForMathJax();
})();
