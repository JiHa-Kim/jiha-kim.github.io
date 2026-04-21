(() => {
  const MATH_SELECTOR = 'mjx-container[data-math-source]';
  const MAX_READY_POLLS = 200;
  const READY_POLL_MS = 100;
  let observingMath = false;
  let copyHandlerInstalled = false;

  function isElement(node) {
    return node && node.nodeType === Node.ELEMENT_NODE;
  }

  function closestMathContainer(node) {
    let current = node;

    while (current) {
      if (isElement(current) && current.matches(MATH_SELECTOR)) {
        return current;
      }

      current = current.parentNode;
    }

    return null;
  }

  function formatMathSource(mathItem) {
    const source = typeof mathItem?.math === 'string' ? mathItem.math.trim() : '';

    if (!source) {
      return '';
    }

    return mathItem.display ? `$$\n${source}\n$$` : `$${source}$`;
  }

  function annotateMathSources() {
    const mathItems = window.MathJax?.startup?.document?.math;

    if (!mathItems) {
      return false;
    }

    for (const mathItem of mathItems) {
      const root = mathItem?.typesetRoot;
      const source = formatMathSource(mathItem);

      if (!isElement(root) || !source) {
        continue;
      }

      root.dataset.mathSource = source;
      root.dataset.mathDisplay = mathItem.display ? 'true' : 'false';
    }

    return true;
  }

  function mutationAddsMath(mutation) {
    return Array.from(mutation.addedNodes).some((node) => {
      if (!isElement(node)) {
        return false;
      }

      return node.matches('mjx-container') || !!node.querySelector('mjx-container');
    });
  }

  function installMathObserver() {
    if (observingMath || !document.body || typeof MutationObserver !== 'function') {
      return;
    }

    const observer = new MutationObserver((mutations) => {
      if (!mutations.some(mutationAddsMath)) {
        return;
      }

      annotateMathSources();
    });

    observer.observe(document.body, { childList: true, subtree: true });
    observingMath = true;
  }

  function normalizeRange(range) {
    const normalized = range.cloneRange();
    const startMath = closestMathContainer(normalized.startContainer);
    const endMath = closestMathContainer(normalized.endContainer);

    if (startMath && normalized.intersectsNode(startMath)) {
      normalized.setStartBefore(startMath);
    }

    if (endMath && normalized.intersectsNode(endMath)) {
      normalized.setEndAfter(endMath);
    }

    return normalized;
  }

  function replaceMathWithSource(fragment) {
    const mathNodes = fragment.querySelectorAll(MATH_SELECTOR);

    for (const node of mathNodes) {
      const source = node.dataset.mathSource;

      if (!source) {
        continue;
      }

      const replacement = node.dataset.mathDisplay === 'true' ? `\n${source}\n` : source;
      node.replaceWith(document.createTextNode(replacement));
    }
  }

  function fragmentToPlainText(fragment) {
    const container = document.createElement('div');
    container.style.position = 'fixed';
    container.style.top = '0';
    container.style.left = '-99999px';
    container.style.whiteSpace = 'pre-wrap';
    container.style.pointerEvents = 'none';
    container.appendChild(fragment);
    document.body.appendChild(container);

    const text = (container.innerText || container.textContent || '').replace(/\u200b/g, '');
    container.remove();
    return text;
  }

  function selectionTouchesMath(selection) {
    if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
      return false;
    }

    const mathNodes = document.querySelectorAll(MATH_SELECTOR);

    for (let i = 0; i < selection.rangeCount; i += 1) {
      const range = selection.getRangeAt(i);

      for (const node of mathNodes) {
        if (range.intersectsNode(node)) {
          return true;
        }
      }
    }

    return false;
  }

  function handleCopy(event) {
    if (!event.clipboardData) {
      return;
    }

    const selection = window.getSelection();

    if (!selectionTouchesMath(selection)) {
      return;
    }

    const copiedRanges = [];

    for (let i = 0; i < selection.rangeCount; i += 1) {
      const range = selection.getRangeAt(i);
      const normalized = normalizeRange(range);
      const fragment = normalized.cloneContents();

      replaceMathWithSource(fragment);
      copiedRanges.push(fragmentToPlainText(fragment).trimEnd());
    }

    const text = copiedRanges.join('\n').trim();

    if (!text) {
      return;
    }

    event.preventDefault();
    event.clipboardData.setData('text/plain', text);
  }

  function startMathCopy() {
    annotateMathSources();
    installMathObserver();

    if (!copyHandlerInstalled) {
      document.addEventListener('copy', handleCopy);
      copyHandlerInstalled = true;
    }
  }

  function waitForMathJax(remainingPolls = MAX_READY_POLLS) {
    if (window.MathJax?.startup?.promise) {
      window.MathJax.startup.promise.then(startMathCopy).catch(() => {});
      return;
    }

    if (remainingPolls <= 0) {
      return;
    }

    window.setTimeout(() => waitForMathJax(remainingPolls - 1), READY_POLL_MS);
  }

  waitForMathJax();
})();
