(() => {
  const MATH_SELECTOR = '.math-inline[data-math-source-b64], .math-block[data-math-source-b64]';
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

  function decodeMathSource(node) {
    const encoded = node?.dataset?.mathSourceB64;

    if (!encoded) {
      return '';
    }

    try {
      const binary = window.atob(encoded);

      if (typeof TextDecoder === 'function') {
        const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
        return new TextDecoder().decode(bytes);
      }

      return binary;
    } catch (_) {
      return '';
    }
  }

  function formatMathSource(node) {
    const source = decodeMathSource(node).trim();

    if (!source) {
      return '';
    }

    return node.classList.contains('math-block') ? `$$\n${source}\n$$` : `$${source}$`;
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
      const source = formatMathSource(node);

      if (!source) {
        continue;
      }

      const replacement = node.classList.contains('math-block') ? `\n${source}\n` : source;
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
    if (copyHandlerInstalled) {
      return;
    }

    document.addEventListener('copy', handleCopy);
    copyHandlerInstalled = true;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startMathCopy, { once: true });
  } else {
    startMathCopy();
  }
})();
