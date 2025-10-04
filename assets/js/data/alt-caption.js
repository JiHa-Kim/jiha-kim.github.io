(function () {
  const hasPositionClass = (img) => {
    const c = img.classList;
    return c.contains('left') || c.contains('right') || c.contains('normal');
  };

  const alreadyHasItalicCaption = (linkOrImg) => {
    const p = linkOrImg.closest('p');
    if (!p) {
      // look for immediate <em> sibling just after the link/img
      const sib = linkOrImg.nextElementSibling;
      return !!(sib && sib.tagName === 'EM' && sib.textContent.trim());
    }
    // In a <p>, check if there is an <em> child with non-empty text
    return Array.from(p.children).some(
      (el) => el.tagName === 'EM' && el.textContent.trim()
    );
  };

  const insertItalicCaption = (linkOrImg, text) => {
    const em = document.createElement('em');
    em.textContent = text;

    const p = linkOrImg.closest('p');
    if (p) {
      // place the <em> right after the link/img inside the same paragraph
      // add a whitespace text node for nicer HTML diffs / readability
      p.insertBefore(document.createTextNode('\n'), linkOrImg.nextSibling);
      p.insertBefore(em, linkOrImg.nextSibling);
    } else {
      // no <p> wrapper: create a new paragraph with the <em> after the image/link
      const para = document.createElement('p');
      para.appendChild(em);
      const parent = linkOrImg.parentElement;
      if (parent) {
        parent.insertBefore(para, linkOrImg.nextSibling);
      }
    }
  };

  const run = () => {
    const scopes = document.querySelectorAll(
      '.post-content, .page-content, .content'
    );
    scopes.forEach((scope) => {
      scope.querySelectorAll('img').forEach((img) => {
        if (img.dataset.altcapProcessed === '1') return;
        if (img.hasAttribute('data-no-caption')) return;
        if (hasPositionClass(img)) return;

        const linkOrImg =
          img.parentElement && img.parentElement.tagName === 'A'
            ? img.parentElement
            : img;

        // If there is already an italic caption, leave it exactly as-is
        if (alreadyHasItalicCaption(linkOrImg)) {
          img.dataset.altcapProcessed = '1';
          return;
        }

        // Fallback to alt text
        const alt = (img.getAttribute('alt') || '').trim();
        if (!alt) {
          img.dataset.altcapProcessed = '1';
          return;
        }

        insertItalicCaption(linkOrImg, alt);
        img.dataset.altcapProcessed = '1';
      });
    });
  };

  document.addEventListener('DOMContentLoaded', run);
  document.addEventListener('pjax:complete', run);
})();
