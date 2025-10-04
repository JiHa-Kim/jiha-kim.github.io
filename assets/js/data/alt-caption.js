(function () {
  const run = () => {
    console.debug('[alt-caption] script running');

    const scopes = document.querySelectorAll(
      '.post-content, .page-content, .content'
    );
    scopes.forEach((scope) => {
      scope.querySelectorAll('img').forEach((img) => {
        if (img.hasAttribute('data-no-caption')) return;

        const next = img.nextElementSibling;
        if (next && next.tagName === 'EM') return; // user-provided caption

        const cls = img.classList;
        if (
          cls.contains('left') ||
          cls.contains('right') ||
          cls.contains('normal')
        )
          return;

        const alt = (img.getAttribute('alt') || '').trim();
        if (!alt) return;

        const parent = img.parentElement;
        const figure = document.createElement('figure');
        figure.className = 'image';

        if (parent && parent.tagName === 'A' && parent.parentElement) {
          parent.parentElement.insertBefore(figure, parent);
          figure.appendChild(parent);
        } else if (parent) {
          parent.insertBefore(figure, img);
          figure.appendChild(img);
        }

        const cap = document.createElement('figcaption');
        cap.className = 'text-center pt-2 pb-2';
        cap.textContent = alt;
        figure.appendChild(cap);
      });
    });
  };

  // First load
  document.addEventListener('DOMContentLoaded', run);
  // PJAX navigations
  document.addEventListener('pjax:complete', run);
})();
