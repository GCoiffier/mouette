(function () {
  const STYLE_ID = "zensical-alloy-search-overlay";
  const SEARCH_SELECTOR = 'input[role="combobox"]';
  const OPEN_STYLE = `
    .n:not(.l) {
      opacity: 1 !important;
      transform: none !important;
      transition: none !important;
    }

    .p:not(.m) {
      opacity: 1 !important;
      transition: none !important;
    }
  `;

  function patchSearchOverlay() {
    for (const host of document.body.children) {
      const root = host.shadowRoot;
      if (!root || !root.querySelector(SEARCH_SELECTOR)) continue;

      host.style.zIndex = "80";

      if (!root.getElementById(STYLE_ID)) {
        const style = document.createElement("style");
        style.id = STYLE_ID;
        style.textContent = OPEN_STYLE;
        root.appendChild(style);
      }
    }
  }

  function schedulePatch() {
    requestAnimationFrame(patchSearchOverlay);
    setTimeout(patchSearchOverlay, 50);
  }

  document.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;

    if (target.closest(".md-header__search-toggle, .md-search--alloy")) {
      schedulePatch();
    }
  }, true);

  new MutationObserver(schedulePatch).observe(document.body, { childList: true });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", schedulePatch, { once: true });
  } else {
    schedulePatch();
  }
})();
