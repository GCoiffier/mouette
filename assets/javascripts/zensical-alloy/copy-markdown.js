(function () {
  const copiedTitle = "Copied Markdown";
  const defaultTitle = "Copy Markdown";

  async function copyMarkdown(event) {
    const button = event.currentTarget;
    const url = button.getAttribute("data-md-copy-markdown-url");

    if (!url || !navigator.clipboard) {
      return;
    }

    event.preventDefault();

    try {
      const response = await fetch(url, { credentials: "omit" });
      if (!response.ok) {
        throw new Error("Unable to fetch markdown");
      }

      await navigator.clipboard.writeText(await response.text());
      button.setAttribute("title", copiedTitle);
      button.setAttribute("aria-label", copiedTitle);
      button.classList.add("md-content__button--copied");

      window.setTimeout(function () {
        button.setAttribute("title", defaultTitle);
        button.setAttribute("aria-label", defaultTitle);
        button.classList.remove("md-content__button--copied");
      }, 1600);
    } catch (_error) {
      window.location.href = url;
    }
  }

  function bind() {
    document.querySelectorAll("[data-md-copy-markdown]").forEach(function (button) {
      button.removeEventListener("click", copyMarkdown);
      button.addEventListener("click", copyMarkdown);
    });
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(bind);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
