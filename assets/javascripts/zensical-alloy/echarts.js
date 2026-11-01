(function () {
  const selector =
    ".zensical-echarts code, code.language-echarts, code[class*='language-echarts']";
  const loaded = { promise: null };

  // Registry of live charts so we can re-render them when the user toggles
  // the color scheme. Entries hold the source code element (for re-parsing
  // the JSON options), the canvas node, the chart instance, and the resize
  // observer attached to the canvas.
  const charts = [];
  let schemeObserver = null;

  function loadECharts() {
    if (window.echarts) return Promise.resolve(window.echarts);
    if (loaded.promise) return loaded.promise;

    loaded.promise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src =
        window.zensicalAlloyEChartsSrc ||
        "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js";
      script.async = true;
      script.onload = () => {
        if (window.echarts && typeof window.echarts.init === "function") {
          resolve(window.echarts);
        } else {
          reject(new Error("Apache ECharts loaded without a usable runtime"));
        }
      };
      script.onerror = () => reject(new Error("Unable to load Apache ECharts"));
      document.head.appendChild(script);
    });

    return loaded.promise;
  }

  function chartTheme() {
    return document.body.getAttribute("data-md-color-scheme") === "slate"
      ? "dark"
      : null;
  }

  function isDarkScheme() {
    return document.body.getAttribute("data-md-color-scheme") === "slate";
  }

  function cssVar(name, fallback) {
    const value = getComputedStyle(document.body).getPropertyValue(name).trim();
    return value || fallback;
  }

  function resolveCssColor(value, fallback) {
    const color = value.trim();
    if (!color) return fallback;
    if (/^(#|rgb\(|rgba\()/i.test(color)) return color;
    if (/^oklch\(/i.test(color)) return oklchToRgb(color, fallback);
    if (!CSS.supports("color", color)) return fallback;

    const probe = document.createElement("span");
    probe.style.color = color;
    probe.style.position = "absolute";
    probe.style.pointerEvents = "none";
    probe.style.visibility = "hidden";
    document.body.appendChild(probe);
    const resolved = getComputedStyle(probe).color;
    probe.remove();

    if (/^oklch\(/i.test(resolved)) return oklchToRgb(resolved, fallback);

    return resolved || fallback;
  }

  function oklchToRgb(value, fallback) {
    const match = value.match(
      /^oklch\(\s*([0-9.]+%?)\s+([0-9.]+)\s+([0-9.]+)(?:deg)?(?:\s*\/\s*([0-9.]+%?))?\s*\)$/i
    );

    if (!match) return fallback;

    const lightness = parsePercentOrNumber(match[1]);
    const chroma = Number.parseFloat(match[2]);
    const hue = Number.parseFloat(match[3]) * Math.PI / 180;
    const alpha = match[4] ? parsePercentOrNumber(match[4]) : 1;

    if (![lightness, chroma, hue, alpha].every(Number.isFinite)) return fallback;

    const okA = chroma * Math.cos(hue);
    const okB = chroma * Math.sin(hue);

    const lPrime = lightness + 0.3963377774 * okA + 0.2158037573 * okB;
    const mPrime = lightness - 0.1055613458 * okA - 0.0638541728 * okB;
    const sPrime = lightness - 0.0894841775 * okA - 1.2914855480 * okB;

    const l = lPrime ** 3;
    const m = mPrime ** 3;
    const s = sPrime ** 3;

    const red = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    const green = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    const blue = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    const rgb = [red, green, blue].map(linearSrgbToByte);
    const opacity = Math.max(0, Math.min(1, alpha));

    return opacity < 1
      ? `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${opacity})`
      : `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  }

  function parsePercentOrNumber(value) {
    return value.endsWith("%")
      ? Number.parseFloat(value) / 100
      : Number.parseFloat(value);
  }

  function linearSrgbToByte(value) {
    const normalized = value <= 0.0031308
      ? 12.92 * value
      : 1.055 * (value ** (1 / 2.4)) - 0.055;

    return Math.round(Math.max(0, Math.min(1, normalized)) * 255);
  }

  function applyThemeDefaults(options) {
    const dark = isDarkScheme();
    const text = dark ? "#a1a1aa" : "#71717a";
    const foreground = dark ? "#f4f4f5" : "#18181b";
    const border = dark ? "rgba(255, 255, 255, 0.18)" : "rgba(24, 24, 27, 0.14)";
    const surface = dark ? "#18181b" : "#ffffff";
    const accent = resolveCssColor(
      cssVar("--brand-accent", ""),
      dark ? "#fb923c" : "#ea580c"
    );
    const palette = dark
      ? [accent, "#38bdf8", "#34d399", "#a78bfa", "#fb7185"]
      : [accent, "#0284c7", "#16a34a", "#7c3aed", "#e11d48"];

    return {
      color: options.color || palette,
      backgroundColor: "transparent",
      textStyle: { color: foreground, ...(options.textStyle || {}) },
      ...options,
      title: normalizeTitle(options.title, foreground, text),
      legend: normalizeLegend(options.legend, text),
      tooltip: normalizeTooltip(options.tooltip, surface, foreground, border),
      xAxis: normalizeAxes(options.xAxis, text, border),
      yAxis: normalizeAxes(options.yAxis, text, border),
      series: normalizeSeries(options.series),
    };
  }

  function normalizeTitle(title, foreground, text) {
    if (!title) return title;

    const normalize = (item) => {
      if (!item || typeof item !== "object") return item;

      return {
        left: "left",
        top: 0,
        ...item,
        textStyle: {
          color: foreground,
          fontSize: 13,
          fontWeight: 600,
          ...(item.textStyle || {}),
        },
        subtextStyle: {
          color: text,
          fontSize: 11,
          ...(item.subtextStyle || {}),
        },
      };
    };

    return Array.isArray(title) ? title.map(normalize) : normalize(title);
  }

  function normalizeLegend(legend, text) {
    if (!legend) return legend;

    const normalize = (item) => {
      if (!item || typeof item !== "object") return item;

      return {
        itemWidth: 10,
        itemHeight: 10,
        itemGap: 12,
        ...item,
        textStyle: {
          color: text,
          fontSize: 11,
          ...(item.textStyle || {}),
        },
      };
    };

    return Array.isArray(legend) ? legend.map(normalize) : normalize(legend);
  }

  function normalizeTooltip(tooltip, surface, text, border) {
    if (tooltip === false) return tooltip;

    const base = {
      trigger: "axis",
      appendToBody: false,
      backgroundColor: surface,
      borderColor: border,
      borderWidth: 1,
      confine: true,
      padding: [8, 10],
      textStyle: { color: text, fontSize: 12 },
      extraCssText:
        "z-index: 2; box-shadow: 0 10px 30px -12px rgba(0, 0, 0, .35); border-radius: 10px;",
      transitionDuration: 0,
      position: placeTooltip,
      axisPointer: {
        type: "line",
        lineStyle: { color: border, width: 1 },
      },
    };

    if (!tooltip) return base;

    return {
      ...base,
      ...tooltip,
      textStyle: { ...base.textStyle, ...(tooltip.textStyle || {}) },
      extraCssText: mergeTooltipCss(base.extraCssText, tooltip.extraCssText),
      axisPointer: {
        ...base.axisPointer,
        ...(tooltip.axisPointer || {}),
        lineStyle: {
          ...base.axisPointer.lineStyle,
          ...((tooltip.axisPointer && tooltip.axisPointer.lineStyle) || {}),
        },
      },
    };
  }

  function mergeTooltipCss(baseCss, customCss) {
    const extra = customCss && customCss.trim();
    return extra ? `${baseCss} ${extra}` : baseCss;
  }

  function placeTooltip(_point, _params, _dom, _rect, size) {
    const margin = 12;
    const [boxWidth, boxHeight] = size.contentSize;
    const [viewWidth, viewHeight] = size.viewSize;

    const left = Math.max(margin, viewWidth - boxWidth - margin);
    const top = Math.min(margin, Math.max(0, viewHeight - boxHeight - margin));

    return [left, top];
  }

  function normalizeSeries(series) {
    if (!series) return series;

    const normalize = (item) => {
      if (!item || typeof item !== "object") return item;

      const base = {
        emphasis: {
          focus: "none",
          scale: false,
          ...(item.emphasis || {}),
          itemStyle: {
            opacity: 1,
            ...(item.itemStyle || {}),
            ...((item.emphasis && item.emphasis.itemStyle) || {}),
          },
        },
        blur: {
          itemStyle: { opacity: 1, ...((item.blur && item.blur.itemStyle) || {}) },
          ...(item.blur || {}),
        },
      };

      return {
        ...base,
        ...item,
        emphasis: base.emphasis,
        blur: base.blur,
      };
    };

    return Array.isArray(series) ? series.map(normalize) : normalize(series);
  }

  function normalizeAxes(axes, text, border) {
    if (!axes) return axes;

    const normalize = (axis) => ({
      axisLabel: { color: text, ...(axis.axisLabel || {}) },
      axisLine: {
        lineStyle: { color: border, ...(axis.axisLine && axis.axisLine.lineStyle || {}) },
        ...(axis.axisLine || {}),
      },
      axisTick: {
        lineStyle: { color: border, ...(axis.axisTick && axis.axisTick.lineStyle || {}) },
        ...(axis.axisTick || {}),
      },
      splitLine: {
        lineStyle: { color: border, type: "solid", ...(axis.splitLine && axis.splitLine.lineStyle || {}) },
        ...(axis.splitLine || {}),
      },
      ...axis,
    });

    return Array.isArray(axes) ? axes.map(normalize) : normalize(axes);
  }

  function parseOptions(code) {
    const text = code.textContent.trim();
    return JSON.parse(text);
  }

  function errorMessage(error) {
    return error && error.message ? error.message : "Unable to render chart";
  }

  function hasDatasetSource(dataset) {
    const datasets = Array.isArray(dataset) ? dataset : dataset ? [dataset] : [];
    return datasets.some((item) => {
      const source = item && item.source;
      return Array.isArray(source) ? source.length > 0 : Boolean(source);
    });
  }

  function hasSeriesData(series) {
    const entries = Array.isArray(series) ? series : series ? [series] : [];
    return entries.some((item) => {
      if (!item || typeof item !== "object") return false;
      const data = item.data;
      if (Array.isArray(data)) return data.length > 0;
      return data != null;
    });
  }

  function hasChartData(options) {
    return hasDatasetSource(options.dataset) || hasSeriesData(options.series);
  }

  function createChartWrapper() {
    const wrapper = document.createElement("div");
    wrapper.className = "zensical-alloy-chart";
    wrapper.setAttribute("role", "img");
    wrapper.setAttribute("aria-label", "Chart");

    const canvas = document.createElement("div");
    canvas.className = "zensical-alloy-chart__canvas";
    wrapper.appendChild(canvas);

    return { wrapper, canvas };
  }

  function setChartState(wrapper, state, message) {
    wrapper.dataset.chartState = state;
    wrapper.setAttribute("aria-busy", state === "loading" ? "true" : "false");

    const existing = wrapper.querySelector(".zensical-alloy-chart__status");
    if (existing) existing.remove();
    if (state === "ready") return;

    const status = document.createElement("div");
    status.className = "zensical-alloy-chart__status";
    status.setAttribute("role", "status");
    status.textContent = message;
    wrapper.appendChild(status);
  }

  function renderCodeBlock(code) {
    if (code.dataset.zensicalAlloyChart) return;
    code.dataset.zensicalAlloyChart = "pending";

    const host = code.closest(".highlight") || code.closest("pre") || code;
    let options;

    try {
      options = parseOptions(code);
    } catch (error) {
      code.dataset.zensicalAlloyChart = "error";
      const { wrapper } = createChartWrapper();
      setChartState(wrapper, "error", errorMessage(error));
      host.replaceWith(wrapper);
      return;
    }

    const { wrapper, canvas } = createChartWrapper();

    if (!hasChartData(options)) {
      code.dataset.zensicalAlloyChart = "empty";
      setChartState(wrapper, "empty", "No chart data");
      host.replaceWith(wrapper);
      return;
    }

    setChartState(wrapper, "loading", "Loading chart");
    host.replaceWith(wrapper);

    loadECharts()
      .then((echarts) => {
        if (!document.body.contains(wrapper)) return;
        try {
          mountChart(echarts, code, canvas, options);
          setChartState(wrapper, "ready");
          code.dataset.zensicalAlloyChart = "done";
          observeScheme();
        } catch (error) {
          code.dataset.zensicalAlloyChart = "error";
          setChartState(wrapper, "error", errorMessage(error));
        }
      })
      .catch((error) => {
        code.dataset.zensicalAlloyChart = "error";
        if (document.body.contains(wrapper)) {
          setChartState(wrapper, "error", errorMessage(error));
        }
      });
  }

  function mountChart(echarts, code, canvas, options) {
    const chart = echarts.init(canvas, chartTheme());
    chart.setOption(applyThemeDefaults(options));
    const resizeObserver = new ResizeObserver(() => chart.resize());
    resizeObserver.observe(canvas);
    charts.push({ code, canvas, chart, resizeObserver });
  }

  // Dispose chart instances whose canvases left the DOM (instant-nav swapped
  // the page out). Keeps the registry from leaking across navigations.
  function pruneDetachedCharts() {
    for (let i = charts.length - 1; i >= 0; i--) {
      if (!document.body.contains(charts[i].canvas)) {
        charts[i].chart.dispose();
        charts[i].resizeObserver.disconnect();
        charts.splice(i, 1);
      }
    }
  }

  // Re-init each live chart with the new scheme. ECharts' built-in theme
  // (light/dark) is fixed at init time, so we dispose and recreate; the
  // wrapper/canvas nodes stay put so layout does not jump.
  function rerenderForScheme() {
    if (!window.echarts) return;
    pruneDetachedCharts();
    charts.forEach((entry) => {
      entry.chart.dispose();
      entry.resizeObserver.disconnect();
      const chart = window.echarts.init(entry.canvas, chartTheme());
      chart.setOption(applyThemeDefaults(parseOptions(entry.code)));
      const resizeObserver = new ResizeObserver(() => chart.resize());
      resizeObserver.observe(entry.canvas);
      entry.chart = chart;
      entry.resizeObserver = resizeObserver;
    });
  }

  function observeScheme() {
    if (schemeObserver) return;
    schemeObserver = new MutationObserver((mutations) => {
      if (mutations.some((m) => m.attributeName === "data-md-color-scheme")) {
        rerenderForScheme();
      }
    });
    schemeObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
  }

  function renderCharts() {
    pruneDetachedCharts();
    document.querySelectorAll(selector).forEach(renderCodeBlock);
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(renderCharts);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderCharts);
  } else {
    renderCharts();
  }
})();
