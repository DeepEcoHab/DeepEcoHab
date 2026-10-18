// Draws the page's plotly outputs as the app's dashboard does: the app's own plot scripts
// (rounded bars, boxes and tiles, event-lane tooltips, square heatmaps centred, wide legends
// moved below), and every figure repainted in the site's light or dark theme from the looks
// it carries in layout.meta.dehLooks (plotting.theme.looks). The scripts are copied next to
// this module's folder at build time (.github/workflows/docs.yml).
const site = new URL("../", import.meta.url);

const load = (name) =>
  new Promise((resolve, reject) =>
    document.head.append(Object.assign(document.createElement("script"), { src: new URL(name, site), onload: resolve, onerror: reject })),
  );

// The theme's renderer imports plotly from a hashed chunk; importing the same URL again
// hands back that module, so relayout acts on the graphs it drew.
async function plotly() {
  for (;;) {
    const chunk = performance.getEntriesByType("resource").find((entry) => /\/plotly-[A-Z0-9]+\.js$/.test(entry.name));
    if (chunk) return (await import(chunk.name)).default;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
}

const theme = () => (document.documentElement.classList.contains("dark") ? "dark" : "light");

function retheme(Plotly, gd) {
  const name = theme();
  const look = gd.layout?.meta?.dehLooks?.[name];
  // Figures are built in the light theme.
  if (!look || (gd._dehTheme ?? "light") === name) return;
  gd._dehTheme = name;
  const { template, shapes } = window.dehRetheme(gd.layout, look);
  Plotly.relayout(gd, { template, shapes });
}

async function setup() {
  await Promise.all([load("plot_dom.js"), load("rounded_corners.js")]);
  const Plotly = (window.Plotly = await plotly());
  const watch = (gd) => {
    if (gd._dehDocs || !gd.on) return;
    gd._dehDocs = true;
    // myst keeps a removed cell's output, hidden, as the source of its embeds: free that copy.
    if (gd.closest(".myst-jp-nb-block.hidden")) return Plotly.purge(gd);
    gd.on("plotly_afterplot", () => retheme(Plotly, gd));
    // A plot drawn in a hidden tab takes its width when the tab opens, not on window resize.
    new ResizeObserver(() => gd.offsetWidth && Plotly.Plots.resize(gd)).observe(gd);
    retheme(Plotly, gd);
  };
  const all = () => document.querySelectorAll(".js-plotly-plot").forEach(watch);
  new MutationObserver(all).observe(document.body, { childList: true, subtree: true });
  new MutationObserver(() => document.querySelectorAll(".js-plotly-plot").forEach((gd) => retheme(Plotly, gd))).observe(
    document.documentElement,
    { attributes: true, attributeFilter: ["class"] },
  );
  all();
}

export default {
  render() {
    window.__dehPlots ??= setup();
  },
};
