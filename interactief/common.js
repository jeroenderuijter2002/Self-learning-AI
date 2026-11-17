// Gedeelde helpers: tooltip en navigatie
(function () {
  const tip = document.createElement("div");
  tip.id = "tip";
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(tip));
  window.showTip = (e, html) => {
    tip.innerHTML = html; tip.style.opacity = 1;
    const x = Math.min(e.clientX + 14, window.innerWidth - 300);
    tip.style.left = x + "px"; tip.style.top = (e.clientY + 14) + "px";
  };
  window.hideTip = () => (tip.style.opacity = 0);

  const pages = [
    ["index.html", "Overzicht"], ["tokenizer.html", "Tokenizer"], ["attention.html", "Attention"],
    ["logitlens.html", "Logit lens"], ["sampling.html", "Sampling"],
    ["basis-vs-assistent.html", "Basis vs assistent"], ["scaling.html", "Scaling"],
  ];
  const MODULE = { "basis-vs-assistent.html": ["02-training", "module 2"], "scaling.html": ["02-training", "module 2"] };
  document.addEventListener("DOMContentLoaded", () => {
    const nav = document.querySelector("nav");
    if (!nav) return;
    const here = location.pathname.split("/").pop() || "index.html";
    nav.innerHTML = pages.map(([href, name]) =>
      `<a href="${href}" class="${href === here ? "active" : ""}">${name}</a>`).join("") +
      `<a href="https://github.com/jeroenderuijter2002/Self-learning-AI/tree/main/modules/${(MODULE[here] || ["01-transformers"])[0]}" style="float:right">← terug naar ${(MODULE[here] || [0, "module 1"])[1]}</a>`;
  });
  window.esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  window.vis = (s) => esc(s).replace(/ /g, "␣").replace(/\n/g, "↵");
})();
