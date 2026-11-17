"""
Exporteert de data voor de speeltuinen van module 2 naar interactief/data/.

    python tools/m2_experiments.py sftdiff gpt2play     # eerst de metingen (~10 min CPU)
    python tools/export_interactive_data_module02.py

De scaling-calculator (interactief/scaling.html) rekent alles zelf uit en heeft geen data nodig.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "tools/results/m2_sftdiff.json").read_text(encoding="utf-8"))
out = []
for r in rows:
    out.append({
        "prompt": r["prompt"],
        "gpt2": r.get("gpt2", []),
        "base": r["base"],
        "urial": r["urial"],
        "instruct": r["instruct"],
        "answer": [[t["t"], t["rank"], round(t["p_base"], 3), round(t["p_inst"], 3), t["base_top"]]
                   for t in r["analysis"]["tokens"]],
    })
p = ROOT / "interactief/data/basis_vs_assistent.js"
p.write_text("window.BVA_DATA = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
print(p, round(p.stat().st_size / 1024), "KB")
