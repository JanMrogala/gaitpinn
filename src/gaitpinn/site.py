"""Build the static website from runs/ only. No number on the site is typed by hand.

Pages: Results (index.html), People, Runs, Method. Plotly is bundled locally (no CDN).
Result pages read `runs/summary/*.json`, written by `gaitpinn.test`; until those exist the
pages say that no results are available yet.
"""

import argparse
import html
import json
import re
from pathlib import Path

import markdown
import plotly.offline

from gaitpinn import runs as runrec

REPO = Path(__file__).resolve().parents[2]
PAGES = [
    ("index.html", "Results"),
    ("people.html", "People"),
    ("runs.html", "Runs"),
    ("method.html", "Method"),
]

CSS = """
:root { --bg:#ffffff; --fg:#1d1f23; --muted:#5f6670; --line:#d9dce1; --accent:#2a62c9;
        --card:#f6f7f9; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#15171a; --fg:#e6e8eb; --muted:#9aa1ab; --line:#2c3036; --accent:#7ea6f0;
          --card:#1d2024; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { border-bottom:1px solid var(--line); }
nav, main { max-width:1000px; margin:0 auto; padding:0 16px; }
nav { display:flex; gap:20px; align-items:center; height:56px; flex-wrap:wrap; }
nav b { margin-right:auto; }
nav a { color:var(--muted); text-decoration:none; }
nav a.on { color:var(--fg); font-weight:600; }
main { padding-top:24px; padding-bottom:48px; }
.note { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:14px 16px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
th { cursor:pointer; user-select:none; color:var(--muted); font-weight:600; }
.wrap { overflow-x:auto; }
a { color:var(--accent); }
code { font-size:13px; }
"""

SORT_JS = """
document.querySelectorAll('table.sortable th').forEach((th, i) => {
  th.addEventListener('click', () => {
    const tb = th.closest('table').tBodies[0];
    const asc = th.dataset.asc !== '1'; th.dataset.asc = asc ? '1' : '0';
    const rows = [...tb.rows].sort((a, b) => {
      const x = a.cells[i].textContent, y = b.cells[i].textContent;
      const nx = parseFloat(x), ny = parseFloat(y);
      const c = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y);
      return asc ? c : -c;
    });
    rows.forEach(r => tb.appendChild(r));
  });
});
"""


def page(name: str, title: str, body: str, plotly: bool = False) -> str:
    nav = "".join(f'<a href="{f}"{" class=on" if f == name else ""}>{t}</a>' for f, t in PAGES)
    js = '<script src="plotly.min.js"></script>' if plotly else ""
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>gaitpinn · {title}</title><style>{CSS}</style>{js}</head><body>"
        f"<header><nav><b>gaitpinn</b>{nav}</nav></header><main>{body}</main>"
        f"<script>{SORT_JS}</script></body></html>"
    )


def load_summary(runs_dir: Path, name: str) -> dict | None:
    p = runs_dir / "summary" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


EMPTY = "<p class=note>No results yet. This page fills in once the training runs finish.</p>"


def results_body(runs_dir: Path) -> tuple[str, bool]:
    s = load_summary(runs_dir, "results")
    if s is None:
        return "<h1>Results</h1>" + EMPTY, False
    # Filled in by step 7: figure JSON, two numbers per model, verdict sentence.
    fig = json.dumps(s["figure"])
    rows = "".join(
        f"<tr><td>{html.escape(m['model'])}</td><td>{m['error_after_switch']}</td>"
        f"<td>{m['time_to_fail']}</td></tr>"
        for m in s["numbers"]
    )
    body = (
        f"<h1>Results</h1><p class=note>{html.escape(s['verdict'])}</p>"
        '<div id=fig style="height:520px"></div>'
        f"<script>const F={fig};Plotly.newPlot('fig',F.data,F.layout)</script>"
        "<h2>Numbers</h2><div class=wrap><table><thead><tr><th>Model</th>"
        "<th>Error after switch (°)</th><th>Time to fail (s)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )
    return body, True


def people_body(runs_dir: Path) -> tuple[str, bool]:
    s = load_summary(runs_dir, "people")
    if s is None:
        return "<h1>People</h1>" + EMPTY, False
    return "<h1>People</h1>" + s.get("html", ""), True


def runs_body(runs_dir: Path) -> str:
    recs = runrec.load_runs(runs_dir)
    if not recs:
        return "<h1>Runs</h1><p class=note>No runs yet.</p>"
    head = "".join(f"<th>{c}</th>" for c in runrec.FIELDS)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in runrec.FIELDS)
        + "</tr>"
        for r in recs
    )
    return (
        "<h1>Runs</h1><p>All runs, including tests. Click a column to sort.</p>"
        f"<div class=wrap><table class=sortable><thead><tr>{head}</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def section(md: str, heading: str) -> str:
    """Return one `## heading` section of a markdown file."""
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    return m.group(1).strip() if m else ""


def method_body() -> str:
    plan = (REPO / "PLAN.md").read_text()
    data = (REPO / "DATA.md").read_text()
    parts = [
        ("Question", section(plan, "Question")),
        ("Data", section(data, "Recommendation: Camargo 2021")),
        ("Metrics", section(plan, "Metrics")),
    ]
    md = "\n\n".join(f"## {h}\n\n{t}" for h, t in parts)
    return "<h1>Method</h1>" + markdown.markdown(md, extensions=["tables"])


def build(runs_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "plotly.min.js").write_text(plotly.offline.get_plotlyjs())
    res, res_plot = results_body(runs_dir)
    ppl, ppl_plot = people_body(runs_dir)
    (out / "index.html").write_text(page("index.html", "Results", res, res_plot))
    (out / "people.html").write_text(page("people.html", "People", ppl, ppl_plot))
    (out / "runs.html").write_text(page("runs.html", "Runs", runs_body(runs_dir)))
    (out / "method.html").write_text(page("method.html", "Method", method_body()))
    (out / ".nojekyll").write_text("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a.runs, a.out)
    print(f"built {a.out}")


if __name__ == "__main__":
    main()
