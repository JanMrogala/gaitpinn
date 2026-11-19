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


def _fmt_ttf(v: float, h: float) -> str:
    return f"> {h:.0f} (end)" if v >= h else f"{v:.1f}"


def verdict_sentence(s: dict) -> str:
    n = s["numbers"]
    b, q = n["base"]["error_after_switch_mean3"], n["pinn"]["error_after_switch_mean3"]
    pr = s["primary"]
    stats = (
        f"mean error on stairs over hip, knee and ankle: base {b:.2f}°, pinn {q:.2f}°; "
        f"Wilcoxon p = {pr['p']:.3f}, rank-biserial r = {pr['rank_biserial_r']:.2f}, "
        f"n = {pr['n']} people"
    )
    lead = {
        "pinn is better": "Yes: pinn kept a lower joint-angle error than base after the switch "
        "to stairs",
        "base is better": "No: base kept a lower joint-angle error than pinn after the switch "
        "to stairs",
        "no clear difference": "No clear difference: pinn did not keep a reliably lower "
        "joint-angle error than base after the switch to stairs",
    }[s["verdict"]]
    return f"{lead} ({stats})."


def results_body(runs_dir: Path) -> tuple[str, bool]:
    s = load_summary(runs_dir, "results")
    figs = load_summary(runs_dir, "figures")
    if s is None or figs is None:
        return "<h1>Results</h1>" + EMPTY, False
    h = s["H"]
    n = s["numbers"]
    joints = ["hip", "knee", "ankle"]
    head = "".join(f"<th>{j}</th>" for j in joints)
    models = s.get("models", ["base", "pinn"])
    rows = ""
    for m in models:
        e = "".join(f"<td>{n[m]['error_after_switch'][j]:.2f}</td>" for j in joints)
        f = "".join(f"<td>{_fmt_ttf(n[m]['time_to_fail_median'][j], h)}</td>" for j in joints)
        w = "".join(f"<td>{n[m]['walk_error'][j]:.2f}</td>" for j in joints)
        rows += (
            f"<tr><td><b>{m}</b></td>{e}<td>{n[m]['error_after_switch_mean3']:.2f}</td>{f}{w}</tr>"
        )
    updown = "".join(
        f"<tr><td><b>{m}</b></td>"
        + "".join(f"<td>{n[m]['error_ascent'][j]:.2f}</td>" for j in joints)
        + "".join(f"<td>{n[m]['error_descent'][j]:.2f}</td>" for j in joints)
        + "</tr>"
        for m in models
    )
    pr = s["primary"]
    sec = "".join(
        f"<tr><td>{k}</td><td>{v['hodges_lehmann']:+.2f}</td><td>{v['rank_biserial_r']:+.2f}"
        f"</td><td>{v['p']:.3f}</td></tr>"
        for k, v in s["secondary"].items()
    )
    sens = s["sensitivity_all_people"]
    strong = ""
    for m, x in s.get("extra", {}).items():
        q = x["primary"]
        lam = s["configs"][m]["lam"]
        lead = {
            f"{m} is better": f"Yes: {m} kept a lower error than base",
            "base is better": f"No: base kept a lower error than {m}",
            "no clear difference": f"No clear difference between {m} and base",
        }[x["verdict"]]
        strong += (
            f"<p class=note><b>Strongly physics-constrained model ({m}, λ = {lam:g}, added "
            f"after the main result on request; not part of the pre-registered verdict).</b> "
            f"{html.escape(lead)} (mean error on stairs: base "
            f"{n['base']['error_after_switch_mean3']:.2f}°, {m} "
            f"{n[m]['error_after_switch_mean3']:.2f}°; Wilcoxon p = {q['p']:.3f}, "
            f"rank-biserial r = {q['rank_biserial_r']:+.2f}, Hodges–Lehmann "
            f"{q['hodges_lehmann']:+.2f}°, n = {q['n']}).</p>"
        )
    body = f"""
<h1>Results</h1>
<p class=note><b>Did pinn last longer?</b> {html.escape(verdict_sentence(s))}</p>
{strong}
<div id=fig1 style="height:520px"></div>
<h2>The two numbers per model</h2>
<p>Error after switch: RMSE (°) over the stairs part, mean over people. Time to fail: median
over people of the time (s) until the rolling 2 s error stays above 1.5 × the model's own
walking error for 2 s; runs that never fail count as the horizon H = {h:.0f} s.</p>
<div class=wrap><table><thead>
<tr><th rowspan=2>Model</th><th colspan=4>Error after switch (°)</th>
<th colspan=3>Time to fail (s)</th><th colspan=3>Walking error (°)</th></tr>
<tr>{head}<th>mean</th>{head}{head}</tr></thead><tbody>{rows}</tbody></table></div>
<p><b>How to read time to fail:</b> both models cross the threshold about 2 s after the switch.
That is before the first stair step (each stair recording starts with 2–3 s of walking up to the
stairs) and within the network's 1.27 s context across the join between recordings, so it
mostly reflects the join and the approach rather than the climbing itself.</p>
<h2>On the stairs only</h2>
<p>Error (°) on samples labelled as climbing, mean over people.</p>
<div class=wrap><table><thead><tr><th>Model</th><th colspan=3>Climbing up</th>
<th colspan=3>Climbing down</th></tr><tr><th></th>{head}{head}</tr></thead><tbody>{updown}
</tbody></table></div>
<h2>Test</h2>
<p>Primary endpoint: error after switch averaged over hip, knee and ankle, paired across
{pr["n"]} people (base − pinn). Wilcoxon signed-rank p = {pr["p"]:.3f}; rank-biserial
r = {pr["rank_biserial_r"]:+.2f}; Hodges–Lehmann difference {pr["hodges_lehmann"]:+.2f}°
(95% bootstrap CI {pr["hl_ci95"][0]:+.2f} to {pr["hl_ci95"][1]:+.2f}). Verdict:
<b>{html.escape(s["verdict"])}</b>.</p>
<p>Excluded by the pre-registered gyro check: {", ".join(s["excluded"]) or "none"}.
Sensitivity (all {len(sens["people"])} people, not the verdict): p = {sens["p"]:.3f},
r = {sens["rank_biserial_r"]:+.2f}.</p>
<details><summary>Secondary results (base − pinn)</summary><div class=wrap><table>
<thead><tr><th>measure</th><th>HL diff</th><th>r</th><th>p</th></tr></thead>
<tbody>{sec}</tbody></table></div></details>
<h2>Around each stair entry</h2>
<div id=fig2 style="height:460px"></div>
<p>All numbers are computed by <code>gaitpinn.test</code> from the final runs listed on the
<a href="runs.html">Runs</a> page; per-person run IDs are on the <a href="people.html">People</a>
page.</p>
<script>
const F={json.dumps(figs)};
const cfg={{responsive:true,displaylogo:false}};
Plotly.newPlot('fig1',F.error_over_time.data,F.error_over_time.layout,cfg);
Plotly.newPlot('fig2',F.error_by_bout.data,F.error_by_bout.layout,cfg);
</script>"""
    return body, True


def people_body(runs_dir: Path) -> tuple[str, bool]:
    ppl = load_summary(runs_dir, "people")
    s = load_summary(runs_dir, "results")
    if ppl is None or s is None:
        return "<h1>People</h1>" + EMPTY, False
    nums = {
        p: {m: v[m]["error_after_switch"] for m in s.get("models", ["base", "pinn"])}
        for p, v in s["per_person"].items()
    }
    opts = "".join(f"<option>{p}</option>" for p in ppl)
    body = f"""
<h1>People</h1>
<p>True vs predicted angles around the switch (seed 0). Pick a person:
<select id=who>{opts}</select></p>
<p id=info class=note></p>
<div id=fig style="height:720px"></div>
<script>
const P={json.dumps(ppl)}; const N={json.dumps(nums)};
const M={json.dumps(s.get("models", ["base", "pinn"]))};
const C={{base:'#4C72B0',pinn:'#DD8452',pinn_strong:'#55A868'}};
function draw(p) {{
  const d=P[p], J=['hip','knee','ankle'], tr=[];
  J.forEach((j,k)=>{{
    const ax=k?String(k+1):'';
    tr.push({{x:d.t,y:d.true[j],name:'true',line:{{color:'#222',width:1.5}},
      xaxis:'x'+ax,yaxis:'y'+ax,showlegend:k==0,legendgroup:'true'}});
    M.filter(m=>d[m]).forEach(m=>tr.push({{x:d.t,y:d[m][j],name:m,
      line:{{color:C[m],width:1.2}},xaxis:'x'+ax,yaxis:'y'+ax,showlegend:k==0,legendgroup:m}}));
  }});
  const L={{template:'plotly_white',hovermode:'x unified',grid:{{rows:3,columns:1,
    pattern:'independent'}},margin:{{l:60,r:20,t:30,b:40}},
    shapes:[0,1,2].map(k=>({{type:'line',x0:0,x1:0,xref:'x'+(k?k+1:''),yref:'y'+(k?k+1:'')+
      ' domain',y0:0,y1:1,line:{{color:'#555',width:1}}}})),
    yaxis:{{title:{{text:'hip (°)'}}}},yaxis2:{{title:{{text:'knee (°)'}}}},
    yaxis3:{{title:{{text:'ankle (°)'}}}},xaxis3:{{title:{{text:'time since switch (s)'}}}}}};
  Plotly.newPlot('fig',tr,L,{{responsive:true,displaylogo:false}});
  let txt='Runs: '+M.filter(m=>d[m+'_run']).map(m=>d[m+'_run']).join(', ')+'.';
  if (d.excluded) txt+=' Excluded from the comparison by the gyro check.';
  if (N[p]) txt+=' Error after switch (mean of 5 seeds): '+M.filter(m=>N[p][m]).map(m=>
    m+' '+J.map(j=>j+' '+N[p][m][j].toFixed(2)+'°').join(', ')).join('; ')+'.';
  document.getElementById('info').textContent=txt;
}}
const sel=document.getElementById('who'); sel.onchange=()=>draw(sel.value); draw(sel.value);
</script>"""
    return body, True


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
        ("Decisions log", section(plan, "Decisions log")),
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
