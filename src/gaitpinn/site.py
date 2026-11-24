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
import numpy as np
import plotly.offline

from gaitpinn import runs as runrec

REPO = Path(__file__).resolve().parents[2]
PAGES = [
    ("index.html", "Results"),
    ("people.html", "People"),
    ("study2.html", "Study 2"),
    ("study3.html", "Study 3"),
    ("study4.html", "Study 4"),
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


def added_section(s: dict, figs: dict, m: str) -> str:
    """A comparison added after the pre-registered result, shown separately and neutrally."""
    x, n, c = s["extra"][m], s["numbers"], s["configs"][m]
    q = x["primary"]
    joints = ["hip", "knee", "ankle"]
    head = "".join(f"<th>{j}</th>" for j in joints)
    rows = ""
    for mm in ["base", m]:
        e = "".join(f"<td>{n[mm]['error_after_switch'][j]:.2f}</td>" for j in joints)
        u = "".join(f"<td>{n[mm]['error_ascent'][j]:.2f}</td>" for j in joints)
        w = "".join(f"<td>{n[mm]['walk_error'][j]:.2f}</td>" for j in joints)
        rows += (
            f"<tr><td><b>{mm}</b></td>{e}<td>{n[mm]['error_after_switch_mean3']:.2f}</td>"
            f"{u}{w}<td>{x['val_rmse'][mm]:.2f}</td></tr>"
        )
    sec = "".join(
        f"<tr><td>{k}</td><td>{v['hodges_lehmann']:+.2f}</td><td>{v['rank_biserial_r']:+.2f}"
        f"</td><td>{v['p']:.3f}</td></tr>"
        for k, v in x["secondary"].items()
    )
    word = {
        "no clear difference": "no clear difference between the two",
        "base is better": "base had the lower error on stairs",
        f"{m} is better": f"{m} had the lower error on stairs",
    }[x["verdict"]]
    return f"""
<h2>Added after the main result: a strongly physics-constrained model</h2>
<p class=note>Not pre-registered. Requested after the main result was known.
<code>{m}</code> is the tuned <code>pinn</code> network (lr {c["lr"]:g}, width {c["width"]})
with a fixed, large physics weight λ = {c["lam"]:g}, chosen from walking-only loss scales and
not tuned. Compared with base by the same test on the same {q["n"]} people:
<b>{html.escape(word)}</b> (Wilcoxon p = {q["p"]:.3f}, rank-biserial r =
{q["rank_biserial_r"]:+.2f}, Hodges–Lehmann difference base − {m}
{q["hodges_lehmann"]:+.2f}°, 95% CI {q["hl_ci95"][0]:+.2f} to {q["hl_ci95"][1]:+.2f}).</p>
<div data-fig="error_over_time_{m}" style="height:520px"></div>
<div class=wrap><table><thead>
<tr><th rowspan=2>Model</th><th colspan=4>Error after switch (°)</th>
<th colspan=3>Climbing up (°)</th><th colspan=3>Walking error (°)</th>
<th rowspan=2>Walking validation (°)</th></tr>
<tr>{head}<th>mean</th>{head}{head}</tr></thead><tbody>{rows}</tbody></table></div>
<p>Time to fail is measured against each model's own walking error, so a model that walks
less accurately gets a looser threshold; it is shown in the secondary results only.</p>
<details><summary>Secondary results (base − {m})</summary><div class=wrap><table>
<thead><tr><th>measure</th><th>HL diff</th><th>r</th><th>p</th></tr></thead>
<tbody>{sec}</tbody></table></div></details>
<div data-fig="error_by_bout_{m}" style="height:460px"></div>"""


def results_body(runs_dir: Path) -> tuple[str, bool]:
    s = load_summary(runs_dir, "results")
    figs = load_summary(runs_dir, "figures")
    if s is None or figs is None:
        return "<h1>Results</h1>" + EMPTY, False
    h = s["H"]
    n = s["numbers"]
    joints = ["hip", "knee", "ankle"]
    head = "".join(f"<th>{j}</th>" for j in joints)
    models = ["base", "pinn"]  # the pre-registered comparison only
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
    ttf = [n[m]["time_to_fail_median"][j] for m in models for j in joints]
    ttf_lo, ttf_hi = min(ttf), max(ttf)
    pr = s["primary"]
    sec = "".join(
        f"<tr><td>{k}</td><td>{v['hodges_lehmann']:+.2f}</td><td>{v['rank_biserial_r']:+.2f}"
        f"</td><td>{v['p']:.3f}</td></tr>"
        for k, v in s["secondary"].items()
    )
    sens = s["sensitivity_all_people"]
    added = "".join(added_section(s, figs, m) for m in s.get("extra", {}))
    body = f"""
<h1>Results</h1>
<p class=note><b>Did pinn last longer?</b> {html.escape(verdict_sentence(s))}</p>
<div id=fig1 style="height:520px"></div>
<h2>The two numbers per model</h2>
<p>Error after switch: RMSE (°) over the stairs part, mean over people. Time to fail: median
over people of the time (s) until the rolling 2 s error stays above 1.5 × the model's own
walking error for 2 s; runs that never fail count as the horizon H = {h:.0f} s.</p>
<div class=wrap><table><thead>
<tr><th rowspan=2>Model</th><th colspan=4>Error after switch (°)</th>
<th colspan=3>Time to fail (s)</th><th colspan=3>Walking error (°)</th></tr>
<tr>{head}<th>mean</th>{head}{head}</tr></thead><tbody>{rows}</tbody></table></div>
<p><b>How to read time to fail:</b> both models cross the threshold {ttf_lo:.1f}–{ttf_hi:.1f} s
(median per joint) after the switch.
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
{added}
<p>All numbers are computed by <code>gaitpinn.test</code> from the final runs listed on the
<a href="runs.html">Runs</a> page; per-person run IDs are on the <a href="people.html">People</a>
page.</p>
<script>
const F={json.dumps(figs)};
const cfg={{responsive:true,displaylogo:false}};
Plotly.newPlot('fig1',F.error_over_time.data,F.error_over_time.layout,cfg);
Plotly.newPlot('fig2',F.error_by_bout.data,F.error_by_bout.layout,cfg);
document.querySelectorAll('[data-fig]').forEach(el=>{{const f=F[el.dataset.fig];
  Plotly.newPlot(el,f.data,f.layout,cfg);}});
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


def study2_body(runs_dir: Path) -> tuple[str, bool]:
    """Study 2 (PLAN2.md): one model for many people, tested on unseen people."""
    s = load_summary(runs_dir, "results2")
    figs = load_summary(runs_dir, "figures2")
    intro = (
        "<h1>Study 2: one model for many people</h1><p>One model is trained on many people "
        "(walking, ramps and stairs) and tested on people it never saw, in 4 folds so that "
        "every person is tested once. <code>pinn_cal</code> is <code>pinn</code> self-"
        "calibrated to each new person on 2 minutes of their <b>unlabelled</b> walking, using "
        "only the physics losses. Pre-registered in <code>PLAN2.md</code>.</p>"
    )
    if s is None or figs is None:
        return intro + EMPTY, False
    n, joints = s["numbers"], ["hip", "knee", "ankle"]
    models = s["models"]
    lines = ""
    for name, w in s["tests"].items():
        a, b = name.split(" vs ")
        lines += (
            f"<li><b>{html.escape(name)}:</b> {html.escape(w['verdict'])} (mean error on stairs "
            f"{n[a]['error_after_switch_mean3']:.2f}° vs {n[b]['error_after_switch_mean3']:.2f}°; "
            f"Wilcoxon p = {w['p']:.3f}, Holm-adjusted p = {w['p_holm']:.3f}, r = "
            f"{w['rank_biserial_r']:+.2f}, Hodges–Lehmann {w['hodges_lehmann']:+.2f}° "
            f"[{w['hl_ci95'][0]:+.2f}, {w['hl_ci95'][1]:+.2f}], n = {w['n']}{nz_note(w)}).</li>"
        )
    head = "".join(f"<th>{j}</th>" for j in joints)
    rows = ""
    for m in models:
        e = "".join(f"<td>{n[m]['error_after_switch'][j]:.2f}</td>" for j in joints)
        u = "".join(f"<td>{n[m]['error_ascent'][j]:.2f}</td>" for j in joints)
        w = "".join(f"<td>{n[m]['walk_error'][j]:.2f}</td>" for j in joints)
        rows += (
            f"<tr><td><b>{m}</b></td>{e}<td>{n[m]['error_after_switch_mean3']:.2f}</td>{u}{w}</tr>"
        )
    folds = "".join(
        f"<tr><td>{html.escape(k)}</td>"
        + "".join(f"<td>{v:+.2f}</td>" for v in w["per_fold_median"].values())
        + "</tr>"
        for k, w in s["tests"].items()
    )
    nf = len(next(iter(s["tests"].values()))["per_fold_median"])
    fold_head = "".join(f"<th>fold {i}</th>" for i in range(nf))
    cal = "".join(
        f"<li><code>{m}</code>: settings by fold {html.escape(json.dumps(c['settings_by_fold']))}"
        f"; gyro term switched off by the label-free gate for {c['joints_gated_off']} of "
        f"{c['joint_checks']} joint checks.</li>"
        for m, c in s["calibration"].items()
        if m in ("pinn_cal", "base_cal")
    )
    gv = s["calibration"].get("gate_vs_true_check")
    if gv:
        cal += (
            f"<li>Label-free gate vs the true-angle gyro check on validation people: both pass "
            f"{gv['both_pass']}, both fail {gv['both_fail']}, gate only {gv['gate_only']}, true "
            f"check only {gv['true_only']}.</li>"
        )
    dg = s["calibration"].get("diagnostic", {})
    if dg:
        gyro_on = [v for v in dg.values() if v["pred"]["gyro"] > 0 or v["true"]["gyro"] > 0]
        better = sum(v["true"]["gyro"] < v["pred"]["gyro"] for v in gyro_on)
        sbetter = sum(v["true"]["swing"] < v["pred"]["swing"] for v in dg.values())
        cal += (
            f"<li>Diagnostic (validation people): the true angles score lower than the "
            f"predictions on the gyro term for {better} of {len(gyro_on)} (the gate switched "
            f"the gyro term off for the other {len(dg) - len(gyro_on)}) and on the swing term "
            f"for {sbetter} of {len(dg)}; calibration can only help where they do.</li>"
        )
    sec = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v['hodges_lehmann']:+.2f}</td>"
        f"<td>{v['rank_biserial_r']:+.2f}</td><td>{v['p']:.3f}</td></tr>"
        for k, v in s["secondary"].items()
    )
    control = '<div data-fig="control" style="height:520px"></div>' if "control" in figs else ""
    body = f"""{intro}
<div class=note><b>Pre-registered tests</b> (paired across people, Holm-corrected over the
three):<ul>{lines}</ul></div>
<div data-fig="error_over_time" style="height:520px"></div>
<h2>Numbers (mean over people)</h2>
<div class=wrap><table><thead>
<tr><th rowspan=2>Model</th><th colspan=4>Error after switch (°)</th>
<th colspan=3>Climbing up (°)</th><th colspan=3>Walking error (°)</th></tr>
<tr>{head}<th>mean</th>{head}{head}</tr></thead><tbody>{rows}</tbody></table></div>
<h2>Per fold</h2>
<p>People in a fold share the same trained models, so the pooled test treats them as more
independent than they are. Median difference (first − second, °) per fold:</p>
<div class=wrap><table><thead><tr><th>test</th>{fold_head}</tr></thead>
<tbody>{folds}</tbody></table></div>
<h2>Self-calibration</h2><ul>{cal}</ul>
{control}
<details><summary>Secondary results</summary>
<p>Differences are first − second, not corrected for multiple tests. For errors a positive
value means the second model is better; for time to fail (longer is better) a positive value
means the first model is better.</p><div class=wrap><table>
<thead><tr><th>comparison / measure / joint</th><th>HL diff</th><th>r</th><th>p</th></tr>
</thead><tbody>{sec}</tbody></table></div></details>
<h2>Around each stair entry</h2>
<div data-fig="error_by_bout" style="height:460px"></div>
<script>
const F={json.dumps(figs)};
const cfg={{responsive:true,displaylogo:false}};
document.querySelectorAll('[data-fig]').forEach(el=>{{const f=F[el.dataset.fig];
  Plotly.newPlot(el,f.data,f.layout,cfg);}});
</script>"""
    return body, True


def _p(v: float) -> str:
    return "< 0.001" if v < 0.001 else f"= {v:.3f}"


def _test_line(label: str, w: dict, n: dict, a: str = "base", b: str = "pinn") -> str:
    v = w.get("verdict")
    head = f"<b>{html.escape(label)}:</b> " + (f"{html.escape(v)} " if v else "")
    holm = f", Holm-adjusted p {_p(w['p_holm'])}" if "p_holm" in w else ""
    return (
        f"<li>{head}(mean error on stairs {a} {n[a]['error_after_switch_mean3']:.2f}° vs {b} "
        f"{n[b]['error_after_switch_mean3']:.2f}°; Wilcoxon p {_p(w['p'])}{holm}, "
        f"r = {w['rank_biserial_r']:+.2f}, Hodges–Lehmann ({a} − {b}) {w['hodges_lehmann']:+.2f}° "
        f"[{w['hl_ci95'][0]:+.2f}, {w['hl_ci95'][1]:+.2f}], n = {w['n']}{nz_note(w)}).</li>"
    )


def study3_body(runs_dir: Path) -> tuple[str, bool]:
    """Study 3 (PLAN3.md): gaps in the true angles of training and validation people."""
    s = load_summary(runs_dir, "results3")
    figs = load_summary(runs_dir, "figures3")
    intro = (
        "<h1>Study 3: gaps in the true angles</h1><p>As in study 2, one model is trained on "
        "many people and tested on people it never saw. Here a share of the training and "
        "validation people's true angles is removed in random 2 s blocks, while their sensors "
        "stay complete. <code>base</code> learns only from the angles that remain; "
        "<code>pinn</code> also applies its physics loss to the stretches without angles. "
        "<code>pinn_lab</code> (secondary) is <code>pinn</code> without those stretches. Every "
        "person keeps a 30 s fully labelled calibration trial. Test people are untouched. "
        "Pre-registered in <code>PLAN3.md</code>.</p>"
    )
    if s is None or figs is None:
        return intro + EMPTY, False
    sh = s["shares"]
    lines = "".join(
        _test_line(f"{round(b['share'] * 100)}% missing", b["test"], b["numbers"])
        for t, b in sh.items()
        if "p_holm" in b["test"]
    )
    rows = ""
    for b in sh.values():
        n = b["numbers"]
        cells = "".join(
            f"<td>{n[m]['error_after_switch_mean3']:.2f}</td>" if m in n else "<td>–</td>"
            for m in ("base", "pinn", "pinn_lab")
        )
        walk = "".join(
            f"<td>{np.mean(list(n[m]['walk_error'].values())):.2f}</td>" for m in ("base", "pinn")
        )
        folds = " ".join(f"{v:+.2f}" for v in b["test"]["per_fold_median"].values())
        rows += f"<tr><td>{round(b['share'] * 100)}%</td>{cells}{walk}<td>{folds}</td></tr>"
    sec_lines = ""
    if "s00" in sh:
        sec_lines += _test_line(
            "0% missing (reference, not in the Holm family)",
            sh["s00"]["test"],
            sh["s00"]["numbers"],
        )
    for b in sh.values():
        w = b["secondary"].get("pinn_lab vs pinn/primary")
        if w:
            sec_lines += _test_line(
                f"{round(b['share'] * 100)}%: pinn_lab vs pinn (does the unlabelled data help?)",
                w,
                b["numbers"],
                "pinn_lab",
                "pinn",
            )
    for b in sh.values():
        w = b["secondary"].get("base vs pinn/primary/gyro-check people")
        if w:
            sec_lines += _test_line(
                f"{round(b['share'] * 100)}%: the {w['n']} people who passed study 1's gyro check",
                w,
                b["numbers"],
            )
    if "trend" in s:
        tr = s["trend"]
        sec_lines += (
            f"<li><b>Does the base − pinn difference grow with the share?</b> Page trend test "
            f"over 50%, 90%, 99%: p = {tr['p']:.3f} (n = {tr['n']}).</li>"
        )
    tune_rows = ""
    for key, row in sorted(s.get("tuning", {}).items()):
        vals = ", ".join(
            f"{k}: {v:.2f}" for k, v in sorted(row.items(), key=lambda kv: float(kv[0]))
        )
        tune_rows += f"<tr><td>{html.escape(key)}</td><td>{html.escape(vals)}</td></tr>"
    mk = s.get("masks")
    mask_html = ""
    if mk:
        mrows = ""
        for key in sorted(k for k in mk["realised_share"] if k.endswith("/d0")):
            ks = mk["kept_seconds"][key]
            mrows += (
                f"<tr><td>{html.escape(key.split('/')[0])}</td>"
                f"<td>{mk['realised_share'][key] * 100:.1f}%</td>"
                + "".join(
                    f"<td>{ks[m]:.0f}</td>" for m in ("treadmill", "levelground", "ramp", "stair")
                )
                + "</tr>"
            )
        phys = "; ".join(f"{f}: {len(v['physics_people'])}" for f, v in sorted(mk["folds"].items()))
        mask_html = (
            "<h2>Masks</h2><p>Draw 0 shown; realised share is outside the calibration trial. "
            "Kept seconds are medians over people, including the 30 s trial. Physics training "
            f"people per fold (from the calibration trials, the same at every share): {phys}.</p>"
            "<div class=wrap><table><thead><tr><th>share</th><th>realised</th><th>treadmill s</th>"
            "<th>level ground s</th><th>ramp s</th><th>stair s</th></tr></thead><tbody>"
            f"{mrows}</tbody></table></div>"
        )
    sec = "".join(
        f"<tr><td>{round(b['share'] * 100)}% {html.escape(k)}</td>"
        f"<td>{v['hodges_lehmann']:+.2f}</td>"
        f"<td>{v['rank_biserial_r']:+.2f}</td><td>{v['p']:.3f}</td></tr>"
        for b in sh.values()
        for k, v in b["secondary"].items()
    )
    opts = "".join(
        f'<option value="error_over_time_{t}">{round(b["share"] * 100)}% missing</option>'
        for t, b in sh.items()
    )
    first = next(iter(sh))
    body = f"""{intro}
<div class=note><b>Pre-registered tests</b> (base vs pinn per share, paired across people,
Holm-corrected over 50%, 90%, 99%; smallest effect of interest ±0.5°):<ul>{lines}</ul></div>
<div data-fig="error_vs_share" style="height:480px"></div>
<p>Error bars: 95% interval of the mean across people. They overlap because people differ a
lot from each other; the tests compare the two models within each person (paired), which is
why small differences can still be clear.</p>
<h2>Numbers (mean over people)</h2>
<div class=wrap><table><thead><tr><th>missing</th><th>base</th><th>pinn</th><th>pinn_lab</th>
<th>base walking</th><th>pinn walking</th><th>per-fold median base − pinn (folds 0–3)</th></tr>
</thead><tbody>{rows}</tbody></table></div>
<p>Error columns: error after switch to stairs, mean of hip, knee and ankle (°). Walking:
level-ground walking error (°). Positive fold medians favour pinn.</p>
<h2>Error over time since the switch</h2>
<p><select id=share>{opts}</select></p>
<div id=eot style="height:520px"></div>
<h2>Secondary results</h2><ul>{sec_lines}</ul>
{mask_html}
<details><summary>Tuning: best validation score per λ (pinn) and weight decay (base)</summary>
<div class=wrap><table><thead><tr><th>share / fold / model</th><th>value: best score (°)</th></tr>
</thead><tbody>{tune_rows}</tbody></table></div></details>
<details><summary>All secondary tests</summary>
<p>Differences are first − second, not corrected for multiple tests; positive favours the
second model.</p><div class=wrap><table>
<thead><tr><th>share, comparison / measure / joint</th><th>HL diff</th><th>r</th><th>p</th></tr>
</thead><tbody>{sec}</tbody></table></div></details>
<script>
const F={json.dumps(figs)};
const cfg={{responsive:true,displaylogo:false}};
document.querySelectorAll('[data-fig]').forEach(el=>{{const f=F[el.dataset.fig];
  Plotly.newPlot(el,f.data,f.layout,cfg);}});
const sel=document.getElementById('share');
function eot(k){{const f=F[k]; Plotly.react('eot',f.data,f.layout,cfg);}}
sel.onchange=()=>eot(sel.value); eot('error_over_time_{first}');
</script>"""
    return body, True


def _gap_set(s: dict, pre: str, primary: bool) -> str:
    """One study-4 input set: tests, headroom, numbers, figures (ids prefixed with pre)."""
    sh = s["shares"]
    tests = ""
    for b in sh.values():
        w = b["test"]
        if "p_holm" not in w:
            continue
        fam = "primary" if w.get("family") == "primary" else "secondary, Holm over 50% and 90%"
        tests += _test_line(f"{round(b['share'] * 100)}% missing ({fam})", w, b["numbers"])
    hr = s.get("headroom", {})
    rows = ""
    for t, b in sh.items():
        n = b["numbers"]
        cells = "".join(
            f"<td>{n[m]['error_after_switch_mean3']:.2f}</td>" if m in n else "<td>–</td>"
            for m in ("base", "pinn", "distill")
        )
        kn = "".join(
            f"<td>{n[m]['error_after_switch'][j]:.2f}</td>"
            for m in ("base", "pinn")
            for j in ("knee", "ankle")
        )
        h = hr.get(t)
        hc = f"{h['mean']:+.2f} [{h['ci95'][0]:+.2f}, {h['ci95'][1]:+.2f}]" if h else "–"
        folds = " ".join(f"{v:+.2f}" for v in b["test"]["per_fold_median"].values())
        rows += (
            f"<tr><td>{round(b['share'] * 100)}%</td>{cells}<td>{hc}</td>{kn}<td>{folds}</td></tr>"
        )
    sec = ""
    for b in sh.values():
        w = b["secondary"].get("distill vs pinn/primary")
        if w:
            sec += _test_line(
                f"{round(b['share'] * 100)}%: distill vs pinn (Holm over the shares)",
                w,
                b["numbers"],
                "distill",
                "pinn",
            )
    if "s00" in sh:
        sec += _test_line("0% missing (reference)", sh["s00"]["test"], sh["s00"]["numbers"])
    for b in sh.values():
        w = b["secondary"].get("base vs pinn/primary/gyro-check people")
        if w and b["share"] == 0.99:
            sec += _test_line(
                f"99%: the {w['n']} people who passed study 1's gyro check", w, b["numbers"]
            )
    if "trend" in s:
        sec += (
            f"<li><b>Does the base − pinn difference grow with the share?</b> Page trend test "
            f"over 50%, 90%, 99%: p {_p(s['trend']['p'])} (n = {s['trend']['n']}).</li>"
        )
    tc = s.get("teacher_check")
    if tc and tc["n"]:
        sec += (
            f"<li><b>Distill teacher vs study 3's base</b> (same share, fold, seed): mean "
            f"|difference| in validation score {tc['mean_abs_diff']:.2f}°, max "
            f"{tc['max_abs_diff']:.2f}° over {tc['n']} runs (GPU noise only expected).</li>"
        )
    tune_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>"
        + html.escape(
            ", ".join(f"{a}: {v:.2f}" for a, v in sorted(r.items(), key=lambda kv: float(kv[0])))
        )
        + "</td></tr>"
        for k, r in sorted(s.get("tuning", {}).items())
    )
    opts = "".join(
        f'<option value="{pre}:error_over_time_{t}">{round(b["share"] * 100)}% missing</option>'
        for t, b in sh.items()
    )
    first = next(iter(sh))
    return f"""<div class=note><b>Tests</b> (base vs pinn, paired across people):
<ul>{tests}</ul></div>
<div data-fig="{pre}:error_vs_share" style="height:480px"></div>
<p>Error bars: 95% interval of the mean across people (the tests are paired). Dotted grey:
study 3's all-sensor <code>base</code>, which sees the gyros as inputs.</p>
<div class=wrap><table><thead><tr><th>missing</th><th>base</th><th>pinn</th><th>distill</th>
<th>headroom base − base@0% [95% CI]</th><th>base knee</th><th>base ankle</th><th>pinn knee</th>
<th>pinn ankle</th><th>per-fold median base − pinn</th></tr></thead><tbody>{rows}</tbody>
</table></div>
<p>Errors: error after switch to stairs (°), mean of hip, knee and ankle unless named.</p>
<h3>Secondary</h3><ul>{sec}</ul>
<p><select class=eotsel data-target="{pre}-eot">{opts}</select></p>
<div id="{pre}-eot" data-first="{pre}:error_over_time_{first}" style="height:500px"></div>
<details><summary>Tuning: best validation score per λ (pinn) and weight decay (base)</summary>
<div class=wrap><table><tbody>{tune_rows}</tbody></table></div></details>"""


def study4_body(runs_dir: Path) -> tuple[str, bool]:
    """Study 4 (PLAN4.md): the network sees no gyros; gyros only in the physics loss."""
    intro = (
        "<h1>Study 4: gyros only in the physics loss</h1><p>As in study 3 (gaps in the "
        "training angles), but the network does not see the gyros: <code>base</code> and "
        "<code>pinn</code> get EMG only (primary) or EMG + accelerometers (secondary). The "
        "gyros enter only <code>pinn</code>'s physics loss, so the physics carries "
        "information the network cannot see. <code>distill</code> (secondary) uses the gyros "
        "another way: an all-sensor teacher fills in the missing angles. The primary test is "
        "<code>base</code> vs <code>pinn</code> at 99% missing, EMG only. Pre-registered in "
        "<code>PLAN4.md</code>, after a feasibility pilot.</p>"
    )
    parts, figs = [], {}
    for pre, title in (("g4", "EMG only (primary)"), ("g4a", "EMG + accelerometers (secondary)")):
        s = load_summary(runs_dir, f"results_{pre}")
        f = load_summary(runs_dir, f"figures_{pre}")
        if s is None or f is None:
            continue
        figs |= {f"{pre}:{k}": v for k, v in f.items()}
        parts.append(f"<h2>{title}</h2>" + _gap_set(s, pre, pre == "g4"))
    if not parts:
        return intro + EMPTY, False
    body = (
        intro
        + "".join(parts)
        + f"""
<script>
const F={json.dumps(figs)};
const cfg={{responsive:true,displaylogo:false}};
document.querySelectorAll('[data-fig]').forEach(el=>{{const f=F[el.dataset.fig];
  Plotly.newPlot(el,f.data,f.layout,cfg);}});
document.querySelectorAll('[data-first]').forEach(el=>{{const f=F[el.dataset.first];
  Plotly.newPlot(el,f.data,f.layout,cfg);}});
document.querySelectorAll('select.eotsel').forEach(sel=>{{sel.onchange=()=>{{
  const f=F[sel.value]; Plotly.react(sel.dataset.target,f.data,f.layout,cfg);}};}});
</script>"""
    )
    return body, True


def nz_note(w: dict) -> str:
    """Wilcoxon drops people with an exactly zero difference; say so when it happens."""
    k = w.get("n_nonzero", w["n"])
    return f"; {w['n'] - k} with zero difference, so the test uses {k}" if k < w["n"] else ""


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
        ("Data source and licence", section(data, "Choice")),
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
    s2, s2_plot = study2_body(runs_dir)
    (out / "study2.html").write_text(page("study2.html", "Study 2", s2, s2_plot))
    s3, s3_plot = study3_body(runs_dir)
    (out / "study3.html").write_text(page("study3.html", "Study 3", s3, s3_plot))
    s4, s4_plot = study4_body(runs_dir)
    (out / "study4.html").write_text(page("study4.html", "Study 4", s4, s4_plot))
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
