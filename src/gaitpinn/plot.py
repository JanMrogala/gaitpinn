"""Figures from runs/summary/results.json (Plotly JSON, rendered by the website).

- error_over_time: rolling error vs time since the switch, base vs pinn, mean over people with
  a 95% CI band; vertical line at the switch; dashed line at each model's walking error.
  Buttons switch between hip, knee and ankle.
- error_by_bout: error vs time since each stair entry, ascent and descent.
- people: true vs predicted angles around the switch, per person (seed 0 runs).
"""

import json
from pathlib import Path

import numpy as np

from gaitpinn.test import JOINTS, MODELS

COLORS = {"base": "#4C72B0", "pinn": "#DD8452", "pinn_strong": "#55A868"}
BANDS = {
    "base": "rgba(76,114,176,0.18)",
    "pinn": "rgba(221,132,82,0.18)",
    "pinn_strong": "rgba(85,168,104,0.18)",
}
LAYOUT = {
    "template": "plotly_white",
    "hovermode": "x unified",
    "margin": {"l": 60, "r": 20, "t": 60, "b": 50},
    "legend": {"orientation": "h", "y": -0.18},
}


def _joint_buttons(n_per_joint: int, default: int, extra_shapes: dict) -> list:
    buttons = []
    for k, j in enumerate(JOINTS):
        vis = [False] * (n_per_joint * len(JOINTS))
        vis[k * n_per_joint : (k + 1) * n_per_joint] = [True] * n_per_joint
        buttons.append(
            {
                "label": j.capitalize(),
                "method": "update",
                "args": [{"visible": vis}, {"shapes": extra_shapes[j]}],
            }
        )
    return [
        {
            "type": "buttons",
            "direction": "right",
            "x": 0,
            "y": 1.15,
            "xanchor": "left",
            "active": default,
            "buttons": buttons,
        }
    ]


def error_over_time(s: dict, default: str = "knee") -> dict:
    c = s["curves"]
    t = c["t"]
    traces, shapes = [], {}
    for j in JOINTS:
        for m in s.get("models", MODELS):
            cur = c[m][j]
            vis = j == default
            traces.append(
                {
                    "x": t + t[::-1],
                    "y": cur["hi"] + cur["lo"][::-1],
                    "fill": "toself",
                    "fillcolor": BANDS[m],
                    "line": {"width": 0},
                    "hoverinfo": "skip",
                    "showlegend": False,
                    "visible": vis,
                    "type": "scatter",
                }
            )
            traces.append(
                {
                    "x": t,
                    "y": cur["mean"],
                    "name": m,
                    "visible": vis,
                    "line": {"color": COLORS[m], "width": 2},
                    "type": "scatter",
                    "hovertemplate": f"{m}: %{{y:.2f}}°<extra></extra>",
                }
            )
        sh = [
            {
                "type": "line",
                "x0": 0,
                "x1": 0,
                "yref": "paper",
                "y0": 0,
                "y1": 1,
                "line": {"color": "#555", "width": 1},
            }
        ]
        for m in s.get("models", MODELS):
            w = s["numbers"][m]["walk_error"][j]
            sh.append(
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": w,
                    "y1": w,
                    "line": {"color": COLORS[m], "dash": "dash", "width": 1},
                }
            )
        shapes[j] = sh
    layout = LAYOUT | {
        "title": {"text": "Joint-angle error over time since the switch to stairs"},
        "xaxis": {"title": {"text": "time since switch (s)"}},
        "yaxis": {"title": {"text": "rolling 2 s RMSE (°)"}, "rangemode": "tozero"},
        "shapes": shapes[default],
        "updatemenus": _joint_buttons(
            2 * len(s.get("models", MODELS)), JOINTS.index(default), shapes
        ),
    }
    return {"data": traces, "layout": layout}


def error_by_bout(s: dict, default: str = "knee") -> dict:
    b = s["bouts"]
    traces, shapes = [], {}
    for j in JOINTS:
        for kind, dash in [("ascent", "solid"), ("descent", "dot")]:
            for m in s.get("models", MODELS):
                traces.append(
                    {
                        "x": b["t"],
                        "y": b[m][kind][j],
                        "name": f"{m}, {kind}",
                        "visible": j == default,
                        "type": "scatter",
                        "line": {"color": COLORS[m], "dash": dash, "width": 2},
                        "hovertemplate": f"{m} {kind}: %{{y:.2f}}°<extra></extra>",
                    }
                )
        shapes[j] = [
            {
                "type": "line",
                "x0": 0,
                "x1": 0,
                "yref": "paper",
                "y0": 0,
                "y1": 1,
                "line": {"color": "#555", "width": 1},
            }
        ]
    layout = LAYOUT | {
        "title": {"text": "Error around each stair entry"},
        "xaxis": {"title": {"text": "time since stair entry (s)"}},
        "yaxis": {"title": {"text": "RMSE over bouts (°)"}, "rangemode": "tozero"},
        "shapes": shapes[default],
        "updatemenus": _joint_buttons(
            2 * len(s.get("models", MODELS)), JOINTS.index(default), shapes
        ),
    }
    return {"data": traces, "layout": layout}


def people(runs_dir: Path, s: dict, t0: float = -10.0, t1: float = 40.0) -> dict:
    """Per person: true and predicted angles (seed 0) for t in [t0, t1], every 2nd sample."""
    out = {}
    for p in s["people"] + s["excluded"]:
        rec = {}
        for m in s.get("models", MODELS):
            ids = [
                r for r in s["per_person"].get(p, {}).get(m, {}).get("run_ids", []) if "-s0-" in r
            ]
            if not ids:
                ids = sorted(d.name for d in runs_dir.glob(f"final-{m}-{p}-s0-*"))
            if not ids:
                continue
            z = np.load(runs_dir / ids[0] / "stream.npz")
            t = z["t"]
            sel = np.flatnonzero((t >= t0) & (t <= t1))[::2]
            rec["t"] = np.round(t[sel], 2).tolist()
            rec["true"] = {j: np.round(z["y"][sel, k], 2).tolist() for k, j in enumerate(JOINTS)}
            rec[m] = {j: np.round(z["pred"][sel, k], 2).tolist() for k, j in enumerate(JOINTS)}
            rec[f"{m}_run"] = ids[0]
        if rec:
            rec["excluded"] = p in s["excluded"]
            out[p] = rec
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    a = ap.parse_args()
    s = json.loads((a.runs / "summary" / "results.json").read_text())
    figs = {"error_over_time": error_over_time(s), "error_by_bout": error_by_bout(s)}
    (a.runs / "summary" / "figures.json").write_text(json.dumps(figs))
    (a.runs / "summary" / "people.json").write_text(json.dumps(people(a.runs, s)))
    print("wrote figures.json and people.json")


if __name__ == "__main__":
    main()
