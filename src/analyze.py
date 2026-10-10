"""Entry point: turn run files into the paper's numbers and figures.

    python -m src.analyze --runs runs --out analysis

Produces:
  analysis/error_mass.csv    the body of Table 1 (rates and shares per scale)
  analysis/elasticity.csv    per-type scale elasticity, the alpha column
  analysis/remedy_gain.csv   the body of Table 2
  analysis/fig2_results.png  the real version of Figure 2

Depends only on the JSONL rows written by `src.evaluate`, so you can re-run it
any time without touching a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict
from typing import Any

from .models import PARAM_COUNTS
from .taxonomy import CASCADE_ORDER, SHORT_LABELS, family_of


def load_runs(runs_dir: str | pathlib.Path) -> list[dict[str, Any]]:
    """Read every .jsonl in `runs_dir` into one flat list of rows."""
    rows: list[dict[str, Any]] = []
    for path in sorted(pathlib.Path(runs_dir).glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(k, "") for k in keys)].append(row)
    return out


def error_mass_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-model, per-type primary error rate and share of error mass."""
    table: list[dict[str, Any]] = []
    for (model,), group in sorted(_group(rows, "model").items()):
        n = len(group)
        n_err = sum(1 for r in group if r["primary"])
        for error in CASCADE_ORDER:
            hits = sum(1 for r in group if r["primary"] == error.value)
            table.append({
                "model": model,
                "error_type": error.value,
                "label": SHORT_LABELS[error],
                "family": family_of(error),
                "rate_pct": round(100.0 * hits / n, 2) if n else 0.0,
                "share_pct": round(100.0 * hits / n_err, 2) if n_err else 0.0,
                "n": n,
            })
    return table


def elasticity_table(mass: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Least-squares slope of log(rate) on log(params), per error type.

    Strongly negative means scale alone dissolves the type. Near zero or
    positive means scale will not save you and an intervention is required.
    """
    by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in mass:
        params = PARAM_COUNTS.get(row["model"])
        rate = row["rate_pct"]
        if params and rate > 0:
            by_type[row["error_type"]].append((math.log(params), math.log(rate)))

    out: list[dict[str, Any]] = []
    for error in CASCADE_ORDER:
        points = by_type.get(error.value, [])
        if len(points) < 2:
            out.append({"error_type": error.value, "alpha": None, "n_points": len(points)})
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in points) / denom if denom else 0.0
        out.append({
            "error_type": error.value,
            "label": SHORT_LABELS[error],
            "family": family_of(error),
            "alpha": round(slope, 3),
            "n_points": len(points),
        })
    return out


def remedy_gain_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accuracy gain of each remedy over the no-remedy baseline, per model."""
    accuracy: dict[tuple[str, str], float] = {}
    for (model, remedy), group in _group(rows, "model", "remedy").items():
        accuracy[(model, remedy)] = 100.0 * sum(
            1 for r in group if r["correct"]
        ) / len(group)

    models = sorted({m for m, _ in accuracy}, key=lambda m: PARAM_COUNTS.get(m, 0))
    remedies = sorted({r for _, r in accuracy if r != "none"})
    out: list[dict[str, Any]] = []
    for remedy in remedies:
        row: dict[str, Any] = {"remedy": remedy}
        gains = []
        for model in models:
            base = accuracy.get((model, "none"))
            treated = accuracy.get((model, remedy))
            gain = round(treated - base, 2) if base is not None and treated is not None else None
            row[model] = gain
            if gain is not None:
                gains.append(gain)
        row["mean"] = round(sum(gains) / len(gains), 2) if gains else None
        out.append(row)
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    import csv  # noqa: PLC0415

    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(mass: list[dict[str, Any]], out: pathlib.Path) -> None:
    """Redraw Figure 2a from measured numbers. Skipped if matplotlib is absent."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("matplotlib not installed, skipping the figure")
        return

    models = sorted({r["model"] for r in mass}, key=lambda m: PARAM_COUNTS.get(m, 0))
    colours = ["#1b4f80", "#2f7cb8", "#8ab9dd", "#98a4ae", "#edb04a", "#d9662c", "#a32f30"]
    order = [e.value for e in CASCADE_ORDER]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bottoms = [0.0] * len(models)
    for error, colour in zip(order, colours):
        values = [
            next((r["share_pct"] for r in mass
                  if r["model"] == m and r["error_type"] == error), 0.0)
            for m in models
        ]
        ax.bar(models, values, bottom=bottoms, color=colour, width=0.58,
               edgecolor="white", linewidth=0.8, label=error)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylabel("share of total error mass (%)")
    ax.set_xlabel("model scale")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="analysis")
    args = parser.parse_args(argv)

    rows = load_runs(args.runs)
    if not rows:
        print(f"no run files in {args.runs}/. Run src.evaluate first.")
        return 1

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mass = error_mass_table(rows)
    gains = remedy_gain_table(rows)
    write_csv(out_dir / "error_mass.csv", mass)
    write_csv(out_dir / "elasticity.csv", elasticity_table(mass))
    write_csv(out_dir / "remedy_gain.csv", gains)
    plot(mass, out_dir / "fig2_results.png")

    print(f"{len(rows)} rows over {len({r['model'] for r in rows})} model(s)")
    print(f"wrote error_mass.csv, elasticity.csv, remedy_gain.csv to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
