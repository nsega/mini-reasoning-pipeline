"""Draw the 17 problems both recorded runs stepped on, before to after.

Writes docs/shared-stepped.html from the two committed runs, so every
number on the figure is computed from the records rather than typed:

    uv run python docs/shared_stepped.py

The comparison is fair only because the runs share their baseline record
for record, so the script checks that before drawing. Scores are
report.py's boxed-reward shares and its seeded sign-flip p, taken over
the shared problems in sorted id order, the order the 2026-09-28 run
note used; the sign-flip p depends on that order in its third decimal.
Each mean carries a 95% t interval.

The fonts and their license come from docs/pipeline.html, so the two
figures match and this script carries no second copy of IBM Plex.

Figure script (Claude-written; docs, not pipeline code).
"""
import json
import math
import re
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# package = false keeps mini_reasoning off the path when run as a script.
sys.path.insert(0, str(ROOT))

from mini_reasoning import report  # noqa: E402

RUNS = (
    ("First run", "control", ROOT / "docs/runs/2026-09-21-float32",
     "--blue"),
    ("Second run", "dynamic sampling",
     ROOT / "docs/runs/2026-09-28-dynamic-sampling", "--teal"),
)
FONTS = ROOT / "docs" / "pipeline.html"
OUT = ROOT / "docs" / "shared-stepped.html"
SAMPLES = 7
T_975 = {16: 2.1199}  # Student's t, two-sided 95%, by degrees of freedom

# Chart geometry, in the SVG's own units: the card's inner width.
WIDTH = 972
PLOT_LEFT, PLOT_RIGHT = 270, 950
DOMAIN = 32  # points either side of zero
TOP = 26
ROW = 210
STEP = 16  # vertical distance between stacked dots
RADIUS = 7

STYLE = """\
:root {
  --ground:#EFF2F5; --grid:#DDE4EB; --ink:#1B2A3A; --muted:#5A6B7E;
  --blue:#1D5FBF; --teal:#0E8074; --line:#E3E8EE;
  --sans:"IBM Plex Sans", system-ui, -apple-system, sans-serif;
  --mono:"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background-color: var(--ground);
  background-image: linear-gradient(var(--grid) 1px, transparent 1px),
                    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 24px 24px; color: var(--ink); font-family: var(--sans);
}
.page { width: 1120px; padding: 26px 48px 22px; }
.eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: .24em; text-transform: uppercase; color: var(--muted); }
h1 { font-size: 27px; font-weight: 700; margin: 6px 0 4px; letter-spacing: -.005em; }
.sub { font-size: 13.5px; color: var(--muted); margin: 0 0 12px; }
.rule { height: 2px; background: #A3B1C2; margin-bottom: 16px; }
.card { background: #fff; border: 2px solid var(--ink); border-radius: 12px; padding: 16px 24px 8px; margin-bottom: 14px; }
svg { display: block; }
svg text { font-family: var(--sans); fill: var(--ink); }
svg .mono { font-family: var(--mono); }
svg .dim { fill: var(--muted); }
.foot { font-family: var(--mono); font-size: 11.5px; line-height: 1.6; color: var(--muted); }"""


def load(run_dir: Path) -> dict:
    """Reads a run's eval records and the problems its training stepped on.

    Args:
        run_dir: A committed run directory holding baseline.json,
            validation.json and training.json.

    Returns:
        The baseline and validation records, the stepped problem ids,
        the number of optimizer updates, and the run's date.
    """
    loaded = {name: json.loads((run_dir / f"{name}.json").read_text())
              for name in ("baseline", "validation", "training")}
    steps = loaded["training"]["steps"]
    ids = [r["unique_id"] for r in loaded["baseline"]["records"]]
    groups = report.training_groups(ids, steps,
                                    loaded["training"].get("held_out"))
    return {"baseline": loaded["baseline"]["records"],
            "validation": loaded["validation"]["records"],
            "stepped": groups["stepped"],
            "updates": sum(s["stepped"] for s in steps),
            "date": run_dir.name[:10]}


def changes(run: Mapping, ids: Sequence[str]) -> list[float]:
    """Per problem, the boxed-reward share after training minus before.

    Args:
        run: A run as load() returns it.
        ids: The problems to score, in the order to return them.

    Returns:
        One change per id, as a share of the problem's samples.
    """
    wanted = set(ids)
    before = report.sample_scores(
        [r for r in run["baseline"] if r["unique_id"] in wanted])
    after = report.sample_scores(
        [r for r in run["validation"] if r["unique_id"] in wanted])
    return [after[i]["reward"] - before[i]["reward"] for i in ids]


def summarize(diffs: Sequence[float]) -> dict:
    """The mean change, its 95% t interval and the sign-flip p.

    Args:
        diffs: Per-problem changes, as shares.

    Returns:
        "mean", "low", "high" and "p".

    Raises:
        ValueError: if no t quantile is recorded for this many problems.
    """
    df = len(diffs) - 1
    if df not in T_975:
        raise ValueError(f"no t quantile recorded for {df} degrees of "
                         "freedom; add it to T_975")
    mean = statistics.fmean(diffs)
    half = T_975[df] * statistics.stdev(diffs) / math.sqrt(len(diffs))
    return {"mean": mean, "low": mean - half, "high": mean + half,
            "p": report.signflip_p(diffs)}


def points(share: float) -> str:
    """A share as signed points to one decimal, with a true minus sign."""
    return f"{100 * share:+.1f}".replace("-", "−")


def x(share: float) -> float:
    """The horizontal position of a change, given as a share."""
    return PLOT_LEFT + (100 * share + DOMAIN) / (2 * DOMAIN) * (
        PLOT_RIGHT - PLOT_LEFT)


def row_svg(top: float, name: str, role: str, color: str, run: Mapping,
            diffs: Sequence[float], stats: Mapping) -> list[str]:
    """One run's row: its labels, a dot per problem, the mean and interval.

    Raises:
        ValueError: if a change is not a whole number of samples, which
            would mean the records hold other than SAMPLES per problem.
    """
    paint = f"var({color})"
    base = top + 150  # the lowest dot in each stack
    bar = top + 180
    out = [
        f'<text x="0" y="{top + 30}" font-size="16" font-weight="700">'
        f'{name}</text>',
        f'<text class="mono dim" x="0" y="{top + 50}" font-size="11.5">'
        f'{run["date"]} &#183; {role}</text>',
        f'<text class="mono dim" x="0" y="{top + 67}" font-size="11.5">'
        f'{run["updates"]} updates</text>',
        f'<text x="0" y="{top + 104}" font-size="15" font-weight="700" '
        f'style="fill:{paint}">mean {points(stats["mean"])} points</text>',
        f'<text x="0" y="{top + 124}" font-size="12.5">95% interval '
        f'{points(stats["low"])} to {points(stats["high"])}</text>',
        f'<text class="dim" x="0" y="{top + 142}" font-size="12.5">'
        f'sign-flip p {stats["p"]:.3f}</text>',
    ]
    moved = []
    for d in diffs:
        k = round(d * SAMPLES)
        if abs(d * SAMPLES - k) > 1e-9:
            raise ValueError(f"a change of {d} is not a whole number of "
                             f"{SAMPLES} samples")
        moved.append(k)
    for k, count in sorted(Counter(moved).items()):
        fill, stroke = ((paint, paint) if k else ("#fff", "var(--muted)"))
        for j in range(count):
            out.append(f'<circle cx="{x(k / SAMPLES):.1f}" '
                       f'cy="{base - j * STEP}" r="{RADIUS}" '
                       f'style="fill:{fill};stroke:{stroke}" '
                       f'stroke-width="1.5"/>')
    lo, hi, mid = x(stats["low"]), x(stats["high"]), x(stats["mean"])
    out += [
        f'<line x1="{lo:.1f}" y1="{bar}" x2="{hi:.1f}" y2="{bar}" '
        f'style="stroke:{paint}" stroke-width="2.5"/>',
        *(f'<line x1="{e:.1f}" y1="{bar - 6}" x2="{e:.1f}" y2="{bar + 6}" '
          f'style="stroke:{paint}" stroke-width="2.5"/>' for e in (lo, hi)),
        f'<rect x="{mid - 6:.1f}" y="{bar - 6}" width="12" height="12" '
        f'transform="rotate(45 {mid:.1f} {bar})" '
        f'style="fill:{paint};stroke:#fff" stroke-width="1.5"/>',
        f'<text x="{mid:.1f}" y="{bar - 11}" text-anchor="middle" '
        f'font-size="12" font-weight="700" style="fill:{paint}">'
        f'{points(stats["mean"])}</text>',
    ]
    return out


def chart_svg(rows: Sequence[tuple]) -> str:
    """The whole chart: rows, column guides, the zero line and the axis."""
    axis = TOP + len(rows) * ROW
    height = axis + 70
    columns = range(-2, 3)
    out = [f'<svg viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" '
           f'height="{height}" role="img">']
    for k in columns:
        if k:
            out.append(f'<line x1="{x(k / SAMPLES):.1f}" y1="{TOP}" '
                       f'x2="{x(k / SAMPLES):.1f}" y2="{axis}" '
                       f'style="stroke:var(--line)" stroke-width="1"/>')
    for i in range(1, len(rows)):
        out.append(f'<line x1="0" y1="{TOP + i * ROW - 5}" x2="{WIDTH}" '
                   f'y2="{TOP + i * ROW - 5}" style="stroke:var(--line)" '
                   f'stroke-width="1"/>')
    zero = x(0)
    out += [f'<line x1="{zero:.1f}" y1="{TOP - 6}" x2="{zero:.1f}" '
            f'y2="{axis}" style="stroke:var(--ink)" stroke-width="1.5" '
            f'stroke-dasharray="4 4"/>',
            f'<text class="mono dim" x="{zero:.1f}" y="{TOP - 12}" '
            f'font-size="11" text-anchor="middle">no change</text>']
    for i, row in enumerate(rows):
        out += row_svg(TOP + i * ROW, *row)
    out.append(f'<line x1="{PLOT_LEFT}" y1="{axis}" x2="{PLOT_RIGHT}" '
               f'y2="{axis}" style="stroke:var(--ink)" stroke-width="1.5"/>')
    for k in columns:
        at = x(k / SAMPLES)
        label = f"{k:+d} of {SAMPLES}".replace("-", "−") if k else "0"
        out += [f'<line x1="{at:.1f}" y1="{axis}" x2="{at:.1f}" '
                f'y2="{axis + 5}" style="stroke:var(--ink)" '
                f'stroke-width="1.5"/>',
                f'<text x="{at:.1f}" y="{axis + 21}" font-size="13" '
                f'text-anchor="middle">{label}</text>',
                f'<text class="mono dim" x="{at:.1f}" y="{axis + 37}" '
                f'font-size="11" text-anchor="middle">'
                f'{points(k / SAMPLES) if k else "0"}</text>']
    out += [f'<text class="dim" x="{(PLOT_LEFT + PLOT_RIGHT) / 2:.1f}" '
            f'y="{axis + 62}" font-size="12.5" text-anchor="middle">'
            f'change per problem, before training to after: samples of '
            f'{SAMPLES} won or lost, and the same in points</text>',
            '</svg>']
    return "\n".join(out)


def fonts() -> tuple[str, str]:
    """The IBM Plex faces and their license notice, from docs/pipeline.html.

    Raises:
        ValueError: if the pipeline figure no longer carries both.
    """
    text = FONTS.read_text()
    faces = re.findall(r"^@font-face \{.*\}$", text, re.M)
    notice = re.search(r"<!--\n  Fonts:.*?-->", text, re.S)
    if not faces or notice is None:
        raise ValueError(f"{FONTS} no longer carries the IBM Plex faces and "
                         "their license notice")
    return "\n".join(faces), notice.group(0)


def main() -> None:
    """Computes the comparison from the runs and writes the figure."""
    runs = [load(path) for _, _, path, _ in RUNS]
    if runs[0]["baseline"] != runs[1]["baseline"]:
        raise ValueError("the runs' baselines differ, so their changes on "
                         "a shared problem are not comparable")
    shared = sorted(set(runs[0]["stepped"]) & set(runs[1]["stepped"]))
    rows = []
    for (name, role, _, color), run in zip(RUNS, runs):
        diffs = changes(run, shared)
        rows.append((name, role, color, run, diffs, summarize(diffs)))
    faces, notice = fonts()
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>mini-reasoning-pipeline: the {len(shared)} shared stepped problems</title>
<!--
  Lesson 4 figure for the blog post. Generated by docs/shared_stepped.py
  from the committed runs: edit the script and rerun it, not this file.
  Written with Claude; see Provenance in the README. IBM Plex is inlined
  from docs/pipeline.html, with its license at the end of this file.

  Rendered to PNG like the pipeline figure, at 1120 CSS px and device
  scale 2, then cropped to the content:
    "Google Chrome" --headless=new --hide-scrollbars --force-device-scale-factor=2 \\
      --window-size=1120,900 --screenshot=shared-stepped.png file://$PWD/docs/shared-stepped.html
-->
<style>
{faces}
{STYLE}
</style>
</head>
<body>
<div class="page">
  <div class="eyebrow">nsega/mini-reasoning-pipeline &middot; boxed reward, per sample &middot; problems both runs stepped on</div>
  <h1>Same {len(shared)} problems, same baseline, opposite signs</h1>
  <p class="sub">Each dot is one problem: how its share of {SAMPLES} samples earning the boxed reward changed from before training to after. Hollow dots did not move. Both runs start from the same baseline, record for record.</p>
  <div class="rule"></div>
  <div class="card">
{chart_svg(rows)}
  </div>
  <div class="foot">Mean with a 95% t interval over the {len(shared)} problems. p: report.py's seeded sign-flip test, as in the run note.<br>Source: docs/runs/2026-09-21-float32 and docs/runs/2026-09-28-dynamic-sampling, via docs/shared_stepped.py.</div>
</div>
{notice}
</body>
</html>
"""
    OUT.write_text(page)
    for name, _, _, _, _, stats in rows:
        print(f"{name}: mean {points(stats['mean'])}, 95% interval "
              f"{points(stats['low'])} to {points(stats['high'])}, "
              f"p {stats['p']:.3f}")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(shared)} shared problems)")


if __name__ == "__main__":
    main()
