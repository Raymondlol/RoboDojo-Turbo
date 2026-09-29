"""README figures from measured runs (standard library only): docs/data/benchmark.json -> docs/img/*.svg (light + dark).

  python scripts/make_figures.py            # rewrites docs/img/{hero,workloads}{,-dark}.svg

Every number drawn comes from the data file, which records where it was measured; nothing is interpolated or projected.
"""
import json
import os
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "docs", "data", "benchmark.json")
IMG = os.path.join(ROOT, "docs", "img")
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"
THEMES = {
    "light": dict(text="#1f2328", muted="#59636e", grid="#d1d9e0", official="#8c959f", turbo="#0969da"),
    "dark": dict(text="#f0f6fc", muted="#9198a1", grid="#3d444d", official="#6e7681", turbo="#4493f8"),
}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w, h, t, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{esc(title)}"><title>{esc(title)}</title>'
            f'<style>text{{font-family:{FONT};fill:{t["text"]}}}.m{{fill:{t["muted"]}}}.b{{font-weight:600}}'
            f'.t{{fill:{t["turbo"]}}}</style>{"".join(body)}</svg>\n')


def text(x, y, s, size=12, cls="", anchor="start"):
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}"{c} text-anchor="{anchor}">{esc(s)}</text>'


def secs(s):
    return str(int(s + 0.5))          # round half up, as in the docs (298.5 s -> 299 s)


def minutes(s):
    return f"{s / 60:.1f} min"


def batch_steps(runs, gap=5.0):
    """Mean completion time of each batch over the repetitions: [(t_s, layouts_done), ...]. A batch's videos are written
    within a second or two of each other; times closer than `gap` seconds belong to the same batch."""
    per_run = []
    for r in runs:
        steps, n, last = [], 0, None
        for t in sorted(r["episode_done_s"].values()):
            n += 1
            if last is not None and t - last < gap:
                steps[-1] = (t, n)
            else:
                steps.append((t, n))
            last = t
        per_run.append(steps)
    shapes = {tuple(n for _, n in s) for s in per_run}
    assert len(shapes) == 1, f"repetitions have different batch structures: {shapes}"
    return [(mean(s[i][0] for s in per_run), per_run[0][i][1]) for i in range(len(per_run[0]))]


def hero(d, t):
    sb = d.get("hero") or d["stack_blocks"]
    up, tu = sb["runs"]["upstream"], sb["runs"]["speedup"]
    same_cache = sb.get("jax_cache") == "same in both arms"
    w_up, w_tu = mean(r["wall_total_s"] for r in up), mean(r["wall_total_s"] for r in tu)
    W, H = 900, 358
    c = d["conditions"]
    b = [text(24, 34, "Same tasks and scoring, about half the wall-clock time", 19, "b"),
         text(24, 56, f"{sb['task']} · {sb['layouts']} official layouts · Pi_05 · 1 × 10 envs · "
                      f"{c['gpu']} + {c['cpu']} · mean of {len(up)} alternating runs per arm", 12, "m")]
    # left: total wall-clock bars
    x0, bw, y = 24, 300, 104
    b.append(text(x0, 88, f"Wall-clock for all {sb['layouts']} layouts", 13, "b"))
    for label, val, col, cls in (("Official RoboDojo", w_up, t["official"], ""), ("RoboDojo-Turbo", w_tu, t["turbo"], "t b")):
        L = bw * val / w_up
        b.append(text(x0, y + 4, label, 13, cls))
        b.append(f'<rect x="{x0}" y="{y + 12}" width="{L:.1f}" height="26" rx="4" fill="{col}"/>')
        b.append(text(x0 + L + 8, y + 30, minutes(val), 13, cls))
        y += 70
    b.append(text(x0, y + 22, f"{w_up / w_tu:.2f}× faster", 30, "t b"))
    b.append(text(x0, y + 44, f"{secs(w_up)} s → {secs(w_tu)} s (policy server start to last client exit)", 11, "m"))
    # right: layouts finished over time
    px, py, pw, ph = 470, 104, 390, 170
    su, st = batch_steps(up), batch_steps(tu)
    xmax = 60 * (int(max(su[-1][0], w_up) / 60) + 1)
    X = lambda s: px + pw * s / xmax
    Y = lambda n: py + ph - ph * n / sb["layouts"]
    b.append(text(px, 88, "Layouts finished over time", 13, "b"))
    for n in (0, 10, 20, sb["layouts"]):
        b.append(f'<line x1="{px}" y1="{Y(n):.1f}" x2="{px + pw}" y2="{Y(n):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        b.append(text(px - 6, Y(n) + 4, n, 11, "m", "end"))
    for m in range(0, xmax // 60 + 1, 2):
        b.append(text(X(60 * m), py + ph + 16, m, 11, "m", "middle"))
    b.append(text(px + pw, py + ph + 32, "minutes since launch (points: batches finishing, measured; lines join them)", 11, "m", "end"))
    for steps, col, cls, label, side in ((su, t["official"], "", "Official RoboDojo", "below"), (st, t["turbo"], "t b", "RoboDojo-Turbo", "above")):
        pts = [f"{X(0):.1f},{Y(0):.1f}"] + [f"{X(s):.1f},{Y(n):.1f}" for s, n in steps]   # measured points, joined
        b.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="3" stroke-linejoin="round"/>')
        for s, n in steps:
            b.append(f'<circle cx="{X(s):.1f}" cy="{Y(n):.1f}" r="4.5" fill="{col}"/>')
        s_k, n_k = steps[-2] if len(steps) > 1 else steps[-1]   # label next to the second-to-last point, off the line
        if side == "above":
            b.append(text(X(s_k) - 10, Y(n_k) - 8, label, 12, cls, "end"))
        else:
            b.append(text(X(s_k) + 10, Y(n_k) + 18, label, 12, cls, "start"))
    b.append(text(24, H - 26, "Same physics settings, dt and render cadence; PhysX state bit-identical under open-loop replay (layouts 0-9). "
                              "Turbo records one camera", 11, "m"))
    b.append(text(24, H - 12, ("video instead of three; both arms use the same JAX compile cache. " if same_cache else
                               "video instead of three and keeps a JAX compile cache (about 6 s of its time). ") +
                  "Conditions and raw data: docs/benchmark.md, docs/data/benchmark.json.", 11, "m"))
    return svg(W, H, t, b, "RoboDojo-Turbo vs official RoboDojo evaluation wall-clock")


def workloads(d, t):
    rows = d.get("workloads") or []
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: -r["official_s"])
    W, rh = 900, 50
    H = 112 + rh * len(rows) + 34
    c = d["conditions"]
    b = [text(24, 34, "Across workloads", 19, "b"),
         text(24, 56, f"{d['workloads_meta']['layouts']} official layouts per task · Pi_05 · 1 × 10 envs (5 for *_random, "
                      f"RoboDojo's cap) · {c['gpu']} + {c['cpu']}", 12, "m"),
         text(24, 74, (f"mean of {len(rows[0]['runs']['upstream'])} alternating runs per arm (official, Turbo, Turbo, official) · "
                       "same JAX compile cache in both arms") if "runs" in rows[0] else
                      ("1 run per arm, official leg first (stack_blocks: mean of 2 alternating runs) · "
                       "Turbo with a warm JAX compile cache"), 12, "m")]
    lx, bx, bw = 24, 330, 440
    vmax = max(r["official_s"] for r in rows)
    y = 100
    for r in rows:
        b.append(text(lx, y + 18, r["task"], 13, "b"))
        b.append(text(lx, y + 34, r["kind"], 11, "m"))
        for i, (val, col, cls) in enumerate(((r["official_s"], t["official"], "m"), (r["turbo_s"], t["turbo"], "t"))):
            L = bw * val / vmax
            b.append(f'<rect x="{bx}" y="{y + 6 + 17 * i}" width="{L:.1f}" height="14" rx="3" fill="{col}"/>')
            b.append(text(bx + L + 6, y + 17 + 17 * i, minutes(val), 11, cls))
        b.append(text(W - 24, y + 28, f"{r['official_s'] / r['turbo_s']:.2f}×", 18, "t b", "end"))
        y += rh
    b.append(f'<rect x="{bx}" y="{y + 8}" width="12" height="10" rx="2" fill="{t["official"]}"/>')
    b.append(text(bx + 18, y + 17, "Official RoboDojo", 11, "m"))
    b.append(f'<rect x="{bx + 140}" y="{y + 8}" width="12" height="10" rx="2" fill="{t["turbo"]}"/>')
    b.append(text(bx + 158, y + 17, "RoboDojo-Turbo (--preset speedup)", 11, "m"))
    return svg(W, H, t, b, "RoboDojo-Turbo vs official RoboDojo wall-clock across tasks")


def main():
    d = json.load(open(DATA))
    os.makedirs(IMG, exist_ok=True)
    for name, fn in (("hero", hero), ("workloads", workloads)):
        for theme, t in THEMES.items():
            s = fn(d, t)
            if s is None:
                continue
            path = os.path.join(IMG, f"{name}{'-dark' if theme == 'dark' else ''}.svg")
            with open(path, "w") as fh:
                fh.write(s)
            print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
