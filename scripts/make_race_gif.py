"""README race GIF: the same recorded actions replayed with every switch off and with the speedup preset, side by side,
each video frame shown at the wall-clock time it was rendered (standard library + Pillow + ffmpeg).

Inputs come from one `scripts/verify_physics.sh` run: the two replay legs' videos of one layout (RoboDojo's
eval_result/..., head camera) and their span traces (`trace_w0.jsonl`). Frame k is shown from the end of the k-th
observation span after the episode span starts.

  python scripts/make_race_gif.py --off-video off.mp4 --off-trace off/trace_w0.jsonl \
      --turbo-video turbo.mp4 --turbo-trace turbo/trace_w0.jsonl --layout 8 \
      --out docs/img/race.gif --times docs/data/race.json [--speed 15 --fps 10]
"""
import argparse
import json
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

PW, PH = 360, 270
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, TXT, MUTED, OFF, TURBO, TRACK = (13, 17, 23), (240, 246, 252), (145, 152, 161), (139, 148, 158), (68, 147, 248), (48, 54, 61)


def frame_times(trace):
    spans = [json.loads(l) for l in open(trace) if l.strip()]
    ep = next(s for s in spans if s.get("kind") == "episode")
    obs = sorted((s for s in spans if s.get("kind") == "obs" and s["t0"] >= ep["t0"] - 1e-6), key=lambda s: s["t0"])
    return [s["t1"] - ep["t0"] for s in obs]


def extract(video, tmp, name):
    d = os.path.join(tmp, name)
    os.makedirs(d)
    subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={PW}:{PH}", os.path.join(d, "%04d.png")], check=True)
    return [os.path.join(d, f) for f in sorted(os.listdir(d))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("off-video", "off-trace", "turbo-video", "turbo-trace", "out", "times"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--layout", type=int, required=True)
    ap.add_argument("--speed", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--hold", type=float, default=2.5, help="seconds of GIF time to hold the end")
    a = ap.parse_args()
    legs = {"off": ("Official RoboDojo (every switch off)", OFF, a.off_video, a.off_trace),
            "turbo": ("RoboDojo-Turbo (speedup preset)", TURBO, a.turbo_video, a.turbo_trace)}
    with tempfile.TemporaryDirectory() as tmp:
        frames = {k: extract(v[2], tmp, k) for k, v in legs.items()}
        n = min(len(f) for f in frames.values())
        times = {k: frame_times(v[3])[:n] for k, v in legs.items()}
        end = {k: t[n - 1] for k, t in times.items()}
        f_small, f_text = ImageFont.truetype(FONT, 12), ImageFont.truetype(FONT, 15)
        f_bold, f_big = ImageFont.truetype(FONT_BOLD, 15), ImageFont.truetype(FONT_BOLD, 20)
        W, H, M = 2 * PW + 60, PH + 150, 20
        out_dir = os.path.join(tmp, "gif")
        os.makedirs(out_dir)
        cache = {}
        for g in range(int((max(end.values()) / a.speed + a.hold) * a.fps) + 1):
            T = g / a.fps * a.speed                      # wall-clock seconds since the episode started
            im = Image.new("RGB", (W, H), BG)
            dr = ImageDraw.Draw(im)
            dr.text((W // 2, 16), f"Same recorded actions, replayed on both   ·   {a.speed:g}x speed", font=f_bold, fill=TXT, anchor="mm")
            for i, (k, (label, col, _, _)) in enumerate(legs.items()):
                x = M + i * (PW + M)
                idx = max(0, min(n - 1, sum(1 for t in times[k] if t <= T) - 1))
                if (k, idx) not in cache:
                    cache[(k, idx)] = Image.open(frames[k][idx]).convert("RGB")
                im.paste(cache[(k, idx)], (x, 58))
                dr.text((x, 44), label, font=f_bold, fill=col, anchor="lm")
                dr.rectangle((x, 58 + PH + 10, x + PW, 58 + PH + 16), fill=TRACK)
                dr.rectangle((x, 58 + PH + 10, x + int(PW * (idx + 1) / n), 58 + PH + 16), fill=col)
                t_show = min(T, end[k])
                clock = f"{int(t_show // 60)}:{int(t_show % 60):02d}"
                if T >= end[k]:
                    dr.text((x, 58 + PH + 38), f"episode done in {clock}", font=f_big, fill=col if k == "turbo" else TXT, anchor="lm")
                else:
                    dr.text((x, 58 + PH + 38), f"{clock}  wall clock", font=f_text, fill=TXT, anchor="lm")
            dr.text((W // 2, H - 34), f"stack_blocks layout {a.layout}, head camera, one RTX 5090 · each frame appears when it was rendered"
                                      " · PhysX state bit-identical", font=f_small, fill=MUTED, anchor="mm")
            dr.text((W // 2, H - 16), "a replay runs no policy inference; full closed-loop evaluations are 1.4-2.4x faster (charts below)",
                    font=f_small, fill=MUTED, anchor="mm")
            im.save(os.path.join(out_dir, f"{g:04d}.png"))
        pal = os.path.join(tmp, "palette.png")
        pattern = os.path.join(out_dir, "%04d.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(a.fps), "-i", pattern,
                        "-vf", "palettegen=max_colors=256:stats_mode=full", pal], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(a.fps), "-i", pattern, "-i", pal,
                        "-lavfi", "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle", "-loop", "0", a.out], check=True)
    json.dump({"_about": "Frame times of the README race GIF (scripts/make_race_gif.py): open-loop replay of one recorded "
                         "closed-loop run, every switch off vs the speedup preset (scripts/verify_physics.sh legs). "
                         "frame_t[k] = seconds from the episode start until the k-th observation of the batch was assembled "
                         "(span traces). The replay legs' PhysX state is bit-identical.",
               "layout": a.layout, "speed": a.speed, "fps": a.fps,
               "legs": {k: {"episode_end_s": round(end[k], 2), "frame_t": [round(t, 3) for t in times[k]]} for k in legs}},
              open(a.times, "w"), indent=1)
    print(f"{n} frames; episode end {end['off']:.1f} s vs {end['turbo']:.1f} s; {os.path.getsize(a.out) // 1024} KiB")


if __name__ == "__main__":
    main()
