"""Assemble the demo video from logged rollout data.

Reads results/summary.json (written by run_rollouts.py) and the per-episode
videos it references. Every label and number in the video comes from that
logged data; the episode shown in each panel is always episode 0 for that
prompt (first by seed order), never hand-picked.

  Shot 1 (~20 s): 2x2 grid, in-distribution prompts, same starting scene.
  Shot 2 (~15 s): 2x2 grid, novel prompts, labeled NOVEL.
  Closing card (~7 s): table of every prompt, group, and success rate.

Outputs media/demo.mp4 (1920x1080 @ 30 fps) and media/shot1.gif.

Usage (client venv):  python scripts/make_video.py [--results results] [--out media]
"""

import argparse
import json
import pathlib

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

WIDTH, HEIGHT = 1920, 1080
FPS = 30
SOURCE_FPS = 20.0  # per-episode videos are written at 20 Hz by run_rollouts.py

TITLE_H = 84
CELL_W, CELL_H = WIDTH // 2, (HEIGHT - TITLE_H) // 2
LABEL_H = 92

BG = (17, 20, 24)
PANEL_BG = (27, 32, 38)
TEXT = (236, 239, 241)
MUTED = (144, 154, 163)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
AMBER = (240, 178, 60)

SHOT1_TARGET_S = 20.0
SHOT2_TARGET_S = 15.0
HOLD_S = 2.0
CARD_S = 7.0


def find_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    try:
        import matplotlib.font_manager as fm

        candidates.insert(0, fm.findfont("DejaVu Sans"))
    except Exception:
        pass
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


F_TITLE = find_font(44)
F_PROMPT = find_font(36)
F_STATS = find_font(30)
F_SMALL = find_font(24)
F_TABLE = find_font(32)


def text_width(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def load_episode_frames(video_path, target_size):
    """All frames of one episode video, resized to a square of target_size."""
    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:
        img = Image.fromarray(frame).resize((target_size, target_size), Image.LANCZOS)
        frames.append(np.asarray(img))
    reader.close()
    return frames


def panel_image(prompt_entry, frame, novel, finished):
    """Compose one grid cell: label strip on top, episode frame below."""
    cell = Image.new("RGB", (CELL_W, CELL_H), PANEL_BG)
    draw = ImageDraw.Draw(cell)

    text = '"{}"'.format(prompt_entry["text"])
    font = F_PROMPT
    if text_width(draw, text, font) > CELL_W - 48:
        font = F_STATS
    draw.text((24, 12), text, font=font, fill=TEXT)

    k, n = prompt_entry["num_successes"], prompt_entry["num_episodes"]
    rate_color = GREEN if prompt_entry["success_rate"] >= 0.5 else RED
    stats = "{}/{} episodes succeeded".format(k, n)
    draw.ellipse((24, 60, 44, 80), fill=rate_color)
    draw.text((56, 56), stats, font=F_STATS, fill=TEXT)
    if novel:
        badge = "NOVEL"
        bw = text_width(draw, badge, F_STATS)
        draw.rectangle((CELL_W - bw - 48, 54, CELL_W - 16, 88), outline=AMBER, width=2)
        draw.text((CELL_W - bw - 32, 58), badge, font=F_STATS, fill=AMBER)

    frame_img = Image.fromarray(frame)
    size = frame_img.width
    x = (CELL_W - size) // 2
    cell.paste(frame_img, (x, LABEL_H))

    if finished:
        ep0 = prompt_entry["episodes"][0]
        ok = ep0["success"]
        tag = "this episode: SUCCESS" if ok else "this episode: FAILURE"
        color = GREEN if ok else RED
        tw = text_width(draw, tag, F_SMALL)
        ty = CELL_H - 40
        draw.rectangle((x + 8, ty - 6, x + tw + 28, ty + 28), fill=(0, 0, 0))
        draw.text((x + 18, ty), tag, font=F_SMALL, fill=color)

    return cell


def render_shot(entries, title, novel, target_s, writer, mock=False):
    """One grid shot: panels play simultaneously at a uniform speed."""
    frame_size = CELL_H - LABEL_H
    panels = []
    for entry in entries:
        video = REPO_ROOT_RESULTS / entry["episodes"][0]["video"]
        panels.append(load_episode_frames(video, frame_size))

    longest_s = max(len(p) for p in panels) / SOURCE_FPS
    speed = max(1.0, longest_s / target_s)
    n_out = int(round((min(longest_s, target_s) + HOLD_S) * FPS))

    for i in range(n_out):
        t_source = i / float(FPS) * speed
        src_idx = int(t_source * SOURCE_FPS)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((32, 18), title, font=F_TITLE, fill=TEXT)
        note = "episode 0 of each prompt, played simultaneously"
        if speed > 1.05:
            note += "  ({:.1f}x speed)".format(speed)
        draw.text((WIDTH - text_width(draw, note, F_SMALL) - 32, 36), note, font=F_SMALL, fill=MUTED)
        if mock:
            draw.text((WIDTH // 2 - 160, 40), "MOCK SERVER DATA", font=F_STATS, fill=RED)

        for j, entry in enumerate(entries):
            frames = panels[j]
            idx = min(src_idx, len(frames) - 1)
            finished = src_idx >= len(frames) - 1
            cell = panel_image(entry, frames[idx], novel, finished)
            x = (j % 2) * CELL_W
            y = TITLE_H + (j // 2) * CELL_H
            canvas.paste(cell, (x, y))
            draw.line(
                [(CELL_W, TITLE_H), (CELL_W, HEIGHT)], fill=BG, width=4
            )

        frame = np.asarray(canvas)
        writer.append_data(frame)
        yield frame


def render_closing_card(summary, writer, mock=False):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((64, 40), "Results: every prompt, all episodes", font=F_TITLE, fill=TEXT)
    if mock:
        draw.text((WIDTH - 420, 48), "MOCK SERVER DATA", font=F_STATS, fill=RED)

    col_prompt, col_group, col_rate = 64, 1280, 1600
    y = 140
    draw.text((col_prompt, y), "prompt", font=F_STATS, fill=MUTED)
    draw.text((col_group, y), "group", font=F_STATS, fill=MUTED)
    draw.text((col_rate, y), "success", font=F_STATS, fill=MUTED)
    y += 52

    for entry in summary["prompts"].values():
        novel = entry["group"] == "novel"
        rate_color = GREEN if entry["success_rate"] >= 0.5 else RED
        draw.text((col_prompt, y), '"{}"'.format(entry["text"]), font=F_TABLE, fill=TEXT)
        draw.text(
            (col_group, y),
            "novel" if novel else "in-distribution",
            font=F_TABLE,
            fill=AMBER if novel else MUTED,
        )
        draw.ellipse((col_rate - 34, y + 6, col_rate - 10, y + 30), fill=rate_color)
        draw.text(
            (col_rate, y),
            "{}/{}".format(entry["num_successes"], entry["num_episodes"]),
            font=F_TABLE,
            fill=TEXT,
        )
        y += 64

    footer = "model: pi05_libero ({})   simulator: {}".format(
        summary["checkpoint"], summary["simulator"]
    )
    draw.text((64, HEIGHT - 120), footer, font=F_SMALL, fill=MUTED)
    footer2 = "openpi by Physical Intelligence (Apache 2.0) - public code and checkpoints only"
    draw.text((64, HEIGHT - 80), footer2, font=F_SMALL, fill=MUTED)

    frame = np.asarray(canvas)
    for _ in range(int(CARD_S * FPS)):
        writer.append_data(frame)


def write_gif(frames, out_path):
    gif_w = 640
    gif_fps = 10
    step = max(1, int(round(FPS / float(gif_fps))))
    small = []
    for frame in frames[::step]:
        img = Image.fromarray(frame)
        img = img.resize((gif_w, int(img.height * gif_w / img.width)), Image.LANCZOS)
        small.append(np.asarray(img))
    imageio.mimwrite(str(out_path), small, fps=gif_fps, loop=0)


def main():
    global REPO_ROOT_RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(REPO_ROOT / "results"))
    parser.add_argument("--out", default=str(REPO_ROOT / "media"))
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results)
    REPO_ROOT_RESULTS = results_dir
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(str(results_dir / "summary.json")) as f:
        summary = json.load(f)

    entries = list(summary["prompts"].values())
    in_dist = [e for e in entries if e["group"] == "in_distribution"][:4]
    novel = [e for e in entries if e["group"] == "novel"][:4]
    if not in_dist or not novel:
        raise SystemExit("summary.json must contain both prompt groups")

    mock = bool((summary.get("server_metadata") or {}).get("mock"))
    mp4_path = out_dir / "demo.mp4"
    writer = imageio.get_writer(
        str(mp4_path),
        fps=FPS,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,  # keep exactly 1920x1080 (1080 % 16 != 0)
    )
    shot1_frames = list(
        render_shot(
            in_dist, "Same scene. Different instruction.", False, SHOT1_TARGET_S, writer, mock=mock
        )
    )
    for _ in render_shot(
        novel,
        "Novel instructions (not in the fine-tuning tasks)",
        True,
        SHOT2_TARGET_S,
        writer,
        mock=mock,
    ):
        pass
    render_closing_card(summary, writer, mock=mock)
    writer.close()
    print("wrote", mp4_path)

    gif_path = out_dir / "shot1.gif"
    write_gif(shot1_frames, gif_path)
    print("wrote", gif_path)


if __name__ == "__main__":
    main()
