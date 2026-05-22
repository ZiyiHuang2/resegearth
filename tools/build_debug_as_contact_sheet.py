#!/usr/bin/env python3
"""
Debug-only: build a contact sheet from existing A_S debug PNGs + summary.jsonl.
No model inference; no visual quality judgments in code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

COL_SPECS: List[Tuple[str, str]] = [
    ("image", "image"),
    ("gt", "gt_mask"),
    ("pred", "pred_mask"),
    ("last", "last_layer_as_overlay"),
    ("mean", "mean_layer_as_overlay"),
    ("loss_style", "loss_style_as_overlay"),
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except OSError:
            return ImageFont.load_default()


def _blank_rgb(size: Tuple[int, int], fill: Tuple[int, int, int] = (200, 200, 200)) -> Image.Image:
    return Image.new("RGB", size, fill)


def _load_cell(path: str, size: Tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = _blank_rgb(size)
    x = (size[0] - im.width) // 2
    y = (size[1] - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description="Build contact sheet from debug_as_quality PNGs.")
    p.add_argument(
        "--input-dir",
        type=str,
        default="/home/wangchengjun/huangziyi/reseg/segearth+cross/outputs/debug_as_quality/test_10_small",
    )
    p.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path to summary.jsonl (default: <input-dir>/summary.jsonl)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG (default: <input-dir>/contact_sheet.png)",
    )
    p.add_argument("--cell-size", type=int, default=280, help="Max width/height of each image cell.")
    p.add_argument("--margin-left", type=int, default=200)
    p.add_argument("--title-height", type=int, default=40)
    p.add_argument("--row-pad", type=int, default=8)
    args = p.parse_args()

    in_dir = os.path.abspath(args.input_dir)
    summary_path = args.summary or os.path.join(in_dir, "summary.jsonl")
    out_path = args.output or os.path.join(in_dir, "contact_sheet.png")

    if not os.path.isdir(in_dir):
        print(f"[missing] input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(summary_path):
        print(f"[missing] summary not found: {summary_path}", file=sys.stderr)
        sys.exit(1)

    rows: List[dict] = []
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["sample_id"])

    cell_w = cell_h = int(args.cell_size)
    title_h = int(args.title_height)
    margin_left = int(args.margin_left)
    row_pad = int(args.row_pad)
    ncols = len(COL_SPECS)
    row_inner_h = title_h + cell_h
    row_h = row_inner_h + row_pad

    sheet_w = margin_left + ncols * cell_w
    sheet_h = len(rows) * row_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font_title = _load_font(14)
    font_side = _load_font(13)

    for ri, meta in enumerate(rows):
        sid = meta["sample_id"]
        pred_iou = meta["pred_iou"]
        fg = meta["foreground_ratio"]
        y0 = ri * row_h

        side_text = f"{sid}\npred_iou={pred_iou}\nforeground_ratio={fg}"
        draw.multiline_text((6, y0 + 8), side_text, fill=(0, 0, 0), font=font_side)

        for ci, (short_name, suffix) in enumerate(COL_SPECS):
            fname = f"{sid}_{suffix}.png"
            fpath = os.path.join(in_dir, fname)
            x0 = margin_left + ci * cell_w
            title = f"{sid} {short_name}"
            draw.text((x0 + 4, y0 + 2), title, fill=(0, 0, 0), font=font_title)

            cell_top = y0 + title_h
            if os.path.isfile(fpath):
                cell = _load_cell(fpath, (cell_w, cell_h))
                sheet.paste(cell, (x0, cell_top))
            else:
                print(f"[missing file] {fpath}", file=sys.stderr)
                sheet.paste(_blank_rgb((cell_w, cell_h)), (x0, cell_top))

    sheet.save(out_path, format="PNG")
    print(out_path)


if __name__ == "__main__":
    main()
