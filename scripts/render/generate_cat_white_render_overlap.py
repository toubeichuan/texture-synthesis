#!/usr/bin/env python3
"""对比 cat_white 训练/测试图与 cat 10 网格渲染图的前景重叠。

用途：
    针对 cat_white/train 和 cat_white/test 的 0-9.png，分别与
    cat 10 renders_gs_mesh/00000-00009.png 做同编号重叠验证；同时会在
    00000-00019.png 中搜索 IoU 最高的渲染图，用于检查视角编号是否错位。

使用方式：
    python scripts/render/generate_cat_white_render_overlap.py

输出目录：
    cat_white_render_overlap_verification/
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import generate_cat10_overlap_verification as overlap


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compare_split(root: Path, split: str, out_root: Path) -> None:
    """处理 train 或 test 一个 split，并写出可视化图和指标文件。"""
    source_dir = root / "cat_white" / split
    render_dir = root / "cat 10 renders_gs_mesh"
    out_dir = out_root / split
    pairs_dir = out_dir / "pairs"
    overlays_dir = out_dir / "overlays"
    best_dir = out_dir / "best_match_checks"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    overview_tiles = []

    for idx in range(10):
        src_w, src_h, src = overlap.read_png_rgba(source_dir / f"{idx}.png")
        render_w, render_h, render = overlap.read_png_rgba(render_dir / f"{idx:05d}.png")
        if (src_w, src_h) != (render_w, render_h):
            render = overlap.resize_nearest(render, render_w, render_h, src_w, src_h)

        overlay_img, metrics = overlap.make_overlay(src, render, src_w, src_h)
        pair_img = overlap.make_pair_sheet(src, render, overlay_img, src_w, src_h)
        overlap.write_png_rgba(
            overlays_dir / f"{idx}_vs_{idx:05d}_overlay.png",
            src_w,
            src_h,
            overlay_img,
        )
        overlap.write_png_rgba(
            pairs_dir / f"{idx}_vs_{idx:05d}_catwhite_render_overlay.png",
            src_w * 3 + 48,
            src_h + 24,
            pair_img,
        )
        rows.append(
            {
                "pair": f"{split}/{idx}.png <-> {idx:05d}.png",
                "width": src_w,
                "height": src_h,
                **metrics,
            }
        )
        overview_tiles.append(overlap.resize_nearest(overlay_img, src_w, src_h, 256, 256))

        best = []
        for render_idx in range(20):
            render_w, render_h, render = overlap.read_png_rgba(render_dir / f"{render_idx:05d}.png")
            if (src_w, src_h) != (render_w, render_h):
                render = overlap.resize_nearest(render, render_w, render_h, src_w, src_h)
            candidate_overlay, candidate_metrics = overlap.make_overlay(src, render, src_w, src_h)
            best.append((candidate_metrics["iou"], render_idx, candidate_metrics, candidate_overlay, render))
        best.sort(key=lambda item: item[0], reverse=True)
        best_iou, best_idx, best_metrics, best_overlay, best_render = best[0]
        best_rows.append(
            {
                "source": f"{split}/{idx}.png",
                "best_render": f"{best_idx:05d}.png",
                "same_index_render": f"{idx:05d}.png",
                "best_iou": best_iou,
                "best_cat_covered_by_render": best_metrics["cat_covered_by_render"],
                "best_render_covered_by_cat": best_metrics["render_covered_by_cat"],
            }
        )
        if best_idx != idx:
            best_pair = overlap.make_pair_sheet(src, best_render, best_overlay, src_w, src_h)
            overlap.write_png_rgba(
                best_dir / f"{idx}_vs_{best_idx:05d}_catwhite_render_overlay.png",
                src_w * 3 + 48,
                src_h + 24,
                best_pair,
            )
            overlap.write_png_rgba(
                best_dir / f"{idx}_vs_{best_idx:05d}_overlay.png",
                src_w,
                src_h,
                best_overlay,
            )

    tile_w = tile_h = 256
    gap = 8
    overview_w = tile_w * 5 + gap * 6
    overview_h = tile_h * 2 + gap * 3
    overview = bytearray([245, 245, 245, 255]) * (overview_w * overview_h)
    for idx, tile in enumerate(overview_tiles):
        col = idx % 5
        row = idx // 5
        overlap.paste(
            overview,
            overview_w,
            tile,
            tile_w,
            tile_h,
            gap + col * (tile_w + gap),
            gap + row * (tile_h + gap),
        )
    overlap.write_png_rgba(out_dir / "overview_overlay_grid.png", overview_w, overview_h, overview)

    write_rows(out_dir / "overlap_metrics.csv", rows)
    write_rows(out_dir / "best_matches.csv", best_rows)
    (out_dir / "overlap_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )
    (out_dir / "best_matches.json").write_text(
        json.dumps(best_rows, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_root = root / "cat_white_render_overlap_verification"
    for split in ("train", "test"):
        compare_split(root, split, out_root)
    print(f"Wrote verification output to {out_root}")


if __name__ == "__main__":
    main()
