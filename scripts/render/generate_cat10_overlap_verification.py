#!/usr/bin/env python3
"""生成 cat 10 图像与渲染结果的重叠验证图。

用途：
  对比 cat 10/train 中的前景图和 cat 10 renders_gs_mesh 中的网格渲染图，
  输出逐对 overlay、三联对比图、总览图，以及 IoU/覆盖率等指标 CSV/JSON。

默认对比关系：
  cat 10/train/0.png       <-> cat 10 renders_gs_mesh/00000.png
  ...
  cat 10/train/9.png       <-> cat 10 renders_gs_mesh/00009.png

使用方式：
  python scripts/render/generate_cat10_overlap_verification.py

输出目录：
  cat10_overlap_verification/

说明：
  这里故意不依赖 Pillow/OpenCV，只支持本数据集中使用的 8-bit、非隔行
  RGB/RGBA PNG，方便在最小环境中运行。
"""

from __future__ import annotations

import csv
import json
import struct
import zlib
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_rgba(path: Path) -> tuple[int, int, bytearray]:
    """读取 PNG，并统一转换为 RGBA 字节数组。"""
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path} is not a PNG file")

    pos = len(PNG_SIGNATURE)
    width = height = color_type = bit_depth = interlace = None
    compressed = bytearray()

    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        pos += 4
        chunk_type = data[pos : pos + 4]
        pos += 4
        chunk = data[pos : pos + length]
        pos += length
        pos += 4  # CRC

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or color_type is None:
        raise ValueError(f"{path} has no IHDR")
    if bit_depth != 8 or interlace != 0 or color_type not in (2, 6):
        raise ValueError(
            f"{path}: unsupported PNG format, bit_depth={bit_depth}, "
            f"color_type={color_type}, interlace={interlace}"
        )

    channels = 3 if color_type == 2 else 4
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    out = bytearray(width * height * 4)
    prev = bytearray(stride)
    src_pos = 0
    dst_pos = 0

    for _ in range(height):
        filter_type = raw[src_pos]
        src_pos += 1
        scanline = bytearray(raw[src_pos : src_pos + stride])
        src_pos += stride

        for i in range(stride):
            left = scanline[i - channels] if i >= channels else 0
            up = prev[i]
            up_left = prev[i - channels] if i >= channels else 0
            if filter_type == 0:
                value = scanline[i]
            elif filter_type == 1:
                value = (scanline[i] + left) & 0xFF
            elif filter_type == 2:
                value = (scanline[i] + up) & 0xFF
            elif filter_type == 3:
                value = (scanline[i] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                value = (scanline[i] + paeth_predictor(left, up, up_left)) & 0xFF
            else:
                raise ValueError(f"{path}: unsupported PNG filter {filter_type}")
            scanline[i] = value

        for x in range(width):
            src = x * channels
            out[dst_pos : dst_pos + 3] = scanline[src : src + 3]
            out[dst_pos + 3] = scanline[src + 3] if channels == 4 else 255
            dst_pos += 4

        prev = scanline

    return width, height, out


def write_png_rgba(path: Path, width: int, height: int, rgba: bytearray | bytes) -> None:
    rows = bytearray()
    stride = width * 4
    for y in range(height):
        rows.append(0)
        start = y * stride
        rows.extend(rgba[start : start + stride])

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    payload = bytearray(PNG_SIGNATURE)
    payload.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
    payload.extend(chunk(b"IDAT", zlib.compress(bytes(rows), level=6)))
    payload.extend(chunk(b"IEND", b""))
    path.write_bytes(payload)


def pixel_at(img: bytearray | bytes, width: int, x: int, y: int) -> tuple[int, int, int, int]:
    i = (y * width + x) * 4
    return img[i], img[i + 1], img[i + 2], img[i + 3]


def is_cat_foreground(pixel: tuple[int, int, int, int]) -> bool:
    _, _, _, a = pixel
    # Source images in these datasets use alpha=0 for background. This also
    # handles all-white objects, where RGB-based background removal would fail.
    return a >= 16


def is_render_foreground(pixel: tuple[int, int, int, int]) -> bool:
    r, g, b, _ = pixel
    # The mesh render images are on a near-black canvas, while the mesh itself
    # can be white, so only black is treated as empty.
    dark_bg = r < 10 and g < 10 and b < 10
    return not dark_bg


def resize_nearest(
    src: bytearray | bytes, src_w: int, src_h: int, dst_w: int, dst_h: int
) -> bytearray:
    dst = bytearray(dst_w * dst_h * 4)
    for y in range(dst_h):
        sy = min(src_h - 1, y * src_h // dst_h)
        for x in range(dst_w):
            sx = min(src_w - 1, x * src_w // dst_w)
            src_i = (sy * src_w + sx) * 4
            dst_i = (y * dst_w + x) * 4
            dst[dst_i : dst_i + 4] = src[src_i : src_i + 4]
    return dst


def paste(
    canvas: bytearray,
    canvas_w: int,
    src: bytearray | bytes,
    src_w: int,
    src_h: int,
    x0: int,
    y0: int,
) -> None:
    for y in range(src_h):
        dst_i = ((y0 + y) * canvas_w + x0) * 4
        src_i = y * src_w * 4
        canvas[dst_i : dst_i + src_w * 4] = src[src_i : src_i + src_w * 4]


def make_overlay(
    cat: bytearray,
    render: bytearray,
    width: int,
    height: int,
) -> tuple[bytearray, dict[str, float | int]]:
    """生成重叠可视化图，并计算前景 IoU 与覆盖率。"""
    out = bytearray(width * height * 4)
    cat_fg = render_fg = overlap = union = abs_diff_sum = 0

    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 4
            cp = tuple(cat[i : i + 4])
            rp = tuple(render[i : i + 4])
            c_fg = is_cat_foreground(cp)
            r_fg = is_render_foreground(rp)
            cat_fg += int(c_fg)
            render_fg += int(r_fg)
            overlap += int(c_fg and r_fg)
            union += int(c_fg or r_fg)

            abs_diff_sum += abs(cp[0] - rp[0]) + abs(cp[1] - rp[1]) + abs(cp[2] - rp[2])

            if c_fg and r_fg:
                # White/gray: both images contain foreground here.
                diff = (abs(cp[0] - rp[0]) + abs(cp[1] - rp[1]) + abs(cp[2] - rp[2])) // 3
                value = max(80, 255 - diff)
                out[i : i + 4] = bytes((value, value, value, 255))
            elif c_fg:
                # Red: only cat 10/train foreground.
                out[i : i + 4] = b"\xff\x30\x30\xff"
            elif r_fg:
                # Cyan: only cat 10 renders_gs_mesh foreground.
                out[i : i + 4] = b"\x20\xd8\xff\xff"
            else:
                out[i : i + 4] = b"\x1b\x1b\x1b\xff"

    metrics = {
        "cat_foreground_pixels": cat_fg,
        "render_foreground_pixels": render_fg,
        "overlap_pixels": overlap,
        "union_pixels": union,
        "iou": round(overlap / union, 6) if union else 1.0,
        "cat_covered_by_render": round(overlap / cat_fg, 6) if cat_fg else 1.0,
        "render_covered_by_cat": round(overlap / render_fg, 6) if render_fg else 1.0,
        "mean_rgb_abs_diff": round(abs_diff_sum / (width * height * 3), 3),
    }
    return out, metrics


def make_pair_sheet(
    cat: bytearray,
    render: bytearray,
    overlay: bytearray,
    width: int,
    height: int,
) -> bytearray:
    gap = 12
    out_w = width * 3 + gap * 4
    out_h = height + gap * 2
    sheet = bytearray([245, 245, 245, 255]) * (out_w * out_h)
    paste(sheet, out_w, cat, width, height, gap, gap)
    paste(sheet, out_w, render, width, height, width + gap * 2, gap)
    paste(sheet, out_w, overlay, width, height, width * 2 + gap * 3, gap)
    return sheet


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cat_dir = root / "cat 10" / "train"
    render_dir = root / "cat 10 renders_gs_mesh"
    out_dir = root / "cat10_overlap_verification"
    pairs_dir = out_dir / "pairs"
    overlays_dir = out_dir / "overlays"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    thumb_w = thumb_h = 256
    gap = 8
    overview_w = thumb_w * 5 + gap * 6
    overview_h = thumb_h * 2 + gap * 3
    overview = bytearray([245, 245, 245, 255]) * (overview_w * overview_h)

    for idx in range(10):
        cat_path = cat_dir / f"{idx}.png"
        render_path = render_dir / f"{idx:05d}.png"
        cat_w, cat_h, cat = read_png_rgba(cat_path)
        render_w, render_h, render = read_png_rgba(render_path)
        if (cat_w, cat_h) != (render_w, render_h):
            render = resize_nearest(render, render_w, render_h, cat_w, cat_h)
            render_w, render_h = cat_w, cat_h

        overlay, metrics = make_overlay(cat, render, cat_w, cat_h)
        write_png_rgba(overlays_dir / f"{idx}_vs_{idx:05d}_overlay.png", cat_w, cat_h, overlay)
        pair = make_pair_sheet(cat, render, overlay, cat_w, cat_h)
        write_png_rgba(pairs_dir / f"{idx}_vs_{idx:05d}_cat_render_overlay.png", cat_w * 3 + 48, cat_h + 24, pair)

        thumb = resize_nearest(overlay, cat_w, cat_h, thumb_w, thumb_h)
        col = idx % 5
        row = idx // 5
        paste(overview, overview_w, thumb, thumb_w, thumb_h, gap + col * (thumb_w + gap), gap + row * (thumb_h + gap))

        rows.append(
            {
                "pair": f"{idx}.png <-> {idx:05d}.png",
                "width": cat_w,
                "height": cat_h,
                **metrics,
            }
        )

    write_png_rgba(out_dir / "overview_overlay_grid.png", overview_w, overview_h, overview)

    with (out_dir / "overlap_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "overlap_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"Wrote verification images to {out_dir}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
