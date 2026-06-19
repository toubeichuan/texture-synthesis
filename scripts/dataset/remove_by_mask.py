#!/usr/bin/env python3
"""根据 mask 删除图片中的白色区域，并输出带透明通道的 PNG。

用途：
    对一批生成图应用对应的 mask：mask 中接近白色的像素会被视为需要删除的区域，
    输出图对应像素的 alpha 会被设为 0，其余区域保持原图颜色。

使用方式：
    python scripts/dataset/remove_by_mask.py \
        --image-dir path/to/inpainted \
        --mask-dir path/to/mask \
        --output-dir path/to/output

常用参数：
    --image-pattern "*_after.png"    原图匹配规则
    --white-threshold 250            mask RGB 三通道都大于等于该值时判定为白色

命名规则：
    默认从原图文件名去掉 "_after" 得到 prefix，然后查找：
    1. {prefix}_quad_0.00000.png
    2. {prefix}_0.00000.png
    3. {prefix}_quad_*.png 的第一个匹配项
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use white regions in mask images to remove pixels from RGBA images."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--mask-dir", required=True, help="Directory containing mask images.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed PNG files.")
    parser.add_argument("--image-pattern", default="*_after.png", help="Glob pattern for source images.")
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=250,
        help="Mask pixels with all RGB channels >= this value are removed.",
    )
    return parser.parse_args()


def process_one_image(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    white_threshold: int = 250,
) -> None:
    """读取一张原图和对应 mask，把 mask 白色区域设为透明。"""
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("RGB")

    if image.size != mask.size:
        raise ValueError(f"image and mask sizes differ: {image_path} vs {mask_path}")

    image_np = np.array(image)
    mask_np = np.array(mask)

    # mask 的 RGB 三通道都接近白色时，认为该像素需要从原图中删除。
    white_region = np.all(mask_np >= white_threshold, axis=-1)
    image_np[white_region, 3] = 0

    Image.fromarray(image_np).save(output_path)
    print(f"wrote {output_path}")


def find_mask(mask_dir: Path, prefix: str) -> Path | None:
    """按照当前工程常见命名规则查找对应 mask。"""
    candidate_mask_names = [
        f"{prefix}_quad_0.00000.png",
        f"{prefix}_0.00000.png",
    ]

    for mask_name in candidate_mask_names:
        candidate = mask_dir / mask_name
        if candidate.exists():
            return candidate

    fallback_matches = sorted(mask_dir.glob(f"{prefix}_quad_*.png"))
    if fallback_matches:
        return fallback_matches[0]
    return None


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob(args.image_pattern))
    if not image_paths:
        raise ValueError(f"No images matched {args.image_pattern!r} in {image_dir}")

    for image_path in image_paths:
        prefix = image_path.stem.replace("_after", "")
        mask_path = find_mask(mask_dir, prefix)
        if mask_path is None:
            print(f"mask not found for {image_path.name}; prefix={prefix}")
            continue

        output_path = output_dir / image_path.name
        process_one_image(image_path, mask_path, output_path, args.white_threshold)


if __name__ == "__main__":
    main()
