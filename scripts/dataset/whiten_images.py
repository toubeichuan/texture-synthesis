#!/usr/bin/env python3
"""把图片前景统一变成白色，同时保留透明通道。

用途：
    对纹理/渲染结果做白模化处理：有 alpha 的图片会保留原 alpha，只把 RGB 改成白色；
    没有 alpha 的图片会整张变成白色 RGB 图。

使用方式：
    python scripts/dataset/whiten_images.py path/to/images --output-dir path/to/white_images
    python scripts/dataset/whiten_images.py a.png b.png --output-dir out
    python scripts/dataset/whiten_images.py path/to/images --recursive --output-dir out
"""

import argparse
from pathlib import Path
from typing import Iterable, List

from PIL import Image


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn all non-transparent pixels in images white and save the results."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input image files and/or directories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write processed images into.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directories for images.",
    )
    return parser.parse_args()


def collect_images(inputs: List[str], recursive: bool) -> List[Path]:
    """收集输入文件或目录中的图片路径。"""
    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            if path.suffix.lower() in VALID_SUFFIXES:
                paths.append(path)
            continue

        if not path.is_dir():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        iterator: Iterable[Path]
        if recursive:
            iterator = path.rglob("*")
        else:
            iterator = path.iterdir()

        for candidate in iterator:
            if candidate.is_file() and candidate.suffix.lower() in VALID_SUFFIXES:
                paths.append(candidate)

    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise ValueError("No images found in the provided inputs.")
    return unique_paths


def whiten_image(input_path: Path, output_path: Path) -> None:
    """将图片非透明部分变白；如果有 alpha，则保留原透明度。"""
    image = Image.open(input_path)

    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.putalpha(alpha)
        white.save(output_path)
        return

    rgb = image.convert("RGB")
    white = Image.new("RGB", rgb.size, (255, 255, 255))
    white.save(output_path)


def main() -> None:
    args = parse_args()
    images = collect_images(args.inputs, args.recursive)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        output_path = output_dir / image_path.name
        whiten_image(image_path, output_path)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
