"""Download transparent Pokemon PNG templates separately from reference images."""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
from pathlib import Path

import requests
from PIL import Image, ImageChops, ImageDraw
from rich.console import Console
from rich.progress import track

DEFAULT_DATABASE = Path("data/pokemon_source/catalog.sqlite")
DEFAULT_OUTPUT_DIR = Path("data/detector_training_data/pokemon_png")
USER_AGENT = "OpenCardCollector/0.1 (local detector template builder)"
CONSOLE = Console()


def valid_png(path: Path) -> bool:
    """Check that a cached template is readable and has transparent card corners."""
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return False
            image.verify()
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            corners = (
                (0, 0),
                (rgba.width - 1, 0),
                (0, rgba.height - 1),
                (rgba.width - 1, rgba.height - 1),
            )
            if any(alpha.getpixel(point) != 0 for point in corners):
                return False
        return True
    except (OSError, SyntaxError):
        return False


def add_rounded_corner_alpha(image: Image.Image) -> Image.Image:
    """Preserve existing alpha or mask square-corner scans to the card silhouette."""
    template = image.convert("RGBA")
    alpha = template.getchannel("A")
    corner_points = (
        (0, 0),
        (template.width - 1, 0),
        (0, template.height - 1),
        (template.width - 1, template.height - 1),
    )
    if all(alpha.getpixel(point) == 0 for point in corner_points):
        return template

    scale = 4
    width, height = template.size
    mask = Image.new("L", (width * scale, height * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width * scale - 1, height * scale - 1),
        radius=round(width * 0.045 * scale),
        fill=255,
    )
    rounded_mask = mask.resize(template.size, Image.Resampling.LANCZOS)
    template.putalpha(ImageChops.multiply(alpha, rounded_mask))
    return template


def download_templates(
    database_path: Path, output_dir: Path, limit: int | None = None
) -> tuple[int, int]:
    """Cache transparent TCGdex PNG templates beside, not inside, the card catalog."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")
    if not database_path.is_file():
        raise FileNotFoundError(
            f"Pokemon catalog not found: {database_path}. "
            "Run data_preparation/catalogs/pokemon/00.sync_catalog.py first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        database_path.resolve().as_uri() + "?mode=ro", uri=True
    )
    downloaded = failed = 0
    try:
        pending: list[tuple[str, str, Path]] = []
        for card_id, raw_json in connection.execute(
            "SELECT card_id, raw_json FROM pokemon_cards ORDER BY card_id"
        ):
            image_url = json.loads(raw_json).get("image")
            if not image_url:
                continue
            destination = output_dir / f"{card_id}.png"
            if valid_png(destination):
                continue
            pending.append((card_id, image_url, destination))
            if limit is not None and len(pending) >= limit:
                break

        with requests.Session() as session:
            session.headers.update({"User-Agent": USER_AGENT})
            for card_id, image_url, destination in track(
                pending, description="Downloading Pokemon PNG templates", console=CONSOLE
            ):
                temporary = destination.with_suffix(".partial")
                try:
                    with session.get(
                        f"{image_url}/high.png", timeout=60
                    ) as response:
                        response.raise_for_status()
                        content = response.content
                    with Image.open(io.BytesIO(content)) as image:
                        if image.format != "PNG":
                            raise ValueError("Response is not a PNG")
                        image.verify()
                    with Image.open(io.BytesIO(content)) as image:
                        add_rounded_corner_alpha(image).save(
                            temporary, format="PNG"
                        )
                    if not valid_png(temporary):
                        raise ValueError(
                            "Response is not a readable PNG with transparent corners"
                        )
                    temporary.replace(destination)
                    downloaded += 1
                except (OSError, ValueError, requests.RequestException) as error:
                    temporary.unlink(missing_ok=True)
                    failed += 1
                    CONSOLE.print(f"Failed {card_id}: {error}", markup=False)
                finally:
                    time.sleep(0.1)
    finally:
        connection.close()

    print(f"Pokemon PNG templates downloaded: {downloaded}; failed: {failed}")
    print(f"Templates: {output_dir}")
    return downloaded, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit", type=int, help="Maximum missing or invalid PNGs to attempt per run."
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    _, failed = download_templates(args.database, args.output_dir, args.limit)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
