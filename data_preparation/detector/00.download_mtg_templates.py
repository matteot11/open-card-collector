"""Download original MTG PNG detector templates without altering their alpha."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import requests
from PIL import Image
from rich.console import Console
from rich.progress import track


def valid_png(path: Path) -> bool:
    """Check that a cached or downloaded file is a readable PNG."""
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return False
            image.verify()
        return True
    except (OSError, SyntaxError):
        return False


def download_templates(
    database_path: Path, output_dir: Path, limit: int | None = None
) -> tuple[int, int]:
    """Cache original single-face PNGs separately from embedder JPEG references."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")
    if not database_path.is_file():
        raise FileNotFoundError(
            f"Catalog not found: {database_path}. "
            "Run data_preparation/catalogs/mtg/00.sync_catalog.py first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        database_path.resolve().as_uri() + "?mode=ro", uri=True
    )
    downloaded = failed = 0
    console = Console()
    try:
        pending: list[tuple[str, str, Path]] = []
        for card_id, raw_json in connection.execute(
            "SELECT scryfall_id, raw_json FROM cards ORDER BY scryfall_id"
        ):
            card = json.loads(raw_json)
            url = (card.get("image_uris") or {}).get("png")
            if not url:
                continue
            destination = output_dir / f"{card_id}.png"
            if valid_png(destination):
                continue
            pending.append((card_id, url, destination))
            if limit is not None and len(pending) >= limit:
                break
        with requests.Session() as session:
            session.headers.update(
                {
                    "User-Agent": "OpenCardCollector/0.1 (local detector template builder)"
                }
            )
            for card_id, url, destination in track(
                pending, description="Downloading PNG templates", console=console
            ):
                temporary = destination.with_suffix(".partial")
                try:
                    with (
                        session.get(url, stream=True, timeout=60) as response,
                        temporary.open("wb") as output,
                    ):
                        response.raise_for_status()
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                    if not valid_png(temporary):
                        raise ValueError("Response is not a valid PNG")
                    temporary.replace(destination)
                    downloaded += 1
                except (OSError, ValueError, requests.RequestException) as error:
                    temporary.unlink(missing_ok=True)
                    failed += 1
                    console.print(f"Failed {card_id}: {error}", markup=False)
                finally:
                    time.sleep(0.1)
    finally:
        connection.close()
    print(f"Original PNG templates downloaded: {downloaded}; failed: {failed}")
    print(f"Templates: {output_dir}")
    return downloaded, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=Path("data/scryfall_source/catalog.sqlite")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/detector_training_data/scryfall_png"),
    )
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
