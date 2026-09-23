"""Synchronize Scryfall's Default Cards bulk export into a local card catalog."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import requests
from PIL import Image

BULK_DATA_URL = "https://api.scryfall.com/bulk-data/default-cards"
USER_AGENT = "OpenCardCollector/0.1 (local card catalog builder)"
DEFAULT_DATA_DIR = Path("data/skryfall_source")


def request(url: str, *, stream: bool = False) -> requests.Response:
    """Request Scryfall data with the project's identifying user agent."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=60,
        stream=stream,
    )
    response.raise_for_status()
    return response


def synchronize_bulk(bulk_dir: Path) -> tuple[Path, bool]:
    """Reuse or download Scryfall's latest Default Cards bulk export."""
    metadata = request(BULK_DATA_URL).json()
    bulk_url = metadata["jsonl_download_uri"]
    metadata_path = bulk_dir / "default-cards.metadata.json"
    bulk_path = bulk_dir / "default-cards.jsonl.gz"
    previous_url = None
    if metadata_path.is_file():
        previous_url = json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "jsonl_download_uri"
        )

    bulk_dir.mkdir(parents=True, exist_ok=True)
    if bulk_path.is_file() and bulk_url == previous_url:
        print(f"Bulk export is current: {bulk_path}")
        return bulk_path, False

    temporary_path = bulk_path.with_suffix(".partial")
    print("Downloading current Scryfall Default Cards export...")
    with (
        request(bulk_url, stream=True) as response,
        temporary_path.open("wb") as output,
    ):
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    temporary_path.replace(bulk_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return bulk_path, True


def iter_cards(bulk_path: Path) -> Iterator[dict[str, Any]]:
    """Stream a plain or gzipped JSONL bulk export without loading it into memory."""
    opener = gzip.open if bulk_path.suffix == ".gz" else open
    with opener(bulk_path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL record in {bulk_path} on line {line_number}"
                ) from error


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the catalog schema used by future embedding and retrieval steps."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS cards (
            scryfall_id TEXT PRIMARY KEY,
            oracle_id TEXT,
            name TEXT NOT NULL,
            set_code TEXT NOT NULL,
            set_name TEXT NOT NULL,
            collector_number TEXT NOT NULL,
            language TEXT NOT NULL,
            layout TEXT NOT NULL,
            released_at TEXT,
            image_url TEXT,
            image_path TEXT,
            image_status TEXT,
            updated_at TEXT NOT NULL,
            raw_json TEXT NOT NULL
        )
        """)
    connection.execute("CREATE INDEX IF NOT EXISTS cards_name_index ON cards(name)")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS card_prices (
            scryfall_id TEXT PRIMARY KEY,
            usd TEXT,
            usd_foil TEXT,
            usd_etched TEXT,
            eur TEXT,
            eur_foil TEXT,
            tix TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (scryfall_id) REFERENCES cards(scryfall_id)
        )
        """)


def image_url(card: dict[str, Any]) -> str | None:
    """Return the full-card image URL; multi-faced cards are deferred for later support."""
    image_uris = card.get("image_uris") or {}
    return image_uris.get("large") or image_uris.get("normal")


def sync_catalog(
    bulk_path: Path,
    database_path: Path,
    images_dir: Path,
    download_images: bool,
    image_limit: int | None,
) -> tuple[int, list[tuple[str, str, Path]]]:
    """Upsert every printing and return reference images that need caching."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    initialize_database(connection)
    pending_images: list[tuple[str, str, Path]] = []
    synced_at = datetime.now(UTC).isoformat()
    count = 0
    new_cards = 0
    changed_cards = 0
    changed_prices = 0

    try:
        for card in iter_cards(bulk_path):
            card_id = card["id"]
            reference_url = image_url(card)
            destination = images_dir / f"{card_id}.jpg"
            previous = connection.execute(
                "SELECT image_url, image_path, raw_json FROM cards WHERE scryfall_id = ?",
                (card_id,),
            ).fetchone()
            previous_url = previous[0] if previous else None
            raw_json = json.dumps(card, separators=(",", ":"))
            if previous is None:
                new_cards += 1
            elif previous[2] != raw_json:
                changed_cards += 1
            if (
                download_images
                and reference_url
                and (previous_url != reference_url or not destination.is_file())
                and (image_limit is None or len(pending_images) < image_limit)
            ):
                pending_images.append((card_id, reference_url, destination))

            connection.execute(
                """
                INSERT INTO cards (
                    scryfall_id, oracle_id, name, set_code, set_name,
                    collector_number, language, layout, released_at, image_url,
                    image_path, image_status, updated_at, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scryfall_id) DO UPDATE SET
                    oracle_id = excluded.oracle_id,
                    name = excluded.name,
                    set_code = excluded.set_code,
                    set_name = excluded.set_name,
                    collector_number = excluded.collector_number,
                    language = excluded.language,
                    layout = excluded.layout,
                    released_at = excluded.released_at,
                    image_url = excluded.image_url,
                    image_path = excluded.image_path,
                    image_status = excluded.image_status,
                    updated_at = excluded.updated_at,
                    raw_json = excluded.raw_json
                """,
                (
                    card_id,
                    card.get("oracle_id"),
                    card["name"],
                    card["set"],
                    card["set_name"],
                    card["collector_number"],
                    card["lang"],
                    card["layout"],
                    card.get("released_at"),
                    reference_url,
                    str(destination) if reference_url else None,
                    card.get("image_status"),
                    synced_at,
                    raw_json,
                ),
            )
            prices = card.get("prices") or {}
            previous_prices = connection.execute(
                "SELECT usd, usd_foil, usd_etched, eur, eur_foil, tix "
                "FROM card_prices WHERE scryfall_id = ?",
                (card_id,),
            ).fetchone()
            price_values = (
                prices.get("usd"),
                prices.get("usd_foil"),
                prices.get("usd_etched"),
                prices.get("eur"),
                prices.get("eur_foil"),
                prices.get("tix"),
            )
            if previous_prices != price_values:
                changed_prices += 1
            connection.execute(
                """
                INSERT INTO card_prices (
                    scryfall_id, usd, usd_foil, usd_etched, eur, eur_foil, tix,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scryfall_id) DO UPDATE SET
                    usd = excluded.usd,
                    usd_foil = excluded.usd_foil,
                    usd_etched = excluded.usd_etched,
                    eur = excluded.eur,
                    eur_foil = excluded.eur_foil,
                        tix = excluded.tix,
                        updated_at = excluded.updated_at
                    """,
                (
                    card_id,
                    *price_values,
                    synced_at,
                ),
            )
            count += 1
            if count % 1000 == 0:
                connection.commit()
        connection.commit()
    finally:
        connection.close()
    return (count, new_cards, changed_cards, changed_prices), pending_images


def cache_images(
    database_path: Path, pending_images: list[tuple[str, str, Path]]
) -> int:
    """Download and validate each new or changed Scryfall reference image."""
    if not pending_images:
        return 0

    connection = sqlite3.connect(database_path)
    downloaded = 0
    try:
        for index, (card_id, url, destination) in enumerate(pending_images, start=1):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination.with_suffix(".partial")
            try:
                with (
                    request(url, stream=True) as response,
                    temporary_path.open("wb") as output,
                ):
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
                with Image.open(temporary_path) as image:
                    image.verify()
                os.replace(temporary_path, destination)
                connection.execute(
                    "UPDATE cards SET image_path = ? WHERE scryfall_id = ?",
                    (str(destination), card_id),
                )
                downloaded += 1
            except (OSError, requests.RequestException) as error:
                temporary_path.unlink(missing_ok=True)
                print(f"[{index}/{len(pending_images)}] Failed {card_id}: {error}")
        connection.commit()
    finally:
        connection.close()
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize Scryfall Default Cards into a local SQLite catalog."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Cache large reference images missing from, or changed in, the catalog.",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        help="Maximum changed or missing images to download; omit for every image.",
    )
    args = parser.parse_args()
    if args.image_limit is not None and args.image_limit <= 0:
        parser.error("--image-limit must be greater than zero")

    bulk_path, downloaded_bulk = synchronize_bulk(args.data_dir / "bulk")
    database_path = args.data_dir / "catalog.sqlite"
    changes, pending_images = sync_catalog(
        bulk_path,
        database_path,
        args.data_dir / "images",
        args.download_images,
        args.image_limit,
    )
    downloaded = cache_images(database_path, pending_images)
    count, new_cards, changed_cards, changed_prices = changes
    print("\nSync report")
    print(f"  Bulk export: {'downloaded' if downloaded_bulk else 'reused'}")
    print(f"  Printings processed: {count}")
    print(f"  New printings: {new_cards}")
    print(f"  Changed printings: {changed_cards}")
    print(f"  Price records changed: {changed_prices}")
    if args.download_images:
        print(f"  Reference images cached: {downloaded}")


if __name__ == "__main__":
    main()
