"""Synchronize the TCGdex catalog and optional Pokemon reference images."""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from rich.progress import track

API_URL = "https://api.tcgdex.net/v2/en"
USER_AGENT = "OpenCardCollector/0.1 (local Pokemon catalog builder)"
DEFAULT_DATA_DIR = Path("data/pokemon_source")


def request_json(session: requests.Session, url: str) -> Any:
    """Fetch one TCGdex JSON resource using a polite, bounded request cadence."""
    response = session.get(url, timeout=60)
    response.raise_for_status()
    time.sleep(0.1)
    return response.json()


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the Pokemon catalog tables without mixing provider-specific IDs."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS pokemon_cards (
            card_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            set_id TEXT NOT NULL,
            set_name TEXT NOT NULL,
            local_id TEXT NOT NULL,
            rarity TEXT,
            image_url TEXT,
            cached_image_url TEXT,
            image_path TEXT,
            updated_at TEXT NOT NULL,
            raw_json TEXT NOT NULL
        )
        """)
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(pokemon_cards)")
    }
    if "cached_image_url" not in columns:
        connection.execute(
            "ALTER TABLE pokemon_cards ADD COLUMN cached_image_url TEXT"
        )
    connection.execute("""
        CREATE TABLE IF NOT EXISTS pokemon_card_prices (
            card_id TEXT PRIMARY KEY,
            currency TEXT NOT NULL,
            average REAL,
            low REAL,
            trend REAL,
            price_updated_at TEXT,
            product_id TEXT,
            FOREIGN KEY (card_id) REFERENCES pokemon_cards(card_id)
        )
        """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS pokemon_tcgplayer_prices (
            card_id TEXT NOT NULL,
            variant TEXT NOT NULL,
            currency TEXT NOT NULL,
            low REAL,
            mid REAL,
            high REAL,
            market REAL,
            direct_low REAL,
            price_updated_at TEXT,
            product_id TEXT,
            PRIMARY KEY (card_id, variant),
            FOREIGN KEY (card_id) REFERENCES pokemon_cards(card_id)
        )
        """)
    connection.commit()


def cache_image(
    session: requests.Session,
    image_url: str,
    image_path: Path,
) -> None:
    """Download a JPEG retrieval reference before replacing its cache."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = image_path.with_suffix(".partial")
    try:
        with session.get(f"{image_url}/high.jpg", timeout=60) as response:
            response.raise_for_status()
            content = response.content
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "JPEG":
                raise ValueError(
                    f"Expected JPEG for {image_path}, got {image.format}"
                )
            image.verify()
        temporary_path.write_bytes(content)
        with Image.open(temporary_path) as image:
            image.verify()
        temporary_path.replace(image_path)
    except (OSError, requests.RequestException, ValueError):
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        time.sleep(0.1)


def sync_catalog(
    data_dir: Path,
    *,
    download_images: bool,
    limit: int | None,
    image_limit: int | None,
    refresh: bool,
) -> tuple[int, int, int]:
    """Append or refresh detailed cards and local Cardmarket/TCGplayer prices."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")
    if image_limit is not None and image_limit <= 0:
        raise ValueError("image_limit must be greater than zero")
    data_dir.mkdir(parents=True, exist_ok=True)
    database_path = data_dir / "catalog.sqlite"
    images_dir = data_dir / "images"
    connection = sqlite3.connect(database_path)
    initialize_database(connection)
    processed = downloaded = failed = image_attempts = 0
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    try:
        set_summaries = request_json(session, f"{API_URL}/sets")
        pending_ids: list[tuple[str, str, str]] = []
        for set_summary in track(set_summaries, description="Listing Pokemon sets"):
            set_id = set_summary.get("id")
            if not set_id:
                continue
            set_data = request_json(session, f"{API_URL}/sets/{set_id}")
            for card in set_data.get("cards", []):
                card_id = card.get("id")
                if not card_id:
                    continue
                existing = connection.execute(
                    "SELECT image_url, image_path, cached_image_url, updated_at "
                    "FROM pokemon_cards WHERE card_id = ?",
                    (card_id,),
                ).fetchone()
                reference_missing = (
                    download_images
                    and existing is not None
                    and existing[0]
                    and (
                        existing[1] is None
                        or not Path(existing[1]).is_file()
                        or Path(existing[1]).suffix.lower() != ".jpg"
                        or existing[2] != existing[0]
                    )
                )
                if refresh or existing is None or reference_missing:
                    pending_ids.append((existing[3] if existing else "", set_id, card_id))
                    if not refresh and limit is not None and len(pending_ids) >= limit:
                        break
            if not refresh and limit is not None and len(pending_ids) >= limit:
                break

        pending_ids.sort()
        if limit is not None:
            pending_ids = pending_ids[:limit]
        for _, set_id, card_id in track(
            pending_ids, description="Syncing Pokemon card details"
        ):
            try:
                detail = request_json(session, f"{API_URL}/cards/{card_id}")
                image_url = detail.get("image")
                image_path = images_dir / f"{card_id}.jpg"
                previous_image = connection.execute(
                    "SELECT cached_image_url, image_path FROM pokemon_cards WHERE card_id = ?",
                    (card_id,),
                ).fetchone()
                cached_image_url = previous_image[0] if previous_image else None
                stored_image_path = previous_image[1] if previous_image else None
                if (
                    download_images
                    and image_url
                    and (
                        not image_path.is_file()
                        or cached_image_url != image_url
                        or stored_image_path != str(image_path)
                    )
                    and (image_limit is None or image_attempts < image_limit)
                ):
                    image_attempts += 1
                    try:
                        cache_image(session, image_url, image_path)
                        cached_image_url = image_url
                        stored_image_path = str(image_path)
                        downloaded += 1
                    except (OSError, requests.RequestException, ValueError) as error:
                        failed += 1
                        print(f"Failed image {card_id}: {error}")
                card_set = detail.get("set") or {}
                cardmarket = (detail.get("pricing") or {}).get("cardmarket") or {}
                tcgplayer = (detail.get("pricing") or {}).get("tcgplayer") or {}
                updated_at = datetime.now(timezone.utc).isoformat()
                connection.execute(
                    """
                    INSERT INTO pokemon_cards (
                        card_id, name, set_id, set_name, local_id, rarity,
                        image_url, cached_image_url, image_path, updated_at, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(card_id) DO UPDATE SET
                        name = excluded.name,
                        set_id = excluded.set_id,
                        set_name = excluded.set_name,
                        local_id = excluded.local_id,
                        rarity = excluded.rarity,
                        image_url = excluded.image_url,
                        cached_image_url = COALESCE(
                            excluded.cached_image_url, pokemon_cards.cached_image_url
                        ),
                        image_path = COALESCE(excluded.image_path, pokemon_cards.image_path),
                        updated_at = excluded.updated_at,
                        raw_json = excluded.raw_json
                    """,
                    (
                        card_id,
                        detail.get("name", card_id),
                        card_set.get("id", set_id),
                        card_set.get("name", set_id),
                        str(detail.get("localId", card_id)),
                        detail.get("rarity"),
                        image_url,
                        cached_image_url,
                        stored_image_path,
                        updated_at,
                        json.dumps(detail, separators=(",", ":")),
                    ),
                )
                if cardmarket:
                    connection.execute(
                        """
                        INSERT INTO pokemon_card_prices (
                            card_id, currency, average, low, trend,
                            price_updated_at, product_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(card_id) DO UPDATE SET
                            currency = excluded.currency,
                            average = excluded.average,
                            low = excluded.low,
                            trend = excluded.trend,
                            price_updated_at = excluded.price_updated_at,
                            product_id = excluded.product_id
                        """,
                        (
                            card_id,
                            cardmarket.get("unit", "EUR"),
                            cardmarket.get("avg"),
                            cardmarket.get("low"),
                            cardmarket.get("trend"),
                            cardmarket.get("updated"),
                            str(cardmarket["idProduct"])
                            if cardmarket.get("idProduct") is not None
                            else None,
                        ),
                    )
                else:
                    connection.execute(
                        "DELETE FROM pokemon_card_prices WHERE card_id = ?",
                        (card_id,),
                    )
                connection.execute(
                    "DELETE FROM pokemon_tcgplayer_prices WHERE card_id = ?",
                    (card_id,),
                )
                for variant, prices in tcgplayer.items():
                    if variant in {"unit", "updated"}:
                        continue
                    if not isinstance(prices, dict):
                        raise ValueError(
                            f"Unexpected TCGplayer price record for {variant}"
                        )
                    connection.execute(
                        """
                        INSERT INTO pokemon_tcgplayer_prices (
                            card_id, variant, currency, low, mid, high, market,
                            direct_low, price_updated_at, product_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            card_id,
                            variant,
                            tcgplayer.get("unit", "USD"),
                            prices.get("lowPrice"),
                            prices.get("midPrice"),
                            prices.get("highPrice"),
                            prices.get("marketPrice"),
                            prices.get("directLowPrice"),
                            tcgplayer.get("updated"),
                            str(prices["productId"])
                            if prices.get("productId") is not None
                            else None,
                        ),
                    )
                connection.commit()
                processed += 1
            except (requests.RequestException, ValueError, KeyError, TypeError) as error:
                connection.rollback()
                failed += 1
                print(f"Failed card {card_id}: {error}")
    finally:
        session.close()
        connection.close()

    print(f"Pokemon cards synced: {processed}; failed requests/images: {failed}")
    if download_images:
        print(f"JPEG reference images downloaded: {downloaded}; cache: {images_dir}")
    print(f"Catalog: {database_path}")
    return processed, downloaded, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Cache high-quality JPEG reference images for embedding and retrieval.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum card details to fetch per run; omit to fetch every remaining card.",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        help="Maximum missing images to download per run; omit to download every remaining image.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch existing card details and update cached metadata and prices.",
    )
    args = parser.parse_args()
    if (args.limit is not None and args.limit <= 0) or (
        args.image_limit is not None and args.image_limit <= 0
    ):
        parser.error("--limit and --image-limit must be greater than zero")
    _, _, failed = sync_catalog(
        args.data_dir,
        download_images=args.download_images,
        limit=args.limit,
        image_limit=args.image_limit,
        refresh=args.refresh,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
