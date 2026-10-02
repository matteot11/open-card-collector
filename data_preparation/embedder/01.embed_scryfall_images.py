"""Embed cached MTG or Pokemon reference images for local card-similarity search."""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
from rich.progress import track

DEFAULT_DATA_DIR = Path("data/scryfall_source")
DEFAULT_POKEMON_DATA_DIR = Path("data/pokemon_source")
DEFAULT_MTG_MODEL = "matteot11/collector-mtg-embedder-dinov3-small"
DEFAULT_POKEMON_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def import_runtime():
    """Load optional embedding dependencies only when building embeddings."""
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as error:
        raise SystemExit(
            "Missing embedder dependencies. Install them with:\n"
            "uv sync --extra embedder-training"
        ) from error
    return torch, AutoModel, AutoProcessor


def select_device(torch, requested_device: str) -> str:
    """Use the requested device or select the best available accelerator."""
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def image_digest(image_path: Path) -> str:
    """Return the SHA-256 digest used to avoid recomputing unchanged images."""
    digest = hashlib.sha256()
    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_embedding_table(connection: sqlite3.Connection, game: str) -> None:
    """Create durable, model-specific vectors in the matching game's catalog."""
    if game not in {"mtg", "pokemon"}:
        raise ValueError(f"Unsupported game: {game}")
    if game == "mtg":
        connection.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                scryfall_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (scryfall_id, model_name),
                FOREIGN KEY (scryfall_id) REFERENCES cards(scryfall_id)
            )
            """)
    else:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS pokemon_embeddings (
                card_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (card_id, model_name),
                FOREIGN KEY (card_id) REFERENCES pokemon_cards(card_id)
            )
            """)
    connection.commit()


def pending_images(
    connection: sqlite3.Connection,
    model_name: str,
    limit: int | None,
    game: str,
) -> list[tuple[str, Path, str]]:
    """Find cached images that are new or whose file content changed."""
    if game not in {"mtg", "pokemon"}:
        raise ValueError(f"Unsupported game: {game}")
    if game == "mtg":
        card_table = "cards"
        card_id_column = "scryfall_id"
        embedding_table = "image_embeddings"
    else:
        card_table = "pokemon_cards"
        card_id_column = "card_id"
        embedding_table = "pokemon_embeddings"
    rows = connection.execute(
        f"SELECT {card_id_column}, image_path FROM {card_table} "
        "WHERE image_path IS NOT NULL"
    ).fetchall()
    pending = []
    for card_id, image_path_text in rows:
        image_path = Path(image_path_text)
        if not image_path.is_file():
            continue
        digest = image_digest(image_path)
        previous = connection.execute(
            f"SELECT image_sha256 FROM {embedding_table} "
            f"WHERE {card_id_column} = ? AND model_name = ?",
            (card_id, model_name),
        ).fetchone()
        if previous is None or previous[0] != digest:
            pending.append((card_id, image_path, digest))
            if limit is not None and len(pending) >= limit:
                break
    return pending


def build_embeddings(
    database_path: Path,
    model_name: str,
    device: str,
    batch_size: int,
    limit: int | None,
    game: str,
) -> tuple[int, int]:
    """Embed new or changed cached card images and store L2-normalized vectors."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Catalog database does not exist: {database_path}")

    torch, AutoModel, AutoProcessor = import_runtime()
    connection = sqlite3.connect(database_path)
    embedded = 0
    failed = 0
    try:
        initialize_embedding_table(connection, game)
        pending = pending_images(connection, model_name, limit, game)
        if not pending:
            return 0, 0
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        for start in track(
            range(0, len(pending), batch_size), description="Embedding images"
        ):
            batch = pending[start : start + batch_size]
            try:
                images = [Image.open(path).convert("RGB") for _, path, _ in batch]
                inputs = processor(images=images, return_tensors="pt")
                inputs = {name: value.to(device) for name, value in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs)
                    vectors = outputs.last_hidden_state[:, 0]
                    vectors = torch.nn.functional.normalize(vectors, dim=1)
                vectors = vectors.cpu().numpy().astype(np.float32)
            except (OSError, ValueError, RuntimeError) as error:
                failed += len(batch)
                print(f"Skipped batch beginning at {batch[0][1]}: {error}")
                continue

            for (card_id, _, digest), vector in zip(batch, vectors, strict=True):
                if game == "mtg":
                    connection.execute(
                        """
                        INSERT INTO image_embeddings (
                            scryfall_id, model_name, image_sha256, dimensions, embedding
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(scryfall_id, model_name) DO UPDATE SET
                            image_sha256 = excluded.image_sha256,
                            dimensions = excluded.dimensions,
                            embedding = excluded.embedding
                        """,
                        (card_id, model_name, digest, len(vector), vector.tobytes()),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO pokemon_embeddings (
                            card_id, model_name, image_sha256, dimensions, embedding
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(card_id, model_name) DO UPDATE SET
                            image_sha256 = excluded.image_sha256,
                            dimensions = excluded.dimensions,
                            embedding = excluded.embedding
                        """,
                        (card_id, model_name, digest, len(vector), vector.tobytes()),
                    )
            connection.commit()
            embedded += len(batch)
    finally:
        connection.close()
    return embedded, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed cached MTG or Pokemon reference images into SQLite."
    )
    parser.add_argument("--game", choices=("mtg", "pokemon"), default="mtg")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, help="Maximum new or changed images to embed."
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--batch-size and --limit must be greater than zero")

    default_data_dir = (
        DEFAULT_DATA_DIR if args.game == "mtg" else DEFAULT_POKEMON_DATA_DIR
    )
    data_dir = args.data_dir or default_data_dir
    model_name = args.model or (
        DEFAULT_MTG_MODEL if args.game == "mtg" else DEFAULT_POKEMON_MODEL
    )
    torch, _, _ = import_runtime()
    device = select_device(torch, args.device)
    embedded, failed = build_embeddings(
        data_dir / "catalog.sqlite",
        model_name,
        device,
        args.batch_size,
        args.limit,
        args.game,
    )
    print(f"Embedded {embedded} reference image(s) using {model_name} on {device}.")
    if failed:
        print(f"Skipped {failed} image(s).")


if __name__ == "__main__":
    main()
