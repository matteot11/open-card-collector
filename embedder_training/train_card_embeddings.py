"""Fine-tune DINOv3 Small for card-image retrieval with synthetic camera views."""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

DEFAULT_MTG_DATA_DIR = Path("data/scryfall_source")
DEFAULT_POKEMON_DATA_DIR = Path("data/pokemon_source")
DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def import_runtime():
    """Load optional training dependencies only when training starts."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise SystemExit(
            "Missing embedder dependencies. Install them with:\n"
            "uv sync --extra embedder-training"
        ) from error
    return (
        torch,
        DataLoader,
        AutoImageProcessor,
        AutoModel,
    )


def select_device(torch, requested_device: str) -> str:
    """Use the requested device or select the best available accelerator."""
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image_paths(database_path: Path, game: str) -> list[Path]:
    """Load cached reference image paths from the selected game's catalog."""
    if game not in {"mtg", "pokemon"}:
        raise ValueError(f"Unsupported game: {game}")
    if not database_path.is_file():
        raise FileNotFoundError(f"Catalog database does not exist: {database_path}")
    table = "cards" if game == "mtg" else "pokemon_cards"
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            f"SELECT image_path FROM {table} WHERE image_path IS NOT NULL"
        ).fetchall()
    return [Path(row[0]) for row in rows if Path(row[0]).is_file()]


def camera_augment(image: Image.Image) -> Image.Image:
    """Simulate common phone-capture degradation while retaining card identity."""
    image = image.copy()
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.35))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        image = image.rotate(random.uniform(-8, 8), resample=Image.Resampling.BICUBIC)
    if random.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.2)))

    pixels = np.asarray(image)
    height, width = pixels.shape[:2]
    source = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    displacement = min(width, height) * 0.08
    destination = source + np.random.uniform(
        -displacement, displacement, source.shape
    ).astype(np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    pixels = cv2.warpPerspective(
        pixels, transform, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    return Image.fromarray(pixels)


class GameBalancedBatchSampler:
    """Yield batches with an equal, fixed number of cards from each game."""

    def __init__(
        self,
        torch,
        indices_by_game: dict[int, list[int]],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.torch = torch
        self.indices_by_game = indices_by_game
        self.samples_per_game = batch_size // len(indices_by_game)
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.batch_count = min(
            len(indices) // self.samples_per_game
            for indices in indices_by_game.values()
        )
        if self.batch_count == 0:
            raise ValueError("Not enough cards to create one balanced batch")

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self):
        game_indices = {}
        for game_id, indices in self.indices_by_game.items():
            values = self.torch.tensor(indices, dtype=self.torch.int64)
            if self.shuffle:
                values = values[
                    self.torch.randperm(len(values), generator=self.generator)
                ]
            game_indices[game_id] = values

        for batch_index in range(self.batch_count):
            start = batch_index * self.samples_per_game
            batch = self.torch.cat(
                [
                    indices[start : start + self.samples_per_game]
                    for indices in game_indices.values()
                ]
            )
            if self.shuffle:
                batch = batch[
                    self.torch.randperm(len(batch), generator=self.generator)
                ]
            yield batch.tolist()


class CardDataset:
    """Produce two independently augmented views for each cached card image."""

    def __init__(self, records: list[tuple[Path, int]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, Image.Image, int]:
        image_path, game_id = self.records[index]
        with Image.open(image_path) as image:
            source = image.convert("RGB")
        return camera_augment(source), camera_augment(source), game_id


def collate_card_views(processor, torch, batch):
    """Convert paired PIL images into model-ready pixel tensors."""
    first_images, second_images, game_ids = zip(*batch, strict=True)
    return (
        processor(images=list(first_images), return_tensors="pt")["pixel_values"],
        processor(images=list(second_images), return_tensors="pt")["pixel_values"],
        torch.tensor(game_ids, dtype=torch.int64),
    )


def split_game_paths(
    image_paths_by_game: dict[str, list[Path]],
    validation_fraction: float,
    seed: int,
    minimum_per_split: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Create reproducible, per-game train/validation splits."""
    training_records = []
    validation_records = []
    for game_id, game in enumerate(("mtg", "pokemon")):
        if game not in image_paths_by_game:
            continue
        paths = image_paths_by_game[game].copy()
        random.Random(f"{seed}:{game}").shuffle(paths)
        if len(paths) < minimum_per_split * 2:
            raise RuntimeError(
                f"{game} needs at least {minimum_per_split * 2} cached images "
                "to make non-empty training and validation splits"
            )
        validation_count = max(
            minimum_per_split, round(len(paths) * validation_fraction)
        )
        validation_count = min(validation_count, len(paths) - minimum_per_split)
        validation_records.extend(
            (path, game_id) for path in paths[:validation_count]
        )
        training_records.extend(
            (path, game_id) for path in paths[validation_count:]
        )
    return training_records, validation_records


def batch_game_metrics(
    torch,
    first_vectors,
    second_vectors,
    game_ids,
    game_names_by_id: dict[int, str],
    temperature: float,
) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    """Compute same-game losses and retrieval Recall@1/@5."""
    losses = {}
    metrics = {}
    for game_id, game_name in game_names_by_id.items():
        selected = torch.nonzero(game_ids == game_id, as_tuple=True)[0]
        first = first_vectors[selected]
        second = second_vectors[selected]
        scores = first @ second.T / temperature
        labels = torch.arange(len(selected), device=scores.device)
        loss = (
            torch.nn.functional.cross_entropy(scores, labels)
            + torch.nn.functional.cross_entropy(scores.T, labels)
        ) / 2
        top_k = min(5, len(selected))
        forward_recall_at_5 = (
            scores.topk(top_k, dim=1)
            .indices.eq(labels[:, None])
            .any(dim=1)
            .float()
            .mean()
        )
        reverse_recall_at_5 = (
            scores.T.topk(top_k, dim=1)
            .indices.eq(labels[:, None])
            .any(dim=1)
            .float()
            .mean()
        )
        recall_at_1 = (
            scores.argmax(dim=1).eq(labels).float().mean()
            + scores.T.argmax(dim=1).eq(labels).float().mean()
        ) / 2
        recall_at_5 = (forward_recall_at_5 + reverse_recall_at_5) / 2
        losses[game_name] = loss
        metrics[game_name] = {
            "loss": float(loss.detach().cpu()),
            "recall_at_1": float(recall_at_1.detach().cpu()),
            "recall_at_5": float(recall_at_5.detach().cpu()),
        }
    return losses, metrics


def run_epoch(
    torch,
    model,
    loader,
    optimizer,
    device: str,
    game_names_by_id: dict[int, str],
    temperature: float,
    *,
    epoch: int,
    epochs: int,
    training: bool,
) -> dict[str, dict[str, float]]:
    """Run one train or validation epoch and return per-game mean metrics."""
    model.train(training)
    totals = {
        game: {"loss": 0.0, "recall_at_1": 0.0, "recall_at_5": 0.0}
        for game in game_names_by_id.values()
    }
    progress = tqdm(
        loader,
        desc=f"{'Train' if training else 'Val'} {epoch}/{epochs}",
        unit="batch",
    )
    for batch_index, (first_pixels, second_pixels, game_ids) in enumerate(
        progress, start=1
    ):
        grad_context = nullcontext() if training else torch.inference_mode()
        with grad_context:
            first_outputs = model(pixel_values=first_pixels.to(device))
            second_outputs = model(pixel_values=second_pixels.to(device))
            first_vectors = torch.nn.functional.normalize(
                first_outputs.last_hidden_state[:, 0], dim=1
            )
            second_vectors = torch.nn.functional.normalize(
                second_outputs.last_hidden_state[:, 0], dim=1
            )
            losses, metrics = batch_game_metrics(
                torch,
                first_vectors,
                second_vectors,
                game_ids.to(device),
                game_names_by_id,
                temperature,
            )
            loss = torch.stack(list(losses.values())).mean()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for game, game_metrics in metrics.items():
            for name, value in game_metrics.items():
                totals[game][name] += value
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            **{
                f"{game}_r1": f"{values['recall_at_1']:.3f}"
                for game, values in metrics.items()
            },
        )

    if not len(loader):
        raise RuntimeError("Training or validation loader produced no batches")
    return {
        game: {metric: value / len(loader) for metric, value in values.items()}
        for game, values in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DINOv3 with paired MTG and/or Pokemon card views."
    )
    parser.add_argument("--game", choices=("mtg", "pokemon", "both"), default="mtg")
    parser.add_argument(
        "--mtg-data-dir",
        "--data-dir",
        dest="mtg_data_dir",
        type=Path,
        default=DEFAULT_MTG_DATA_DIR,
        help="MTG data directory containing catalog.sqlite.",
    )
    parser.add_argument(
        "--pokemon-data-dir",
        type=Path,
        default=DEFAULT_POKEMON_DATA_DIR,
        help="Pokemon data directory containing catalog.sqlite.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/embedder_training/dinov3-card-small"),
    )
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Per-game fraction of reference images held out for validation.",
    )
    args = parser.parse_args()
    if (
        args.epochs <= 0
        or args.batch_size < 2
        or args.learning_rate <= 0
        or args.temperature <= 0
        or not 0.0 < args.validation_fraction < 0.5
    ):
        parser.error(
            "--epochs, --learning-rate, and --temperature must be positive; "
            "--batch-size must be at least 2; --validation-fraction must be "
            "between 0 and 0.5"
        )
    if args.game == "both" and (args.batch_size < 4 or args.batch_size % 2):
        parser.error("Joint training requires an even --batch-size of at least 4")

    (
        torch,
        DataLoader,
        AutoImageProcessor,
        AutoModel,
    ) = import_runtime()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch, args.device)
    image_paths_by_game = {}
    if args.game in ("mtg", "both"):
        image_paths_by_game["mtg"] = load_image_paths(
            args.mtg_data_dir / "catalog.sqlite", "mtg"
        )
    if args.game in ("pokemon", "both"):
        image_paths_by_game["pokemon"] = load_image_paths(
            args.pokemon_data_dir / "catalog.sqlite", "pokemon"
        )
    if args.game == "both" and any(
        not paths for paths in image_paths_by_game.values()
    ):
        raise RuntimeError(
            "Joint training needs cached reference images for both MTG and Pokemon"
        )
    minimum_per_split = (
        args.batch_size // 2 if args.game == "both" else args.batch_size
    )
    training_records, validation_records = split_game_paths(
        image_paths_by_game,
        args.validation_fraction,
        args.seed,
        minimum_per_split,
    )
    game_id_by_name = {"mtg": 0, "pokemon": 1}
    game_names_by_id = {
        game_id_by_name[game]: game for game in image_paths_by_game
    }
    training_dataset = CardDataset(training_records)
    validation_dataset = CardDataset(validation_records)
    processor = AutoImageProcessor.from_pretrained(args.model)
    if args.game == "both":
        training_indices = {
            game_id: [
                index
                for index, (_, record_game_id) in enumerate(training_records)
                if record_game_id == game_id
            ]
            for game_id in game_names_by_id
        }
        validation_indices = {
            game_id: [
                index
                for index, (_, record_game_id) in enumerate(validation_records)
                if record_game_id == game_id
            ]
            for game_id in game_names_by_id
        }
        training_batches = GameBalancedBatchSampler(
            torch,
            training_indices,
            args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
        validation_batches = GameBalancedBatchSampler(
            torch,
            validation_indices,
            args.batch_size,
            shuffle=False,
            seed=args.seed,
        )
        training_loader = DataLoader(
            training_dataset,
            batch_sampler=training_batches,
            num_workers=args.workers,
            collate_fn=partial(collate_card_views, processor, torch),
            pin_memory=device == "cuda",
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_sampler=validation_batches,
            num_workers=args.workers,
            collate_fn=partial(collate_card_views, processor, torch),
            pin_memory=device == "cuda",
        )
    else:
        training_loader = DataLoader(
            training_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            collate_fn=partial(collate_card_views, processor, torch),
            pin_memory=device == "cuda",
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
            collate_fn=partial(collate_card_views, processor, torch),
            pin_memory=device == "cuda",
        )

    model = AutoModel.from_pretrained(args.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.05
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.csv"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metric_columns = [
        f"{phase}_{game}_{metric}"
        for phase in ("train", "val")
        for game in ("mtg", "pokemon")
        for metric in ("loss", "recall_at_1", "recall_at_5")
    ]
    fieldnames = ["run_id", "epoch", "train_loss", "val_loss", *metric_columns]
    include_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    if include_header:
        with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            csv.DictWriter(metrics_file, fieldnames=fieldnames).writeheader()

    print(
        "Training/validation images: "
        + ", ".join(
            f"{game}="
            f"{sum(record_game_id == game_id_by_name[game] for _, record_game_id in training_records)}"
            f"/{sum(record_game_id == game_id_by_name[game] for _, record_game_id in validation_records)}"
            for game in image_paths_by_game
        )
    )
    print(f"Per-epoch metrics: {metrics_path}")
    for epoch in range(1, args.epochs + 1):
        training_metrics = run_epoch(
            torch,
            model,
            training_loader,
            optimizer,
            device,
            game_names_by_id,
            args.temperature,
            epoch=epoch,
            epochs=args.epochs,
            training=True,
        )
        random.seed(args.seed + 100_000 + epoch)
        np.random.seed(args.seed + 100_000 + epoch)
        validation_metrics = run_epoch(
            torch,
            model,
            validation_loader,
            optimizer,
            device,
            game_names_by_id,
            args.temperature,
            epoch=epoch,
            epochs=args.epochs,
            training=False,
        )
        train_loss = float(
            np.mean([values["loss"] for values in training_metrics.values()])
        )
        validation_loss = float(
            np.mean([values["loss"] for values in validation_metrics.values()])
        )
        row = {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation_loss,
        }
        for phase, game_metrics in (
            ("train", training_metrics),
            ("val", validation_metrics),
        ):
            for game, values in game_metrics.items():
                for metric, value in values.items():
                    row[f"{phase}_{game}_{metric}"] = value
        with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            csv.DictWriter(metrics_file, fieldnames=fieldnames).writerow(row)
        print(
            f"Epoch {epoch}/{args.epochs}: train loss={train_loss:.4f}, "
            f"val loss={validation_loss:.4f}"
        )
        for game in game_names_by_id.values():
            train = training_metrics[game]
            validation = validation_metrics[game]
            print(
                f"  {game}: train loss={train['loss']:.4f}, "
                f"R@1={train['recall_at_1']:.3f}, R@5={train['recall_at_5']:.3f}; "
                f"val loss={validation['loss']:.4f}, "
                f"R@1={validation['recall_at_1']:.3f}, "
                f"R@5={validation['recall_at_5']:.3f}"
            )
        checkpoint_dir = args.output_dir / f"checkpoint-{epoch:03d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
                "train_metrics": training_metrics,
                "validation_metrics": validation_metrics,
                "args": vars(args),
            },
            checkpoint_dir / "training_state.pt",
        )

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
