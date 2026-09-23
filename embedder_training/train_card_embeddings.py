"""Fine-tune DINOv3 Small for card-image retrieval with synthetic camera views."""

from __future__ import annotations

import argparse
import random
import sqlite3
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

DEFAULT_DATA_DIR = Path("data/skryfall_source")
DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def import_runtime():
    """Load optional training dependencies only when training starts."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise SystemExit(
            "Missing embedder dependencies. Install them with:\n"
            "uv sync --extra embedder-training"
        ) from error
    return torch, Dataset, DataLoader, AutoImageProcessor, AutoModel


def select_device(torch, requested_device: str) -> str:
    """Use the requested device or select the best available accelerator."""
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image_paths(database_path: Path) -> list[Path]:
    """Load every valid cached catalog image path from SQLite."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Catalog database does not exist: {database_path}")
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT image_path FROM cards WHERE image_path IS NOT NULL"
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


def contrastive_loss(torch, first_vectors, second_vectors, temperature: float):
    """Return symmetric InfoNCE loss for paired image views in one batch."""
    logits = first_vectors @ second_vectors.T / temperature
    labels = torch.arange(len(first_vectors), device=first_vectors.device)
    return (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.T, labels)
    ) / 2


class CardDataset:
    """Produce two independently augmented views for each cached card image."""

    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, Image.Image]:
        with Image.open(self.image_paths[index]) as image:
            source = image.convert("RGB")
        return camera_augment(source), camera_augment(source)


def collate_card_views(processor, batch):
    """Convert paired PIL images into model-ready pixel tensors."""
    first_images, second_images = zip(*batch, strict=True)
    return (
        processor(images=list(first_images), return_tensors="pt")["pixel_values"],
        processor(images=list(second_images), return_tensors="pt")["pixel_values"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DINOv3 Small with paired synthetic card captures."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
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
    args = parser.parse_args()
    if (
        args.epochs <= 0
        or args.batch_size < 2
        or args.learning_rate <= 0
        or args.temperature <= 0
    ):
        parser.error(
            "--epochs, --learning-rate, and --temperature must be positive; --batch-size must be at least 2"
        )

    torch, Dataset, DataLoader, AutoImageProcessor, AutoModel = import_runtime()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch, args.device)
    image_paths = load_image_paths(args.data_dir / "catalog.sqlite")
    if len(image_paths) < args.batch_size:
        raise RuntimeError(
            "Not enough cached catalog images for the requested batch size"
        )

    processor = AutoImageProcessor.from_pretrained(args.model)

    loader = DataLoader(
        CardDataset(image_paths),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=partial(collate_card_views, processor),
        pin_memory=device == "cuda",
    )
    model = AutoModel.from_pretrained(args.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.05
    )

    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_total = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch_index, (first_pixels, second_pixels) in enumerate(progress, start=1):
            first_outputs = model(pixel_values=first_pixels.to(device))
            second_outputs = model(pixel_values=second_pixels.to(device))
            first_vectors = torch.nn.functional.normalize(
                first_outputs.last_hidden_state[:, 0], dim=1
            )
            second_vectors = torch.nn.functional.normalize(
                second_outputs.last_hidden_state[:, 0], dim=1
            )
            loss = contrastive_loss(
                torch, first_vectors, second_vectors, args.temperature
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            progress.set_postfix(loss=f"{loss_total / batch_index:.4f}")
        print(f"Epoch {epoch}/{args.epochs}: loss {loss_total / len(loader):.4f}")
        checkpoint_dir = args.output_dir / f"checkpoint-{epoch:03d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_total / len(loader),
                "args": vars(args),
            },
            checkpoint_dir / "training_state.pt",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
