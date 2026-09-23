"""Compute DINOv3 cosine similarity between two card images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MODEL = "matteot11/collector-mtg-embedder-dinov3-small"


def import_runtime():
    """Load optional embedder dependencies only when comparison runs."""
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


def embed_images(
    images: list[Image.Image], processor, model, torch, device: str
) -> np.ndarray:
    """Return L2-normalized image embeddings."""
    inputs = processor(images=images, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        vectors = outputs.last_hidden_state[:, 0]
        vectors = torch.nn.functional.normalize(vectors, dim=1)
    return vectors.cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two card images with DINOv3 cosine similarity."
    )
    parser.add_argument("first_image", type=Path)
    parser.add_argument("second_image", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or cuda")
    args = parser.parse_args()

    for image_path in (args.first_image, args.second_image):
        if not image_path.is_file():
            parser.error(f"Image does not exist: {image_path}")

    torch, AutoModel, AutoProcessor = import_runtime()
    device = select_device(torch, args.device)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    with Image.open(args.first_image) as image:
        first_image = image.convert("RGB")
    with Image.open(args.second_image) as image:
        second_image = image.convert("RGB")

    first_vectors = embed_images(
        [first_image, first_image.rotate(180)], processor, model, torch, device
    )
    second_vector = embed_images([second_image], processor, model, torch, device)[0]
    upright_score = float(first_vectors[0] @ second_vector)
    rotated_score = float(first_vectors[1] @ second_vector)

    print(f"First image:  {args.first_image}")
    print(f"Second image: {args.second_image}")
    print(f"Upright cosine similarity: {upright_score:.6f}")
    print(f"180-degree cosine similarity: {rotated_score:.6f}")
    print(f"Best cosine similarity: {max(upright_score, rotated_score):.6f}")


if __name__ == "__main__":
    main()
