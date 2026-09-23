"""Create a small, auditable YOLO-OBB dataset from local MTG card templates.

It uses local background photos when available, otherwise generates a textured
fallback. It varies card scale, rotation, perspective, and light camera degradation
while preserving exact quadrilateral labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.progress import track

CARD_CLASS_ID = 0
BACKGROUND_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_templates(images_dir: Path) -> list[dict[str, str]]:
    """Load all cached Scryfall scans directly from the embedder image cache."""
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Embedder images directory does not exist: {images_dir}"
        )
    templates = [
        {
            "scryfall_id": path.stem,
            "filename": path.name,
            "template_path": str(path),
        }
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in BACKGROUND_EXTENSIONS
    ]
    if not templates:
        raise RuntimeError(f"No readable template images found in {images_dir}")
    return templates


def load_background_paths(backgrounds_dir: Path) -> list[Path]:
    """Find locally supplied background photos without treating them as labels."""
    if not backgrounds_dir.is_dir():
        return []
    return sorted(
        path
        for path in backgrounds_dir.iterdir()
        if path.is_file() and path.suffix.lower() in BACKGROUND_EXTENSIONS
    )


def procedural_background(
    canvas_width: int, canvas_height: int, randomizer: random.Random
) -> np.ndarray:
    """Create a subdued textured fallback while no local photos are available."""
    base_color = np.array(
        [
            randomizer.randint(45, 160),
            randomizer.randint(45, 160),
            randomizer.randint(45, 160),
        ],
        dtype=np.float32,
    )
    generator = np.random.default_rng(randomizer.randrange(2**32))
    texture = generator.normal(
        0, randomizer.uniform(8, 20), (canvas_height, canvas_width)
    )
    texture = cv2.GaussianBlur(texture, (0, 0), randomizer.uniform(5, 18))
    return np.clip(base_color + texture[..., np.newaxis], 0, 255).astype(np.uint8)


def photo_background(
    path: Path, canvas_width: int, canvas_height: int, randomizer: random.Random
) -> np.ndarray:
    """Crop a matching aspect ratio from a local photo and resize to the canvas."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load background photo: {path}")

    height, width = image.shape[:2]
    target_ratio = canvas_width / canvas_height
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_height = height
        crop_width = round(crop_height * target_ratio)
    else:
        crop_width = width
        crop_height = round(crop_width / target_ratio)
    top = randomizer.randint(0, height - crop_height)
    left = randomizer.randint(0, width - crop_width)
    crop = image[top : top + crop_height, left : left + crop_width]
    return cv2.resize(crop, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)


def make_background(
    background_paths: list[Path],
    canvas_width: int,
    canvas_height: int,
    randomizer: random.Random,
) -> tuple[np.ndarray, str]:
    """Choose a user-supplied photo or create a deterministic procedural fallback."""
    if background_paths:
        background_path = randomizer.choice(background_paths)
        return (
            photo_background(background_path, canvas_width, canvas_height, randomizer),
            background_path.name,
        )
    return procedural_background(canvas_width, canvas_height, randomizer), "procedural"


def apply_camera_degradation(
    image: np.ndarray, randomizer: random.Random
) -> np.ndarray:
    """Apply modest blur, sensor noise, and JPEG compression to the finished scene."""
    blur_sigma = randomizer.uniform(0.0, 0.7)
    if blur_sigma > 0.1:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)

    noise_generator = np.random.default_rng(randomizer.randrange(2**32))
    noise = noise_generator.normal(0, randomizer.uniform(0, 3), image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    quality = randomizer.randint(88, 100)
    encoded, compressed = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not encoded:
        raise OSError("Could not apply JPEG compression")
    return cv2.imdecode(compressed, cv2.IMREAD_COLOR)


def card_quad_in_region(
    region: tuple[float, float, float, float],
    card_height: float,
    angle_degrees: float,
    randomizer: random.Random,
) -> np.ndarray:
    """Generate a rotated card quadrilateral contained by one placement region."""
    left, top, region_width, region_height = region
    card_width = card_height * 2.5 / 3.5
    base_quad = np.array(
        [
            [-card_width / 2, -card_height / 2],
            [card_width / 2, -card_height / 2],
            [card_width / 2, card_height / 2],
            [-card_width / 2, card_height / 2],
        ],
        dtype=np.float32,
    )

    radians = math.radians(angle_degrees)
    rotation = np.array(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
        dtype=np.float32,
    )
    quad = base_quad @ rotation.T

    perspective_jitter = card_width * 0.08
    quad += np.array(
        [
            [
                randomizer.uniform(-perspective_jitter, perspective_jitter),
                randomizer.uniform(-perspective_jitter, perspective_jitter),
            ],
            [
                randomizer.uniform(-perspective_jitter, perspective_jitter),
                randomizer.uniform(-perspective_jitter, perspective_jitter),
            ],
            [
                randomizer.uniform(-perspective_jitter, perspective_jitter),
                randomizer.uniform(-perspective_jitter, perspective_jitter),
            ],
            [
                randomizer.uniform(-perspective_jitter, perspective_jitter),
                randomizer.uniform(-perspective_jitter, perspective_jitter),
            ],
        ],
        dtype=np.float32,
    )

    center = np.array(
        [
            left
            + region_width / 2
            + randomizer.uniform(-region_width * 0.08, region_width * 0.08),
            top
            + region_height / 2
            + randomizer.uniform(-region_height * 0.08, region_height * 0.08),
        ],
        dtype=np.float32,
    )
    quad += center

    min_x, min_y = quad.min(axis=0)
    max_x, max_y = quad.max(axis=0)
    shift_x = min(max(left - min_x, 0), left + region_width - max_x)
    shift_y = min(max(top - min_y, 0), top + region_height - max_y)
    return quad + np.array([shift_x, shift_y], dtype=np.float32)


def sample_card_rotation(
    full_rotation_probability: float,
    upright_rotation_degrees: float,
    randomizer: random.Random,
) -> float:
    """Sample either any card orientation or a smaller near-upright rotation."""
    if randomizer.random() < full_rotation_probability:
        return randomizer.uniform(-180.0, 180.0)
    return randomizer.uniform(-upright_rotation_degrees, upright_rotation_degrees)


def maximum_card_height(
    region_width: float, region_height: float, angle_degrees: float
) -> float:
    """Return the largest rotated portrait card height that fits inside a region."""
    aspect_ratio = 2.5 / 3.5
    radians = math.radians(angle_degrees)
    sine = abs(math.sin(radians))
    cosine = abs(math.cos(radians))
    rotated_width_per_height = aspect_ratio * cosine + sine
    rotated_height_per_height = cosine + aspect_ratio * sine
    return 0.88 * min(
        region_width / rotated_width_per_height,
        region_height / rotated_height_per_height,
    )


def composite_card(
    canvas: np.ndarray, template_path: Path, destination_quad: np.ndarray
) -> None:
    """Warp a template scan into the canvas at an exact known quadrilateral."""
    template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if template is None:
        raise ValueError(f"Could not load template: {template_path}")

    height, width = template.shape[:2]
    source_quad = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_quad, destination_quad)
    canvas_height, canvas_width = canvas.shape[:2]
    warped = cv2.warpPerspective(template, transform, (canvas_width, canvas_height))
    if template.ndim == 3 and template.shape[2] == 4:
        alpha = warped[..., 3:4].astype(np.float32) / 255.0
        canvas[:] = (
            warped[..., :3].astype(np.float32) * alpha
            + canvas.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        return

    mask = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        transform,
        (canvas_width, canvas_height),
    )
    cv2.copyTo(warped, mask, canvas)


def yolo_obb_line(quad: np.ndarray, canvas_width: int, canvas_height: int) -> str:
    """Serialize [top-left, top-right, bottom-right, bottom-left] as a YOLO-OBB label."""
    normalized = np.clip(quad / np.array([canvas_width, canvas_height]), 0.0, 1.0)
    values = " ".join(
        f"{coordinate:.6f}" for point in normalized for coordinate in point
    )
    return f"{CARD_CLASS_ID} {values}"


def placement_regions(
    canvas_width: int,
    canvas_height: int,
    cards_per_scene: int,
    randomizer: random.Random,
) -> list[tuple[float, float, float, float]]:
    """Allocate distinct grid regions so dense scenes always have room to render."""
    columns = math.ceil(math.sqrt(cards_per_scene))
    rows = math.ceil(cards_per_scene / columns)
    cell_width = canvas_width / columns
    cell_height = canvas_height / rows
    inset = max(2.0, min(canvas_width, canvas_height) * 0.003)
    regions = []
    for index in range(cards_per_scene):
        row, column = divmod(index, columns)
        regions.append(
            (
                column * cell_width + inset,
                row * cell_height + inset,
                cell_width - 2 * inset,
                cell_height - 2 * inset,
            )
        )
    randomizer.shuffle(regions)
    return regions


def introduce_occlusion(
    quad: np.ndarray,
    existing_quads: list[np.ndarray],
    canvas_width: int,
    canvas_height: int,
    randomizer: random.Random,
) -> np.ndarray:
    """Move a card toward its nearest predecessor while keeping it in frame."""
    if not existing_quads:
        return quad

    center = quad.mean(axis=0)
    reference = min(
        existing_quads,
        key=lambda existing: np.linalg.norm(existing.mean(axis=0) - center),
    )
    shift = (reference.mean(axis=0) - center) * randomizer.uniform(0.42, 0.62)
    min_x, min_y = quad.min(axis=0)
    max_x, max_y = quad.max(axis=0)
    shift[0] = np.clip(shift[0], -min_x, canvas_width - 1 - max_x)
    shift[1] = np.clip(shift[1], -min_y, canvas_height - 1 - max_y)
    return quad + shift


def intersecting_card_indices(
    quad: np.ndarray, existing_quads: list[np.ndarray]
) -> list[int]:
    """Return previous card indices whose full quadrilaterals are covered in part."""
    candidate = quad.reshape((-1, 1, 2)).astype(np.float32)
    indices = []
    for index, existing_quad in enumerate(existing_quads):
        existing = existing_quad.reshape((-1, 1, 2)).astype(np.float32)
        overlap_area, _ = cv2.intersectConvexConvex(candidate, existing)
        if overlap_area > 1.0:
            indices.append(index)
    return indices


def create_scene(
    templates: list[dict[str, Any]],
    background_paths: list[Path],
    canvas_width: int,
    canvas_height: int,
    cards_per_scene: int,
    occlusion_probability: float,
    full_rotation_probability: float,
    upright_rotation_degrees: float,
    randomizer: random.Random,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    """Compose one scene and return its image plus source/projection provenance."""
    canvas, background_source = make_background(
        background_paths, canvas_width, canvas_height, randomizer
    )
    selected_templates = randomizer.sample(templates, k=cards_per_scene)
    scene_cards: list[dict[str, Any]] = []
    regions = placement_regions(
        canvas_width, canvas_height, cards_per_scene, randomizer
    )
    placed_quads: list[np.ndarray] = []

    for template, region in zip(selected_templates, regions, strict=True):
        _, _, region_width, region_height = region
        close_up = cards_per_scene == 1
        angle_degrees = sample_card_rotation(
            full_rotation_probability, upright_rotation_degrees, randomizer
        )
        max_height = maximum_card_height(region_width, region_height, angle_degrees)
        min_height = max_height * (0.84 if close_up else 0.78)
        quad = card_quad_in_region(
            region=region,
            card_height=randomizer.uniform(min_height, max_height),
            angle_degrees=angle_degrees,
            randomizer=randomizer,
        )
        if randomizer.random() < occlusion_probability:
            quad = introduce_occlusion(
                quad, placed_quads, canvas_width, canvas_height, randomizer
            )

        occluded_card_indices = intersecting_card_indices(quad, placed_quads)

        composite_card(canvas, Path(template["template_path"]), quad)
        placed_quads.append(quad)
        for index in occluded_card_indices:
            scene_cards[index]["occluded_by_card_indices"].append(len(scene_cards))
        scene_cards.append(
            {
                "template_scryfall_id": template["scryfall_id"],
                "template_filename": template["filename"],
                "rotation_degrees": round(angle_degrees, 3),
                "quadrilateral_pixels": quad.round(3).tolist(),
                "occludes_card_indices": occluded_card_indices,
                "occluded_by_card_indices": [],
            }
        )

    return apply_camera_degradation(canvas, randomizer), scene_cards, background_source


def create_dataset(
    images_dir: Path,
    backgrounds_dir: Path,
    output_dir: Path,
    image_count: int,
    canvas_dimensions: list[tuple[int, int]],
    min_cards: int,
    max_cards: int,
    occlusion_probability: float,
    full_rotation_probability: float,
    upright_rotation_degrees: float,
    seed: int,
) -> None:
    """Create image, label, and scene-provenance files for a synthetic OBB dataset."""
    if min_cards <= 0 or max_cards < min_cards:
        raise ValueError("Card count must satisfy 0 < min_cards <= max_cards")
    if not 0.0 <= occlusion_probability <= 1.0:
        raise ValueError("occlusion_probability must be between zero and one")
    if not 0.0 <= full_rotation_probability <= 1.0:
        raise ValueError("full_rotation_probability must be between zero and one")
    if not 0.0 <= upright_rotation_degrees <= 180.0:
        raise ValueError("upright_rotation_degrees must be between zero and 180")

    templates = load_templates(images_dir)
    if max_cards > len(templates):
        raise ValueError("max_cards cannot exceed the number of available templates")

    background_paths = load_background_paths(backgrounds_dir)

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "scenes.jsonl"
    randomizer = random.Random(seed)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in track(range(image_count), description="Generating scenes"):
            cards_per_scene = randomizer.randint(min_cards, max_cards)
            canvas_width, canvas_height = randomizer.choice(canvas_dimensions)
            image, scene_cards, background_source = create_scene(
                templates,
                background_paths,
                canvas_width,
                canvas_height,
                cards_per_scene,
                occlusion_probability,
                full_rotation_probability,
                upright_rotation_degrees,
                randomizer,
            )
            stem = f"scene_{index:05d}"
            image_path = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"

            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"Could not write scene image: {image_path}")

            with label_path.open("w", encoding="utf-8") as labels:
                for card in scene_cards:
                    quad = np.array(card["quadrilateral_pixels"], dtype=np.float32)
                    labels.write(
                        yolo_obb_line(quad, canvas_width, canvas_height) + "\n"
                    )

            manifest.write(
                json.dumps(
                    {
                        "scene": stem,
                        "seed": seed,
                        "canvas_width": canvas_width,
                        "canvas_height": canvas_height,
                        "aspect_ratio": round(canvas_width / canvas_height, 6),
                        "background_source": background_source,
                        "occlusion_probability": occlusion_probability,
                        "full_rotation_probability": full_rotation_probability,
                        "cards": scene_cards,
                    }
                )
                + "\n"
            )

    print(f"Created {image_count} synthetic scenes in {output_dir}")
    print(f"Images: {images_dir}")
    print(f"YOLO-OBB labels: {labels_dir}")
    print(f"Scene provenance: {manifest_path}")


def parse_canvas_sizes(value: str) -> list[int]:
    """Parse a comma-separated set of positive square scene dimensions."""
    try:
        sizes = [int(size) for size in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Canvas sizes must be comma-separated integers"
        ) from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("Canvas sizes must be greater than zero")
    return sizes


def parse_canvas_dimensions(value: str) -> list[tuple[int, int]]:
    """Parse comma-separated positive WIDTHxHEIGHT scene dimensions."""
    dimensions = []
    try:
        for item in value.split(","):
            width_text, height_text = item.lower().split("x", maxsplit=1)
            width, height = int(width_text), int(height_text)
            if width <= 0 or height <= 0:
                raise ValueError
            dimensions.append((width, height))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Canvas dimensions must be comma-separated WIDTHxHEIGHT values"
        ) from error
    if not dimensions:
        raise argparse.ArgumentTypeError("At least one canvas dimension is required")
    return dimensions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a geometric synthetic MTG YOLO-OBB dataset."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/skryfall_source/images"),
        help="Cached Scryfall scans used as card templates.",
    )
    parser.add_argument(
        "--backgrounds-dir",
        type=Path,
        default=Path("data/detector_training_data/backgrounds"),
        help="Directory of local background photos; procedural textures are used when empty.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/detector_training_data/synthetic/mtg_mobile_v2"),
    )
    parser.add_argument(
        "--count", type=int, default=10, help="Number of scenes to create."
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        help="Use one square output dimension for every scene.",
    )
    parser.add_argument(
        "--canvas-sizes",
        type=parse_canvas_sizes,
        default=parse_canvas_sizes("1024,1280,1536,1920"),
        help="Comma-separated square output dimensions chosen per scene.",
    )
    parser.add_argument(
        "--canvas-dimensions",
        type=parse_canvas_dimensions,
        help="Comma-separated WIDTHxHEIGHT output dimensions chosen per scene.",
    )
    parser.add_argument("--min-cards", type=int, default=1)
    parser.add_argument("--max-cards", type=int, default=24)
    parser.add_argument(
        "--occlusion-probability",
        type=float,
        default=0.35,
        help="Probability that each card after the first partially covers an earlier card.",
    )
    parser.add_argument(
        "--full-rotation-probability",
        type=float,
        default=0.7,
        help="Probability that a card uses a full [-180, 180] degree rotation.",
    )
    parser.add_argument(
        "--upright-rotation-degrees",
        type=float,
        default=25.0,
        help="Maximum absolute rotation for the remaining near-upright cards.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.count <= 0 or (args.canvas_size is not None and args.canvas_size <= 0):
        parser.error("--count and --canvas-size must be greater than zero")

    create_dataset(
        images_dir=args.images_dir,
        backgrounds_dir=args.backgrounds_dir,
        output_dir=args.output_dir,
        image_count=args.count,
        canvas_dimensions=(
            [(args.canvas_size, args.canvas_size)]
            if args.canvas_size
            else (
                args.canvas_dimensions
                if args.canvas_dimensions
                else [(size, size) for size in args.canvas_sizes]
            )
        ),
        min_cards=args.min_cards,
        max_cards=args.max_cards,
        occlusion_probability=args.occlusion_probability,
        full_rotation_probability=args.full_rotation_probability,
        upright_rotation_degrees=args.upright_rotation_degrees,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
