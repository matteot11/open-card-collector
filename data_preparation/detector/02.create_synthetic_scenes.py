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

GAME_CLASS_IDS = {"mtg": 0, "pokemon": 1}
BACKGROUND_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_templates(
    images_dir: Path, game: str, class_id: int | None = None
) -> list[dict[str, Any]]:
    """Load local card templates, preferably original PNGs with transparency."""
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Template images directory does not exist: {images_dir}"
        )
    if game not in GAME_CLASS_IDS:
        raise ValueError(f"Unsupported template game: {game}")
    templates = [
        {
            "card_id": path.stem,
            "filename": path.name,
            "template_path": str(path),
            "game": game,
            "class_id": GAME_CLASS_IDS[game] if class_id is None else class_id,
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
        for path in backgrounds_dir.rglob("*")
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
    """Vary illumination, camera resolution, focus, noise, and JPEG compression."""
    height, width = image.shape[:2]
    horizontal = np.linspace(-1, 1, width, dtype=np.float32)[None, :, None]
    vertical = np.linspace(-1, 1, height, dtype=np.float32)[:, None, None]
    illumination = (
        randomizer.uniform(0.65, 1.25)
        + horizontal * randomizer.uniform(-0.20, 0.20)
        + vertical * randomizer.uniform(-0.20, 0.20)
    )
    balance = np.array([randomizer.uniform(0.90, 1.10) for _ in range(3)])
    image = np.clip(image.astype(np.float32) * illumination * balance, 0, 255).astype(
        np.uint8
    )
    if randomizer.random() < 0.35:
        resolution = randomizer.uniform(0.45, 0.80)
        reduced = cv2.resize(
            image,
            (max(1, round(width * resolution)), max(1, round(height * resolution))),
        )
        image = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    blur_sigma = randomizer.uniform(0.0, 1.2) * max(width, height) / 640
    if blur_sigma > 0.1:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)

    noise_generator = np.random.default_rng(randomizer.randrange(2**32))
    if randomizer.random() < 0.15:
        kernel_size = randomizer.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if randomizer.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
        else:
            kernel[:, kernel_size // 2] = 1.0 / kernel_size
        image = cv2.filter2D(image, -1, kernel)
    noise = noise_generator.normal(0, randomizer.uniform(0, 6), image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    quality = randomizer.randint(65, 100)
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

    perspective_jitter = card_width * 0.02
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
            + randomizer.uniform(-region_width * 0.02, region_width * 0.02),
            top
            + region_height / 2
            + randomizer.uniform(-region_height * 0.02, region_height * 0.02),
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


def sample_scene_card_height(
    regions: list[tuple[float, float, float, float]],
    full_rotation_probability: float,
    upright_rotation_degrees: float,
    randomizer: random.Random,
    scale_range: tuple[float, float] = (0.90, 0.98),
) -> float:
    """Sample one scale that fits every region at any permitted rotation."""
    aspect_ratio = 2.5 / 3.5
    limit = math.radians(
        90.0 if full_rotation_probability > 0 else min(upright_rotation_degrees, 90.0)
    )
    width_angle = min(limit, math.atan(1.0 / aspect_ratio))
    height_angle = min(limit, math.atan(aspect_ratio))
    jitter_extent = 2.0 * aspect_ratio * 0.02
    width_extent = (
        aspect_ratio * math.cos(width_angle) + math.sin(width_angle) + jitter_extent
    )
    height_extent = (
        math.cos(height_angle) + aspect_ratio * math.sin(height_angle) + jitter_extent
    )
    maximum = min(
        min(width / width_extent, height / height_extent)
        for _, _, width, height in regions
    )
    return maximum * randomizer.uniform(*scale_range)


def preserves_card_visibility(quads: list[np.ndarray]) -> bool:
    """Conservatively retain at least 60% of each card after all later overlays."""
    for index, quad in enumerate(quads):
        area = cv2.contourArea(quad)
        if area <= 0:
            return False
        covered_area = sum(
            cv2.intersectConvexConvex(quad, later)[0] for later in quads[index + 1 :]
        )
        if covered_area > area * 0.40:
            return False
    return True


def project_scene_quads(
    quads: list[np.ndarray],
    canvas_width: int,
    canvas_height: int,
    randomizer: random.Random,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Apply a shared oblique camera view and keep every card fully in frame."""
    source = np.array(
        [[0, 0], [canvas_width, 0], [canvas_width, canvas_height], [0, canvas_height]],
        dtype=np.float32,
    )
    destination = source.copy()
    strength = randomizer.uniform(0.0, 0.18)
    destination += np.array(
        [
            [
                randomizer.uniform(-1, 1) * canvas_width * strength,
                randomizer.uniform(-1, 1) * canvas_height * strength,
            ]
            for _ in range(4)
        ],
        dtype=np.float32,
    )
    perspective = cv2.getPerspectiveTransform(source, destination)
    rotation = np.eye(3)
    rotation[:2] = cv2.getRotationMatrix2D(
        (canvas_width / 2, canvas_height / 2), randomizer.uniform(-35, 35), 1.0
    )
    transform = rotation @ perspective
    projected = [cv2.perspectiveTransform(quad[None], transform)[0] for quad in quads]
    points = np.concatenate(projected)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    margin = min(canvas_width, canvas_height) * 0.02
    scale = min(
        1.0,
        (canvas_width - 2 * margin) / (upper[0] - lower[0]),
        (canvas_height - 2 * margin) / (upper[1] - lower[1]),
    )
    center = (lower + upper) / 2
    fit = np.array(
        [
            [scale, 0, canvas_width / 2 - scale * center[0]],
            [0, scale, canvas_height / 2 - scale * center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    transform = fit @ transform
    return [
        cv2.perspectiveTransform(quad[None], transform)[0] for quad in quads
    ], transform


def composite_card(
    canvas: np.ndarray,
    template_path: Path,
    destination_quad: np.ndarray,
    randomizer: random.Random | None = None,
    sleeved: bool = False,
    binder: bool = False,
) -> None:
    """Warp a template scan into the canvas at an exact known quadrilateral."""
    template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if template is None:
        raise ValueError(f"Could not load template: {template_path}")

    height, width = template.shape[:2]
    if template.ndim == 2:
        template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    alpha = (
        template[..., 3:4].astype(np.float32) / 255.0
        if template.shape[2] == 4
        else np.ones((height, width, 1), dtype=np.float32)
    )
    colors = template[..., :3].astype(np.float32)
    if randomizer is not None:
        colors *= randomizer.uniform(0.8, 1.15)
        if randomizer.random() < (0.65 if sleeved or binder else 0.12):
            horizontal = np.linspace(0, 1, width, dtype=np.float32)[None, :]
            vertical = np.linspace(0, 1, height, dtype=np.float32)[:, None]
            band = np.exp(
                -(
                    (
                        (
                            horizontal
                            + vertical * randomizer.uniform(-0.7, 0.7)
                            - randomizer.uniform(0.1, 0.9)
                        )
                        / randomizer.uniform(0.04, 0.15)
                    )
                    ** 2
                )
            )
            glare = band[..., None] * randomizer.uniform(0.15, 0.50)
            colors = colors * (1 - glare) + 255 * glare
        overlay = canvas.copy()
        shadow = destination_quad + np.array([3, 5], dtype=np.float32)
        cv2.fillConvexPoly(overlay, shadow.astype(np.int32), (15, 15, 15))
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, dst=canvas)
        if sleeved or binder:
            center = destination_quad.mean(axis=0)
            border = (destination_quad - center) * (1.06 if binder else 1.035) + center
            color = randomizer.choice([(40, 40, 40), (180, 180, 180), (110, 80, 45)])
            thickness = max(
                1,
                round(
                    np.linalg.norm(destination_quad[1] - destination_quad[0]) * 0.012
                ),
            )
            cv2.polylines(
                canvas, [border.astype(np.int32)], True, color, thickness, cv2.LINE_AA
            )
    source_quad = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_quad, destination_quad)
    canvas_height, canvas_width = canvas.shape[:2]
    warped_colors = cv2.warpPerspective(
        np.clip(colors, 0, 255) * alpha, transform, (canvas_width, canvas_height)
    )
    warped_alpha = cv2.warpPerspective(alpha, transform, (canvas_width, canvas_height))[
        ..., None
    ]
    canvas[:] = np.clip(
        warped_colors + canvas.astype(np.float32) * (1 - warped_alpha), 0, 255
    ).astype(np.uint8)


def yolo_obb_line(
    quad: np.ndarray, canvas_width: int, canvas_height: int, class_id: int
) -> str:
    """Serialize [top-left, top-right, bottom-right, bottom-left] as a YOLO-OBB label."""
    normalized = np.clip(quad / np.array([canvas_width, canvas_height]), 0.0, 1.0)
    values = " ".join(
        f"{coordinate:.6f}" for point in normalized for coordinate in point
    )
    return f"{class_id} {values}"


def placement_regions(
    canvas_width: int,
    canvas_height: int,
    cards_per_scene: int,
    randomizer: random.Random,
    layout: str = "binder",
) -> list[tuple[float, float, float, float]]:
    """Center a compact portrait-cell grid resembling adjacent binder pockets."""
    cell_aspect_ratio = 0.85
    columns = max(
        range(1, cards_per_scene + 1),
        key=lambda candidate: min(
            canvas_width / (candidate * cell_aspect_ratio),
            canvas_height / math.ceil(cards_per_scene / candidate),
        ),
    )
    rows = math.ceil(cards_per_scene / columns)
    cell_height = min(
        canvas_width / (columns * cell_aspect_ratio), canvas_height / rows
    )
    cell_width = cell_height * cell_aspect_ratio
    if layout == "loose":
        cell_width = canvas_width / columns
        cell_height = canvas_height / rows
    grid_left = (canvas_width - columns * cell_width) / 2
    grid_top = (canvas_height - rows * cell_height) / 2
    inset = max(2.0, min(canvas_width, canvas_height) * 0.003)
    regions = []
    for index in range(cards_per_scene):
        row, column = divmod(index, columns)
        regions.append(
            (
                grid_left + column * cell_width + inset,
                grid_top + row * cell_height + inset,
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
    future_quads: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Try an overlap without excessively covering any current or future card."""
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
    candidate = quad + shift
    if preserves_card_visibility(existing_quads + [candidate] + (future_quads or [])):
        return candidate
    return quad


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
    layout: str = "mixed",
    camera_view_probability: float = 0.75,
    sleeve_probability: float = 0.5,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    """Compose one scene and return its image plus source/projection provenance."""
    canvas, background_source = make_background(
        background_paths, canvas_width, canvas_height, randomizer
    )
    if cards_per_scene == 0:
        return apply_camera_degradation(canvas, randomizer), [], background_source
    layout = (
        randomizer.choice(["binder", "loose", "loose"]) if layout == "mixed" else layout
    )
    rotation_probability = (
        1.0 if randomizer.random() < full_rotation_probability else 0.0
    )
    local_rotation_limit = upright_rotation_degrees
    if layout == "binder":
        rotation_probability = 0.0
        local_rotation_limit = min(5.0, upright_rotation_degrees)
    templates_by_class: dict[int, list[dict[str, Any]]] = {}
    for template in templates:
        templates_by_class.setdefault(template["class_id"], []).append(template)
    class_ids = sorted(templates_by_class)
    if len(class_ids) == 1:
        selected_templates = randomizer.sample(templates, k=cards_per_scene)
    else:
        class_counts = dict.fromkeys(class_ids, 0)
        class_sequence = []
        for _ in range(cards_per_scene):
            least_used = min(class_counts.values())
            eligible_classes = [
                class_id
                for class_id, count in class_counts.items()
                if count == least_used
            ]
            selected_class = randomizer.choice(eligible_classes)
            class_sequence.append(selected_class)
            class_counts[selected_class] += 1
        selected_templates = [
            randomizer.choice(templates_by_class[class_id])
            for class_id in class_sequence
        ]
    scene_cards: list[dict[str, Any]] = []
    regions = placement_regions(
        canvas_width, canvas_height, cards_per_scene, randomizer, layout=layout
    )
    card_height = sample_scene_card_height(
        regions,
        rotation_probability,
        local_rotation_limit,
        randomizer,
        scale_range=(0.80, 0.98) if layout == "binder" else (0.45, 0.95),
    )
    angles = [
        sample_card_rotation(rotation_probability, local_rotation_limit, randomizer)
        for _ in regions
    ]
    planned_quads = [
        card_quad_in_region(region, card_height, angle, randomizer)
        for region, angle in zip(regions, angles, strict=True)
    ]
    camera_transform = np.eye(3)
    if randomizer.random() < camera_view_probability:
        planned_quads, camera_transform = project_scene_quads(
            planned_quads, canvas_width, canvas_height, randomizer
        )
    placed_quads: list[np.ndarray] = []

    for card_index, (template, angle_degrees, quad) in enumerate(
        zip(selected_templates, angles, planned_quads, strict=True)
    ):
        if layout != "binder" and randomizer.random() < occlusion_probability:
            quad = introduce_occlusion(
                quad,
                placed_quads,
                canvas_width,
                canvas_height,
                randomizer,
                future_quads=planned_quads[card_index + 1 :],
            )

        occluded_card_indices = intersecting_card_indices(quad, placed_quads)

        sleeved = randomizer.random() < sleeve_probability
        composite_card(
            canvas,
            Path(template["template_path"]),
            quad,
            randomizer,
            sleeved=sleeved,
            binder=layout == "binder",
        )
        placed_quads.append(quad)
        for index in occluded_card_indices:
            scene_cards[index]["occluded_by_card_indices"].append(len(scene_cards))
        scene_cards.append(
            {
                "template_card_id": template["card_id"],
                "template_scryfall_id": (
                    template["card_id"] if template["game"] == "mtg" else None
                ),
                "template_filename": template["filename"],
                "game": template["game"],
                "class_id": template["class_id"],
                "rotation_degrees": round(angle_degrees, 3),
                "layout": layout,
                "sleeved": sleeved,
                "camera_transform": camera_transform.round(6).tolist(),
                "card_height_pixels": round(card_height, 3),
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
    layout: str = "mixed",
    camera_view_probability: float = 0.75,
    sleeve_probability: float = 0.5,
    empty_scene_probability: float = 0.05,
    game: str = "mtg",
    pokemon_images_dir: Path | None = None,
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
    if layout not in {"mixed", "binder", "loose"}:
        raise ValueError("layout must be mixed, binder, or loose")
    for probability in (
        camera_view_probability,
        sleeve_probability,
        empty_scene_probability,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Scene augmentation probabilities must be between zero and one"
            )
    if (
        image_count <= 0
        or not canvas_dimensions
        or any(width <= 0 or height <= 0 for width, height in canvas_dimensions)
    ):
        raise ValueError("Image count and canvas dimensions must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"Output directory must be empty; choose a fresh path: {output_dir}"
        )

    if game not in {"mtg", "pokemon", "both"}:
        raise ValueError("game must be mtg, pokemon, or both")
    templates = (
        load_templates(images_dir, "pokemon", class_id=0)
        if game == "pokemon"
        else load_templates(images_dir, "mtg")
    )
    if game == "both":
        if pokemon_images_dir is None:
            raise ValueError("pokemon_images_dir is required when game is both")
        templates.extend(load_templates(pokemon_images_dir, "pokemon"))
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
            if randomizer.random() < empty_scene_probability:
                cards_per_scene = 0
            elif min_cards <= 4 and randomizer.random() < 0.35:
                cards_per_scene = randomizer.randint(min_cards, min(4, max_cards))
            else:
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
                layout=layout,
                camera_view_probability=camera_view_probability,
                sleeve_probability=sleeve_probability,
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
                        yolo_obb_line(
                            quad,
                            canvas_width,
                            canvas_height,
                            card["class_id"],
                        )
                        + "\n"
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
                        "generation_recipe": "camera_domain_v3",
                        "layout": scene_cards[0]["layout"] if scene_cards else "empty",
                        "camera_view_probability": camera_view_probability,
                        "sleeve_probability": sleeve_probability,
                        "empty_scene_probability": empty_scene_probability,
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
        description="Create synthetic MTG, Pokemon, or mixed-game YOLO-OBB scenes."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/detector_training_data/scryfall_png"),
        help="Primary game's card templates; alpha preserves rounded corners when available.",
    )
    parser.add_argument(
        "--game",
        choices=("mtg", "pokemon", "both"),
        default="mtg",
        help="Template game; both assigns balanced two-class labels.",
    )
    parser.add_argument(
        "--pokemon-images-dir",
        type=Path,
        default=Path("data/pokemon_source/images"),
        help="Pokemon image templates, used when --game both.",
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
        default=Path("data/detector_training_data/synthetic/mtg_mobile_v3"),
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
    parser.add_argument("--max-cards", type=int, default=20)
    parser.add_argument(
        "--layout", choices=["mixed", "binder", "loose"], default="mixed"
    )
    parser.add_argument("--camera-view-probability", type=float, default=0.75)
    parser.add_argument("--sleeve-probability", type=float, default=0.5)
    parser.add_argument(
        "--empty-scene-probability",
        type=float,
        default=0.05,
        help="Fraction of card-free background scenes for false-positive training.",
    )
    parser.add_argument(
        "--occlusion-probability",
        type=float,
        default=0.35,
        help="Probability of attempting an overlap; every card retains at least 60%% visibility.",
    )
    parser.add_argument(
        "--full-rotation-probability",
        type=float,
        default=0.1,
        help="Probability of a full-orientation scene; otherwise near-upright local rotations.",
    )
    parser.add_argument(
        "--upright-rotation-degrees",
        type=float,
        default=25.0,
        help="Maximum local near-upright rotation, before the shared camera transform.",
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
        layout=args.layout,
        camera_view_probability=args.camera_view_probability,
        sleeve_probability=args.sleeve_probability,
        empty_scene_probability=args.empty_scene_probability,
        game=args.game,
        pokemon_images_dir=args.pokemon_images_dir,
    )


if __name__ == "__main__":
    main()
