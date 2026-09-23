"""Detect cards from a live camera and save perspective-corrected capture crops."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CARD_WIDTH = 1371
CARD_HEIGHT = 1920
DEFAULT_RECOGNITION_DATA_DIR = Path("data/skryfall_source")
DEFAULT_EMBEDDING_MODEL = "matteot11/collector-mtg-embedder-dinov3-small"
DEFAULT_DETECTOR_MODEL = "matteot11/collector-mtg-detector-yolo11n-obb"


def import_runtime():
    """Load optional inference dependencies only when the preview is started."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Missing optional training dependencies. Install them with:\n"
            "uv sync --extra training\n"
            "Then rerun this command."
        ) from error
    return torch, YOLO


def select_device(torch, requested_device: str) -> str:
    """Use the requested device or select the best available accelerator."""
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_detector_checkpoint(model: str) -> Path:
    """Return a local checkpoint, downloading a Hugging Face model when needed."""
    local_path = Path(model).expanduser()
    if local_path.is_file():
        return local_path
    if "/" not in model:
        raise FileNotFoundError(f"Model checkpoint does not exist: {local_path}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit(
            "Missing Hugging Face Hub support. Install project dependencies with:\n"
            "uv sync"
        ) from error
    return Path(hf_hub_download(repo_id=model, filename="best.pt"))


def order_card_corners(points: np.ndarray) -> np.ndarray:
    """Order corners cyclically with a short card edge first."""
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    cyclic = points[np.argsort(angles)]
    edge_lengths = np.linalg.norm(cyclic - np.roll(cyclic, -1, axis=0), axis=1)
    return np.roll(cyclic, -int(np.argmin(edge_lengths)), axis=0).astype(np.float32)


def crop_card(frame: np.ndarray, quadrilateral: np.ndarray) -> np.ndarray:
    """Warp a detected card to the standard portrait recognition dimensions."""
    source = order_card_corners(quadrilateral.astype(np.float32))
    destination = np.array(
        [
            [0, 0],
            [CARD_WIDTH - 1, 0],
            [CARD_WIDTH - 1, CARD_HEIGHT - 1],
            [0, CARD_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(frame, transform, (CARD_WIDTH, CARD_HEIGHT))


def orient_frame(frame: np.ndarray, orientation: str) -> np.ndarray:
    """Keep native orientation or rotate a frame to the requested aspect direction."""
    if orientation == "auto":
        return frame

    height, width = frame.shape[:2]
    is_portrait = height > width
    if (orientation == "portrait" and not is_portrait) or (
        orientation == "landscape" and is_portrait
    ):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def load_catalog_embeddings(
    database_path: Path, model_name: str
) -> tuple[np.ndarray, list[tuple[str, str, str, str, str | None, str | None]]]:
    """Load normalized image embeddings and their card details into memory."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Recognition catalog does not exist: {database_path}")

    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """
                 SELECT embedding, scryfall_id, name, set_code, collector_number,
                     usd, eur
            FROM image_embeddings
            JOIN cards USING (scryfall_id)
                 LEFT JOIN card_prices USING (scryfall_id)
            WHERE model_name = ?
            """,
            (model_name,),
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise RuntimeError(
            "No matching reference embeddings found. Run "
            "data_preparation/embedder/01.embed_scryfall_images.py first."
        )
    vectors = np.vstack([np.frombuffer(row[0], dtype=np.float32) for row in rows])
    details = [(row[1], row[2], row[3], row[4], row[5], row[6]) for row in rows]
    return vectors, details


def embed_capture(
    image: np.ndarray, processor, embedding_model, torch, device: str
) -> np.ndarray:
    """Embed upright and upside-down versions of a captured BGR crop."""
    upright = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    images = [upright, upright.rotate(180)]
    inputs = processor(images=images, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = embedding_model(**inputs)
        vectors = outputs.last_hidden_state[:, 0]
        vectors = torch.nn.functional.normalize(vectors, dim=1)
    return vectors.cpu().numpy().astype(np.float32)


def retrieve_cards(
    image: np.ndarray,
    catalog_vectors: np.ndarray,
    catalog_details: list[tuple[str, str, str, str, str | None, str | None]],
    processor,
    embedding_model,
    torch,
    device: str,
    top_k: int,
) -> list[tuple[float, bool, tuple[str, str, str, str, str | None, str | None]]]:
    """Return top catalog matches across upright and 180-degree orientations."""
    query_vectors = embed_capture(image, processor, embedding_model, torch, device)
    scores = query_vectors @ catalog_vectors.T
    candidates = []
    for orientation, orientation_scores in enumerate(scores):
        for index in np.argpartition(orientation_scores, -top_k)[-top_k:]:
            candidates.append(
                (
                    float(orientation_scores[index]),
                    orientation == 1,
                    catalog_details[index],
                )
            )
    return sorted(candidates, reverse=True, key=lambda candidate: candidate[0])[:top_k]


def format_prices(usd: str | None, eur: str | None) -> str:
    """Format available Scryfall price estimates for display."""
    prices = []
    if usd is not None:
        prices.append(f"${usd}")
    if eur is not None:
        prices.append(f"EUR {eur}")
    return " | ".join(prices) if prices else "price unavailable"


def draw_capture_metadata(
    frame: np.ndarray,
    polygon: np.ndarray,
    match: tuple[float, bool, tuple[str, str, str, str, str | None, str | None]],
) -> None:
    """Draw a captured card's top retrieval result inside its detection box."""
    _, _, details = match
    _, name, set_code, collector_number, usd, eur = details
    points = np.round(polygon).astype(np.int32)
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    label_lines = [
        name,
        f"{set_code.upper()} #{collector_number}",
        format_prices(usd, eur),
    ]
    line_height = 23
    label_top = max(int(top) + 8, 8)
    label_bottom = min(label_top + line_height * len(label_lines) + 8, int(bottom))
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (int(left) + 4, label_top - 4),
        (int(right) - 4, label_bottom),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    for line_index, label in enumerate(label_lines):
        cv2.putText(
            frame,
            label,
            (int(left) + 10, label_top + 17 + line_index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 230, 70),
            2,
            cv2.LINE_AA,
        )


def draw_predictions(
    frame: np.ndarray, result, confidence_threshold: float
) -> list[tuple[np.ndarray, float]]:
    """Draw one oriented polygon and confidence tag for each predicted card."""
    if result.obb is None:
        return []

    polygons = result.obb.xyxyxyxy.cpu().numpy()
    confidences = result.obb.conf.cpu().numpy()
    class_ids = result.obb.cls.cpu().numpy().astype(int)
    detections: list[tuple[np.ndarray, float]] = []

    for polygon, confidence, class_id in zip(
        polygons, confidences, class_ids, strict=True
    ):
        if confidence < confidence_threshold:
            continue
        points = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=(0, 230, 70), thickness=3)
        anchor_x, anchor_y = points[0, 0]
        label = (
            f"card {confidence:.0%}"
            if class_id == 0
            else f"class {class_id} {confidence:.0%}"
        )
        cv2.putText(
            frame,
            label,
            (int(anchor_x), max(24, int(anchor_y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 230, 70),
            2,
            cv2.LINE_AA,
        )
        detections.append((polygon, float(confidence)))
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect cards from a camera and save recognition-ready crops."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_DETECTOR_MODEL,
        help="Local Ultralytics OBB .pt checkpoint or Hugging Face model repository.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, help="Optional requested camera width.")
    parser.add_argument("--height", type=int, help="Optional requested camera height.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size.")
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument(
        "--device", default="auto", help="auto, mps, cpu, or CUDA index."
    )
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument(
        "--captures-dir",
        type=Path,
        default=Path("data/captures"),
        help="Directory for perspective-corrected card crops saved with c.",
    )
    parser.add_argument(
        "--recognition-data-dir",
        type=Path,
        default=DEFAULT_RECOGNITION_DATA_DIR,
        help="Directory containing catalog.sqlite and reference embeddings.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.imgsz <= 0 or args.max_det <= 0 or args.top_k <= 0:
        parser.error("--imgsz, --max-det, and --top-k must be greater than zero")
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    if args.width is not None and (args.width <= 0 or args.height <= 0):
        parser.error("--width and --height must be greater than zero")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between zero and one")

    torch, YOLO = import_runtime()
    device = select_device(torch, args.device)
    try:
        detector_checkpoint = resolve_detector_checkpoint(args.model)
    except FileNotFoundError as error:
        parser.error(str(error))
    model = YOLO(str(detector_checkpoint))
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as error:
        raise SystemExit(
            "Missing recognition dependencies. Install them with:\n"
            "uv sync --extra recognition"
        ) from error
    catalog_vectors, catalog_details = load_catalog_embeddings(
        args.recognition_data_dir / "catalog.sqlite", args.embedding_model
    )
    if args.top_k > len(catalog_details):
        parser.error(
            f"--top-k cannot exceed {len(catalog_details)} available embeddings"
        )
    processor = AutoProcessor.from_pretrained(args.embedding_model)
    embedding_model = AutoModel.from_pretrained(args.embedding_model).to(device).eval()
    camera = cv2.VideoCapture(args.camera)
    if args.width is not None:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    for _ in range(10):
        success, frame = camera.read()
        if success:
            break
    else:
        raise SystemExit(f"Could not read a frame from camera index {args.camera}")

    window_name = "Open Card Collector - Camera Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(
        f"Camera capture on {device}. Press c to capture and retrieve, q or Escape to quit."
    )
    previous_time = time.perf_counter()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Camera frame read failed; stopping capture.")
                break

            capture_frame = frame.copy()
            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.confidence,
                max_det=args.max_det,
                device=device,
                verbose=False,
            )[0]
            detections = draw_predictions(frame, result, args.confidence)
            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            cv2.putText(
                frame,
                f"{len(detections)} cards | {fps:.1f} FPS | c: capture | q: quit",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                if not detections:
                    print("No card detected to capture.")
                    continue
                args.captures_dir.mkdir(parents=True, exist_ok=True)
                capture_id = time.time_ns()
                saved = 0
                frozen_frame = capture_frame.copy()
                frozen_matches = []
                for index, (polygon, _) in enumerate(detections, start=1):
                    output_path = (
                        args.captures_dir / f"capture_{capture_id}_{index:02d}.jpg"
                    )
                    crop = crop_card(capture_frame, polygon)
                    if cv2.imwrite(str(output_path), crop):
                        saved += 1
                        matches = retrieve_cards(
                            crop,
                            catalog_vectors,
                            catalog_details,
                            processor,
                            embedding_model,
                            torch,
                            device,
                            args.top_k,
                        )
                        if matches:
                            frozen_matches.append((polygon, matches[0]))
                        print(f"\n{output_path}")
                        for rank, (score, rotated, details) in enumerate(
                            matches, start=1
                        ):
                            _, name, set_code, collector_number, usd, eur = details
                            orientation = "rotated" if rotated else "upright"
                            print(
                                f"  {rank}. {name} | {set_code.upper()} #{collector_number} "
                                f"| {format_prices(usd, eur)} | {score:.3f} | {orientation}"
                            )
                    else:
                        print(f"Could not write capture: {output_path}")
                print(
                    f"Captured {saved}/{len(detections)} cards in {args.captures_dir}"
                )
                for polygon, match in frozen_matches:
                    points = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        frozen_frame,
                        [points],
                        isClosed=True,
                        color=(0, 230, 70),
                        thickness=3,
                    )
                    draw_capture_metadata(frozen_frame, polygon, match)
                cv2.putText(
                    frozen_frame,
                    "Captured - c or r: resume | q: quit",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                while True:
                    cv2.imshow(window_name, frozen_frame)
                    frozen_key = cv2.waitKey(1) & 0xFF
                    if frozen_key in (27, ord("q")):
                        return
                    if frozen_key in (ord("c"), ord("r")):
                        break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
