# Detector Training Workflow

Run all commands from the repository root with `uv run`.

The generated scans, backgrounds, datasets, checkpoints, and training runs are local artifacts. Review generated backgrounds before using them for training.

## Scripts in This Folder

- `train_yolo_obb.py` trains the oriented YOLO card detector from a generated dataset, creates deterministic train/validation split files, and writes the training run under `runs/obb`.

The data-preparation scripts used before training live under `data_preparation/detector`; they generate backgrounds and synthetic card scenes.

## 1. Prepare Card Images

Synthetic scenes use original high-resolution PNG card images from [Scryfall](https://scryfall.com/), including their supplied transparency and rounded corners. These templates are separate from the JPEG reference cache used by the embedder. First synchronize the local catalog:

```bash
uv run python data_preparation/00.sync_scryfall_catalog.py
```

Download a smaller initial batch of original PNG templates:

```bash
uv run python data_preparation/detector/00.download_scryfall_png.py \
  --limit 2000
```

The downloader reads `image_uris.png` from the catalog and saves the original bytes under `data/detector_training_data/scryfall_png`. It validates downloads, preserves alpha without resizing or converting images, and skips valid cached PNGs on subsequent runs. `--limit` caps attempted missing or invalid downloads per run; omit it to download all eligible templates. Records without a top-level PNG URL, including separately imaged multi-face cards, are skipped.

To use another location, pass `--output-dir` to the downloader and the matching `--images-dir` to the scene generator. The embedder's JPEG cache is unchanged.

## 2. Generate Backgrounds

Install the optional FLUX.2 klein 4B runtime once:

```bash
uv sync --extra background-generation
```

Generate a varied set of card-free backgrounds. The first execution downloads the model weights.

```bash
uv run python data_preparation/detector/01.generate_diffusion_backgrounds.py \
  --output-dir data/detector_training_data/backgrounds/generated_v1 \
  --count 100 \
  --dimensions 720x1280,1088x1920,1024x1024,1280x720,1920x1088 \
  --seed 42
```

Generate a small review batch before a large run:

```bash
uv run python data_preparation/detector/01.generate_diffusion_backgrounds.py \
  --output-dir data/detector_training_data/backgrounds/review \
  --count 8 \
  --dimensions 720x1280,1280x720,1024x1024 \
  --seed 42
```

Custom background prompts use one nonempty prompt per line:

```bash
uv run python data_preparation/detector/01.generate_diffusion_backgrounds.py \
  --output-dir data/detector_training_data/backgrounds/custom \
  --count 24 \
  --prompts-file prompts.txt \
  --dimensions 720x1280,1280x720,1024x1024
```

On Apple Silicon, `--device auto` selects MPS. Specify a device explicitly when needed:

```bash
uv run python data_preparation/detector/01.generate_diffusion_backgrounds.py \
  --output-dir data/detector_training_data/backgrounds/generated_v3 \
  --count 24 \
  --device mps
```

## 3. Generate Synthetic Scenes

The v3 recipe mixes aligned binder-pocket grids (one third) and looser tabletop layouts (two thirds), with 1-20 cards and additional weighting toward 1-4 cards. All cards share one nominal physical scale before camera projection. Zoom varies between frames, including smaller-card examples. The shared camera view rotates by up to +/-35 degrees and perturbs the frame corners by up to 18%; projected cards are fitted back inside the frame, so this recipe does not label truncated cards. Binder cards remain locally aligned within +/-5 degrees; loose cards use +/-25 degrees, with occasional full-orientation scenes.

Half of cards receive simplified sleeve borders, binder scenes receive pocket seams, and some cards receive directional highlights. Contact shadows, exposure gradients, white-balance changes, reduced resolution, defocus/motion blur, sensor noise, and JPEG compression vary across scenes. Five percent of scenes contain no cards, providing background negatives. These are approximations, not a physical renderer of plastic, foil, hands, or glare.

Control the mixture with `--layout mixed|binder|loose`, `--camera-view-probability`, `--sleeve-probability`, and `--empty-scene-probability`. `--full-rotation-probability` now controls the fraction of loose scenes with unrestricted local orientations, rather than shrinking all near-upright scenes to accommodate that possibility. Binder scenes do not add card-on-card overlap; loose overlap attempts retain at least 60% visibility.

Background discovery includes subdirectories. Use reviewed, card-free photos of real desks, playmats, and empty binders, not just generated backgrounds. Visible cards in a background would become unlabeled positives. Output directories must be empty to avoid mixing old and new recipes.

Generate a square baseline dataset:

```bash
uv run python data_preparation/detector/02.create_synthetic_scenes.py \
  --images-dir data/detector_training_data/scryfall_png \
  --backgrounds-dir data/detector_training_data/backgrounds/generated_v3 \
  --output-dir data/detector_training_data/synthetic/mtg_upright_v1 \
  --count 1700 \
  --canvas-sizes 1024,1280,1536,1920 \
  --max-cards 4 \
  --occlusion-probability 0.35 \
  --seed 42
```

Generate the mixed v3 smartphone dataset. Repeat portrait dimensions to weight the generated scenes toward portrait capture. Override `--images-dir` if your existing PNG cache uses a different location.

```bash
uv run python data_preparation/detector/02.create_synthetic_scenes.py \
  --images-dir data/detector_training_data/scryfall_png \
  --backgrounds-dir data/detector_training_data/backgrounds/custom \
  --output-dir data/detector_training_data/synthetic/mtg_mobile_v3 \
  --count 8000 \
  --canvas-dimensions 720x1280,1088x1920,1280x720,1920x1088,1088x1088 \
  --max-cards 20 \
  --layout mixed \
  --occlusion-probability 0.35 \
  --seed 42
```

Generate a separate dense-card stress dataset rather than making dense scenes the default PoC training distribution:

```bash
uv run python data_preparation/detector/02.create_synthetic_scenes.py \
  --images-dir data/detector_training_data/scryfall_png \
  --backgrounds-dir data/detector_training_data/backgrounds/generated_v3 \
  --output-dir data/detector_training_data/synthetic/mtg_dense \
  --count 500 \
  --canvas-dimensions 720x1280,1280x720,1024x1024 \
  --min-cards 12 \
  --max-cards 40 \
  --occlusion-probability 0.55 \
  --seed 43
```

Each dataset contains `images/`, `labels/`, and `scenes.jsonl`. OBB labels use `class x1 y1 x2 y2 x3 y3 x4 y4`, with coordinates normalized to each image's actual width and height.

Scene provenance records the shared nominal card height alongside each card's rotation and quadrilateral. A requested occlusion probability controls overlap attempts, not the final fraction of overlapping cards.

## 4. Train YOLO OBB

Install the optional Ultralytics runtime once:

```bash
uv sync --extra detector-training
```

Start a lightweight model training run on Apple Silicon:

```bash
uv run python detector_training/train_yolo_obb.py \
  --dataset-dir data/detector_training_data/synthetic/mtg_mobile_v3 \
  --epochs 60 \
  --imgsz 896 \
  --batch 8 \
  --device mps \
  --output-dir runs/obb \
  --name mtg_mobile_v3_yolo11n_obb
```

CUDA run with automatic batch-size selection:

```bash
uv run python detector_training/train_yolo_obb.py \
  --dataset-dir data/detector_training_data/synthetic/mtg_mobile_v3 \
  --epochs 60 \
  --imgsz 896 \
  --batch -1 \
  --device 0 \
  --output-dir runs/obb \
  --name mtg_mobile_v3_yolo11n_obb
```

Resume an interrupted run by explicitly selecting its checkpoint:

```bash
uv run python detector_training/train_yolo_obb.py \
  --resume runs/obb/mtg_mobile_v3_yolo11n_obb/weights/last.pt \
  --device mps
```

Resume reuses the checkpoint's original settings, dataset YAML, and existing split lists; it does not regenerate splits. Keep the original dataset unchanged. Completed checkpoints without optimizer state are rejected. To start new fine-tuning from a completed checkpoint, use `--model PATH/TO/best.pt` without `--resume` instead. Replace the example path with your actual run directory.

The trainer defaults to 60 epochs, `imgsz=640`, and rectangular, aspect-ratio-grouped batches to reduce padding without stretching cards. Mixed aspect ratios can still leave padding. Whole-frame scale varies from 75% to 125%, translation is limited to 8%, rotation varies by +/-15 degrees, and perspective is set to 0.0005. These perturbations transform labels together with images. Color augmentation remains enabled. Mirroring, mosaic, mixup, CutMix, and copy-paste remain disabled to avoid mirrored text or unnatural mixtures of independently scaled scenes. The commands above use `--imgsz 896` for more detail; start with batch 8 on MPS and lower it if memory is insufficient.

Synthetic validation is a pipeline check, not evidence of real-camera accuracy. Compare v1 and v3 on the same held-out, labeled full camera frames: loose cards, binders, sleeves, glare, oblique views, small cards, and card-free scenes. Report recall and false positives at the same inference size and confidence threshold. Start a new run for this recipe rather than resuming an old checkpoint's training configuration.

The trainer writes an 85/15 deterministic scene-level split to `splits/train.txt` and `splits/val.txt`, then creates `dataset.yaml`. The main output checkpoint is:

```text
runs/obb/mtg_mobile_v3_yolo11n_obb/weights/best.pt
```

The published detector checkpoint is available as the [collector-mtg-detector-yolo11n-obb model](https://huggingface.co/matteot11/collector-mtg-detector-yolo11n-obb) on Hugging Face.

## Script Help

Every script exposes its available options through `--help`:

```bash
uv run python data_preparation/detector/01.generate_diffusion_backgrounds.py --help
uv run python data_preparation/detector/02.create_synthetic_scenes.py --help
uv run python detector_training/train_yolo_obb.py --help
```
