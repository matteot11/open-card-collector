# Detector Training Workflow

Run all commands from the repository root with `uv run`.

The generated scans, backgrounds, datasets, checkpoints, and training runs are local artifacts. Review generated backgrounds before using them for training.

## Scripts in This Folder

- `train_yolo_obb.py` trains the oriented YOLO card detector from a generated
  dataset, creates deterministic train/validation split files, and writes the
  training run under `runs/obb`.

The data-preparation scripts used before training live under
`data_preparation/detector`; they generate backgrounds and synthetic card scenes.

## 1. Prepare Card Images

Synthetic scenes use locally cached card images downloaded from [Scryfall](https://scryfall.com/)
under
`data/skryfall_source/images`. Synchronize the catalog and download those images
before generating scenes:

```bash
uv run python data_preparation/00.sync_scryfall_catalog.py --download-images
```

The complete image cache is large. To prepare a smaller local dataset, add an
image limit:

```bash
uv run python data_preparation/00.sync_scryfall_catalog.py \
  --download-images \
  --image-limit 10000
```

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

Generate the existing square baseline dataset:

```bash
uv run python data_preparation/detector/02.create_synthetic_scenes.py \
  --images-dir data/skryfall_source/images \
  --backgrounds-dir data/detector_training_data/backgrounds/generated_v3 \
  --output-dir data/detector_training_data/synthetic/mtg_v1 \
  --count 1700 \
  --canvas-sizes 1024,1280,1536,1920 \
  --max-cards 8 \
  --occlusion-probability 0.35 \
  --seed 42
```

Generate a smartphone-oriented dataset. Repeating portrait dimensions deliberately weights the generated scenes toward portrait capture.

```bash
uv run python data_preparation/detector/02.create_synthetic_scenes.py \
  --images-dir data/skryfall_source/images \
  --backgrounds-dir data/detector_training_data/backgrounds/generated_v3 \
  --output-dir data/detector_training_data/synthetic/mtg_mobile_v2 \
  --count 1700 \
  --canvas-dimensions 720x1280,1088x1920,1280x720,1920x1088,1088x1088 \
  --max-cards 8 \
  --occlusion-probability 0.35 \
  --seed 42
```

Generate a dense-card stress dataset:

```bash
uv run python data_preparation/detector/02.create_synthetic_scenes.py \
  --images-dir data/skryfall_source/images \
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

## 4. Train YOLO OBB

Install the optional Ultralytics runtime once:

```bash
uv sync --extra detector-training
```

Start a lightweight model training run on Apple Silicon:

```bash
uv run python detector_training/train_yolo_obb.py \
  --dataset-dir data/detector_training_data/synthetic/mtg_mobile_v2 \
  --epochs 60 \
  --imgsz 640 \
  --batch 16 \
  --device mps \
  --output-dir runs/obb \
  --name mtg_mobile_v2_yolo11n_obb
```

CUDA run with automatic batch-size selection:

```bash
uv run python detector_training/train_yolo_obb.py \
  --dataset-dir data/detector_training_data/synthetic/mtg_mobile_v2 \
  --epochs 60 \
  --imgsz 640 \
  --batch -1 \
  --device 0 \
  --output-dir runs/obb \
  --name mtg_mobile_v2_yolo11n_obb
```

Resume an interrupted run. Keep the same output directory and run name:

```bash
uv run python detector_training/train_yolo_obb.py \
  --dataset-dir data/detector_training_data/synthetic/mtg_mobile_v2 \
  --output-dir runs/obb \
  --name mtg_mobile_v2_yolo11n_obb \
  --resume
```

The trainer writes an 85/15 deterministic scene-level split to `splits/train.txt` and `splits/val.txt`, then creates `dataset.yaml`. The main output checkpoint is:

```text
runs/obb/mtg_mobile_v2_yolo11n_obb/weights/best.pt
```

The published detector checkpoint is available as the [collector-mtg-detector-yolo11n-obb model](https://huggingface.co/matteot11/collector-mtg-detector-yolo11n-obb) on Hugging Face.

## Script Help

Every script exposes its available options through `--help`:

```bash
uv run python data_preparation/detector/01.generate_diffusion_backgrounds.py --help
uv run python data_preparation/detector/02.create_synthetic_scenes.py --help
uv run python detector_training/train_yolo_obb.py --help
```
