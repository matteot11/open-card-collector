# Open Card Collector

A local prototype for detecting Magic: The Gathering and Pokemon cards with YOLO OBB and retrieving candidate printings with DINOv3.

> [!NOTE]
> Work in progress. Accuracy, performance, and workflows may change. The published detector and shared embedder support MTG and Pokemon, but the Pokemon catalog currently uses English-language TCGdex data only. Collection management, OCR, and multilingual retrieval are not implemented.

## Demo

![Camera pipeline demo](resources/sample_video.gif)

## Start Here

Follow **steps 1–4 to use the camera with the published models**. You do not need to train anything. Steps 5–8 are optional and explain how to build your own models. Run every command from the repository root.

- [1. Install](#1-install)
- [2. Download the catalog and reference images](#2-download-the-catalog-and-reference-images)
- [3. Build reference embeddings](#3-build-reference-embeddings)
- [4. Run the camera](#4-run-the-camera)
- [5. Prepare a detector dataset (optional)](#5-prepare-a-detector-dataset-optional)
- [6. Train the detector (optional)](#6-train-the-detector-optional)
- [7. Train the embedder (optional)](#7-train-the-embedder-optional)
- [8. Use and test your own models (optional)](#8-use-and-test-your-own-models-optional)

## Project Layout

Catalog synchronization is grouped by game, while workflows that apply to both games use shared scripts:

| Area | Purpose |
| ---- | ------- |
| `data_preparation/catalogs/mtg/` | Scryfall catalog synchronization |
| `data_preparation/catalogs/pokemon/` | TCGdex catalog synchronization |
| `data_preparation/detector/` | Game-specific template downloads and shared scene generation |
| `data_preparation/embedder/` | Shared MTG/Pokemon reference embedding builder |
| `detector_training/`, `embedder_training/` | Shared detector and embedder trainers |
| `pipeline/` | Camera inference and per-game catalog routing |

The existing `data/scryfall_source/` directory is retained as the MTG data default so existing catalogs and embeddings remain usable.

## 1. Install

You need Python 3.10 or newer, [uv](https://docs.astral.sh/uv/), and a desktop with a camera.

```bash
uv sync --extra detector-training --extra embedder-training
```

Despite their names, these two extras also install the packages needed to run the published models. The defaults use the [YOLO11n OBB detector](https://huggingface.co/matteot11/collector-mtg-pkm-detector-yolo11n-obb) and [DINOv3 Small embedder](https://huggingface.co/matteot11/collector-mtg-pkm-embedder-dinov3-small); weights download on first use. Neither model includes the searchable catalog, which you prepare next.

Device selection is automatic: NVIDIA CUDA, Apple Silicon MPS, then CPU. Commands below use `--device auto`; substitute `mps`, `cpu`, or `cuda` when needed. YOLO commands also accept a CUDA index such as `0`.

For NVIDIA GPUs, install a compatible CUDA-enabled PyTorch build. For example, if CUDA 12.4 suits your system:

```bash
uv sync --extra detector-training --extra embedder-training --torch-backend=cu124
```

Choose the build using the [PyTorch installation selector](https://pytorch.org/get-started/locally/). Keep GUI-enabled `opencv-python`; do not install `opencv-python-headless` alongside it, because camera windows need GUI support.

## 2. Download the Catalog and Reference Images

The catalog contains card names, printing details, prices, and image URLs from Scryfall. The reference images are the scans against which camera captures will be matched.

```bash
uv run python data_preparation/catalogs/mtg/00.sync_catalog.py --download-images
```

This downloads Scryfall's Default Cards export, imports it into SQLite, and downloads JPEG reference images. The full bulk export, catalog, and reference image cache together use about 20 GB of disk space, growing as Scryfall adds cards. To try a smaller subset, add `--image-limit 10000`: this run will attempt up to 10,000 missing or changed images. Running again skips successful downloads and continues with the remaining images. Leave the limit out to download all remaining images.

| Location                                | Contents                                    |
| --------------------------------------- | ------------------------------------------- |
| `data/scryfall_source/bulk/`          | Bulk archive and export metadata            |
| `data/scryfall_source/catalog.sqlite` | Card metadata, prices, and later embeddings |
| `data/scryfall_source/images/`        | JPEG reference images                       |

Use `--data-dir PATH` to choose another location. Without `--download-images`, the script updates metadata only. Cards without a top-level full-card image URL, including separately imaged multi-face cards, do not get a reference image through this workflow.

### Keeping the Catalog Current

Rerun this command when you want updated cards or prices, then rerun step 3. An unchanged bulk archive is reused. Missing images and changed image URLs are downloaded; unchanged cached images are skipped.

The database keeps two URLs: `image_url` is the latest URL from Scryfall, and `cached_image_url` is the URL of the last successful image download. If a replacement fails, the old image stays on disk and the next run retries. Each successful download is committed immediately, so interrupting the command does not lose completed download records.

### Pokemon Catalog (Optional)

Pokemon data currently comes from TCGdex's English-language endpoint (`/v2/en`); other languages are not yet supported by this workflow. By default, each run fetches every remaining card detail and price into a separate local catalog; use `--limit N` to fetch fewer cards. Use `--refresh` to refresh existing details and prices. `--download-images` caches high-quality JPEG references for the embedder, attempting every remaining image by default; use `--image-limit N` to bound downloads. The sync uses eight concurrent workers by default while pacing requests globally; adjust with `--workers N`. PNG detector templates are downloaded separately in step 5.1.

```bash
uv run python data_preparation/catalogs/pokemon/00.sync_catalog.py --download-images
```

| Location | Contents |
| -------- | -------- |
| `data/pokemon_source/catalog.sqlite` | Pokemon card metadata, local Cardmarket/TCGplayer prices, and later embeddings |
| `data/pokemon_source/images/` | Cached TCGdex JPEG reference images used for embedding and retrieval |

The script downloads available Cardmarket and TCGplayer pricing through TCGdex into the user's local SQLite database. Cardmarket summary values are stored in `pokemon_card_prices`; TCGplayer values are stored separately per finish (such as normal or reverse-holofoil) in `pokemon_tcgplayer_prices`, including currency, product IDs, update timestamps, and low/mid/high/market/direct-low prices. The full card response is retained in `raw_json`, preserving additional Cardmarket fields and variant-specific pricing. Missing prices remain unavailable, and `--refresh` replaces local price snapshots rather than retaining stale variants. Camera retrieval currently displays the Cardmarket summary estimate.

This repository distributes downloader code, not downloaded pricing databases. TCGdex asks bulk API users to cache locally; users must still follow TCGdex and applicable upstream provider access and usage terms. The project's MIT license does not license marketplace data or Pokemon artwork. Publishing local price dumps, cached images, database exports, or derived datasets is a separate action requiring appropriate permission.

## 3. Build Reference Embeddings

An embedding is a numerical representation of an image. This command turns the downloaded reference images into vectors that can be searched quickly:

```bash
uv run python data_preparation/embedder/01.build_reference_embeddings.py --device auto
```

Vectors are stored in `catalog.sqlite`. **Only cards with reference embeddings can be retrieved.** Downloading model weights or catalog metadata alone is not enough. Images are needed to create or refresh embeddings, but not to search vectors already stored in the database.

| Option                         | Meaning                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------- |
| `--data-dir PATH`            | Use the same data directory as step 2                                                       |
| `--batch-size 8`             | Process fewer images at once to reduce memory use; default is 16                            |
| `--limit 1000`               | Process up to 1,000 images that do not have a current embedding; omit for all cached images |
| `--model MODEL_OR_DIRECTORY` | Use a different published embedder or local model; see step 8                               |

Rerunning skips unchanged images already embedded with the same model name. If you used download or embedding limits for a trial, rerun both steps without limits to complete the index.

For Pokemon, build vectors using the same selected model name that you will use for inference:

```bash
uv run python data_preparation/embedder/01.build_reference_embeddings.py \
	--game pokemon \
	--model facebook/dinov3-vits16-pretrain-lvd1689m
```

To test one shared embedder for both games, use the same `--model` value for both catalogs. Each catalog keeps its own model-keyed vector table:

```bash
uv run python data_preparation/embedder/01.build_reference_embeddings.py \
	--game mtg \
	--model matteot11/collector-mtg-pkm-embedder-dinov3-small
uv run python data_preparation/embedder/01.build_reference_embeddings.py \
	--game pokemon \
	--model matteot11/collector-mtg-pkm-embedder-dinov3-small
```

Both games now default to the published shared embedder, so `--model` can be omitted unless using a different checkpoint.

## 4. Run the Camera

```bash
uv run python pipeline/capture_camera.py --device auto
```

The default camera is index 0. Use `--camera 1` to try another camera. The program detects cards continuously; retrieval starts when you capture a frame.

The default published detector and shared embedder support both MTG and Pokemon, and the camera defaults to `--game both`. In a single-game run, detections of the other named class are ignored. Mixed mode routes each detected class to its corresponding catalog and requires detector class names `mtg,pokemon` in either order:

```bash
uv run python pipeline/capture_camera.py
```

Run the catalog sync and build reference embeddings for the selected games first. In mixed mode, YOLO's predicted class routes each crop to the corresponding SQLite catalog; detections use different colors for each class. Use `--model` or `--embedding-model` to override the published defaults.

| Key                          | Action                                                     |
| ---------------------------- | ---------------------------------------------------------- |
| `c` during live preview    | Capture all detected cards and look up candidate printings |
| `c` or `r` during review | Resume the live preview                                    |
| `q` or Escape              | Quit                                                       |

The captured frame pauses with the top match over each card. The terminal lists candidate names, set codes, collector numbers, available prices, similarity scores, and orientation. These are suggestions to inspect, not confirmed identifications. There is no in-app confirmation, correction, or collection saving yet.

### Camera Options

| Option                          | Default / meaning                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| `--captures-dir PATH`         | No files saved by default; supply a directory to save JPEG crops                          |
| `--top-k N`                   | Show 5 distinct candidate printings; cannot exceed the number of indexed cards            |
| `--camera N`                  | Camera index 0                                                                            |
| `--width W --height H`        | Request a camera resolution; supply both                                                  |
| `--imgsz N`                   | Detector input size 640                                                                   |
| `--confidence SCORE`          | Detection threshold 0.75                                                                  |
| `--max-det N`                 | At most 100 detections per frame                                                          |
| `--device DEVICE`             | `auto`, `mps`, `cpu`, or CUDA such as `0` / `cuda:0`                            |
| `--recognition-data-dir PATH` | `data/scryfall_source`; must contain your prepared `catalog.sqlite`                   |
| `--model MODEL`               | Published detector by default; accepts a local OBB`.pt` file or Hugging Face repository |
| `--embedding-model MODEL`     | Published embedder by default; custom model usage is explained in step 8                  |

For example, to save crops as well as retrieve them:

```bash
uv run python pipeline/capture_camera.py --device auto --captures-dir data/captures
```

Camera frames can be portrait or landscape. Requested resolution and phone/Continuity Camera behavior depend on the device.

### What Happens After Capture

YOLO detects oriented rectangles. Their corners are ordered and warped to 1371x1920 portrait crops in memory; this is not a separate estimate of the card's true perspective corners. DINOv3 embeds each crop upright and rotated 180 degrees, then compares its L2-normalized CLS vectors to the catalog using cosine similarity. Each printing keeps its best orientation before selecting the top matches.

Cards are processed one at a time, and review appears after all finish. Optional JPEG saving is separate from retrieval: failed writes do not suppress matches.

**You can stop here if you only want to use the published models.** The remaining steps are for training your own.

## 5. Prepare a Detector Dataset (Optional)

The detector learns card geometry, not card names or printings. Its training images are synthetic scenes built from transparent card scans and backgrounds.

### 5.1. Download PNG Card Templates

You need the catalog from step 2, but JPEG downloads and reference embeddings are not required if you are only training the detector. For metadata alone, run step 2's command without `--download-images`.

```bash
uv run python data_preparation/detector/00.download_mtg_templates.py --limit 2000
```

This tries to download up to 2,000 missing or invalid PNGs. Already downloaded valid PNGs are skipped and do not count toward the limit; failed attempts do count. Run again to continue downloading, or remove `--limit 2000` to try every remaining PNG.

Use original Scryfall PNGs with their alpha channel (RGBA), which preserves transparent rounded corners. Do not substitute the JPEG reference cache from step 2. Opaque images can be loaded, but the compositor treats their entire rectangle as visible.

The downloader reads `image_uris.png`, validates files, and saves original bytes without resizing or conversion under `data/detector_training_data/scryfall_png`. It skips cards without a top-level PNG URL, including separately imaged multi-face cards. Use `--database PATH` for a custom catalog and `--output-dir PATH` for another template directory; pass that directory to the scene generator's `--mtg-images-dir`.

Pokemon detector templates are downloaded separately from the synced catalog, mirroring the MTG workflow:

```bash
uv run python data_preparation/detector/00.download_pokemon_templates.py --limit 1000
```

`--limit 1000` bounds each run to 1,000 missing templates, so rerunning resumes with the next batch and eventually reports `Pokemon PNG templates downloaded: 0`. Omit `--limit` to attempt every remaining template in one run. Files are stored under `data/detector_training_data/pokemon_png`. TCGdex PNG alpha is preserved; for legacy scans whose PNG has an opaque background, this detector downloader adds an antialiased rounded-corner alpha mask while retaining the visible card border. Embedder references remain separate JPEG files in `data/pokemon_source/images/`. For a Pokemon-only dataset, use `--game pokemon --pokemon-imgs-dir data/detector_training_data/pokemon_png`; train it with `--class-names pokemon` and run the camera with `--game pokemon`.

### 5.2. Prepare Backgrounds

Use reviewed photos of card-free desks, playmats, or empty binders. Background discovery includes subdirectories. Cards already visible in a background would become unlabeled training examples.

You can optionally generate backgrounds with FLUX.2 klein 4B:

```bash
uv sync --extra detector-training --extra embedder-training --extra background-generation
uv run python data_preparation/detector/01.generate_diffusion_backgrounds.py \
	--output-dir data/detector_training_data/backgrounds/generated_v3 \
	--count 8 \
	--dimensions 720x1280,1088x1920,1024x1024,1280x720,1920x1088 \
	--seed 42
```

The first run downloads weights. Review these eight images before increasing `--count`. To use your own prompts, add `--prompts-file prompts.txt`, with one prompt per line; blank lines and lines beginning with `#` are ignored. Use the same background directory in the next command. Without local background images, the scene generator uses a procedural textured fallback.

### 5.3. Generate Scenes

Choose a new, empty output directory so different recipes are not mixed:

```bash
uv run python data_preparation/detector/02.create_synthetic_scenes.py \
	--mtg-images-dir data/detector_training_data/scryfall_png \
	--backgrounds-dir data/detector_training_data/backgrounds/generated_v3 \
	--output-dir data/detector_training_data/synthetic/mtg_mobile_v3 \
	--count 8000 \
	--canvas-dimensions 720x1280,1088x1920,1280x720,1920x1088,1088x1088 \
	--max-cards 20 \
	--layout mixed \
	--occlusion-probability 0.35 \
	--seed 42
```

The result contains `images/`, `labels/`, and `scenes.jsonl`. Labels use YOLO OBB rows `class x1 y1 x2 y2 x3 y3 x4 y4`, with coordinates normalized by image width and height. The JSONL file records scene provenance, nominal card height, rotations, and quadrilaterals. Repeat dimensions in `--canvas-dimensions` to sample them more often.

To create a mixed dataset with both detector classes, supply both image directories. By default, scenes are split as evenly as possible among MTG-only, Pokemon-only, and mixed compositions. Mixed scenes balance card classes as evenly as the scene's card count allows; labels use class 0 for MTG and class 1 for Pokemon:

```bash
uv run python data_preparation/detector/02.create_synthetic_scenes.py \
	--game both \
	--mtg-images-dir data/detector_training_data/scryfall_png \
	--pokemon-imgs-dir data/detector_training_data/pokemon_png \
	--output-dir data/detector_training_data/synthetic/mtg_pokemon_v1 \
	--count 8000 \
	--canvas-dimensions 720x1280,1088x1920,1280x720,1920x1088,1088x1088 \
	--max-cards 20 \
	--layout mixed \
	--occlusion-probability 0.35 \
	--seed 42
```

### Scene Recipe and Variations

| Property              | Default v3 behavior / control                                                                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Layout                | One-third binder grids, two-thirds loose tabletop;`--layout mixed\|binder\|loose`                                                                                     |
| Count and scale       | 1–20 cards, weighted toward 1–4;`--min-cards`, `--max-cards`. One physical scale per scene before camera projection; zoom changes between scenes                |
| Camera                | Shared rotation up to +/-35 degrees, frame-corner perturbations up to 18%;`--camera-view-probability`                                                               |
| Card rotation         | Binder +/-5 degrees, loose +/-25 degrees;`--full-rotation-probability` adds unrestricted loose scenes                                                               |
| Visibility            | Cards fit inside the frame, so no truncated-card labels. Binder cards do not overlap; loose overlap retains at least 60% visibility                                   |
| Overlap               | `--occlusion-probability` sets overlap attempts, not the final fraction of overlapping cards                                                                        |
| Plastic and negatives | Half receive simplified sleeve borders; binder pockets add seams.`--sleeve-probability` controls sleeves; `--empty-scene-probability` defaults to 5% empty scenes |
| Image effects         | Highlights, contact shadows, exposure gradients, white balance, low resolution, defocus/motion blur, noise, and JPEG compression                                      |

These effects are approximations, not physically accurate plastic, foil, hands, or glare. Full rotation affects a fraction of loose scenes; it does not shrink every near-upright layout.

For a square baseline, replace `--canvas-dimensions` with `--canvas-sizes 1024,1280,1536,1920`, use `--count 1700 --max-cards 4`, and choose a fresh output directory. For a separate dense stress dataset, use `--count 500 --canvas-dimensions 720x1280,1280x720,1024x1024 --min-cards 12 --max-cards 40 --occlusion-probability 0.55 --seed 43`. Keep stress scenes separate from the main training distribution.

## 6. Train the Detector (Optional)

Use the dataset created in step 5:

```bash
uv run python detector_training/train_yolo_obb.py \
	--dataset-dir data/detector_training_data/synthetic/mtg_mobile_v3 \
	--epochs 60 \
	--imgsz 896 \
	--batch 8 \
	--device auto \
	--output-dir runs/obb \
	--name mtg_mobile_v3_yolo11n_obb
```

The example uses 896px input for extra detail and batch 8 to reduce memory use. Script defaults are 60 epochs, 640px, batch 16, four workers, and seed 42. Lower the batch if memory is insufficient. For CUDA automatic batch sizing, use `--device 0 --batch -1`.

Train a two-class model on the mixed dataset from the previous section with:

```bash
uv run python detector_training/train_yolo_obb.py \
	--dataset-dir data/detector_training_data/synthetic/mtg_pokemon_v1 \
	--class-names mtg,pokemon \
	--epochs 60 \
	--imgsz 896 \
	--batch 8 \
	--device auto \
	--output-dir runs/obb \
	--name mtg_pokemon_v1_yolo11n_obb
```

The trainer creates a deterministic 85/15 scene split in `splits/train.txt` and `splits/val.txt`, plus `dataset.yaml`; `--validation-fraction` changes the ratio. It starts from `yolo11n-obb.pt` unless you supply `--model`.

Rectangular batches group aspect ratios to reduce padding without stretching cards. Augmentation uses scale 75–125%, translation up to 8%, rotation +/-15 degrees, perspective 0.0005, and HSV changes 0.015/0.5/0.35. Mirroring, mosaic, mixup, CutMix, and copy-paste are disabled. Training uses pretrained weights, a deterministic seed, cosine learning rate, and patience 15.

The usual result is `runs/obb/mtg_mobile_v3_yolo11n_obb/weights/best.pt`. Ultralytics can suffix existing run names; use the actual path printed by the trainer.

### Resume an Interrupted Run

```bash
uv run python detector_training/train_yolo_obb.py \
	--resume runs/obb/mtg_mobile_v3_yolo11n_obb/weights/last.pt \
	--device auto
```

Resume requires an interrupted checkpoint with optimizer state and its original dataset, YAML, and split lists. It reuses the saved configuration; it does not create new splits. A completed checkpoint cannot resume; use `--model PATH/TO/best.pt` without `--resume` to start new fine-tuning. Start a new run when changing the training recipe.

Synthetic validation is not proof of real-camera accuracy. Compare models on the same held-out labeled camera frames, with equal input size and confidence threshold. Include loose cards, binders, sleeves, glare, oblique views, small cards, and empty scenes; report recall and false positives.

## 7. Train the Embedder (Optional)

This workflow uses the JPEG references from step 2, not the synthetic detector scenes. Reference embeddings from step 3 are not needed for training. You need at least as many cached images as the requested batch size.

```bash
uv run python embedder_training/train_card_embeddings.py \
	--mtg-data-dir data/scryfall_source \
	--output-dir runs/embedder_training/dinov3-card-small \
	--epochs 3 \
	--batch-size 16 \
	--device auto
```

The base model is `facebook/dinov3-vits16-pretrain-lvd1689m`. If Hugging Face requires access, accept its terms and authenticate in your terminal; never put tokens in scripts. Defaults are three epochs, batch 16, learning rate `1e-5`, temperature `0.07`, zero workers, seed 42, and a 10% validation split. The batch must be at least 2; joint training requires an even batch size of at least 4. Use `--model` to start from another compatible model or directory.

For each card, two independently augmented views form a matching pair. Brightness, contrast, color, rotation, blur, and perspective vary. L2-normalized CLS embeddings are trained with symmetric InfoNCE. In joint training, every batch contains exactly half MTG and half Pokemon cards, and the loss is computed separately per game so cards from the other game are not treated as retrieval negatives. Every epoch reports per-game train/validation loss and augmented-view Recall@1/Recall@5; `training_metrics.csv` in the output directory records these metrics. Validation cards are held out from training per game. These in-batch augmented-view metrics are useful for monitoring regressions, but do not replace evaluation against held-out real camera captures and a realistic full catalog. The optimizer is AdamW with weight decay 0.05.

For a shared model, train on both cached image catalogs with per-batch game balancing:

```bash
uv run python embedder_training/train_card_embeddings.py \
	--game both \
	--mtg-data-dir data/scryfall_source \
	--pokemon-data-dir data/pokemon_source \
	--output-dir runs/embedder_training/dinov3-mtg-pokemon_v2 \
	--epochs 5 \
	--batch-size 32 \
	--device auto
```

The published shared embedder was trained with this configuration from the DINOv3 base checkpoint. Every joint batch contains exactly half MTG and half Pokemon examples. Contrastive loss and augmented-view metrics are calculated separately per game. Each run holds out 10% of each game's images and logs per-game train/validation loss and Recall@1/Recall@5 in `training_metrics.csv`. These pairwise augmented-view metrics are monitoring signals, not full-catalog retrieval accuracy; evaluate with held-out real camera captures.

Use a new output directory for a fresh run so previous weights and logs are preserved. Use that output directory as `--model` when building embeddings for both catalogs and as `--embedding-model` during inference.

The output directory holds the final model and processor. Each epoch also saves `checkpoint-NNN/` and `training_state.pt` containing epoch, optimizer state, loss, and arguments. There is no resume option: supplying a checkpoint through `--model` starts new training without restoring optimizer state.

## 8. Use and Test Your Own Models (Optional)

### Detector

Pass your trained checkpoint to [capture_camera.py](pipeline/capture_camera.py):

```bash
uv run python pipeline/capture_camera.py --model runs/obb/mtg_mobile_v3_yolo11n_obb/weights/best.pt
```

Changing only the detector does not require new reference embeddings.

### Embedder

First build reference vectors with your trained embedder, then use the same directory for camera queries:

```bash
uv run python data_preparation/embedder/01.build_reference_embeddings.py \
	--model runs/embedder_training/dinov3-card-small
uv run python pipeline/capture_camera.py \
	--embedding-model runs/embedder_training/dinov3-card-small
```

`--model` in [01.build_reference_embeddings.py](data_preparation/embedder/01.build_reference_embeddings.py) and `--embedding-model` in [capture_camera.py](pipeline/capture_camera.py) must identify the same embedder, because reference and camera vectors must come from the same model. Both defaults use the published shared MTG/Pokemon embedder.

Cache reuse checks image content and the model name or directory string, not model weights. If you train new weights, save them under a new directory and use it in both commands. Otherwise unchanged references could reuse vectors from the old weights. Existing vectors remain usable with the model that created them.

### Compare Two Images

```bash
uv run python embedder_training/utils/compute_pair_similarity.py \
	path/to/camera-crop.jpg path/to/reference.jpg \
	--model runs/embedder_training/dinov3-card-small \
	--device auto
```

Replace the two image paths with your files. The utility reports upright, 180-degree, and best cosine similarity; omit `--model` to use the published embedder. A pair score is not calibrated confidence or a full retrieval benchmark. Evaluate exact-printing ranking on held-out camera captures, especially similar editions and difficult lighting.

## Performance and Limitations

The demo FPS counter measures the live loop, including camera delivery and display. The following are **historical component timings**, not current mixed-model live-preview guarantees: Apple M2/MPS, an older local MTG-only v2 detector, a saved 1280x720 synthetic scene, 640px inference, and 20 warmed iterations.

| Measurement                                     | Result                  |
| ----------------------------------------------- | ----------------------- |
| Detector without camera acquisition or display  | 23.0 ms/frame, 43.4 FPS |
| Upright and rotated embedding of one saved crop | 61.2 ms                 |
| Top-five search over 113,993 reference vectors  | 14.6 ms                 |
| Per-card embedding and search together          | 75.9 ms                 |

Retrieval timings exclude cropping, JPEG writes, and model loading. Cold starts, card count, hardware, and frame contents affect real capture latency.

- Pokemon catalog synchronization currently uses TCGdex's English endpoint only; other languages and multilingual retrieval are not yet supported.
- The published detector and embedder support both MTG and Pokemon. Retrieval still depends on having cached reference images and embeddings for the selected language and card printing.
- Similar reprints and variants can be confused, particularly when only small edition details differ.
- Synthetic training may not cover glare, sleeves, blur, occlusion, extreme perspective, or very small cards well.
- The intended collection workflow is detect, retrieve, review, correct, and save confirmed cards. Only detection and visual retrieval exist today.

## Next Steps

- Add user review, OCR, and edition-aware reranking for exact printing identification.
- Evaluate held-out real captures and version model releases with source and evaluation results.
- Evaluate mixed-game detection and retrieval on held-out captures, including similar Pokemon variants and reprints.
- Add collection management and multilingual support.

## License

Original code and documentation in this repository are licensed under the [MIT License](LICENSE), copyright (c) 2026 Matteo Tomei. This license does not cover third-party dependencies, pretrained or fine-tuned model weights, datasets, or card images and artwork; those remain subject to their applicable licenses and terms. Using or distributing this project must also comply with those terms, including any applicable Ultralytics AGPL-3.0 obligations. The MIT license does not override them.

## Help and Data Rights

Every script supports `--help`, for example:

```bash
uv run python pipeline/capture_camera.py --help
```

Release metadata is maintained in the [detector model card](detector_training/huggingface_model_card.md) and [embedder model card](embedder_training/huggingface_model_card.md).

MTG metadata and reference images come from [Scryfall](https://scryfall.com/docs/api/bulk-data), which does not endorse this project. Pokemon metadata, prices, and reference image URLs come from [TCGdex](https://tcgdex.dev/), which is not affiliated with Nintendo or The Pokemon Company. Review the [Scryfall terms](https://scryfall.com/docs/terms) and [TCGdex FAQ](https://tcgdex.dev/faq), [market-price documentation](https://tcgdex.dev/markets-prices), and the terms of upstream marketplace and artwork rights holders.

Magic card names, artwork, and related intellectual property belong to Wizards of the Coast and their respective rights holders. Catalog exports, downloaded images, synthetic datasets, local databases, and catalog embeddings are local artifacts, not distributed here. You are responsible for rights governing their use, storage, and redistribution. Published model license and attribution details are in their model cards.
