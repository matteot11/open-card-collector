# Open Card Collector

Open source prototype for detecting Magic: The Gathering cards in camera frames, rectifying detected cards, and retrieving local [Scryfall](https://scryfall.com/) catalog matches. Pokemon support and collection-management workflows are planned but not implemented.

## What This Project Wants to Do

Open Card Collector aims to become a local-first tool for turning camera captures
of physical trading cards into a reviewable collection. A user should be able to
point a camera at one or more cards, confirm the detected cards, inspect likely
catalog matches, correct uncertain results, and eventually save the confirmed
cards to a collection database.

The project is intentionally built as several replaceable stages rather than as
one closed-set classifier:

- A detector finds card boundaries and does not need to know the card's identity.
- A perspective-correction step turns each detected quadrilateral into a standard
  card crop.
- An embedder maps reference and camera images into the same visual feature space.
- A local catalog search returns candidate printings for user review.
- Future OCR and reranking stages will use text and edition-specific details to
  distinguish near-duplicate printings.

The current repository contains the camera and visual-retrieval prototype. The
full collection workflow, multilingual support, Pokemon support, OCR, and
edition-aware reranking remain future work.

Published weights are available on Hugging Face:

- [Magic card detector](https://huggingface.co/matteot11/collector-mtg-detector-yolo11n-obb)
- [DINOv3 card embedder](https://huggingface.co/matteot11/collector-mtg-embedder-dinov3-small)

## Repository Workflows

- [Data preparation](data_preparation): synchronize the Scryfall catalog,
  download reference images, generate detector backgrounds and synthetic scenes,
  and build catalog embeddings.
- [Detector training](detector_training/README.md): train the YOLO OBB model with
  `train_yolo_obb.py`.
- [Embedder training](embedder_training/README.md): fine-tune DINOv3 with
  `train_card_embeddings.py` and compare image pairs with the utility under
  `utils/`.
- [Camera pipeline](pipeline/README.md): run `capture_camera.py` for live
  detection, perspective correction, and retrieval.

## Architecture & Workflow

1. **Camera Detection**:
   - Processes live OpenCV camera frames with a YOLO11 OBB detector.
   - Draws oriented card polygons and detector confidence scores.
   - Press `c` to capture all cards detected in the current frame.
2. **Perspective Correction**:
   - Warps each detected quadrilateral into a standardized 1371x1920 px portrait crop.
   - Saves the crops under `data/captures` by default.
3. **Visual Retrieval**:
   - Embeds each captured crop with the published DINOv3-based embedder.
   - Compares it with normalized reference embeddings stored in the local SQLite catalog.
   - Prints ranked Scryfall matches, set codes, collector numbers, prices, scores, and orientation.
4. **Review**:
   - Displays the captured frame with the top match overlaid while the camera is paused.
   - Press `c` or `r` to resume, or `q`/Escape to quit.

## End-to-End Workflow

### 1. Prepare the Local Catalog

The project uses [Scryfall&#39;s Default Cards export](https://scryfall.com/docs/api/bulk-data)
for local Magic card metadata.
The synchronizer stores the bulk archive, SQLite catalog, and reference-image
cache under `data/skryfall_source`.

```bash
uv run python data_preparation/00.sync_scryfall_catalog.py \
   --download-images
```

The published [detector](https://huggingface.co/matteot11/collector-mtg-detector-yolo11n-obb)
and [embedder](https://huggingface.co/matteot11/collector-mtg-embedder-dinov3-small)
are downloaded from Hugging Face when inference starts, while the Scryfall
metadata and images must be prepared locally.

### 2. Build Reference Embeddings

The embedder processes the cached reference images and stores normalized vectors
in `data/skryfall_source/catalog.sqlite`. Retrieval itself uses the stored
embedding and catalog metadata; the reference image is needed when creating or
refreshing that embedding. Therefore, every card that should be searchable
needs a corresponding embedding, while its source image only needs to remain
available if you may need to regenerate or update that embedding.

```bash
uv run python data_preparation/embedder/01.embed_scryfall_images.py \
   --device auto
```

Rerun `00.sync_scryfall_catalog.py` when Scryfall publishes updated catalog data.
Use `--download-images` if that update includes new or changed reference images.
Then rerun `01.embed_scryfall_images.py` whenever new reference images have been
added to the local cache, or when you want to populate embeddings that are
missing from the local database. The embedding script skips unchanged images.

### 3. Run Camera Detection

The camera pipeline loads the published YOLO OBB detector, selects CUDA, MPS, or
CPU, and processes live OpenCV frames. Press `c` to save every card detected in
the current frame and start retrieval for those crops.

```bash
uv run python pipeline/capture_camera.py
```

### 4. Rectify and Save Card Crops

For each detection, the pipeline orders the four predicted corners and applies a
perspective transform to produce a 1371x1920 portrait crop. Crops are saved in
`data/captures` by default and are the inputs to visual retrieval.

### 5. Retrieve Candidate Cards

Each crop is embedded with the published DINOv3-based model. The query vector is
L2-normalized and compared with the normalized vectors in the local SQLite
catalog using cosine similarity. The pipeline prints the top candidates with
card name, set code, collector number, available prices, score, and orientation.

### 6. Review the Result

The camera pauses on the captured frame and overlays the highest-ranked result.
The terminal also lists the requested number of candidates, controlled by
`--top-k`. The current prototype does not provide in-app confirmation,
correction, or candidate selection; results must be inspected outside the
application, especially for reprints, alternate treatments, foils, and unclear
images.

### 7. Improve and Retrain the Models

Training is a separate data-preparation workflow. The detector workflow creates
synthetic scenes from cached card images and backgrounds, then trains YOLO OBB.
The embedder workflow fine-tunes DINOv3 with augmented views of cached card
images. See the [detector training workflow](detector_training/README.md) and
[embedder workflow](embedder_training/README.md) for the reproducible commands.

## Installation

Install the base project and the detector/embedder dependencies:

```bash
uv sync --extra detector-training --extra embedder-training
```

### GPU Acceleration

The scripts automatically select CUDA when an NVIDIA GPU is available, MPS on
supported Apple Silicon systems, and CPU otherwise. The default PyTorch package
resolution does not necessarily install CUDA support. NVIDIA users should select
the PyTorch CUDA build matching their environment before syncing dependencies:

```bash
uv sync --extra detector-training --extra embedder-training --torch-backend=cu124
```

Replace `cu124` with the CUDA build supported by your system. See the
[PyTorch installation selector](https://pytorch.org/get-started/locally/) for
available builds. Use `--device cpu`, `--device mps`, or `--device cuda` to
override automatic device selection for supported commands.

## Quick Camera Usage

```bash
# Detect, capture, and retrieve cards from the default camera
uv run python pipeline/capture_camera.py

# Use an explicit camera and accelerator
uv run python pipeline/capture_camera.py --camera 0 --device mps
```

The catalog must be synchronized, reference images must be downloaded, and
embeddings must be computed before starting camera retrieval. See the
[embedder workflow](embedder_training/README.md) and [camera pipeline
README](pipeline/README.md).

## Data Provenance and Third-Party Rights

The catalog synchronizer uses [Scryfall's API and Default Cards bulk
data](https://scryfall.com/docs/api/bulk-data). Scryfall is an independent
third-party service and does not endorse this project. Review [Scryfall's
terms](https://scryfall.com/docs/terms) and applicable API/data guidance before
using or redistributing downloaded data.

Card names, artwork, logos, and other card-related intellectual property belong
to Wizards of the Coast and other respective rights holders. This repository
does not distribute Scryfall catalog exports, downloaded card images, generated
datasets containing card imagery, local SQLite catalogs, or derived catalog
embeddings. Users are responsible for checking the rights and terms that apply
to their use, storage, and redistribution of locally downloaded data.

## Current Limitations

- The published models and current retrieval evaluation target English-language Magic: The Gathering cards. The catalog synchronizer can store other Scryfall languages, but multilingual retrieval is not yet supported or evaluated.
- Pokemon and other trading-card games are not supported by the current published models, despite the planned roadmap work.
- Recognition depends on a locally synchronized Scryfall catalog, cached reference images, and matching local embeddings. The published models do not include this catalog or image data.
- DINOv3 retrieval usually identifies the correct card, but can confuse reprints and variants with the same name when they differ only in small edition-specific details such as a set symbol, collector number, border, or layout.
- The detector and embedder were trained primarily on synthetic camera views and may degrade with glare, sleeves, motion blur, strong occlusion, unusual lighting, extreme perspective, or very small cards.
- Retrieval results are candidates rather than guaranteed identifications. The current prototype does not let users confirm or correct a result in the application; close matches require external review, especially for printings, languages, foils, showcase cards, and alternate treatments.
- The current camera workflow is local-first and expects a desktop environment with an accessible OpenCV camera. Camera orientation and device-specific behavior may vary, especially with phone or Continuity Camera sources.

## Roadmap

- [ ] Improve Magic printing identification with a candidate reranker to distinguish editions and variants.
- [ ] Add user review for uncertain retrieval results: show ranked candidates, allow search and manual selection, and record corrections for evaluation.
- [ ] Add Pokemon support in stages:

  - Add a Pokemon catalog provider and local image/metadata synchronization workflow.
  - Build and evaluate a labeled mixed-game camera dataset.
  - Decide from benchmarks whether the detector needs `mtg` and `pokemon` classes, or whether a game-agnostic `card` detector plus a separate game classifier is more accurate and maintainable.
  - Train game-specific retrieval catalogs and use game classification only to select or prioritize the appropriate catalog.
- [ ] Add reproducible retrieval benchmarks for exact printing identification, including held-out camera captures, alternate artwork, glare, sleeves, occlusion, rotation, and near-duplicate printings.
- [ ] Version published detector and embedder weights, record their source commit and evaluation results, and keep model cards synchronized with each release.
- [ ] Document data provenance and licensing boundaries, including that card images, catalog exports, generated datasets, local SQLite catalogs, and catalog embeddings are not distributed with this repository.
