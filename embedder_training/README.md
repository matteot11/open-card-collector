# Embedder Training Workflow

Run commands from the repository root with `uv run`.

## Scripts in This Folder

- `train_card_embeddings.py` fine-tunes the DINOv3 image model with paired,
  augmented views of cached card images and writes a model directory under
  `runs/embedder_training`.
- `utils/compute_pair_similarity.py` compares two card images with the published
  embedder and reports upright, rotated, and best cosine similarity scores.

The catalog synchronization and reference-image embedding scripts live under
`data_preparation`; they prepare the SQLite catalog used by the camera pipeline.

## How DINOv3 Fine-Tuning Works

`train_card_embeddings.py` starts from the pretrained DINOv3 Small image model
and fine-tunes it on the locally cached card images. For each card, the training
loader creates two independently augmented views using changes such as
brightness, contrast, color, rotation, blur, and perspective distortion.

The model produces a CLS embedding for each view. The two views of the same card
form a positive pair, while the other cards in the batch act as negatives. A
symmetric InfoNCE contrastive loss trains the model to bring matching card views
together and separate different cards. Embeddings are L2-normalized during
training and retrieval, and the fine-tuned model is written to the configured
`runs/embedder_training` output directory.

## Sync Scryfall Catalog

Synchronize [Scryfall&#39;s Default Cards export](https://scryfall.com/docs/api/bulk-data)
into a local SQLite catalog:

```bash
uv sync
uv run python data_preparation/00.sync_scryfall_catalog.py
```

Each run checks Scryfall for a new bulk export. An unchanged local archive is
reused; a new archive is downloaded and imported automatically. The bulk archive,
SQLite catalog, and image cache are stored in `data/skryfall_source`.

Cache reference images as well:

```bash
uv run python data_preparation/00.sync_scryfall_catalog.py --download-images
```

Only missing images and images whose Scryfall URL changed are downloaded.

Reference-image caching is only optional if you need the catalog metadata without
building a retrieval index. To make every catalog card searchable, download and
embed every desired reference image. A complete cache is large, so you can use a
limit for a smaller test index:

```bash
uv run python data_preparation/00.sync_scryfall_catalog.py \
  --download-images \
  --image-limit 10000
```

## Embed Reference Images

Install the embedder runtime once, then embed the cached card images:

```bash
uv sync --extra embedder-training
uv run python data_preparation/embedder/01.embed_scryfall_images.py --device mps
```

Embeddings are stored in `data/skryfall_source/catalog.sqlite`. Rerun
`00.sync_scryfall_catalog.py` when Scryfall publishes updated catalog data, using
`--download-images` when new or changed reference images should be cached. Rerun
this embedding command after those local images are added, or when embeddings
are missing. Later runs skip unchanged image files.

Start with a smaller batch if memory is limited:

```bash
uv run python data_preparation/embedder/01.embed_scryfall_images.py \
  --device mps \
  --batch-size 8 \
  --limit 1000
```

The published fine-tuned model is available as the [collector-mtg-embedder-dinov3-small model](https://huggingface.co/matteot11/collector-mtg-embedder-dinov3-small) on Hugging Face.
