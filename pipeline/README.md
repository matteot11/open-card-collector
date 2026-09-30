# Camera Capture Pipeline

## Script in This Folder

`capture_camera.py` is the end-user prototype pipeline. It loads a YOLO OBB detector and the DINOv3 embedder, detects cards from a live OpenCV camera, perspective-corrects captured cards, optionally saves crops, and prints local Scryfall candidate matches. By default it downloads the published [YOLO card detector](https://huggingface.co/matteot11/collector-mtg-detector-yolo11n-obb) and [DINOv3 card embedder](https://huggingface.co/matteot11/collector-mtg-embedder-dinov3-small). The catalog must be synced and embedded first.

Run commands from the repository root with `uv run`:

```bash
uv run python pipeline/capture_camera.py \
  --device mps
```

To use your new local detector, add `--model PATH/TO/weights/best.pt`. A missing local checkpoint is reported as a file error, not treated as a model repository. CUDA device indices such as `--device 0` are normalized for both YOLO and PyTorch.

Press `q` or Escape to stop the camera.

Press `c` to capture and retrieve cards in memory. Saving is disabled by default. To also save JPEG crops, pass the destination directory:

```bash
uv run python pipeline/capture_camera.py --device mps --captures-dir data/captures
```

The default catalog is `data/scryfall_source/catalog.sqlite`. Use `--top-k 10` to print more distinct candidate printings. Each printing uses its best score across upright and 180-degree views.

Reference embeddings are selected by model name. Use the same `--model` value in the embedding script and `--embedding-model` value here. Weights are assumed to stay unchanged for that name; existing embeddings remain usable without a migration or rebuild.

The pipeline accepts portrait or landscape camera frames. Specify `--width` and `--height` only when you need to request a particular camera resolution.
