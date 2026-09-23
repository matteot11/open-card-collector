# Camera Capture Pipeline

## Script in This Folder

`capture_camera.py` is the end-user prototype pipeline. It loads a YOLO OBB
detector and the DINOv3 embedder, detects cards from a live OpenCV camera,
perspective-corrects captured cards, saves crops, and prints local Scryfall
candidate matches. By default it downloads the published [YOLO card
detector](https://huggingface.co/matteot11/collector-mtg-detector-yolo11n-obb)
and [DINOv3 card embedder](https://huggingface.co/matteot11/collector-mtg-embedder-dinov3-small).
The catalog must be synced and embedded first.

Run commands from the repository root with `uv run`:

```bash
uv run python pipeline/capture_camera.py \
  --model runs/obb/mtg_mobile_v2_yolo11n_obb/weights/best.pt \
  --device mps
```

Press `q` or Escape to stop the camera.

The default catalog is `data/skryfall_source/catalog.sqlite`.
Use `--top-k 10` to print more candidates.

The pipeline accepts portrait or landscape camera frames. Specify `--width` and
`--height` only when you need to request a particular camera resolution.