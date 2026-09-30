---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - magic-the-gathering
  - trading-cards
  - object-detection
  - oriented-object-detection
  - yolo
---
# MTG Card Detector - YOLO11n OBB

This model detects Magic: The Gathering cards in camera frames and returns oriented bounding boxes. It is intended for use before perspective correction and card-image retrieval.

## Model

- Architecture: Ultralytics YOLO11n OBB
- Base checkpoint: `yolo11n-obb.pt`
- Classes: `card`
- Input size: 640 pixels
- Weights: `best.pt`

## Training

The detector was fine-tuned for 5 epochs with a batch size of 16 and seed `42`. Training used synthetic camera scenes containing rendered Magic card images, generated or custom backgrounds, perspective transformations, lighting effects, occlusion, and rotation.

Training used rectangular batches on Apple MPS, rotation of +/-15 degrees, translation of 0.08, scale of 0.25, and perspective of 0.0005. Mirroring, mosaic, mixup, CutMix, and copy-paste were disabled.

The synthetic scenes were created with the scripts in the [Open Card Collector](https://github.com/matteot11/open-card-collector) repository. The recorded Ultralytics run settings are included in `args.yaml`.

## Synthetic Validation

The final epoch's recorded validation metrics were:

| Metric    | Value   |
| --------- | ------- |
| Precision | 0.99949 |
| Recall    | 0.99977 |
| mAP50     | 0.99500 |
| mAP50-95  | 0.98727 |

These results are from the synthetic validation split, not a real-camera benchmark, and do not establish equivalent performance on real footage.

## Usage

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("frame.jpg")
```

## Limitations

This model was trained for Magic cards and synthetic camera conditions. Detection quality may degrade with severe motion blur, glare, heavy occlusion, very small cards, unusual lighting, or scenes unlike its synthetic backgrounds. It detects card geometry; it does not identify a card printing.

## License and Attribution

This model is released under the AGPL-3.0 license. It is fine-tuned from Ultralytics `yolo11n-obb.pt`; use and redistribution must comply with the Ultralytics license terms.

The training workflow uses Magic card imagery obtained through Scryfall. Magic: The Gathering card names, artwork, and related intellectual property belong to Wizards of the Coast and their respective rights holders. This repository does not distribute the training images or Scryfall catalog data. Scryfall does not endorse this project.
