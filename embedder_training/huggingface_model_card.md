---
license: other
library_name: transformers
pipeline_tag: image-feature-extraction
base_model: facebook/dinov3-vits16-pretrain-lvd1689m
tags:
  - magic-the-gathering
  - trading-cards
  - image-retrieval
  - image-feature-extraction
  - dinov3
---
# MTG Card Embedder - DINOv3 Small

Built with DINOv3.

This model produces normalized visual embeddings for Magic: The Gathering card images. It is intended for nearest-neighbor retrieval against a separately created local card catalog; it does not directly classify or identify a card printing.

## Model

- Architecture: DINOv3 ViT-S/16
- Base model: [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
- Image size: 224 x 224 pixels
- Embedding: final-layer CLS token, L2-normalized by the retrieval workflow
- Runtime: Hugging Face Transformers

## Training

The base model was fine-tuned using paired synthetic camera views of cached Magic card images. Each view can include brightness, contrast, and color changes; small rotation; blur; and perspective distortion. Training optimizes symmetric InfoNCE loss between embeddings of two augmented views of the same card.

The default training configuration uses 3 epochs, batch size 16, learning rate `1e-5`, temperature `0.07`, zero data-loader workers, and seed `42`. The complete training workflow is available in the [Open Card Collector](https://github.com/matteot11/open-card-collector) repository.

## Usage

```python
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

model_id = "matteot11/collector-mtg-embedder-dinov3-small"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).eval()

image = Image.open("card.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.inference_mode():
    vector = model(**inputs).last_hidden_state[:, 0]
    vector = torch.nn.functional.normalize(vector, dim=1)
```

## Limitations

The model was fine-tuned for Magic card retrieval, not general-purpose image recognition or authoritative card identification. Retrieval quality can degrade with glare, motion blur, occlusion, low resolution, heavy cropping, uncommon printings, or reference catalogs that do not contain the target card. The nearest retrieved catalog item should be reviewed by a user.

## License and Attribution

This model is a derivative of DINOv3 and is distributed under the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/). Use, modification, and redistribution must comply with those terms.

The training workflow uses Magic card imagery obtained through Scryfall. Magic: The Gathering card names, artwork, and related intellectual property belong to Wizards of the Coast and their respective rights holders. This repository does not distribute training images, Scryfall catalog data, or derived catalog embeddings. Scryfall and Wizards of the Coast do not endorse this project.
