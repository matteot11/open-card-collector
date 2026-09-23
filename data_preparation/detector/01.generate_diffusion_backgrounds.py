"""Generate card-free synthetic backgrounds with a local Diffusers model.

The generated images are intended as compositor backgrounds only. Review outputs
before training and remove images that contain trading cards or obvious artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

BACKGROUND_PROMPTS = [
    "top-down camera photograph, an empty warm oak desk surface fills the entire frame, natural window light, subtle grain",
    "top-down camera photograph, a clean white laminate work desk surface fills the entire frame, soft indoor light",
    "top-down camera photograph, a dark walnut table surface with mild wear fills the entire frame, warm lamp light",
    "top-down camera photograph, a light gray woven fabric surface fills the entire frame, soft diffuse daylight",
    "top-down camera photograph, a neutral charcoal carpet surface fills the entire frame, uneven natural room light",
    "top-down camera photograph, a pale stone countertop surface fills the entire frame, fine texture, soft shadows",
    "top-down camera photograph, a black rubber desk mat surface fills the entire frame, subtle texture, indoor light",
    "top-down camera photograph, a gently worn wooden coffee table surface fills the entire frame, daylight, soft shadows",
    "top-down camera photograph, a navy fabric tablecloth surface fills the entire frame, fine weave, diffuse indoor light",
    "top-down camera photograph, a clean beige cork board surface fills the entire frame, subtle camera grain, daylight",
    "top-down camera photograph, an unmarked muted green cutting mat surface fills the entire frame, indoor light",
    "top-down camera photograph, a pale concrete floor surface fills the entire frame, subtle variation, natural room light",
    "top-down camera photograph, a matte red painted tabletop surface fills the entire frame, faint brush texture",
    "top-down camera photograph, a matte royal blue desk surface fills the entire frame, soft texture",
    "top-down camera photograph, a dark olive green felt playmat surface fills the entire frame, fine fibers",
    "top-down camera photograph, a brushed stainless steel work surface fills the entire frame, soft reflections",
    "top-down camera photograph, a white marble surface with gray veins fills the entire frame, realistic texture",
    "top-down camera photograph, a speckled blue terrazzo surface fills the entire frame, realistic texture",
    "top-down camera photograph, a pale pink linen surface fills the entire frame, fine woven fibers",
    "top-down camera photograph, a faded denim blue fabric surface fills the entire frame, subtle weave",
    "top-down camera photograph, a black leather desk pad surface fills the entire frame, fine grain",
    "top-down camera photograph, a honey bamboo tabletop surface fills the entire frame, natural grain",
    "top-down camera photograph, a dark slate tile surface fills the entire frame, subtle mineral texture",
    "top-down camera photograph, a light blue painted plywood surface fills the entire frame, gentle imperfections",
]

COLOR_LIGHTING_VARIANTS = [
    "soft daylight with neutral white balance",
    "warm amber indoor lamp light",
    "cool blue morning window light",
    "late-afternoon golden daylight",
    "bright diffuse overcast daylight",
    "dim warm room light with gentle falloff",
    "vivid teal surface color under neutral daylight",
    "deep cobalt blue surface color under soft indoor light",
    "forest green surface color under diffuse daylight",
    "muted burgundy surface color under warm lamp light",
    "terracotta red surface color under soft daylight",
    "mustard yellow surface color under neutral indoor light",
    "lavender fabric surface color under cool daylight",
    "bright coral surface color under diffuse room light",
    "rich emerald green surface color under balanced daylight",
    "dusty rose surface color under soft morning light",
    "sunlit turquoise surface color with gentle contrast",
    "indigo surface color under cool overcast daylight",
    "warm cream surface color under indirect window light",
    "dark charcoal surface color under a soft overhead light",
    "bright mint green surface color under neutral daylight",
    "deep red surface color under low warm evening light",
]

CAMERA_CONDITION_VARIANTS = [
    "sharp focus, subtle natural camera grain",
    "slight lens softness near the frame edges",
    "gentle uneven illumination from one side",
    "soft vignette and realistic mobile-camera exposure",
    "mild ISO noise in dim indoor lighting",
    "crisp diffuse studio-like illumination",
    "faint soft cast shadows from off-frame lighting",
    "slightly cool automatic white balance",
    "slightly warm automatic white balance",
    "realistic JPEG compression and subtle sensor noise",
]

CARD_FREE_SUFFIX = (
    ", only the bare surface is visible; no objects, smartphone, mobile phone, "
    "camera, cards, sleeves, text, logos, hands, books, documents, packages, "
    "furniture, tools, or decorations"
)


def import_runtime():
    """Load optional model dependencies only when this script is executed."""
    try:
        import torch
        from diffusers import Flux2KleinPipeline
    except ImportError as error:
        raise SystemExit(
            "Missing optional background-generation dependencies. Install them with:\n"
            "uv sync --extra background-generation\n"
            "Then rerun this command."
        ) from error
    return torch, Flux2KleinPipeline


def select_device(torch, requested_device: str) -> str:
    """Choose the requested accelerator or the best locally available option."""
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_prompts(prompts_file: Path | None) -> list[str]:
    """Load one prompt per line, or use the built-in card-free surface prompts."""
    if prompts_file is None:
        return BACKGROUND_PROMPTS
    prompts = [
        line.strip() for line in prompts_file.read_text(encoding="utf-8").splitlines()
    ]
    prompts = [prompt for prompt in prompts if prompt and not prompt.startswith("#")]
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")
    return prompts


def parse_sizes(value: str) -> list[int]:
    """Parse comma-separated, positive square output dimensions."""
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Sizes must be comma-separated integers"
        ) from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("Sizes must be greater than zero")
    return sizes


def parse_dimensions(value: str) -> list[tuple[int, int]]:
    """Parse comma-separated widthxheight dimensions supported by Flux."""
    dimensions = []
    try:
        for item in value.split(","):
            width_text, height_text = item.lower().split("x", maxsplit=1)
            width, height = int(width_text), int(height_text)
            if width <= 0 or height <= 0 or width % 16 or height % 16:
                raise ValueError
            dimensions.append((width, height))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Dimensions must be comma-separated WIDTHxHEIGHT values divisible by 16"
        ) from error
    if not dimensions:
        raise argparse.ArgumentTypeError("At least one dimension is required")
    return dimensions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate card-free compositor backgrounds with a local Diffusers model."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/detector_training_data/backgrounds/custom"),
    )
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        help="Comma-separated square dimensions chosen independently for each image.",
    )
    parser.add_argument(
        "--dimensions",
        type=parse_dimensions,
        help="Comma-separated WIDTHxHEIGHT dimensions chosen independently for each image.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-id",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="A compatible FLUX.2 klein Hugging Face model ID.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--prompts-file", type=Path, help="Optional text file with one prompt per line."
    )
    args = parser.parse_args()

    if args.count <= 0 or args.size <= 0 or args.steps <= 0:
        parser.error("--count, --size, and --steps must be greater than zero")

    torch, Flux2KleinPipeline = import_runtime()
    device = select_device(torch, args.device)
    if device == "cpu":
        print(
            "Warning: CPU diffusion generation is very slow. Use Apple Silicon MPS or CUDA when possible."
        )

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    pipeline = Flux2KleinPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    if device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    prompts = load_prompts(args.prompts_file)
    randomizer = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in range(args.count):
            base_prompt = randomizer.choice(prompts)
            color_lighting = randomizer.choice(COLOR_LIGHTING_VARIANTS)
            camera_condition = randomizer.choice(CAMERA_CONDITION_VARIANTS)
            prompt = (
                f"{base_prompt}, {color_lighting}, {camera_condition}"
                + CARD_FREE_SUFFIX
            )
            width, height = (
                randomizer.choice(args.dimensions)
                if args.dimensions
                else (
                    (randomizer.choice(args.sizes),) * 2
                    if args.sizes
                    else (args.size, args.size)
                )
            )
            image_seed = randomizer.randrange(2**32)
            generator = torch.Generator(device=device).manual_seed(image_seed)
            image = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            filename = f"background_{index:05d}.jpg"
            image.save(args.output_dir / filename, quality=95)
            manifest.write(
                json.dumps(
                    {
                        "filename": filename,
                        "model_id": args.model_id,
                        "prompt": prompt,
                        "color_lighting": color_lighting,
                        "camera_condition": camera_condition,
                        "seed": image_seed,
                        "width": width,
                        "height": height,
                        "aspect_ratio": round(width / height, 6),
                        "steps": args.steps,
                        "guidance_scale": args.guidance_scale,
                        "device": device,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                + "\n"
            )
            print(f"Generated {index + 1}/{args.count}: {filename}")

    print(f"Background images: {args.output_dir}")
    print(f"Provenance manifest: {manifest_path}")


if __name__ == "__main__":
    main()
