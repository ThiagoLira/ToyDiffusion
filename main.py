"""DiffusionGenealogy: 6 Diffusion Variants on Homer Simpson.

Usage:
    python main.py                      # Run all 6 variants
    python main.py ddpm rectified_flow  # Run specific ones
    python main.py --epochs 100         # Override epochs
    python main.py --list               # List available variants
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

from DiffusionGenealogy import VARIANTS
from DiffusionGenealogy.shared import load_homer_data, assign_colors, create_trajectory_gif


VARIANT_DISPLAY = {
    "ddpm": "DDPM (Ho et al., 2020)",
    "ddim": "DDIM (Song et al., 2020)",
    "score_sde": "Score-SDE (Song et al., 2021)",
    "edm": "EDM (Karras et al., 2022)",
    "rectified_flow": "Rectified Flow (Liu et al., 2022)",
    "ot_cfm": "OT-CFM (Tong et al., 2023)",
}

# Default epochs per variant (tuned for convergence on Homer data)
VARIANT_EPOCHS = {
    "ddpm": 5000,
    "ddim": 5000,
    "score_sde": 5000,
    "edm": 5000,
    "rectified_flow": 5000,
    "ot_cfm": 3000,
}


def run_variant(name, cls, data, device, epochs, output_dir):
    """Train and generate for a single variant, save GIF."""
    print(f"\n{'='*60}")
    print(f"  {VARIANT_DISPLAY.get(name, name)}")
    print(f"{'='*60}")

    start = time.time()

    # Instantiate
    diffusion = cls(device=device)

    # Train
    losses = diffusion.train(data, epochs=epochs)
    train_time = time.time() - start
    print(f"  Training: {train_time:.1f}s | Final loss: {losses[-1]:.6f}")

    # Generate trajectory
    n_samples = data.shape[0]
    trajectory = diffusion.generate(n_samples)
    gen_time = time.time() - start - train_time
    print(f"  Generation: {gen_time:.1f}s | {len(trajectory)} steps")

    # Assign colors based on final positions (the generated Homer)
    final_positions = trajectory[-1]
    colors = assign_colors(final_positions)

    # Save GIF
    gif_path = output_dir / name / f"{name}_generation.gif"
    create_trajectory_gif(
        trajectory, colors, gif_path, axes,
        title=VARIANT_DISPLAY.get(name, name),
    )
    print(f"  GIF saved: {gif_path}")

    return {
        "name": name,
        "final_loss": losses[-1],
        "train_time": train_time,
        "gen_time": gen_time,
        "n_steps": len(trajectory) - 1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffusionGenealogy: 6 Diffusion Variants on Homer Simpson")
    parser.add_argument("variants", nargs="*", help="Variant names to run (default: all)")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (overrides per-variant defaults)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    parser.add_argument("--homer", type=str, default=None, help="Path to homer.png")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available variants")
    args = parser.parse_args()

    if args.list:
        print("Available variants:")
        for name, display in VARIANT_DISPLAY.items():
            print(f"  {name:20s} {display}")
        sys.exit(0)

    # Select device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    data, axes = load_homer_data(args.homer)
    print(f"Loaded Homer data: {data.shape[0]} points")

    # Select variants
    if args.variants:
        selected = []
        for v in args.variants:
            if v not in VARIANTS:
                print(f"Unknown variant '{v}'. Available: {list(VARIANTS.keys())}")
                sys.exit(1)
            selected.append(v)
    else:
        selected = list(VARIANTS.keys())

    output_dir = Path(args.output_dir)
    results = []

    for name in selected:
        cls = VARIANTS[name]
        epochs = args.epochs if args.epochs is not None else VARIANT_EPOCHS.get(name, 5000)
        result = run_variant(name, cls, data, device, epochs, output_dir)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"{'Variant':20s} {'Loss':>12s} {'Train(s)':>10s} {'Gen(s)':>10s} {'Steps':>8s}")
    print("-" * 62)
    for r in results:
        print(f"{r['name']:20s} {r['final_loss']:12.6f} {r['train_time']:10.1f} {r['gen_time']:10.1f} {r['n_steps']:8d}")
