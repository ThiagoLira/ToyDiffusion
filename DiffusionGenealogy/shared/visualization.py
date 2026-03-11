import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pathlib import Path
import io


def assign_colors(positions):
    """Assign colors to points based on angle from centroid using HSV colormap.

    Args:
        positions: (N, 2) numpy array of initial positions

    Returns:
        colors: (N, 4) RGBA array
    """
    centroid = positions.mean(axis=0)
    dx = positions[:, 0] - centroid[0]
    dy = positions[:, 1] - centroid[1]
    angles = np.arctan2(dy, dx)
    hue = (angles + np.pi) / (2 * np.pi)
    cmap = plt.cm.hsv
    colors = cmap(hue)
    return colors


def _render_frame(pts, colors, axes, title, point_size, dpi, figsize):
    """Render a single frame to a PIL Image."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(axes["xlim"])
    ax.set_ylim(axes["ylim"])
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.5)

    ax.scatter(pts[:, 0], pts[:, 1], s=point_size, c=colors, marker=".")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).convert("RGB")


def create_trajectory_gif(
    trajectories,
    colors,
    path,
    axes,
    title="Diffusion",
    fps=24,
    max_frames=60,
    point_size=3.0,
    hold_last_frames=36,
):
    """Create a GIF showing the generative trajectory with colored points.

    The last frame is held for `hold_last_frames` extra frames (~1.5s at 24fps)
    so the final result is clearly visible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(trajectories)
    if n_total <= max_frames:
        indices = list(range(n_total))
    else:
        indices = np.linspace(0, n_total - 1, max_frames, dtype=int).tolist()

    frame_duration_ms = int(1000 / fps)
    pil_frames = []

    for frame_idx, step_idx in enumerate(indices):
        pts = trajectories[step_idx]
        frame_title = f"{title}  (step {step_idx}/{n_total - 1})"
        img = _render_frame(pts, colors, axes, frame_title, point_size, dpi=150, figsize=(6, 6))
        pil_frames.append(img)

    # Build durations: normal for all frames, long hold on the last one
    durations = [frame_duration_ms] * len(pil_frames)
    durations[-1] = frame_duration_ms * hold_last_frames

    pil_frames[0].save(
        str(path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
    )
