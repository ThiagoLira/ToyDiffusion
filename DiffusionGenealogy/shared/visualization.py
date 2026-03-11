import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


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
    # Normalize to [0, 1]
    hue = (angles + np.pi) / (2 * np.pi)
    cmap = plt.cm.hsv
    colors = cmap(hue)
    return colors


def create_trajectory_gif(
    trajectories,
    colors,
    path,
    axes,
    title="Diffusion",
    fps=24,
    max_frames=60,
    point_size=1.5,
):
    """Create a GIF showing the generative trajectory with colored points.

    Args:
        trajectories: list of (N, 2) numpy arrays (from noise to data)
        colors: (N, 4) RGBA color array for each point
        path: output path for the GIF
        axes: dict with 'xlim' and 'ylim'
        title: title shown on the GIF
        fps: frames per second
        max_frames: subsample to at most this many frames
        point_size: matplotlib scatter point size
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(trajectories)
    if n_total <= max_frames:
        indices = list(range(n_total))
    else:
        indices = np.linspace(0, n_total - 1, max_frames, dtype=int).tolist()

    frames = [trajectories[i] for i in indices]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(axes["xlim"])
    ax.set_ylim(axes["ylim"])
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    scatter = ax.scatter([], [], s=point_size, c=[], marker=".")

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return (scatter,)

    def update(frame_idx):
        pts = frames[frame_idx]
        scatter.set_offsets(pts)
        scatter.set_color(colors)
        step = indices[frame_idx]
        ax.set_title(f"{title}  (step {step}/{n_total - 1})", color="white",
                      fontsize=14, fontweight="bold")
        return (scatter,)

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(frames), blit=True)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)
