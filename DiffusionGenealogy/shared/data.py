import torch
import numpy as np
from PIL import Image
from pathlib import Path

IMG_SIZE = 250


def load_homer_data(img_path=None):
    """Load Homer Simpson image and return (N,2) tensor of normalized coordinates.

    Returns:
        data: torch.FloatTensor of shape (N, 2) with normalized (x, y) coords
        axes: dict with 'xlim' and 'ylim' tuples for consistent plotting
    """
    if img_path is None:
        img_path = Path(__file__).parent.parent.parent / "homer.png"
    else:
        img_path = Path(img_path)

    if not img_path.exists():
        raise FileNotFoundError(
            f"Homer image not found at {img_path}.\n"
            "Download from: https://www.infomoney.com.br/wp-content/uploads/2019/06/"
            "homer-simpson.jpg?resize=900%2C515&quality=50&strip=all\n"
            "and save as 'homer.png' in the project root."
        )

    w = IMG_SIZE
    img = Image.open(img_path).resize((w, w)).convert("1")
    pels = img.load()
    black_pels = [(x, y) for x in range(w) for y in range(w) if pels[x, y] == 0]

    xs = np.array([p[0] for p in black_pels], dtype=np.float32)
    ys = np.array([w - p[1] for p in black_pels], dtype=np.float32)  # invert Y

    # Normalize to roughly [-3, 3] range (matching original notebook)
    xs = xs / 25.0 - 3.0
    ys = ys / 25.0 - 2.0

    data = np.stack([xs, ys], axis=1)
    data_tensor = torch.from_numpy(data)

    axes = {
        "xlim": (float(xs.min()) - 0.3, float(xs.max()) + 0.3),
        "ylim": (float(ys.min()) - 0.3, float(ys.max()) + 0.3),
    }

    return data_tensor, axes
