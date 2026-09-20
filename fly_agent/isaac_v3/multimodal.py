from pathlib import Path
import numpy as np


class V35DemoBuffer:
    def __init__(self):
        self.dn, self.visual, self.grid, self.y = [], [], [], []

    def __len__(self):
        return len(self.dn)

    def append(self, dn, visual, grid, labels):
        self.dn.append(np.asarray(dn, dtype=np.float32).copy())
        self.visual.append(np.asarray(visual, dtype=np.float32).copy())
        self.grid.append(np.asarray(grid, dtype=np.float32).copy())
        self.y.append(np.asarray(labels, dtype=np.float32).copy())

    def arrays(self):
        if not self.dn:
            e = np.empty((0, 0), dtype=np.float32)
            return e, e.copy(), e.copy(), e.copy()
        return np.stack(self.dn), np.stack(self.visual), np.stack(self.grid), np.stack(self.y)

    def save(self, path, metadata=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        dn, visual, grid, y = self.arrays()
        np.savez_compressed(
            path, x=dn, visual=visual, grid=grid, y=y,
            format_version=np.asarray([35], dtype=np.int32),
            metadata=np.asarray(["" if metadata is None else str(metadata)]),
        )
        print(
            f"Dataset saved: {path} ({len(dn)} samples, "
            f"DN={dn.shape[1] if dn.ndim==2 else 0}, "
            f"VIS={visual.shape[1] if visual.ndim==2 else 0}, "
            f"GRID={grid.shape[1] if grid.ndim==2 else 0})"
        )
        return path


def load_v35_dataset(path):
    path = Path(path); data = np.load(path, allow_pickle=False)
    version = int(data["format_version"][0]) if "format_version" in data else 0
    if version != 35 or "grid" not in data or "visual" not in data:
        raise ValueError(f"{path.name} is not a V3.5 DN+VISUAL+GRID dataset")
    dn = np.asarray(data["x"], dtype=np.float32)
    visual = np.asarray(data["visual"], dtype=np.float32)
    grid = np.asarray(data["grid"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    if not (len(dn) == len(visual) == len(grid) == len(y)):
        raise ValueError(f"Sample count mismatch in {path}")
    return dn, visual, grid, y


def load_v34_visual_dataset(path):
    """V3.4 data can still augment PURE VISUAL training because VIS width is unchanged."""
    path = Path(path); data = np.load(path, allow_pickle=False)
    version = int(data["format_version"][0]) if "format_version" in data else 0
    if version != 34 or "visual" not in data:
        raise ValueError(f"{path.name} is not V3.4")
    return (
        np.asarray(data["x"], dtype=np.float32),
        np.asarray(data["visual"], dtype=np.float32),
        np.asarray(data["y"], dtype=np.float32),
    )


def source_vector(mode, dn, visual, grid):
    mode = str(mode).lower()
    dn = np.asarray(dn, dtype=np.float32).reshape(-1)
    visual = np.asarray(visual, dtype=np.float32).reshape(-1)
    grid = np.asarray(grid, dtype=np.float32).reshape(-1)
    if mode == "visual": return visual
    if mode == "grid": return grid
    if mode == "visual_grid": return np.concatenate([visual, grid]).astype(np.float32, copy=False)
    if mode == "visual_grid_dn": return np.concatenate([visual, grid, dn]).astype(np.float32, copy=False)
    raise ValueError(f"Unknown V3.5 input mode: {mode}")


def source_matrix(mode, dn, visual, grid):
    mode = str(mode).lower()
    dn = np.asarray(dn, dtype=np.float32)
    visual = np.asarray(visual, dtype=np.float32)
    grid = np.asarray(grid, dtype=np.float32)
    if mode == "visual": return visual
    if mode == "grid": return grid
    if mode == "visual_grid": return np.concatenate([visual, grid], axis=1).astype(np.float32, copy=False)
    if mode == "visual_grid_dn": return np.concatenate([visual, grid, dn], axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unknown V3.5 input mode: {mode}")


def mode_base_dim(mode, dn_dim, visual_dim, grid_dim):
    return {
        "visual": visual_dim,
        "grid": grid_dim,
        "visual_grid": visual_dim + grid_dim,
        "visual_grid_dn": visual_dim + grid_dim + dn_dim,
    }[str(mode).lower()]
