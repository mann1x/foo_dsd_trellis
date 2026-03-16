"""
foo_dsd_trellis — PyTorch Dataset for DSD training pairs.

Pre-slices all data into fixed-size blocks at load time, stores them
as GPU tensors. The model is small — the entire training set of blocks
fits comfortably in VRAM.

V3: Supports both causal (left context only) and non-causal (symmetric
context) models. For non-causal, history is split equally left/right.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class DSDPairDataset:
    """Pre-sliced blocks stored directly on GPU.

    At init:
      1. Load all .npy pairs
      2. Slice each into non-overlapping blocks of (history + block_size)
      3. Stack all blocks into two GPU tensors: x_blocks, y_blocks
      4. Training loop just shuffles and indexes — zero data loading overhead
    """

    def __init__(self, data_dir: str, block_size: int = 4096,
                 history: int = 62,
                 dsd_rates: Optional[list[str]] = None,
                 device: Optional[torch.device] = None,
                 stride: Optional[int] = None,
                 max_vram_mb: int = 6000):
        self.block_size = block_size
        self.history = history
        self.device = device or torch.device('cpu')
        total_len = history + block_size
        if stride is None:
            stride = block_size // 2  # 50% overlap by default

        # Discover pairs
        pair_paths = []
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        for rate_dir in sorted(data_path.iterdir()):
            if not rate_dir.is_dir():
                continue
            if dsd_rates and rate_dir.name not in dsd_rates:
                continue

            for config_dir in sorted(rate_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                for signal_dir in sorted(config_dir.iterdir()):
                    if not signal_dir.is_dir():
                        continue
                    dsd_path = signal_dir / "dsd_input.npy"
                    ref_path = signal_dir / "fir_reference.npy"
                    if dsd_path.exists() and ref_path.exists():
                        pair_paths.append((str(dsd_path), str(ref_path)))

        if not pair_paths:
            raise RuntimeError(f"No training pairs found in {data_dir}")

        # Pre-slice all pairs into blocks
        print(f"Loading {len(pair_paths)} pairs, slicing into blocks "
              f"(block={block_size}, history={history}, stride={stride})...")
        x_blocks = []
        y_blocks = []

        for dsd_path, ref_path in pair_paths:
            dsd = np.load(dsd_path)
            ref = np.load(ref_path)
            min_len = min(len(dsd), len(ref))

            # Slide a window of total_len across the signal
            pos = 0
            while pos + total_len <= min_len:
                x_blocks.append(dsd[pos:pos + total_len])
                y_blocks.append(ref[pos + history:pos + total_len])
                pos += stride

        if not x_blocks:
            raise RuntimeError("No usable blocks (signals too short)")

        n_blocks = len(x_blocks)

        # If blocks exceed VRAM budget, subsample uniformly
        bytes_per_block = (total_len + block_size) * 4  # x + y, float32
        total_bytes = n_blocks * bytes_per_block
        budget_bytes = max_vram_mb * 1024 * 1024
        if total_bytes > budget_bytes and max_vram_mb > 0:
            keep = int(n_blocks * budget_bytes / total_bytes)
            rng = np.random.default_rng(42)
            indices = rng.choice(n_blocks, keep, replace=False)
            indices.sort()
            x_blocks = [x_blocks[i] for i in indices]
            y_blocks = [y_blocks[i] for i in indices]
            print(f"  Subsampled {n_blocks} -> {keep} blocks to fit "
                  f"{max_vram_mb} MB VRAM budget")
            n_blocks = keep

        # Stack into tensors and move to device
        self.x = torch.from_numpy(np.stack(x_blocks)).float().unsqueeze(1)
        self.y = torch.from_numpy(np.stack(y_blocks)).float().unsqueeze(1)
        del x_blocks, y_blocks

        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)

        mem_mb = (self.x.nbytes + self.y.nbytes) / (1024 * 1024)
        print(f"DSDPairDataset: {n_blocks} blocks from {len(pair_paths)} pairs, "
              f"x={list(self.x.shape)}, y={list(self.y.shape)}, "
              f"memory={mem_mb:.0f} MB ({self.device})")

    def __len__(self):
        return self.x.shape[0]

    def generate_batch(self, batch_size: int):
        """Generate a random batch by indexing pre-sliced GPU tensors."""
        idx = torch.randint(len(self), (batch_size,), device=self.device)
        return self.x[idx], self.y[idx]


def create_dataloaders(data_dir: str, block_size: int = 4096,
                       batch_size: int = 256,
                       val_split: float = 0.1,
                       dsd_rates: Optional[list[str]] = None,
                       device: Optional[torch.device] = None,
                       stride: Optional[int] = None,
                       history: Optional[int] = None,
                       **kwargs):
    """Create train and validation datasets on GPU.

    Returns (train_ds, val_ds, train_batches, val_batches).
    """
    if history is None:
        history = 62  # default for compact model

    # Load to CPU first, split, then move to device
    cpu_ds = DSDPairDataset(data_dir, block_size=block_size,
                             history=history,
                             dsd_rates=dsd_rates,
                             device=torch.device('cpu'),
                             stride=stride,
                             max_vram_mb=6000)

    n = len(cpu_ds)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    # Shuffle before split for diversity
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # Create GPU datasets
    train_ds = DSDPairDataset.__new__(DSDPairDataset)
    train_ds.block_size = block_size
    train_ds.history = history
    train_ds.device = device
    train_ds.x = cpu_ds.x[train_idx].to(device)
    train_ds.y = cpu_ds.y[train_idx].to(device)

    val_ds = DSDPairDataset.__new__(DSDPairDataset)
    val_ds.block_size = block_size
    val_ds.history = history
    val_ds.device = device
    val_ds.x = cpu_ds.x[val_idx].to(device)
    val_ds.y = cpu_ds.y[val_idx].to(device)

    # Free CPU copy
    del cpu_ds

    # Cap batches — random sampling means we don't need to exhaust the dataset
    train_batches = min(200, max(100, n_train // batch_size))
    val_batches = min(50, max(10, n_val // batch_size))

    train_mb = (train_ds.x.nbytes + train_ds.y.nbytes) / (1024 * 1024)
    val_mb = (val_ds.x.nbytes + val_ds.y.nbytes) / (1024 * 1024)

    print(f"Train: {n_train} blocks ({train_mb:.0f} MB), {train_batches} batches | "
          f"Val: {n_val} blocks ({val_mb:.0f} MB), {val_batches} batches")

    return train_ds, val_ds, train_batches, val_batches


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/pairs")
    parser.add_argument("--history", type=int, default=62)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DSDPairDataset(args.data, history=args.history, device=device)
    x, y = ds.generate_batch(4)
    print(f"Input:  {x.shape} range [{x.min():.2f}, {x.max():.2f}] device={x.device}")
    print(f"Target: {y.shape} range [{y.min():.2f}, {y.max():.2f}] device={y.device}")
