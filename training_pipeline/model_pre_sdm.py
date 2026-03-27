"""
Pre-SDM Enhancement Model — tiny adaptive pre-emphasis.

Instead of a fixed k per rate, learns a small causal FIR that
conditions the signal before SDM encoding. The FIR adapts to
local signal characteristics via a lightweight feature extractor.

Architecture:
  Input: boxcar output (multi-bit, DSD rate)
  → Learned causal FIR (5-15 taps)
  → Output: pre-emphasized signal for SDM

Two variants:
  1. FixedFIR: learns one FIR per rate (no signal adaptation)
  2. AdaptiveFIR: tiny CNN extracts features → modulates FIR coefficients
"""

import torch
import torch.nn as nn


class FixedPreEmphasis(nn.Module):
    """Learns a fixed causal FIR filter for pre-SDM enhancement.

    The filter is applied as a convolution. The first tap is fixed at 1.0
    (identity) and the remaining taps learn the pre-emphasis shape.
    This is equivalent to: y[n] = x[n] + sum(k[i] * x[n-i])
    """

    def __init__(self, num_taps=7):
        super().__init__()
        self.num_taps = num_taps
        # Initialize: identity (tap 0 = 1.0) + small random perturbation
        # The model learns deviations from identity
        self.taps = nn.Parameter(torch.zeros(num_taps, dtype=torch.float64))
        # Initialize tap 0 to 1.0 (identity passthrough)
        with torch.no_grad():
            self.taps[0] = 1.0
            # Small pre-emphasis seed (tap 1 = 0.01, like our sweep winner)
            if num_taps > 1:
                self.taps[1] = -0.007

    def forward(self, x):
        """
        x: (batch, seq_len) float64 — boxcar output
        Returns: (batch, seq_len) float64 — pre-emphasized signal
        """
        # Causal convolution: pad on left only
        x_pad = torch.nn.functional.pad(x.unsqueeze(1), (self.num_taps - 1, 0))
        # Flip taps for conv1d (it does cross-correlation, not convolution)
        kernel = self.taps.flip(0).unsqueeze(0).unsqueeze(0)
        out = torch.nn.functional.conv1d(x_pad, kernel).squeeze(1)
        return out


class AdaptivePreEmphasis(nn.Module):
    """Signal-adaptive pre-emphasis with learned FIR modulation.

    A tiny feature extractor analyzes local signal statistics and
    modulates the FIR coefficients. This allows different pre-emphasis
    for loud vs quiet passages, tonal vs noisy content, etc.

    Architecture:
      Signal → AvgPool(block_size) → small MLP → FIR modulation weights
      Signal → FIR(base_taps * modulation) → output
    """

    def __init__(self, num_taps=7, block_size=256, hidden=16):
        super().__init__()
        self.num_taps = num_taps
        self.block_size = block_size

        # Base FIR taps (learned, initialized to identity + pre-emphasis)
        self.base_taps = nn.Parameter(torch.zeros(num_taps, dtype=torch.float64))
        with torch.no_grad():
            self.base_taps[0] = 1.0
            if num_taps > 1:
                self.base_taps[1] = -0.007

        # Feature extractor: local RMS + spectral tilt → modulation
        self.feature_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_taps),
        ).double()

        # Initialize feature_net to output near-zero (base_taps dominate)
        with torch.no_grad():
            self.feature_net[-1].weight.zero_()
            self.feature_net[-1].bias.zero_()

    def _extract_features(self, x):
        """Extract local signal features per block.

        Returns: (batch, num_blocks, 2) — [RMS, spectral_tilt] per block
        """
        B, N = x.shape
        nblocks = N // self.block_size
        x_blocks = x[:, :nblocks * self.block_size].reshape(B, nblocks, self.block_size)

        # RMS per block
        rms = x_blocks.pow(2).mean(dim=2).sqrt()  # (B, nblocks)

        # Spectral tilt: ratio of high-freq to low-freq energy
        half = self.block_size // 2
        lo = x_blocks[:, :, :half].pow(2).mean(dim=2)
        hi = x_blocks[:, :, half:].pow(2).mean(dim=2)
        tilt = (hi / (lo + 1e-10)).log()  # log ratio

        features = torch.stack([rms, tilt], dim=2)  # (B, nblocks, 2)
        return features

    def forward(self, x):
        """
        x: (batch, seq_len) float64
        Returns: (batch, seq_len) float64
        """
        B, N = x.shape
        nblocks = N // self.block_size
        usable = nblocks * self.block_size

        # Extract features
        features = self._extract_features(x)  # (B, nblocks, 2)
        modulation = self.feature_net(features)  # (B, nblocks, num_taps)

        # Apply modulated FIR per block
        output = torch.zeros_like(x)
        taps_base = self.base_taps  # (num_taps,)

        for b in range(nblocks):
            start = b * self.block_size
            end = start + self.block_size

            # Modulated taps: base + signal-adaptive offset
            mod_taps = taps_base + modulation[:, b, :]  # (B, num_taps)

            # Apply per-sample (using the block's taps)
            # For efficiency, use conv1d with the block's taps
            block = x[:, max(0, start - self.num_taps + 1):end]
            if block.shape[1] < self.block_size + self.num_taps - 1:
                block = torch.nn.functional.pad(block, (self.num_taps - 1 - (block.shape[1] - self.block_size), 0))

            # Batched conv1d: each batch element has its own kernel
            for bi in range(B):
                kernel = mod_taps[bi].flip(0).unsqueeze(0).unsqueeze(0)
                inp = block[bi:bi+1].unsqueeze(1)
                out_block = torch.nn.functional.conv1d(inp, kernel).squeeze()
                output[bi, start:end] = out_block[-self.block_size:]

        # Handle remainder
        if usable < N:
            output[:, usable:] = x[:, usable:]

        return output


def create_model(model_type='fixed', **kwargs):
    """Factory function for pre-SDM models."""
    if model_type == 'fixed':
        return FixedPreEmphasis(**kwargs)
    elif model_type == 'adaptive':
        return AdaptivePreEmphasis(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
