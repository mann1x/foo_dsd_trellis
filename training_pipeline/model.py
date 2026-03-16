"""
foo_dsd_trellis — CNN models for DSD noise reduction.

V4: Large non-causal model with bidirectional context.
  - Non-causal convolutions see both past AND future DSD samples
  - Large receptive field (8191+ samples, ~2.9 ms at DSD64)
  - The Viterbi trellis only looks forward; non-causal model can use
    future context to make better decisions about current sample
  - Temperature-annealed quantization: tanh(x * T), T: 1 → 100

Model variants:
  - CausalDilatedCNN: 5-layer causal, RF=63 (~22 μs) — baseline
  - CausalDilatedCNNLarge: 10-layer causal, RF=2047 (~725 μs)
  - NonCausalDilatedCNN: 12-layer non-causal, RF=8191 (~2.9 ms)
    Uses symmetric (centered) padding — each output sample sees
    equal past and future context.

Non-causal implication: the C runtime must buffer look-ahead samples.
Look-ahead = receptive_field // 2 samples of latency.
At DSD64: 4095 samples = ~1.45 ms — negligible for playback.
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1D convolution with causal (left-only) padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block with dilated conv + skip connection.

    Supports both causal (left-pad) and non-causal (symmetric-pad) modes.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 causal: bool = True):
        super().__init__()
        self.causal = causal
        if causal:
            self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        else:
            # Symmetric padding: conv sees equal left and right context
            self.pad_left = (kernel_size - 1) * dilation // 2
            self.pad_right = (kernel_size - 1) * dilation - self.pad_left
            self.conv = nn.Conv1d(channels, channels, kernel_size,
                                  dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)  # instance norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            return x + torch.relu(self.norm(self.conv(x)))
        else:
            h = nn.functional.pad(x, (self.pad_left, self.pad_right))
            return x + torch.relu(self.norm(self.conv(h)))


class CausalDilatedCNN(nn.Module):
    """Causal dilated CNN for DSD noise reduction — compact version.

    5 layers, receptive field = 63 samples (~22 μs at DSD64).
    """

    DILATIONS = [1, 2, 4, 8, 16]
    KERNEL_SIZE = 3
    HIDDEN_CH = 32
    RECEPTIVE_FIELD = (KERNEL_SIZE - 1) * sum(DILATIONS) + 1  # 63

    def __init__(self):
        super().__init__()
        self.temperature = 1.0

        self.input_conv = CausalConv1d(1, self.HIDDEN_CH, self.KERNEL_SIZE, 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(self.HIDDEN_CH, self.KERNEL_SIZE, d, causal=True)
            for d in self.DILATIONS[1:]
        ])
        self.output_conv = nn.Conv1d(self.HIDDEN_CH, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_conv(x))
        for block in self.blocks:
            h = block(h)
        out = self.output_conv(h)
        return torch.tanh(out * self.temperature)

    def set_temperature(self, t: float):
        self.temperature = t

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def receptive_field() -> int:
        return CausalDilatedCNN.RECEPTIVE_FIELD


class CausalDilatedCNNLarge(nn.Module):
    """Causal dilated CNN — large version.

    10 layers, receptive field = 2047 samples (~725 μs at DSD64).
    """

    DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    KERNEL_SIZE = 3
    HIDDEN_CH = 32
    RECEPTIVE_FIELD = (KERNEL_SIZE - 1) * sum(DILATIONS) + 1  # 2047

    def __init__(self):
        super().__init__()
        self.temperature = 1.0

        self.input_conv = CausalConv1d(1, self.HIDDEN_CH, self.KERNEL_SIZE, 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(self.HIDDEN_CH, self.KERNEL_SIZE, d, causal=True)
            for d in self.DILATIONS[1:]
        ])
        self.output_conv = nn.Conv1d(self.HIDDEN_CH, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_conv(x))
        for block in self.blocks:
            h = block(h)
        out = self.output_conv(h)
        return torch.tanh(out * self.temperature)

    def set_temperature(self, t: float):
        self.temperature = t

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def receptive_field() -> int:
        return CausalDilatedCNNLarge.RECEPTIVE_FIELD


class NonCausalDilatedCNN(nn.Module):
    """Non-causal (bidirectional) dilated CNN — large receptive field.

    12 layers with symmetric padding. Each output sample sees equal
    past and future context. This is the key advantage over the trellis
    SDM encoder which only has forward (causal) context.

    Dilations: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    Kernel size: 3
    RF = (K-1) * sum(dilations) + 1 = 2 * 4095 + 1 = 8191 samples
    Look-ahead = RF // 2 = 4095 samples (~1.45 ms at DSD64)
    Hidden channels: 48 (wider than causal models for more capacity)

    At DSD64 (2.8224 MHz): 8191 samples = 2.9 ms total context
    For comparison, the trellis Viterbi has ~latency*cands causal context.
    """

    DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    KERNEL_SIZE = 3
    HIDDEN_CH = 48
    RECEPTIVE_FIELD = (KERNEL_SIZE - 1) * sum(DILATIONS) + 1  # 8191
    LOOK_AHEAD = RECEPTIVE_FIELD // 2  # 4095

    def __init__(self):
        super().__init__()
        self.temperature = 1.0

        # Input conv — symmetric padding
        self.input_pad_left = (self.KERNEL_SIZE - 1) // 2
        self.input_pad_right = (self.KERNEL_SIZE - 1) - self.input_pad_left
        self.input_conv = nn.Conv1d(1, self.HIDDEN_CH, self.KERNEL_SIZE)

        self.blocks = nn.ModuleList([
            ResidualBlock(self.HIDDEN_CH, self.KERNEL_SIZE, d, causal=False)
            for d in self.DILATIONS[1:]
        ])
        self.output_conv = nn.Conv1d(self.HIDDEN_CH, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.pad(x, (self.input_pad_left, self.input_pad_right))
        h = torch.relu(self.input_conv(h))
        for block in self.blocks:
            h = block(h)
        out = self.output_conv(h)
        return torch.tanh(out * self.temperature)

    def set_temperature(self, t: float):
        self.temperature = t

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def receptive_field() -> int:
        return NonCausalDilatedCNN.RECEPTIVE_FIELD

    @staticmethod
    def look_ahead() -> int:
        return NonCausalDilatedCNN.LOOK_AHEAD


def build_model(large: bool = False, noncausal: bool = False):
    """Create and initialize the model."""
    if noncausal:
        model = NonCausalDilatedCNN()
    elif large:
        model = CausalDilatedCNNLarge()
    else:
        model = CausalDilatedCNN()

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


if __name__ == "__main__":
    for name, kwargs in [("Compact", {}),
                          ("Large causal", {"large": True}),
                          ("Non-causal", {"noncausal": True})]:
        model = build_model(**kwargs)
        rf = model.receptive_field()
        print(f"\n{name} model:")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Receptive field: {rf} samples ({rf / 2822400 * 1e6:.0f} us at DSD64)")
        if hasattr(model, 'look_ahead'):
            la = model.look_ahead()
            print(f"  Look-ahead: {la} samples ({la / 2822400 * 1e6:.0f} us)")

        hist = rf - 1
        x = torch.randn(1, 1, 4096 + hist)

        for T in [1.0, 10.0, 100.0]:
            model.set_temperature(T)
            y = model(x)
            mn, mx = y.min().item(), y.max().item()
            # Non-causal: output length = input length
            # Causal: output length = input length (due to causal padding)
            print(f"  T={T:5.1f}: output shape {list(y.shape)}, "
                  f"range [{mn:.4f}, {mx:.4f}]")

        print(f"  {name} model OK")
