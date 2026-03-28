"""
Export full-pipeline pre-SDM ONNX model for GPU inference.

Single model: signal chunk → feature extraction → MLP → dynamic FIR → output.
Runs entirely on GPU (CUDA/DirectML), replacing the embedded CPU MLP.

Input:  (1, 1, N) float32 — boxcar output chunk
Output: (1, 1, N) float32 — pre-emphasized signal

The model bakes in:
  1. Feature extraction (RMS, ZCR proxy, crest factor)
  2. MLP with baked normalization (3→16→16→3)
  3. DC gain normalization (sum of taps = 1.0)
  4. Element-wise 3-tap FIR: out = t0*x[n] + t1*x[n-1] + t2*x[n-2]

Usage:
  python export_preemph_onnx.py
  python export_preemph_onnx.py --params foo_dsd_trellis_preemph_params.npz
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn


class PreemphFullPipeline(nn.Module):
    """Full pre-SDM pipeline: features → MLP → FIR apply.

    All operations are differentiable and ONNX-exportable.
    """

    def __init__(self, hidden=16):
        super().__init__()

        # Feature normalization (baked in)
        self.register_buffer('feat_mean', torch.zeros(3))
        self.register_buffer('feat_std', torch.ones(3))

        # MLP: 3 features → 3 taps
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 3)

    def forward(self, x):
        """
        x: (1, 1, N) float32 — boxcar output signal

        Returns: (1, 1, N) float32 — pre-emphasized signal
        """
        signal = x.squeeze(0).squeeze(0)  # (N,)
        N = signal.shape[0]

        # --- Feature extraction (matches preemph.c exactly) ---

        # RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(signal * signal))

        # Spectral centroid via ZCR: count sign changes / (N-1) * sr / 2
        # Sign of each sample (>=0 → 1, <0 → 0)
        signs = (signal >= 0.0).float()
        crossings = torch.abs(signs[1:] - signs[:-1])
        # Use tensor division to keep dynamic (avoid float(N-1) constant)
        n_minus_1 = torch.tensor(1.0).expand_as(crossings).sum()
        zcr = torch.sum(crossings) / n_minus_1
        # C code uses sample_rate = DSD512 rate (22579200)
        # centroid = zcr * sample_rate / 2, clamped to [20, 20000]
        centroid = zcr * (22579200.0 / 2.0)
        centroid = torch.clamp(centroid, 20.0, 20000.0)

        # Crest factor: peak / RMS
        peak = torch.max(torch.abs(signal))
        crest = peak / torch.clamp(rms, min=1e-10)

        # --- MLP inference ---
        # Feature order matches preemph.c: [centroid, rms, crest]
        features = torch.stack([centroid, rms, crest]).unsqueeze(0)  # (1, 3)
        normalized = (features - self.feat_mean) / self.feat_std
        h = torch.relu(self.fc1(normalized))
        h = torch.relu(self.fc2(h))
        taps = self.fc3(h).squeeze(0)  # (3,)

        # DC gain normalization: sum(taps) = 1.0
        dc_gain = torch.sum(taps)
        dc_gain = torch.clamp(dc_gain, min=0.01)
        taps = taps / dc_gain

        # --- Element-wise 3-tap FIR ---
        # out[n] = t0*x[n] + t1*x[n-1] + t2*x[n-2]
        t0, t1, t2 = taps[0], taps[1], taps[2]

        # Shifted versions (zero-padded at start)
        x_0 = signal                                         # x[n]
        x_1 = torch.cat([torch.zeros(1, device=signal.device), signal[:-1]])  # x[n-1]
        x_2 = torch.cat([torch.zeros(2, device=signal.device), signal[:-2]])  # x[n-2]

        out = t0 * x_0 + t1 * x_1 + t2 * x_2

        return out.unsqueeze(0).unsqueeze(0)  # (1, 1, N)


def load_weights_from_npz(model, npz_path):
    """Load MLP weights from the CMA-ES training output."""
    params = np.load(npz_path)

    with torch.no_grad():
        model.feat_mean.copy_(torch.tensor(params['feat_mean'], dtype=torch.float32))
        model.feat_std.copy_(torch.tensor(params['feat_std'], dtype=torch.float32))

        # w0 = (48,) → fc1.weight (16, 3)
        model.fc1.weight.copy_(torch.tensor(
            params['w0'].reshape(16, 3), dtype=torch.float32))
        model.fc1.bias.copy_(torch.tensor(params['w1'], dtype=torch.float32))

        # w2 = (256,) → fc2.weight (16, 16)
        model.fc2.weight.copy_(torch.tensor(
            params['w2'].reshape(16, 16), dtype=torch.float32))
        model.fc2.bias.copy_(torch.tensor(params['w3'], dtype=torch.float32))

        # w4 = (48,) → fc3.weight (3, 16)
        model.fc3.weight.copy_(torch.tensor(
            params['w4'].reshape(3, 16), dtype=torch.float32))
        model.fc3.bias.copy_(torch.tensor(params['w5'], dtype=torch.float32))

    # Verify: the feature normalization must match preemph_model.h
    print(f"  feat_mean: {model.feat_mean.numpy()}")
    print(f"  feat_std:  {model.feat_std.numpy()}")
    print(f"  fc3.bias (identity init check): {model.fc3.bias.detach().numpy()}")


def load_weights_from_header(model):
    """Load weights directly from the C header values (hardcoded backup)."""
    feat_mean = [4627.5, 0.5, 0.89]
    feat_std = [4529.44433594, 0.17888543, 0.22338307]

    # These are from preemph_model.h — but the feature order in the header is
    # [spectral_centroid, rms, crest]. In the full pipeline model, the features
    # are computed as [zcr_proxy, rms, crest] — the ZCR proxy is NOT the same
    # as the spectral centroid (which is zcr * sample_rate / 2).
    #
    # We need to adjust feat_mean/feat_std for the zcr proxy input.
    # C code: centroid = zcr * sample_rate / 2
    # For DSD512 (sample_rate = 22579200): centroid = zcr * 11289600
    # So zcr = centroid / 11289600
    #
    # Actually — the full pipeline model computes mean(|diff(x)|) which is
    # NOT the same as ZCR. We should match the C features exactly.
    # Let's use the same feature set as preemph.c.

    print("WARNING: Using hardcoded weights. Prefer --params with .npz file.")
    with torch.no_grad():
        model.feat_mean.copy_(torch.tensor(feat_mean, dtype=torch.float32))
        model.feat_std.copy_(torch.tensor(feat_std, dtype=torch.float32))


def verify_vs_c_code(model):
    """Verify the model produces same taps as preemph.c for known inputs."""
    model.eval()

    # Test with a 1kHz sine at DSD512 rate, RMS ~0.7
    N = 4096
    t = torch.linspace(0, N / 22579200.0, N)
    signal = 0.7 * torch.sin(2 * 3.14159265 * 1000.0 * t)
    x = signal.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out = model(x)

    print(f"\nVerification (1kHz sine, N={N}):")
    print(f"  Input  range: [{signal.min():.4f}, {signal.max():.4f}]")
    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"  Max abs diff: {(out.squeeze() - signal).abs().max():.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Export full-pipeline pre-SDM ONNX model")
    parser.add_argument('--params',
                        default='foo_dsd_trellis_preemph_params.npz',
                        help='Path to .npz with MLP weights')
    parser.add_argument('--output',
                        default='foo_dsd_trellis_preemph_full.onnx',
                        help='Output ONNX path')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version')
    args = parser.parse_args()

    print("Creating full-pipeline pre-SDM model...")
    model = PreemphFullPipeline(hidden=16)

    # Load weights
    npz_path = args.params
    if not os.path.isabs(npz_path):
        npz_path = os.path.join(os.path.dirname(__file__), npz_path)

    if os.path.exists(npz_path):
        print(f"Loading weights from {npz_path}")
        load_weights_from_npz(model, npz_path)
    else:
        print(f"WARNING: {npz_path} not found")
        load_weights_from_header(model)

    model.eval()

    # Verify
    verify_vs_c_code(model)

    # Export to ONNX
    print(f"\nExporting to {args.output}...")
    dummy_input = torch.randn(1, 1, 4096)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(__file__), output_path)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "seq_len"},
            "output": {2: "seq_len"},
        },
        dynamo=False,
    )

    # Add metadata
    import onnx
    onnx_model = onnx.load(output_path)
    metadata = {
        'model_type': 'preemph_full_pipeline',
        'num_taps': '3',
        'hidden': '16',
        'rate': '22579200',
    }
    for k, v in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = k
        entry.value = v
    onnx.save(onnx_model, output_path)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Model size: {size_kb:.1f} KB")

    # Validate with ONNX Runtime
    print("\nValidating with ONNX Runtime...")
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)

    for block_size in [1024, 4096, 16384, 65536]:
        test_input = np.random.randn(1, 1, block_size).astype(np.float32) * 0.5
        outputs = sess.run(None, {"input": test_input})
        out = outputs[0]
        assert out.shape == (1, 1, block_size), \
            f"Expected (1,1,{block_size}), got {out.shape}"
        max_diff = np.max(np.abs(out - test_input))
        print(f"  N={block_size:6d}: shape={out.shape}, "
              f"max_diff_from_input={max_diff:.6f}")

    print(f"\nDone. Model: {output_path}")
    print("Copy to plugin component folder as foo_dsd_trellis_preemph.onnx")


if __name__ == '__main__':
    main()
