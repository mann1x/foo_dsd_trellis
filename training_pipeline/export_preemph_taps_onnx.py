"""
Export tap-prediction ONNX model for GPU inference.

Lightweight model: signal features → MLP → 3 taps.
The FIR apply step stays on CPU (trivial for 3-tap).

Input:  (1, 3) float32 — [spectral_centroid, rms, crest_factor]
Output: (1, 3) float32 — [tap0, tap1, tap2] (DC-gain-normalized)

Usage:
  python export_preemph_taps_onnx.py
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn


class PreemphTapPredictor(nn.Module):
    """MLP only: signal features → FIR taps with DC gain normalization."""

    def __init__(self, hidden=16):
        super().__init__()
        self.register_buffer('feat_mean', torch.zeros(3))
        self.register_buffer('feat_std', torch.ones(3))
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 3)

    def forward(self, features):
        """
        features: (1, 3) float32 — [centroid, rms, crest]
        Returns: (1, 3) float32 — [tap0, tap1, tap2] normalized
        """
        normalized = (features - self.feat_mean) / self.feat_std
        h = torch.relu(self.fc1(normalized))
        h = torch.relu(self.fc2(h))
        taps = self.fc3(h)

        # DC gain normalization
        dc_gain = torch.sum(taps, dim=1, keepdim=True)
        dc_gain = torch.clamp(dc_gain, min=0.01)
        taps = taps / dc_gain

        return taps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='foo_dsd_trellis_preemph_params.npz')
    parser.add_argument('--output', default='foo_dsd_trellis_preemph_taps.onnx')
    args = parser.parse_args()

    model = PreemphTapPredictor(hidden=16)

    npz_path = args.params
    if not os.path.isabs(npz_path):
        npz_path = os.path.join(os.path.dirname(__file__), npz_path)

    params = np.load(npz_path)
    with torch.no_grad():
        model.feat_mean.copy_(torch.tensor(params['feat_mean']))
        model.feat_std.copy_(torch.tensor(params['feat_std']))
        model.fc1.weight.copy_(torch.tensor(params['w0'].reshape(16, 3)))
        model.fc1.bias.copy_(torch.tensor(params['w1']))
        model.fc2.weight.copy_(torch.tensor(params['w2'].reshape(16, 16)))
        model.fc2.bias.copy_(torch.tensor(params['w3']))
        model.fc3.weight.copy_(torch.tensor(params['w4'].reshape(3, 16)))
        model.fc3.bias.copy_(torch.tensor(params['w5']))

    model.eval()

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(__file__), output_path)

    dummy = torch.tensor([[5000.0, 0.5, 1.4]])
    torch.onnx.export(
        model, dummy, output_path,
        input_names=['features'], output_names=['taps'],
        dynamic_axes={'features': {0: 'batch'}},
        opset_version=13, dynamo=False,
    )

    import onnx
    m = onnx.load(output_path)
    for k, v in {'model_type': 'preemph_taps', 'num_taps': '3'}.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, output_path)

    print(f"Model: {output_path} ({os.path.getsize(output_path)} bytes)")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    test_features = np.array([[5000.0, 0.5, 1.4]], dtype=np.float32)
    taps = sess.run(None, {'features': test_features})[0]
    print(f"Test: features={test_features[0]} -> taps={taps[0]} (sum={taps[0].sum():.4f})")


if __name__ == '__main__':
    main()
