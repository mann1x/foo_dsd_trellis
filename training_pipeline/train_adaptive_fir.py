"""
Train adaptive FIR model from CMA-ES training data.

Input: signal features (spectral centroid, RMS, crest factor)
Output: optimal 3-tap FIR pre-emphasis coefficients

Model: tiny MLP (~50 params) → export to ONNX for C runtime.

Usage:
  python -u train_adaptive_fir.py --data I:/foo_dsd_trellis/checkpoints/cmaes/training_data_dsd512.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class AdaptiveFIR(nn.Module):
    """Tiny MLP: signal features → FIR taps.

    Input: (batch, 3) — [spectral_centroid_hz, rms, crest_factor]
    Output: (batch, num_taps) — FIR pre-emphasis coefficients
    """

    def __init__(self, num_features=3, num_taps=3, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_taps),
        )

        # Initialize to output identity FIR [1.0, 0.0, 0.0]
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
            self.net[-1].bias[0] = 1.0

    def forward(self, features):
        return self.net(features)


def train(args):
    # Load training data
    data = np.load(args.data)
    features = data['features'].astype(np.float32)
    taps = data['taps'].astype(np.float32)
    names = data['names']
    num_taps = int(data['num_taps'])

    num_features = features.shape[1]
    print(f"Training data: {len(features)} examples, {num_features} features, {num_taps} taps")
    print(f"Feature ranges:")
    feat_names = ['centroid', 'rms', 'crest', 'rate_mhz'][:num_features]
    for i, fn in enumerate(feat_names):
        print(f"  {fn}: [{features[:,i].min():.3f} - {features[:,i].max():.3f}]")

    # Normalize features for training
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0
    features_norm = (features - feat_mean) / feat_std

    X = torch.tensor(features_norm, dtype=torch.float32)
    Y = torch.tensor(taps, dtype=torch.float32)

    # Model
    model = AdaptiveFIR(num_features=features.shape[1], num_taps=num_taps, hidden=args.hidden)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params} parameters (hidden={args.hidden})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                pred_np = pred.numpy()
                max_err = np.abs(pred_np - taps).max()
                mean_err = np.abs(pred_np - taps).mean()
            print(f"  [{epoch:4d}] loss={loss.item():.6f} "
                  f"max_err={max_err:.4f} mean_err={mean_err:.4f}")

    # Evaluate per-example
    model.eval()
    print(f"\nPer-example predictions:")
    with torch.no_grad():
        pred = model(X).numpy()
    for i in range(len(names)):
        tgt = taps[i]
        prd = pred[i]
        err = np.abs(tgt - prd).max()
        print(f"  {names[i]:20s}: target=[{', '.join(f'{t:.4f}' for t in tgt)}] "
              f"pred=[{', '.join(f'{t:.4f}' for t in prd)}] err={err:.4f}")

    # Export to ONNX
    model.eval()
    dummy = torch.randn(1, features.shape[1])
    onnx_path = args.output

    # Wrap model with normalization baked in
    class ExportModel(nn.Module):
        def __init__(self, model, feat_mean, feat_std):
            super().__init__()
            self.model = model
            self.register_buffer('feat_mean', torch.tensor(feat_mean, dtype=torch.float32))
            self.register_buffer('feat_std', torch.tensor(feat_std, dtype=torch.float32))

        def forward(self, raw_features):
            normalized = (raw_features - self.feat_mean) / self.feat_std
            return self.model(normalized)

    export_model = ExportModel(model, feat_mean, feat_std)
    export_model.eval()

    dummy_raw = torch.randn(1, features.shape[1])  # match feature count
    # Use legacy exporter to avoid Unicode issues on Windows
    torch.onnx.export(
        export_model, dummy_raw, onnx_path,
        input_names=['features'],
        output_names=['taps'],
        dynamic_axes={'features': {0: 'batch'}},
        opset_version=13,
        dynamo=False,
    )

    # Add metadata
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(key='model_type', value='preemph_taps'))
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(key='num_taps', value=str(num_taps)))
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(key='num_features', value=str(features.shape[1])))
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(key='rate', value=str(int(data['rate']))))
    onnx.save(onnx_model, onnx_path)

    print(f"\nExported to {onnx_path}")
    print(f"Model size: {os.path.getsize(onnx_path)} bytes")

    # Also save the normalization params and taps for C integration
    params = {f'w{i}': p.detach().numpy() for i, p in enumerate(model.parameters())}
    params['feat_mean'] = feat_mean
    params['feat_std'] = feat_std
    np.savez(onnx_path.replace('.onnx', '_params.npz'), **params)
    print(f"Saved params to {onnx_path.replace('.onnx', '_params.npz')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='../training_pipeline/foo_dsd_trellis_preemph.onnx')
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
