"""
foo_dsd_trellis — Export trained model to ONNX format.

Produces foo_dsd_trellis_ml.onnx compatible with the C plugin's
onnx_filter.c runtime loader.

Model contract:
  Input:  "input"  — [1, 1, N] float32 (N = history + block_size, variable)
  Output: "output" — [1, 1, N] float32 (same length, values are ±1.0)

Usage:
    python export_onnx.py --checkpoint checkpoints/best.pt --output ../bin/Release/x64/foo_dsd_trellis_ml.onnx
    python export_onnx.py --checkpoint checkpoints/best.pt  # outputs to foo_dsd_trellis_ml.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort

from model import build_model, CausalDilatedCNN, CausalDilatedCNNLarge, NonCausalDilatedCNN


def export_onnx(checkpoint_path: str, output_path: str,
                opset_version: int = 18):
    """Export trained model to ONNX."""

    # Detect model type from checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    large = ckpt.get('large', False)
    noncausal = ckpt.get('noncausal', False)

    model = build_model(large=large, noncausal=noncausal)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    rf = model.receptive_field()
    history = rf - 1
    look_ahead = model.look_ahead() if hasattr(model, 'look_ahead') else 0

    if noncausal:
        model_name = "Non-causal"
    elif large:
        model_name = "Large"
    else:
        model_name = "Compact"

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Model: {model_name}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Val loss: {ckpt.get('val_loss', '?')}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Receptive field: {rf} samples")
    if noncausal:
        print(f"  Look-ahead: {look_ahead} samples")

    # Dynamic input length (history + variable block_size)
    dummy_block = 4096
    dummy_input = torch.randn(1, 1, history + dummy_block)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "seq_len"},
            "output": {2: "seq_len"},
        },
    )

    # Add metadata so C runtime knows history size and model type
    onnx_model = onnx.load(output_path)
    meta = onnx_model.metadata_props.add()
    meta.key = "receptive_field"
    meta.value = str(rf)
    if noncausal:
        meta2 = onnx_model.metadata_props.add()
        meta2.key = "look_ahead"
        meta2.value = str(look_ahead)
        meta3 = onnx_model.metadata_props.add()
        meta3.key = "noncausal"
        meta3.value = "1"
    onnx.save(onnx_model, output_path)

    print(f"\nExported ONNX model: {output_path}")
    print(f"  Metadata: receptive_field={rf}")
    if noncausal:
        print(f"  Metadata: look_ahead={look_ahead}, noncausal=1")

    # Validate
    onnx_model = onnx.load(output_path)  # reload with metadata
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation: OK")

    # File size
    size_kb = Path(output_path).stat().st_size / 1024
    print(f"Model size: {size_kb:.1f} KB")

    # Verify with ONNX Runtime
    print("\nVerifying with ONNX Runtime...")
    sess = ort.InferenceSession(output_path)

    # Test with different input lengths
    for block_size in [256, 1024, 4096, 8192]:
        seq_len = history + block_size
        test_input = np.random.randn(1, 1, seq_len).astype(np.float32)
        test_input = np.sign(test_input)  # ±1.0 DSD-like

        outputs = sess.run(None, {"input": test_input})
        out = outputs[0]

        assert out.shape == (1, 1, seq_len), \
            f"Expected (1,1,{seq_len}), got {out.shape}"

        # Verify output is ±1.0 (STE model)
        unique_vals = np.unique(out)
        is_binary = all(v in [-1.0, 0.0, 1.0] for v in unique_vals)

        print(f"  block_size={block_size}: input={test_input.shape} -> "
              f"output={out.shape}, binary={is_binary}, "
              f"unique_vals={unique_vals.tolist()[:5]}")

    # Compare PyTorch vs ONNX output
    print("\nPyTorch vs ONNX numerical check...")
    test_in = np.sign(np.random.randn(1, 1, history + 4096)).astype(np.float32)
    torch_out = model(torch.from_numpy(test_in)).detach().numpy()
    onnx_out = sess.run(None, {"input": test_in})[0]
    max_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"  Max absolute difference: {max_diff:.2e}")
    assert max_diff < 1e-3, f"ONNX output diverges from PyTorch: {max_diff}"
    print("  Match: OK")

    print(f"\nModel ready for deployment: {output_path}")
    print("Copy to plugin directory as foo_dsd_trellis_ml.onnx")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained model to ONNX")
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--output", "-o", default="foo_dsd_trellis_ml.onnx",
                        help="Output ONNX path (default: foo_dsd_trellis_ml.onnx)")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (default: 18)")
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
