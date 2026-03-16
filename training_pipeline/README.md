# Training Pipeline — foo_dsd_trellis ML Noise Filter

Trains a causal dilated CNN to reduce SDM quantization noise in DSD bitstreams.

## Prerequisites

- NVIDIA GPU with CUDA support (tested: RTX 5080, 16 GB VRAM)
- Conda (Miniconda or Anaconda)
- Built `foo_dsd_trellis_test.exe` (Release x64)

## Environment Setup

```bash
# Windows
setup_env.bat

# Or manually:
conda create -n foo_dsd_trellis python=3.11 -y
conda activate foo_dsd_trellis
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install numpy scipy soundfile librosa onnx onnxruntime onnxruntime-directml onnxscript matplotlib tqdm tensorboard
```

Note: PyTorch nightly with CUDA 12.8 is required for RTX 50-series (Blackwell/sm_120) GPUs.

## Pipeline Steps

### 1. Generate Synthetic Signals

```bash
conda activate foo_dsd_trellis
cd training_pipeline
python generate_signals.py
```

Creates 171 WAV files (57 signal types x 3 PCM rates: 44100, 88200, 176400 Hz) in `data/synthetic/`. Signal types include sines, multitone, sweeps, noise, bursts, IMD/TIM patterns.

### 2. Generate Training Pairs

Uses the real C encoder (`foo_dsd_trellis_test.exe --encode`) to produce paired (DSD_input, FIR_reference) data across SDM configurations.

```bash
# DSD64, all 23 configs, 2s max duration (fast, ~11 min)
python generate_pairs.py --rates DSD64 --max-duration 2

# DSD64, specific configs only
python generate_pairs.py --rates DSD64 --configs trellis_clans8 precorr

# All rates (slower)
python generate_pairs.py --rates DSD64 DSD128 DSD256 DSD512

# With real music files
python generate_pairs.py --rates DSD64 --music /path/to/song.flac /path/to/other.wav

# List available SDM configs
python generate_pairs.py --list-configs
```

Output: `data/pairs/<DSD_RATE>/<CONFIG_NAME>/<SIGNAL_NAME>/` with `dsd_input.npy` and `fir_reference.npy`.

23 SDM configs: 15 Trellis variants (different NTF filters, candidates, latency, state limiter, gain) + 5 PreCorr variants.

### 3. Train

```bash
# Compact model (recommended, fast inference)
python train.py --data data/pairs --epochs 500 --batch 128

# Large model (more context, slower inference)
python train.py --data data/pairs --epochs 500 --batch 128 --large

# Resume from checkpoint
python train.py --data data/pairs --epochs 500 --resume checkpoints/latest.pt

# Tune loss weights
python train.py --data data/pairs --spectral-weight 1.0 --time-weight 0.5
```

**Key parameters:**
- `--batch 128`: Batch size
- `--lr 0.0003`: Initial learning rate (cosine annealing to 1e-6)
- `--block-size 4096`: Training block size in samples
- `--rates DSD64`: Filter by DSD rate
- `--large`: Use 10-layer model (RF=2047) instead of 5-layer (RF=63)
- `--spectral-weight`, `--time-weight`: Loss component weights
- `--stride`: Dataset block stride (default: block_size/2, 50% overlap)

**Monitoring:**
- Progress: `cat training_progress.log` (updated every epoch)
- TensorBoard: `tensorboard --logdir runs`
- GPU: `nvidia-smi`

### 4. Export to ONNX

```bash
python export_onnx.py --checkpoint checkpoints/best.pt --output foo_dsd_trellis_ml.onnx
```

Validates the exported model with ONNX checker and ONNX Runtime inference at multiple block sizes. Embeds `receptive_field` in ONNX model metadata so the C runtime knows the correct history size.

### 5. Deploy

Copy the ONNX model next to the plugin DLL:

```bash
cp foo_dsd_trellis_ml.onnx /path/to/foobar2000/user-components-x64/foo_dsd_trellis/
```

Also requires `onnxruntime.dll` in the same directory or system PATH.

## Model Architecture (V2)

Two model sizes available, both using **STE sign()** for ±1.0 output during training and inference:

### Compact (default)

- 5 layers: dilations [1, 2, 4, 8, 16], kernel_size=3, hidden_channels=32
- Residual blocks with GroupNorm
- Receptive field: 63 samples (~22 us at DSD64)
- Parameters: ~5,000
- Fast inference, suitable for real-time DSD processing

### Large (`--large`)

- 10 layers: dilations [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], kernel_size=3, hidden_channels=32
- Residual blocks with GroupNorm
- Receptive field: 2047 samples (~725 us at DSD64)
- Parameters: ~25,000
- More temporal context, heavier inference cost

### I/O Contract

- Input: `[1, 1, N]` float32 (±1.0 DSD samples with causal history prepended)
- Output: `[1, 1, N]` float32 (±1.0 refined DSD — already quantized by STE sign())
- ONNX metadata: `receptive_field` key stores the model's RF (C runtime reads this)

## V1 → V2 Changes (Why V1 Failed)

V1 produced a -62 dB SINAD regression. Three root causes were identified and fixed:

### 1. History/Receptive Field Mismatch

V1 dataset hardcoded `history=30` samples, but the model had `RECEPTIVE_FIELD=2047`. During training, the model saw zero-padded context for most of its receptive field. During inference in the C plugin, `onnx_filter.c` provided the full 2046 samples of real history. This train/inference mismatch meant the model operated in a completely different regime at deployment.

**Fix:** Dataset `history` parameter now matches the model's actual receptive field.

### 2. Train/Deploy Output Mismatch (tanh vs sign)

V1 model output continuous values via `tanh [-1, +1]` during training, optimized with lowpass MSE against the multi-bit FIR reference. At inference, hard `sign()` quantization converted these to ±1.0. The sign() operation destroyed the nuanced continuous predictions the model learned — the model never trained for the actual binary output it would produce.

**Fix:** V2 uses **Straight-Through Estimator (STE)** — `sign()` in the forward pass with identity gradient in the backward pass. The model outputs ±1.0 during both training and inference. No mismatch.

### 3. Loss Function

V1's lowpass MSE in time domain didn't correlate with SINAD after sign() quantization. A model could achieve low lowpass MSE with continuous tanh values but produce terrible spectral characteristics after hard quantization.

**Fix:** V2 uses a **CombinedLoss**:
- **SpectralLoss** (primary): FFT-based MSE on magnitude spectrum in the audio band (0-24kHz). Phase-tolerant, directly penalizes noise energy where it matters.
- **Time-domain lowpass MSE** (secondary): Weighted complement for sample-level guidance.
- SINAD-like SNR validation metric computed every 5 epochs for monitoring.

### 4. Model Architecture

V1 used plain sequential convolutions with no skip connections. V2 uses **residual blocks** (conv + GroupNorm + skip connection) for more stable gradient flow, especially important with STE gradients.

## File Overview

| File | Purpose |
|------|---------|
| `setup_env.bat` | Create conda environment |
| `environment.yml` | Conda env spec (reference) |
| `generate_signals.py` | Generate synthetic test signals |
| `generate_pairs.py` | Generate (DSD, FIR) training pairs via C encoder |
| `dataset.py` | GPU-preloaded dataset with pre-sliced blocks |
| `model.py` | CausalDilatedCNN (compact + large) with STE sign() |
| `train.py` | Training loop with spectral loss, cosine LR, TensorBoard |
| `export_onnx.py` | Export to ONNX with metadata + validation |
