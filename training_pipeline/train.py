"""
foo_dsd_trellis — Training script V3: Temperature-annealed quantization.

Key design:
  1. Model outputs tanh(x * T) where T increases from 1 → T_max over training
  2. At T=1: smooth gradients, model learns audio-band structure
  3. At T=100: output ≈ ±1.0, model learns to make good binary decisions
  4. Loss: lowpass MSE (audio band) — same for all temperatures
  5. Validation measures both continuous and quantized (sign) output quality

Usage:
    python train.py --data I:/foo_dsd_trellis/data/pairs --epochs 500 --batch 128
    python train.py --data I:/foo_dsd_trellis/data/pairs --large --epochs 500
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from scipy.signal import firwin

from model import build_model, CausalDilatedCNN, CausalDilatedCNNLarge, NonCausalDilatedCNN
from dataset import create_dataloaders


class AudioBandLoss(nn.Module):
    """Lowpass-filtered MSE in the audio band.

    Works with both soft (tanh) and hard (sign) model outputs.
    The lowpass focuses optimization on the audio band where it matters.
    """

    def __init__(self, fs: int = 2822400, cutoff: float = 24000.0,
                 num_taps: int = 127):
        super().__init__()
        nyq = fs / 2.0
        taps = firwin(num_taps, cutoff / nyq, window=('kaiser', 8.0),
                      pass_zero=True)
        kernel = torch.from_numpy(taps.astype(np.float32)).reshape(1, 1, -1)
        self.register_buffer('kernel', kernel)
        self.pad = num_taps // 2

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv1d(x, self.kernel, padding=self.pad)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lp = self._lowpass(pred)
        target_lp = self._lowpass(target)
        return nn.functional.mse_loss(pred_lp, target_lp)


def temperature_schedule(epoch: int, total_epochs: int,
                         t_start: float = 1.0, t_end: float = 100.0) -> float:
    """Exponential temperature annealing.

    Ramps from t_start to t_end over training.
    First 10% of epochs: stay at t_start (let model learn structure)
    Next 80%: exponential ramp
    Last 10%: stay at t_end (fine-tune at near-binary)
    """
    warmup = int(total_epochs * 0.1)
    cooldown_start = int(total_epochs * 0.9)

    if epoch < warmup:
        return t_start
    if epoch >= cooldown_start:
        return t_end

    # Exponential interpolation
    progress = (epoch - warmup) / (cooldown_start - warmup)
    log_t = math.log(t_start) + progress * (math.log(t_end) - math.log(t_start))
    return math.exp(log_t)


def trim_prediction(pred, x, y, noncausal=False):
    """Trim model prediction to match target y length.

    Causal: model prepends (RF-1) history samples → trim from left.
    Non-causal: model uses symmetric context → trim from both sides.
    """
    excess = pred.shape[2] - y.shape[2]
    if excess == 0:
        return pred
    if noncausal:
        left = excess // 2
        right = excess - left
        return pred[:, :, left:pred.shape[2] - right]
    else:
        return pred[:, :, excess:]


def train_epoch(model, dataset, batch_size, n_batches, optimizer, criterion,
                device, noncausal=False):
    model.train()
    total_loss = 0.0

    for _ in range(n_batches):
        x, y = dataset.generate_batch(batch_size)

        pred = model(x)
        pred_trimmed = trim_prediction(pred, x, y, noncausal)

        loss = criterion(pred_trimmed, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_batches


@torch.no_grad()
def validate(model, dataset, batch_size, n_batches, criterion, device,
             noncausal=False):
    model.eval()
    total_loss = 0.0
    total_input_mse = 0.0
    total_soft_mse = 0.0
    total_hard_mse = 0.0
    count = 0

    for _ in range(n_batches):
        x, y = dataset.generate_batch(batch_size)

        pred = model(x)
        pred_trimmed = trim_prediction(pred, x, y, noncausal)

        loss = criterion(pred_trimmed, y)
        total_loss += loss.item()

        # For input comparison, trim x the same way
        excess = x.shape[2] - y.shape[2]
        if noncausal:
            left = excess // 2
            x_trimmed = x[:, :, left:left + y.shape[2]]
        else:
            x_trimmed = x[:, :, excess:]

        input_lp = criterion._lowpass(x_trimmed)
        ref_lp = criterion._lowpass(y)
        soft_lp = criterion._lowpass(pred_trimmed)
        hard_lp = criterion._lowpass(torch.sign(pred_trimmed))

        total_input_mse += nn.functional.mse_loss(input_lp, ref_lp).item()
        total_soft_mse += nn.functional.mse_loss(soft_lp, ref_lp).item()
        total_hard_mse += nn.functional.mse_loss(hard_lp, ref_lp).item()
        count += 1

    avg_loss = total_loss / max(n_batches, 1)
    input_mse = total_input_mse / max(count, 1)
    soft_mse = total_soft_mse / max(count, 1)
    hard_mse = total_hard_mse / max(count, 1)

    soft_db = (10.0 * math.log10(input_mse / max(soft_mse, 1e-30))
               if input_mse > 0 else 0.0)
    hard_db = (10.0 * math.log10(input_mse / max(hard_mse, 1e-30))
               if input_mse > 0 else 0.0)
    return avg_loss, soft_db, hard_db


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path,
                    large=False, noncausal=False):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'large': large,
        'noncausal': noncausal,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('val_loss', float('inf'))


def main():
    parser = argparse.ArgumentParser(
        description="Train DSD noise reduction model (V3: temp annealing)")
    parser.add_argument("--data", default="data/pairs",
                        help="Training pairs directory")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--block-size", type=int, default=4096,
                        help="Training block size (default: 4096)")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--rates", nargs="+", default=None,
                        help="DSD rates to train on (default: all)")
    parser.add_argument("--log-dir", default="runs",
                        help="TensorBoard log directory")
    parser.add_argument("--large", action="store_true",
                        help="Use large model (10 layers, RF=2047)")
    parser.add_argument("--noncausal", action="store_true",
                        help="Use non-causal model (12 layers, RF=8191, bidirectional)")
    parser.add_argument("--t-start", type=float, default=1.0,
                        help="Starting temperature (default: 1.0)")
    parser.add_argument("--t-end", type=float, default=100.0,
                        help="Final temperature (default: 100.0)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Dataset stride (default: block_size // 2)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    noncausal = args.noncausal
    model = build_model(large=args.large, noncausal=noncausal).to(device)
    rf = model.receptive_field()
    history = rf - 1
    if noncausal:
        model_name = "Non-causal"
    elif args.large:
        model_name = "Large"
    else:
        model_name = "Compact"
    print(f"Model: {model_name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Receptive field: {rf} samples ({rf / 2822400 * 1e6:.0f} us at DSD64)")
    if noncausal:
        la = model.look_ahead()
        print(f"  Look-ahead: {la} samples ({la / 2822400 * 1e6:.0f} us)")
    print(f"  History: {history} samples")
    print(f"  Temperature: {args.t_start} -> {args.t_end}")

    stride = args.stride or args.block_size // 2
    train_ds, val_ds, train_batches, val_batches = create_dataloaders(
        args.data, block_size=args.block_size,
        batch_size=args.batch,
        dsd_rates=args.rates, device=device,
        history=history,
        stride=stride)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = AudioBandLoss(fs=2822400, cutoff=24000.0, num_taps=127).to(device)

    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}, val_loss={best_val_loss:.6f}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    log_path = Path("training_progress.log")
    log_file = open(log_path, "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\nTraining V4 (temp annealing) for {args.epochs} epochs")
    log(f"Model: {model_name}, "
        f"params={model.count_parameters():,}, RF={rf}")
    log(f"Temperature: {args.t_start} -> {args.t_end}")
    log(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Soft':>7} {'Hard':>7} "
        f"{'Temp':>7} {'LR':>10} {'Time':>6}")
    log("-" * 72)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Set temperature for this epoch
        temp = temperature_schedule(epoch, args.epochs, args.t_start, args.t_end)
        model.set_temperature(temp)

        train_loss = train_epoch(model, train_ds, args.batch, train_batches,
                                 optimizer, criterion, device, noncausal)
        val_loss, soft_db, hard_db = validate(model, val_ds, args.batch,
                                                val_batches, criterion, device,
                                                noncausal)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        log(f"{epoch+1:>6} {train_loss:>10.6f} {val_loss:>10.6f} "
            f"{soft_db:>+6.1f}dB {hard_db:>+6.1f}dB "
            f"{temp:>7.1f} {lr:>10.2e} {elapsed:>5.1f}s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/soft_improvement_db", soft_db, epoch)
        writer.add_scalar("Metrics/hard_improvement_db", hard_db, epoch)
        writer.add_scalar("Temperature", temp, epoch)
        writer.add_scalar("LR", lr, epoch)

        save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss,
                        ckpt_dir / "latest.pt", args.large, noncausal)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss,
                            ckpt_dir / "best.pt", args.large, noncausal)
            log(f"       ^ new best (val_loss={best_val_loss:.6f})")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss,
                            ckpt_dir / f"epoch_{epoch+1:04d}.pt", args.large,
                            noncausal)

    writer.close()
    log_file.close()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Best model: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
