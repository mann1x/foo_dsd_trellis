"""
Train pre-SDM enhancement model end-to-end through differentiable SDM.

Pipeline:
  reference signal → boxcar → [learned pre-emphasis] → diff SDM → audio-band loss
  ↑ known                     ↑ trainable              ↑ fixed     ↑ gradient

The model learns to produce an input signal that, when fed through
the SDM, minimizes audio-band noise relative to the reference.
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from diff_sdm import DiffSDM, AudioBandLoss
from model_pre_sdm import create_model


def generate_boxcar_from_sine(dsd_rate, freq_hz, amplitude, n_samples, box_taps, gain=0.708):
    """Generate a test signal: sine → DSD-like boxcar output.

    For training, we use the clean sine as reference and the boxcar
    output (simulating DSD→boxcar) as model input.
    """
    t = torch.arange(n_samples, dtype=torch.float64) / dsd_rate
    reference = amplitude * torch.sin(2 * math.pi * freq_hz * t)

    # Simulate boxcar smoothing of a DSD bitstream
    # In production: DSD(±1) → boxcar → multi-bit
    # For training: we use the reference directly as the "ideal" boxcar output
    # and add boxcar-like frequency response characteristics

    # Simple boxcar: running average of the reference (simulates the smoothing)
    kernel = torch.ones(box_taps, dtype=torch.float64) / box_taps
    padded = torch.nn.functional.pad(reference.unsqueeze(0).unsqueeze(0),
                                      (box_taps - 1, 0))
    boxcar_out = torch.nn.functional.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0))
    boxcar_out = boxcar_out.squeeze() * gain

    return boxcar_out, reference


def generate_training_batch(batch_size, seq_len, dsd_rate, box_taps, gain=0.708):
    """Generate a batch of diverse training signals."""
    signals_in = []
    signals_ref = []

    for _ in range(batch_size):
        # Random frequency 20-15000 Hz
        freq = 20 + torch.rand(1).item() * 14980
        # Random amplitude 0.1-0.5
        amp = 0.1 + torch.rand(1).item() * 0.4

        boxcar, ref = generate_boxcar_from_sine(dsd_rate, freq, amp, seq_len, box_taps, gain)
        signals_in.append(boxcar)
        signals_ref.append(ref)

    return torch.stack(signals_in), torch.stack(signals_ref)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # DSD rate config
    dsd_rates = {
        'dsd64': 2822400,
        'dsd128': 5644800,
        'dsd256': 11289600,
        'dsd512': 22579200,
    }
    box_taps = {
        'dsd64': 32,
        'dsd128': 64,
        'dsd256': 64,
        'dsd512': 16,
    }

    rate_name = args.rate
    dsd_rate = dsd_rates[rate_name]
    btaps = box_taps[rate_name]

    print(f"\nTraining pre-SDM model for {rate_name.upper()}")
    print(f"  DSD rate: {dsd_rate} Hz")
    print(f"  Boxcar taps: {btaps}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model: {args.model_type}, {args.num_taps} taps")

    # Create model
    model_kwargs = {'num_taps': args.num_taps}
    if args.model_type == 'adaptive':
        model_kwargs['block_size'] = 256
        model_kwargs['hidden'] = 16
    model = create_model(args.model_type, **model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params}")

    # Create differentiable SDM
    ntf = 'clans-6' if rate_name in ('dsd64', 'dsd128', 'dsd256') else 'sdm-6'
    sdm = DiffSDM(ntf_name=ntf, temperature=args.temperature).to(device)
    print(f"  NTF: {ntf}, temperature: {args.temperature}")

    # Loss function
    loss_fn = AudioBandLoss(dsd_rate=dsd_rate).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        # Temperature annealing
        if args.anneal_temp:
            progress = epoch / max(args.epochs - 1, 1)
            if progress < 0.1:
                T = 1.0
            elif progress < 0.9:
                T = 1.0 * (args.temperature / 1.0) ** ((progress - 0.1) / 0.8)
            else:
                T = args.temperature
            sdm.temperature = T

        # Generate training batch
        boxcar_in, reference = generate_training_batch(
            args.batch_size, args.seq_len, dsd_rate, btaps
        )
        boxcar_in = boxcar_in.to(device)
        reference = reference.to(device)

        # Forward: boxcar → pre-emphasis → SDM → loss
        model.train()
        optimizer.zero_grad()

        pre_emphasized = model(boxcar_in)
        sdm_output = sdm(pre_emphasized)
        loss = loss_fn(sdm_output, reference)

        # Also compute baseline loss (no pre-emphasis)
        with torch.no_grad():
            sdm_baseline = sdm(boxcar_in)
            baseline_loss = loss_fn(sdm_baseline, reference)

        # Improvement in dB
        if baseline_loss.item() > 0 and loss.item() > 0:
            improvement_db = 10 * math.log10(baseline_loss.item() / loss.item())
        else:
            improvement_db = 0.0

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        dt = time.time() - t0

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            if args.checkpoint_dir:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'model_type': args.model_type,
                    'num_taps': args.num_taps,
                    'rate': rate_name,
                    'loss': best_loss,
                    'improvement_db': improvement_db,
                }, os.path.join(args.checkpoint_dir, f'best_{rate_name}.pt'))

        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            # Print learned taps
            if hasattr(model, 'taps'):
                taps_str = ', '.join(f'{t:.6f}' for t in model.taps.detach().cpu().tolist())
            elif hasattr(model, 'base_taps'):
                taps_str = ', '.join(f'{t:.6f}' for t in model.base_taps.detach().cpu().tolist())
            else:
                taps_str = 'N/A'

            print(f"  [{epoch:4d}/{args.epochs}] loss={loss.item():.2e} "
                  f"baseline={baseline_loss.item():.2e} "
                  f"delta={improvement_db:+.2f} dB  "
                  f"T={sdm.temperature:.1f} lr={scheduler.get_last_lr()[0]:.1e} "
                  f"({dt:.1f}s)")
            if epoch % (args.log_interval * 5) == 0:
                print(f"         taps=[{taps_str}]")

    print(f"\nBest: epoch {best_epoch}, loss={best_loss:.2e}")
    if hasattr(model, 'taps'):
        print(f"Learned taps: {model.taps.detach().cpu().tolist()}")
    elif hasattr(model, 'base_taps'):
        print(f"Learned base_taps: {model.base_taps.detach().cpu().tolist()}")


def main():
    parser = argparse.ArgumentParser(description='Train pre-SDM enhancement model')
    parser.add_argument('--rate', default='dsd512',
                        choices=['dsd64', 'dsd128', 'dsd256', 'dsd512'])
    parser.add_argument('--model-type', default='fixed', choices=['fixed', 'adaptive'])
    parser.add_argument('--num-taps', type=int, default=7)
    parser.add_argument('--seq-len', type=int, default=2048,
                        help='Sequence length per sample (DSD samples)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=50.0,
                        help='SDM quantizer temperature (higher = sharper)')
    parser.add_argument('--anneal-temp', action='store_true',
                        help='Anneal temperature from 1.0 to --temperature')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--checkpoint-dir', default='I:/foo_dsd_trellis/checkpoints/pre_sdm')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
