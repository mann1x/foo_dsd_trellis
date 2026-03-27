"""
Differentiable Sigma-Delta Modulator for end-to-end pre-SDM training.
Uses tanh(x*T) as soft quantizer instead of sign(x).

Matches the C trellis.c NTF filter structure:
  new_state[0] = state[0] - g[0]*state[1] + x
  v = x + a[0]*new_state[0]
  for i in 1..order-2:
      new_state[i] = state[i] + state[i-1] - g[i]*state[i+1]
      v += a[i]*new_state[i]
  new_state[order-1] = state[order-1] + state[order-2]
  v += a[order-1]*new_state[order-1]
  y = tanh(v * T)  # soft 1-bit quantizer
  new_state[0] -= y
"""

import torch
import torch.nn as nn
import math


# NTF coefficients from ntf.c (CLANS-6 @ DSD64, the production default)
NTF_CONFIGS = {
    'clans-6': {
        'order': 6,
        'a': [0.003046, 0.01905, 0.06099, 0.1225, 0.1604, 0.1371],
        'g': [0.7891, 0.0, 0.5731, 0.0, 0.1667, 0.0],
    },
    'sdm-6': {
        'order': 6,
        'a': [0.003629, 0.02155, 0.06592, 0.1288, 0.1645, 0.1365],
        'g': [0.7853, 0.0, 0.5597, 0.0, 0.1534, 0.0],
    },
    'sdm-4': {
        'order': 4,
        'a': [0.01587, 0.07086, 0.1694, 0.2333],
        'g': [0.7419, 0.0, 0.1847, 0.0],
    },
}


class DiffSDM(nn.Module):
    """Differentiable greedy SDM with NTF filter.

    Processes one sample at a time (sequential), returns soft ±1 output.
    Temperature T controls quantizer sharpness:
      T=1: smooth (good gradients, poor binary approximation)
      T=100: sharp (weak gradients, near-binary)
    """

    def __init__(self, ntf_name='clans-6', temperature=10.0, input_scale=0.5):
        super().__init__()
        cfg = NTF_CONFIGS[ntf_name]
        self.order = cfg['order']
        self.register_buffer('a', torch.tensor(cfg['a'], dtype=torch.float64))
        self.register_buffer('g', torch.tensor(cfg['g'], dtype=torch.float64))
        self.temperature = temperature
        self.input_scale = input_scale

    def forward(self, x):
        """
        x: (batch, seq_len) float64 — pre-SDM signal (multi-bit)
        Returns: (batch, seq_len) float64 — soft DSD output (near ±1)
        """
        B, N = x.shape
        device = x.device

        state = torch.zeros(B, self.order, dtype=torch.float64, device=device)
        output = torch.zeros(B, N, dtype=torch.float64, device=device)

        a = self.a
        g = self.g
        order = self.order
        T = self.temperature

        for n in range(N):
            xn = x[:, n] * self.input_scale  # match C code: in[i] * 0.5

            # NTF filter update
            new_state = torch.zeros_like(state)
            new_state[:, 0] = state[:, 0] - g[0] * state[:, 1] + xn
            v = xn + a[0] * new_state[:, 0]

            for i in range(1, order - 1):
                if i + 1 < order:
                    new_state[:, i] = state[:, i] + state[:, i-1] - g[i] * state[:, i+1]
                else:
                    new_state[:, i] = state[:, i] + state[:, i-1]
                v = v + a[i] * new_state[:, i]

            new_state[:, order-1] = state[:, order-1] + state[:, order-2]
            v = v + a[order-1] * new_state[:, order-1]

            # Soft quantizer: tanh(v * T) approximates sign(v)
            y = torch.tanh(v * T)

            # Feedback
            new_state[:, 0] = new_state[:, 0] - y

            state = new_state
            output[:, n] = y

        return output


class AudioBandLoss(nn.Module):
    """Loss that measures noise in the audio band (0-24 kHz).

    Uses a Kaiser lowpass FIR to extract audio-band content from the
    DSD-rate signal, then computes MSE between SDM output and reference.
    """

    def __init__(self, dsd_rate=2822400, cutoff_hz=24000, ntaps=127):
        super().__init__()
        # Design Kaiser lowpass
        fc = cutoff_hz / (dsd_rate / 2)
        beta = 10.0
        M = ntaps - 1
        h = torch.zeros(ntaps, dtype=torch.float64)
        I0_beta = self._bessel_I0(beta)

        for n in range(ntaps):
            x = n - M / 2
            if abs(x) < 1e-10:
                sinc = 2 * fc
            else:
                sinc = math.sin(2 * math.pi * fc * x) / (math.pi * x)
            t = 2 * n / M - 1
            arg = 1 - t * t
            w = self._bessel_I0(beta * math.sqrt(max(arg, 0))) / I0_beta
            h[n] = sinc * w

        h = h / h.sum()  # normalize
        # Conv1d kernel: (out_ch, in_ch, kernel_size)
        self.register_buffer('kernel', h.unsqueeze(0).unsqueeze(0))
        self.pad = ntaps // 2

    @staticmethod
    def _bessel_I0(x):
        s, t = 1.0, 1.0
        xh2 = (x / 2) ** 2
        for k in range(1, 26):
            t *= xh2 / (k * k)
            s += t
            if t < 1e-20 * s:
                break
        return s

    def forward(self, sdm_output, reference):
        """
        sdm_output: (batch, seq_len) — soft DSD from differentiable SDM
        reference: (batch, seq_len) — original signal (pre-SDM input)
        Returns: scalar loss (audio-band MSE)
        """
        # Apply lowpass to extract audio band
        out_lp = torch.nn.functional.conv1d(
            sdm_output.unsqueeze(1), self.kernel, padding=self.pad
        ).squeeze(1)
        ref_lp = torch.nn.functional.conv1d(
            reference.unsqueeze(1), self.kernel, padding=self.pad
        ).squeeze(1)

        # Trim edges (filter transient)
        trim = self.pad
        out_lp = out_lp[:, trim:-trim]
        ref_lp = ref_lp[:, trim:-trim]

        return torch.nn.functional.mse_loss(out_lp, ref_lp)
