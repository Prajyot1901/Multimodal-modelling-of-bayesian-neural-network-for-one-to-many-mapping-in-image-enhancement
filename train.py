#!/usr/bin/env python3
"""
Single-file Multimodal Bayesian Enhancement Model (MM-BEM)
Self-contained, bug-fixed standalone script.
"""

import copy
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# HARD-CODED EXPERIMENT CONFIGURATION
# =============================================================================

SEED = 1234

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
PIN_MEMORY = True

DATA_ROOT = r"E:\PWP_project\UIEB"

TRAIN_INPUT_DIR = os.path.join(DATA_ROOT, "train", "lq")
TRAIN_TARGET_DIR = os.path.join(DATA_ROOT, "train", "gt")
VAL_INPUT_DIR = os.path.join(DATA_ROOT, "val", "lq")
VAL_TARGET_DIR = os.path.join(DATA_ROOT, "val", "gt")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

PATCH_SIZE = 256
BATCH_SIZE = 16

STAGE1_ITERATIONS = 18000
STAGE2_ITERATIONS = 18000

LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 1e-6

GRAD_CLIP_NORM = 1.0

DOWNSAMPLE_FACTOR = 1
ALPHA = 0.025
INFERENCE_SAMPLES = 25

NUM_MIXTURE_COMPONENTS = 3
KL_MC_SAMPLES = 5

SIGMA_INIT = 0.05
MIXTURE_INIT_JITTER = 0.01

ADAPTIVE_PRIOR_DECAY = 0.998

CHECKPOINT_DIR = "./experiments/mm_bem"

SAVE_EVERY = 10_000
VALIDATE_EVERY = 5_000
LOG_EVERY = 100

STAGE1_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage1_latest_modified_.pth")
STAGE1_FAKE_CKPT = os.path.join(CHECKPOINT_DIR, "stage1_latest_modified_1.pth")
STAGE2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage2_latest_modified_1.pth")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# =============================================================================
# BASIC IMAGE DATASET
# =============================================================================

def list_images(folder: str) -> List[str]:
    paths = []
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset directory does not exist: {folder}")

    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            paths.append(os.path.join(folder, name))

    if not paths:
        raise RuntimeError(f"No images found in: {folder}")
    return paths


def load_rgb(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


class PairedEnhancementDataset(Dataset):
    def __init__(self, input_dir: str, target_dir: str, patch_size: int = 128, train: bool = True):
        self.input_paths = list_images(input_dir)
        self.target_paths = list_images(target_dir)

        target_by_stem = {Path(p).stem: p for p in self.target_paths}
        pairs = []

        for input_path in self.input_paths:
            stem = Path(input_path).stem
            if stem in target_by_stem:
                pairs.append((input_path, target_by_stem[stem]))

        if not pairs:
            raise RuntimeError("No input/target filename matches were found.")

        self.pairs = pairs
        self.patch_size = patch_size
        self.train = train

    @staticmethod
    def _random_crop(x: torch.Tensor, y: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h, w = x.shape
        if h < size or w < size:
            scale = max(size / h, size / w)
            new_h = int(math.ceil(h * scale))
            new_w = int(math.ceil(w * scale))
            x = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
            y = F.interpolate(y.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
            _, h, w = x.shape

        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        return x[:, top:top + size, left:left + size], y[:, top:top + size, left:left + size]

    @staticmethod
    def _center_crop(x: torch.Tensor, y: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h, w = x.shape
        if h < size or w < size:
            new_h = max(size, h)
            new_w = max(size, w)
            x = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
            y = F.interpolate(y.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
            _, h, w = x.shape

        top = (h - size) // 2
        left = (w - size) // 2
        return x[:, top:top + size, left:left + size], y[:, top:top + size, left:left + size]

    @staticmethod
    def _augment(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        return x, y

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        input_path, target_path = self.pairs[index]
        x = load_rgb(input_path)
        y = load_rgb(target_path)

        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y.unsqueeze(0), size=x.shape[-2:], mode="bilinear", align_corners=False)[0]

        if self.train:
            x, y = self._random_crop(x, y, self.patch_size)
            x, y = self._augment(x, y)
        else:
            x, y = self._center_crop(x, y, self.patch_size)

        return {"lq": x, "gt": y, "input_path": input_path, "target_path": target_path}


# =============================================================================
# LOW-PASS FILTER + DOWN/UPSAMPLING
# =============================================================================

def fft_low_pass(x: torch.Tensor, cutoff_ratio: float = 0.5) -> torch.Tensor:
    b, c, h, w = x.shape
    freq = torch.fft.fft2(x, norm="ortho")
    freq = torch.fft.fftshift(freq)

    keep_h = max(1, int(h * cutoff_ratio))
    keep_w = max(1, int(w * cutoff_ratio))

    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - keep_h // 2), min(h, cy + keep_h // 2)
    x0, x1 = max(0, cx - keep_w // 2), min(w, cx + keep_w // 2)

    mask = torch.zeros(h, w, device=x.device, dtype=freq.dtype)
    mask[y0:y1, x0:x1] = 1

    freq = freq * mask[None, None]
    freq = torch.fft.ifftshift(freq)
    return torch.fft.ifft2(freq, norm="ortho").real


def downsample_low_pass(x: torch.Tensor, factor: int) -> torch.Tensor:
    filtered = fft_low_pass(x, cutoff_ratio=1.0 / factor)
    h, w = x.shape[-2] // factor, x.shape[-1] // factor
    return F.interpolate(filtered, size=(h, w), mode="bilinear", align_corners=False)


def upsample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


# =============================================================================
# BASIC BACKBONE
# =============================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x, **kwargs):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-6)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x, step: int = 0, decay: float = ADAPTIVE_PRIOR_DECAY, deterministic: bool = False):
        identity = x
        x = self.norm(x)

        if isinstance(self.conv1, BayesianConv2d):
            x = self.conv1(x, step=step, decay=decay, deterministic=deterministic)
        else:
            x = self.conv1(x)

        x = self.act(x)

        if isinstance(self.conv2, BayesianConv2d):
            x = self.conv2(x, step=step, decay=decay, deterministic=deterministic)
        else:
            x = self.conv2(x)

        return x + identity


class CustomSequential(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self:
            if isinstance(module, (ConvBlock, EncoderDecoder, BayesianConv2d, CustomSequential)):
                input = module(input, **kwargs)
            else:
                input = module(input)
        return input


class EncoderDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 48, depth: int = 3):
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = base_channels
        for _ in range(depth):
            self.enc_blocks.append(CustomSequential(ConvBlock(channels), ConvBlock(channels)))
            self.downs.append(nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1))
            channels *= 2

        self.middle = CustomSequential(ConvBlock(channels), ConvBlock(channels))
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for _ in range(depth):
            self.ups.append(nn.ConvTranspose2d(channels, channels // 2, 2, stride=2))
            channels //= 2
            self.dec_blocks.append(CustomSequential(ConvBlock(channels), ConvBlock(channels)))

        self.tail = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, step: int = 0, decay: float = ADAPTIVE_PRIOR_DECAY, deterministic: bool = False):
        kwargs = {"step": step, "decay": decay, "deterministic": deterministic}

        x = self.head(x, **kwargs) if isinstance(self.head, BayesianConv2d) else self.head(x)
        skips = []

        for block, down in zip(self.enc_blocks, self.downs):
            x = block(x, **kwargs)
            skips.append(x)
            x = down(x, **kwargs) if isinstance(down, BayesianConv2d) else down(x)

        x = self.middle(x, **kwargs)

        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = up(x, **kwargs) if isinstance(up, BayesianConv2d) else up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + skip
            x = block(x, **kwargs)

        return self.tail(x, **kwargs) if isinstance(self.tail, BayesianConv2d) else self.tail(x)


# =============================================================================
# MULTIMODAL BAYESIAN LAYERS
# =============================================================================

class MultimodalBayesianMixin:
    def _initialize_mixture(self, parameter_shape, sigma_init, n_components, init_jitter):
        self.n_components = n_components
        self.sigma_init = sigma_init
        self.init_jitter = init_jitter

        self.mixture_logits = nn.Parameter(torch.zeros(n_components))
        self.mu_weight = nn.Parameter(torch.empty(n_components, *parameter_shape))
        self.rho_weight = nn.Parameter(torch.full((n_components, *parameter_shape), inverse_softplus(sigma_init)))

        self.register_buffer("prior_mu_weight", torch.empty(*parameter_shape))
        self.register_buffer("prior_rho_weight", torch.full(parameter_shape, inverse_softplus(sigma_init)))

        self._cached_weight = None

    @property
    def mixture_probs(self):
        return torch.softmax(self.mixture_logits, dim=0)

    def component_sigmas(self):
        return F.softplus(self.rho_weight).clamp(1e-5, 10.0)

    def mixture_moments(self):
        pi = self.mixture_probs
        sigma = self.component_sigmas()
        shape = (self.n_components,) + (1,) * (self.mu_weight.ndim - 1)
        pi = pi.view(shape)

        mean = (pi * self.mu_weight).sum(dim=0)
        second = (pi * (sigma.square() + self.mu_weight.square())).sum(dim=0)
        variance = (second - mean.square()).clamp(1e-8, 100.0)

        return mean, torch.sqrt(variance)

    def sample_weight(self):
        component = torch.multinomial(self.mixture_probs, 1).item()
        sigma = F.softplus(self.rho_weight[component]).clamp(1e-5, 10.0)
        eps = torch.randn_like(self.mu_weight[component])
        return self.mu_weight[component] + sigma * eps

    @staticmethod
    def log_gaussian(value, mu, sigma):
        dist = torch.distributions.Normal(mu, sigma)
        return dist.log_prob(value).sum()

    def log_q(self, value):
        mus = self.mu_weight.float()
        sigmas = F.softplus(self.rho_weight.float()).clamp(1e-5, 10.0)

        flat_val = value.float().unsqueeze(0)
        dist = torch.distributions.Normal(mus, sigmas)
        component_log_probs = dist.log_prob(flat_val).flatten(start_dim=1).sum(dim=1)

        log_pi = F.log_softmax(self.mixture_logits.float(), dim=0)
        return torch.logsumexp(log_pi + component_log_probs, dim=0)

    def update_adaptive_prior(self, step: int, decay: float):
        with torch.no_grad():
            effective_decay = min(decay, (1.0 + step) / (10.0 + step))
            mean, sigma = self.mixture_moments()

            self.prior_mu_weight.mul_(effective_decay).add_(mean, alpha=1.0 - effective_decay)
            sigma_rho = inverse_softplus_tensor(sigma)
            self.prior_rho_weight.mul_(effective_decay).add_(sigma_rho, alpha=1.0 - effective_decay)

    def kl_loss(self, samples: int = 1):
        value = self._cached_weight if self._cached_weight is not None else self.sample_weight()
        prior_sigma = F.softplus(self.prior_rho_weight).clamp(1e-5, 10.0)

        estimates = []
        for i in range(samples):
            sample = value if i == 0 else self.sample_weight()
            log_q = self.log_q(sample)
            log_p = self.log_gaussian(sample, self.prior_mu_weight, prior_sigma)
            estimates.append((log_q - log_p) / sample.numel())

        return torch.stack(estimates).mean().to(value.dtype)


def inverse_softplus(x: float) -> float:
    return x if x > 20 else math.log(math.expm1(x))


def inverse_softplus_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20, x, torch.log(torch.expm1(x.clamp(1e-5, 20.0))))


class BayesianConv2d(MultimodalBayesianMixin, nn.Module):
    def __init__(self, original: nn.Conv2d, n_components: int, sigma_init: float, init_jitter: float):
        super().__init__()
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.kernel_size = original.kernel_size
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        self.has_bias = original.bias is not None

        self._initialize_mixture(original.weight.shape, sigma_init, n_components, init_jitter)

        with torch.no_grad():
            self.mu_weight[0].copy_(original.weight)
            for k in range(1, n_components):
                self.mu_weight[k].copy_(original.weight + init_jitter * sigma_init * torch.randn_like(original.weight))

        if self.has_bias:
            self.mu_bias = nn.Parameter(torch.empty(n_components, self.out_channels))
            self.rho_bias = nn.Parameter(torch.full((n_components, self.out_channels), inverse_softplus(sigma_init)))
            self.register_buffer("prior_mu_bias", torch.empty(self.out_channels))
            self.register_buffer("prior_rho_bias", torch.full((self.out_channels,), inverse_softplus(sigma_init)))

            with torch.no_grad():
                self.mu_bias[0].copy_(original.bias)
                for k in range(1, n_components):
                    self.mu_bias[k].copy_(original.bias + init_jitter * sigma_init * torch.randn_like(original.bias))
                self.prior_mu_bias.copy_(original.bias)

    def forward(self, x, step: int = 0, decay: float = ADAPTIVE_PRIOR_DECAY, deterministic: bool = False):
        if deterministic:
            weight, _ = self.mixture_moments()
            bias = None
            if self.has_bias:
                pi = self.mixture_probs.view(self.n_components, 1)
                bias = (pi * self.mu_bias).sum(dim=0)
            return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        self.update_adaptive_prior(step, decay)
        weight = self.sample_weight()
        self._cached_weight = weight
        bias = None

        if self.has_bias:
            component = torch.multinomial(self.mixture_probs, 1).item()
            sigma = F.softplus(self.rho_bias[component]).clamp(1e-5, 10.0)
            bias = self.mu_bias[component] + sigma * torch.randn_like(self.mu_bias[component])
            self._cached_bias = bias

        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_total(self, samples: int = 1):
        kl = self.kl_loss(samples)
        if not self.has_bias:
            return kl

        value = getattr(self, "_cached_bias", None)
        if value is None:
            component = torch.multinomial(self.mixture_probs, 1).item()
            sigma = F.softplus(self.rho_bias[component]).clamp(1e-5, 10.0)
            value = self.mu_bias[component] + sigma * torch.randn_like(self.mu_bias[component])

        sigmas = F.softplus(self.rho_bias).clamp(1e-5, 10.0)
        dist = torch.distributions.Normal(self.mu_bias, sigmas)
        component_log_probs = dist.log_prob(value.unsqueeze(0)).flatten(start_dim=1).sum(dim=1)

        log_q = torch.logsumexp(F.log_softmax(self.mixture_logits.float(), dim=0) + component_log_probs, dim=0)
        prior_sigma = F.softplus(self.prior_rho_bias).clamp(1e-5, 10.0)
        log_p = self.log_gaussian(value, self.prior_mu_bias, prior_sigma)

        return kl + ((log_q - log_p) / value.numel()).to(kl.dtype)


# =============================================================================
# CONVERT CONVOLUTIONAL LAYERS TO MULTIMODAL BAYESIAN LAYERS
# =============================================================================

def convert_to_multimodal_bnn(module: nn.Module, n_components: int, sigma_init: float, init_jitter: float):
    for name, child in list(module.named_children()):
        if isinstance(child, BayesianConv2d):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, BayesianConv2d(child, n_components, sigma_init, init_jitter))
        else:
            convert_to_multimodal_bnn(child, n_components, sigma_init, init_jitter)


def bayesian_layers(module: nn.Module):
    for m in module.modules():
        if isinstance(m, BayesianConv2d):
            yield m


# =============================================================================
# STAGE I & II NETWORKS
# =============================================================================

class Stage1BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = EncoderDecoder(in_channels=3, out_channels=3, base_channels=48, depth=3)
        convert_to_multimodal_bnn(self.network, NUM_MIXTURE_COMPONENTS, SIGMA_INIT, MIXTURE_INIT_JITTER)

    def forward(self, x, step: int = 0):
        low = downsample_low_pass(x, DOWNSAMPLE_FACTOR)
        z_low = self.network(low, step=step, decay=ADAPTIVE_PRIOR_DECAY, deterministic=False)
        return upsample(z_low, x.shape[-2:])

    def kl_loss(self):
        total = None
        for layer in bayesian_layers(self.network):
            value = layer.kl_total(KL_MC_SAMPLES)
            total = value if total is None else total + value
        if total is None:
            raise RuntimeError("No Bayesian layers found.")
        return total


class Stage2DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = EncoderDecoder(in_channels=6, out_channels=3, base_channels=48, depth=3)

    def forward(self, x, z):
        return self.network(torch.cat([x, z], dim=1))


# =============================================================================
# PAPER EQUATIONS & UTILS
# =============================================================================

def construct_coarse_prediction(x, z):
    return (x + ALPHA * z) * z


def ground_truth_coarse_condition(x, y):
    inside = (x.square() + 4.0 * ALPHA * y).clamp_min(1e-8)
    z = (torch.sqrt(inside) - x) / (2.0 * ALPHA)
    return fft_low_pass(z, cutoff_ratio=1.0 / DOWNSAMPLE_FACTOR)


def cosine_lr(step: int, total_steps: int):
    progress = min(max(step / total_steps, 0.0), 1.0)
    return MIN_LEARNING_RATE + 0.5 * (LEARNING_RATE - MIN_LEARNING_RATE) * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(path, model, optimizer, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": SEED}, path)


def load_checkpoint(path, model, optimizer=None):
    if not os.path.isfile(path):
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def make_train_loader():
    dataset = PairedEnhancementDataset(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, PATCH_SIZE, True)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True, persistent_workers=(NUM_WORKERS > 0))


# =============================================================================
# STAGE I & II TRAINING
# =============================================================================

def train_stage1():
    print("=" * 80 + "\nMM-BEM STAGE I\n" + "=" * 80)
    train_loader = make_train_loader()
    model = Stage1BNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_step = load_checkpoint(STAGE1_CHECKPOINT, model, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    model.train()
    loader_iter = iter(train_loader)
    running_loss, running_recon, running_kl = 0.0, 0.0, 0.0
    start_time = time.time()

    for step in range(start_step + 1, STAGE1_ITERATIONS + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        x = batch["lq"].to(DEVICE, non_blocking=True)
        y = batch["gt"].to(DEVICE, non_blocking=True)

        lr = cosine_lr(step, STAGE1_ITERATIONS)
        set_learning_rate(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            z = model(x, step=step)
            y_tilde = construct_coarse_prediction(x, z)
            reconstruction = F.mse_loss(y_tilde, y)
            kl = model.kl_loss()
            loss = reconstruction + (0.01 / BATCH_SIZE) * kl

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.detach().item()
        running_recon += reconstruction.detach().item()
        running_kl += kl.detach().item()

        if step % LOG_EVERY == 0 or step == STAGE1_ITERATIONS:
            elapsed = time.time() - start_time
            divisor = min(step, LOG_EVERY)
            print(f"[Stage I] {step:07d}/{STAGE1_ITERATIONS} loss={running_loss/divisor:.6f} recon={running_recon/divisor:.6f} kl={running_kl/divisor:.6f} lr={lr:.3e} time={elapsed:.1f}s")
            running_loss, running_recon, running_kl = 0.0, 0.0, 0.0

        if step % SAVE_EVERY == 0:
            save_checkpoint(STAGE1_FAKE_CKPT, model, optimizer, step)

    save_checkpoint(STAGE1_FAKE_CKPT, model, optimizer, STAGE1_ITERATIONS)
    return model


def train_stage2():
    print("=" * 80 + "\nMM-BEM STAGE II\n" + "=" * 80)
    train_loader = make_train_loader()
    
    # Load Stage 1 model to generate sampled inputs
    stage1_model = Stage1BNN().to(DEVICE)
    load_checkpoint(STAGE1_FAKE_CKPT, stage1_model)
    stage1_model.eval()
    
    model = Stage2DNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_step = load_checkpoint(STAGE2_CHECKPOINT, model, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    model.train()
    loader_iter = iter(train_loader)
    running_loss = 0.0

    for step in range(start_step + 1, STAGE2_ITERATIONS + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        x = batch["lq"].to(DEVICE, non_blocking=True)
        y = batch["gt"].to(DEVICE, non_blocking=True)

        lr = cosine_lr(step, STAGE2_ITERATIONS)
        set_learning_rate(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            # Randomly sample 5 outputs from the Stage 1 model and take their mean
            with torch.no_grad():
                z_samples = []
                for _ in range(5):
                    z_samples.append(stage1_model(x))
                z_mean = torch.stack(z_samples).mean(dim=0)
                
            prediction = model(x, z_mean)
            loss = F.l1_loss(prediction, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.detach().item()

        if step % LOG_EVERY == 0 or step == STAGE2_ITERATIONS:
            divisor = min(step, LOG_EVERY)
            print(f"[Stage II] {step:07d}/{STAGE2_ITERATIONS} L1={running_loss/divisor:.6f} lr={lr:.3e}")
            running_loss = 0.0

        if step % SAVE_EVERY == 0:
            save_checkpoint(STAGE2_CHECKPOINT, model, optimizer, step)

    save_checkpoint(STAGE2_CHECKPOINT, model, optimizer, STAGE2_ITERATIONS)
    return model


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("Standalone MM-BEM\nUsing 3 Gaussian modes for q(w).\nAdaptive prior: single diagonal Gaussian.\n")
    train_stage1()
    train_stage2()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
