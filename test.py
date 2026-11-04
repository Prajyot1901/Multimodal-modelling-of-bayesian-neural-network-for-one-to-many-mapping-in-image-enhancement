#!/usr/bin/env python3
"""
Standalone Inference and Evaluation Script for MM-BEM
Features:
 - Multi-sample Monte Carlo inference via Stage I BNN.
 - Expectation (Mean) and Uncertainty Map estimation.
 - Full-resolution evaluation with PSNR and SSIM calculation.
 - Image saving (LQ, GT, Predicted Mean, Uncertainty Map).
"""

import math
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

# Paths (Match these with your experiment paths)
DATA_ROOT = r"E:\PWP_project\UIEB"
VAL_INPUT_DIR = os.path.join(DATA_ROOT, "val", "lq")
VAL_TARGET_DIR = os.path.join(DATA_ROOT, "val", "gt")
OUTPUT_DIR = "./experiments/mm_bem/inference_results"

CHECKPOINT_DIR = "./experiments/mm_bem"
STAGE1_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage1_latest_modified_1.pth")
STAGE2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage2_latest_modified_1.pth")

# Model Specs
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DOWNSAMPLE_FACTOR = 16
NUM_MIXTURE_COMPONENTS = 3
SIGMA_INIT = 0.05
MIXTURE_INIT_JITTER = 0.01

# Inference Settings
MC_SAMPLES = 150  # Number of Monte Carlo draws from Stage 1 BNN
SAVE_VISUALS = True


# =============================================================================
# METRICS (PSNR & SSIM)
# =============================================================================
def save_random_predictions_with_best(
    predictions,
    psnr_scores,
    ssim_scores,
    lq,
    gt,
    filename,
    output_dir,
    num_random=4,
    seed=None
):
    """
    Save the best prediction and randomly selected predictions
    in a comparison grid.

    Best prediction is selected by highest PSNR.
    Random predictions are selected from the remaining samples.
    """

    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    num_predictions = len(predictions)

    # -------------------------------------------------
    # 1. Find best prediction based on PSNR
    # -------------------------------------------------
    best_idx = int(np.argmax(psnr_scores))

    # -------------------------------------------------
    # 2. Randomly select predictions excluding best
    # -------------------------------------------------
    available_indices = [
        i for i in range(num_predictions)
        if i != best_idx
    ]

    rng = np.random.default_rng(seed)

    num_random = min(num_random, len(available_indices))

    random_indices = rng.choice(
        available_indices,
        size=num_random,
        replace=False
    ).tolist()

    # Best prediction first, followed by random predictions
    selected_indices = [best_idx] + random_indices

    # -------------------------------------------------
    # 3. Create comparison grid
    # -------------------------------------------------
    num_images = 2 + len(selected_indices)  # LQ + GT + predictions

    cols = 4
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 5 * rows)
    )

    axes = np.array(axes).reshape(-1)

    # LQ input
    lq_img = lq.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(np.clip(lq_img, 0, 1))
    axes[0].set_title("Input (LQ)")
    axes[0].axis("off")

    # Ground truth
    gt_img = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[1].imshow(np.clip(gt_img, 0, 1))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Best + random predictions
    for rank, idx in enumerate(selected_indices):
        pred = predictions[idx].squeeze(0)
        pred = pred.permute(1, 2, 0).cpu().numpy()
        pred = np.clip(pred, 0, 1)

        ax = axes[rank + 2]
        ax.imshow(pred)

        if idx == best_idx:
            title = "Best Prediction"
        else:
            title = f"Random Prediction {rank}"

        ax.set_title(
            f"{title}\n"
            f"PSNR: {psnr_scores[idx]:.2f} dB | "
            f"SSIM: {ssim_scores[idx]:.4f}"
        )
        ax.axis("off")

    # Hide unused panels
    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        f"{Path(filename).stem}_comparison.png"
    )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved best + random comparison: {save_path}")

    return best_idx, random_indices


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    data_range = 1.0
    return 20 * math.log10(data_range / math.sqrt(mse))


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculates Structural Similarity Index (SSIM) for 3-channel RGB tensors [1, 3, H, W]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    kernel = torch.tensor([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=torch.float32, device=img1.device) / 256.0
    kernel = kernel.repeat(3, 1, 1, 1)

    mu1 = F.conv2d(img1, kernel, padding=2, groups=3)
    mu2 = F.conv2d(img2, kernel, padding=2, groups=3)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=2, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=2, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# =============================================================================
# DATA UTILITIES
# =============================================================================

def load_rgb(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_rgb(tensor: torch.Tensor, path: str):
    """Saves a [3, H, W] tensor in range [0, 1] to disc."""
    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def save_heatmap(tensor: torch.Tensor, path: str):
    """Saves a [1, H, W] or [3, H, W] variance tensor as a normalized grayscale heatmap."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    variance = tensor.mean(dim=0).cpu().numpy()  # Average over C
    norm_var = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
    array = np.clip(norm_var * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


class InferenceDataset(Dataset):
    def __init__(self, input_dir: str, target_dir: str):
        self.input_paths = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.lower().endswith(IMAGE_EXTENSIONS)]
        self.target_paths = [os.path.join(target_dir, f) for f in sorted(os.listdir(target_dir)) if f.lower().endswith(IMAGE_EXTENSIONS)]
        
        target_map = {Path(p).stem: p for p in self.target_paths}
        self.pairs = [(p, target_map[Path(p).stem]) for p in self.input_paths if Path(p).stem in target_map]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        in_path, gt_path = self.pairs[idx]
        x = load_rgb(in_path)
        y = load_rgb(gt_path)
        
        # Ensure divisible by downsample factor for clean upsampling
        _, h, w = x.shape
        new_h = (h // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
        new_w = (w // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
        
        if new_h != h or new_w != w:
            x = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
            y = F.interpolate(y.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]

        return {"lq": x, "gt": y, "filename": Path(in_path).name}


# =============================================================================
# MODEL ARCHITECTURE DEFS (MATCHES TRAINING)
# =============================================================================

def fft_low_pass(x: torch.Tensor, cutoff_ratio: float = 0.5) -> torch.Tensor:
    b, c, h, w = x.shape
    freq = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
    keep_h, keep_w = max(1, int(h * cutoff_ratio)), max(1, int(w * cutoff_ratio))
    cy, cx = h // 2, w // 2
    mask = torch.zeros(h, w, device=x.device, dtype=freq.dtype)
    mask[cy - keep_h//2 : cy + keep_h//2, cx - keep_w//2 : cx + keep_w//2] = 1
    freq = torch.fft.ifftshift(freq * mask[None, None])
    return torch.fft.ifft2(freq, norm="ortho").real

def downsample_low_pass(x: torch.Tensor, factor: int) -> torch.Tensor:
    filtered = fft_low_pass(x, cutoff_ratio=1.0 / factor)
    return F.interpolate(filtered, size=(x.shape[-2] // factor, x.shape[-1] // factor), mode="bilinear", align_corners=False)

def upsample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

def inverse_softplus(x: float) -> float:
    return x if x > 20 else math.log(math.expm1(x))

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
    def forward(self, x, **kwargs):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        return ((x - mean) / torch.sqrt(var + 1e-6)) * self.weight[:, None, None] + self.bias[:, None, None]

class BayesianConv2d(nn.Module):
    def __init__(self, original: nn.Conv2d, n_components: int, sigma_init: float, init_jitter: float):
        super().__init__()
        self.in_channels, self.out_channels = original.in_channels, original.out_channels
        self.kernel_size, self.stride, self.padding = original.kernel_size, original.stride, original.padding
        self.dilation, self.groups, self.has_bias = original.dilation, original.groups, original.bias is not None
        self.n_components = n_components

        # Parameters
        self.mixture_logits = nn.Parameter(torch.zeros(n_components))
        self.mu_weight = nn.Parameter(torch.empty(n_components, *original.weight.shape))
        self.rho_weight = nn.Parameter(torch.full((n_components, *original.weight.shape), inverse_softplus(sigma_init)))

        # Register prior buffers expected by state_dict
        self.register_buffer("prior_mu_weight", torch.empty(*original.weight.shape))
        self.register_buffer("prior_rho_weight", torch.full(original.weight.shape, inverse_softplus(sigma_init)))

        if self.has_bias:
            self.mu_bias = nn.Parameter(torch.empty(n_components, self.out_channels))
            self.rho_bias = nn.Parameter(torch.full((n_components, self.out_channels), inverse_softplus(sigma_init)))
            
            # Register bias prior buffers expected by state_dict
            self.register_buffer("prior_mu_bias", torch.empty(self.out_channels))
            self.register_buffer("prior_rho_bias", torch.full((self.out_channels,), inverse_softplus(sigma_init)))

    @property
    def mixture_probs(self):
        return torch.softmax(self.mixture_logits, dim=0)

    def sample_weight(self):
        comp = torch.multinomial(self.mixture_probs, 1).item()
        sigma = F.softplus(self.rho_weight[comp]).clamp(1e-5, 10.0)
        return self.mu_weight[comp] + sigma * torch.randn_like(self.mu_weight[comp])

    def forward(self, x, deterministic: bool = False, **kwargs):
        if deterministic:
            pi = self.mixture_probs.view((self.n_components,) + (1,) * self.mu_weight.ndim)
            weight = (pi * self.mu_weight).sum(dim=0)
            bias = (self.mixture_probs.view(self.n_components, 1) * self.mu_bias).sum(dim=0) if self.has_bias else None
            return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        weight = self.sample_weight()
        bias = None
        if self.has_bias:
            comp = torch.multinomial(self.mixture_probs, 1).item()
            sigma = F.softplus(self.rho_bias[comp]).clamp(1e-5, 10.0)
            bias = self.mu_bias[comp] + sigma * torch.randn_like(self.mu_bias[comp])

        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
class ConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x, **kwargs):
        res = x
        x = self.norm(x)
        x = self.conv1(x, **kwargs) if isinstance(self.conv1, BayesianConv2d) else self.conv1(x)
        x = self.act(x)
        x = self.conv2(x, **kwargs) if isinstance(self.conv2, BayesianConv2d) else self.conv2(x)
        return x + res

class CustomSequential(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs) if isinstance(module, (ConvBlock, BayesianConv2d, CustomSequential)) else module(input)
        return input

class EncoderDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 48, depth: int = 3):
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc_blocks, self.downs = nn.ModuleList(), nn.ModuleList()
        channels = base_channels
        for _ in range(depth):
            self.enc_blocks.append(CustomSequential(ConvBlock(channels), ConvBlock(channels)))
            self.downs.append(nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1))
            channels *= 2

        self.middle = CustomSequential(ConvBlock(channels), ConvBlock(channels))
        self.ups, self.dec_blocks = nn.ModuleList(), nn.ModuleList()
        for _ in range(depth):
            self.ups.append(nn.ConvTranspose2d(channels, channels // 2, 2, stride=2))
            channels //= 2
            self.dec_blocks.append(CustomSequential(ConvBlock(channels), ConvBlock(channels)))

        self.tail = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, **kwargs):
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
            x = block(x + skip, **kwargs)

        return self.tail(x, **kwargs) if isinstance(self.tail, BayesianConv2d) else self.tail(x)

def convert_to_multimodal_bnn(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, BayesianConv2d):
            setattr(module, name, BayesianConv2d(child, NUM_MIXTURE_COMPONENTS, SIGMA_INIT, MIXTURE_INIT_JITTER))
        else:
            convert_to_multimodal_bnn(child)

class Stage1BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = EncoderDecoder(in_channels=3, out_channels=3, base_channels=48, depth=3)
        convert_to_multimodal_bnn(self.network)

    def forward(self, x, deterministic: bool = False):
        low = downsample_low_pass(x, DOWNSAMPLE_FACTOR)
        z_low = self.network(low, deterministic=deterministic)
        return upsample(z_low, x.shape[-2:])

class Stage2DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = EncoderDecoder(in_channels=6, out_channels=3, base_channels=48, depth=3)

    def forward(self, x, z):
        return self.network(torch.cat([x, z], dim=1))


# =============================================================================
# INFERENCE RUNNER
# =============================================================================

def run_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Models
    print("Loading models...")
    stage1 = Stage1BNN().to(DEVICE)
    stage2 = Stage2DNN().to(DEVICE)

    if os.path.exists(STAGE1_CHECKPOINT):
        ckpt1 = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE)
        stage1.load_state_dict(ckpt1["model"], strict=False)
        print(f"Loaded Stage I from: {STAGE1_CHECKPOINT}")

    if os.path.exists(STAGE2_CHECKPOINT):
        ckpt2 = torch.load(STAGE2_CHECKPOINT, map_location=DEVICE)
        stage2.load_state_dict(ckpt2["model"])
        print(f"Loaded Stage II from: {STAGE2_CHECKPOINT}")
    else:
        print(f"Warning: Stage II checkpoint missing at {STAGE2_CHECKPOINT}")

    stage1.eval()
    stage2.eval()

    # 2. Data Loader
    val_dataset = InferenceDataset(VAL_INPUT_DIR, VAL_TARGET_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Evaluation samples found: {len(val_dataset)}")

    psnr_scores, ssim_scores = [], []
    start_time = time.time()

    print("\nStarting inference...")
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            x = batch["lq"].to(DEVICE)
            y = batch["gt"].to(DEVICE)
            filename = batch["filename"][0]

            # Stage I MC Sampling for Multimodal Latent Condition (z)
            # Stage I MC Sampling
            z_samples = []

            for _ in range(MC_SAMPLES):
                z_s = stage1(x, deterministic=False)
                z_samples.append(z_s)

            z_stack = torch.stack(z_samples, dim=0)

            z_mean = z_stack.mean(dim=0)
            z_var = z_stack.var(dim=0)

            # Stage II predictions for all MC samples
            predictions = []
            psnrs = []
            ssims = []

            for z in z_stack:
                pred_y = stage2(x, z)
                pred_y = torch.clamp(pred_y, 0.0, 1.0)

                predictions.append(pred_y)

                psnr_val = calculate_psnr(pred_y, y)
                ssim_val = calculate_ssim(pred_y, y)

                psnrs.append(psnr_val)
                ssims.append(ssim_val)

            # Best prediction metrics
            best_idx = int(np.argmax(psnrs))

            psnr_scores.append(psnrs[best_idx])
            ssim_scores.append(ssims[best_idx])

            print(
                f"[{idx+1:03d}/{len(val_dataset):03d}] "
                f"{filename} -> "
                f"Best PSNR: {psnrs[best_idx]:.2f} dB | "
                f"SSIM: {ssims[best_idx]:.4f}"
            )

            # Save visualizations
            if SAVE_VISUALS:
                file_stem = Path(filename).stem

                # Save best prediction
                save_rgb(
                    predictions[best_idx],
                    os.path.join(OUTPUT_DIR, f"{file_stem}_pred.png")
                )

                # Save uncertainty map
                save_heatmap(
                    z_var,
                    os.path.join(OUTPUT_DIR, f"{file_stem}_uncertainty.png")
                )

                # Save top 5 predictions in one image
                save_random_predictions_with_best(
                    predictions=predictions,
                    psnr_scores=psnrs,
                    ssim_scores=ssims,
                    lq=x,
                    gt=y,
                    filename=filename,
                    output_dir=OUTPUT_DIR,
                    num_random=4
                )

            print(f"[{idx+1:03d}/{len(val_dataset):03d}] {filename} -> PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

    elapsed = time.time() - start_time
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Processed Images : {len(val_dataset)}")
    print(f"Average PSNR           : {avg_psnr:.3f} dB")
    print(f"Average SSIM           : {avg_ssim:.4f}")
    print(f"Total Time Taken       : {elapsed:.2f} seconds ({elapsed/len(val_dataset):.2f}s/img)")
    print(f"Saved Visuals Directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    run_inference()