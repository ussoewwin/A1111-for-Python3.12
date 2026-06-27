# A1111-for-Python3.12 — img2img 16GB VRAM: Forge Tiled VAE + MultiDiffusion Integration

**Purpose:** What broke on ~16 GB VRAM img2img upscales with MultiDiffusion + ControlNet tile + Noise Inversion, what we changed, and the **full text** of every added or modified line.

**UI unchanged:** MultiDiffusion tile 96×96 overlap 48 batch 4, ControlNet tile_resample, Noise Inversion Euler, Tiled VAE enabled — fixes are code-only.

---

## 1. What was wrong

### 1-1. VRAM spikes at VAE encode/decode (bookends)

MultiDiffusion tiles the **UNet** in latent space. ControlNet and Noise Inversion run inside that loop. **VAE encode** (img2img start) and **VAE decode** (end) still ran at **full image resolution**.

On 16 GB, large upscales cause the largest VRAM peaks at those bookends. MultiDiffusion **VAEHook** tiles encoder/decoder `forward` but uses megatile sizes from UI sliders. We needed Forge-style 3-pass tiled encode/decode on **`encode_first_stage` / `decode_first_stage`** class hooks.

### 1-2. Latent canvas width used floor; VAE width uses ceil

`AbstractDiffusion` used `width // 8` and `height // 8`. When pixel width is not a multiple of 8, latent width is **ceil(width/8)**. Example: **2325 px → latent W = 291**, but `2325//8 = 290`. Canvas was one column short.

### 1-3. Size mismatch triggered full UNet fallback

When `(H, W) != (self.h, self.w)`, `multidiffusion.py` called `org_func(x_in)` — **full UNet** — instead of tiling. Forge encode and canvas fixes often change latent size; this path OOM’d on 16 GB.

### 1-4. Forge encode `downscale=True` shifted latent indices

Tile positions were divided by 8 again when mapping into the latent buffer. Encoded H/W no longer matched MultiDiffusion canvas.

### 1-5. Last VAE tile left uncovered cells → NaN

When width mod 8 ≠ 0, the last tile did not cover the rightmost latent column. `out_div == 0` → `out / out_div` → **NaN**.

### 1-6. ControlNet hints at wrong resolution

Hints are often pixel-sized `(H*8, W*8)` while canvas is latent `(H, W)`. Old code cropped incorrectly or assumed latent-sized hints only.

### 1-7. Noise Inversion: noise/x floor-sized vs init_latent ceil-sized

`create_random_tensors` uses floor. Forge tiled encode uses ceil width. Example **1853×1254**: `init_latent` **157×232**, `noise`/`x` **156×231** → broadcast failure at renoise mask.

### 1-8. Batched multi-tile UNet + ControlNet VRAM

Several tiles per `apply_model` with batched ControlNet hints exceeded 16 GB. Need **one tile per forward** when ControlNet is on.

---

## 2. What we did (countermeasures)

| Problem | Fix |
|---------|-----|
| Full-res VAE bookends | `forge_tiled_vae.py`: Forge 3-pass tiled encode/decode; CPU accumulation |
| VAEHook conflict | `tilevae.py`: Forge path skips VAEHook; sets `VAE_ALWAYS_TILED` |
| Patch install | `sd_models_xl.py`: `apply_all_vae_patches()` at import |
| Floor width | `pixel_to_latent_w` (ceil), `pixel_to_latent_h` (floor) |
| org_func OOM | `_rebuild_latent_canvas`; multidiffusion rebuilds instead of `org_func` |
| Encode downscale bug | `downscale=False` on encode |
| VAE edge NaN | End-align last tile in `tiled_scale_multidim` |
| ControlNet hints | `set_controlnet_tensors_for_size`, `_crop_controlnet_tile`, pixel branches |
| NI 231 vs 232 | `_align_latent_to_canvas` on `noise` and `x` in `sample_img2img` |
| ControlNet VRAM | One tile per `repeat_func`; `micro_plan = [1]*tb` when ControlNet on |

---

## 3. Files added or modified

| File | Action |
|------|--------|
| `modules/forge_tiled_vae.py` | **New** (768 lines) |
| `modules/sd_models_xl.py` | Import + `apply_all_vae_patches()` |
| `tile_utils/utils.py` | `pixel_to_latent_h`, `pixel_to_latent_w` |
| `scripts/tilevae.py` | Forge routing, VRAM caps, postprocess |
| `tile_methods/abstractdiffusion.py` | Canvas, rebuild, ControlNet, NI alignment |
| `tile_methods/multidiffusion.py` | Rebuild, per-tile forwards, hints |
| `tile_methods/demofusion.py` | Pixel-space hint branch |
| `tile_methods/mixtureofdiffusers.py` | Pixel-space hint branch |

Paths are under `extensions-builtin/multidiffusion-upscaler-for-automatic1111/` except `modules/`.

---

## 4. Full text of added or modified code

### 4-1. `modules/forge_tiled_vae.py` (entire new file)

```python
"""
Forge (ComfyUI-style) tiled VAE — ported from Forge backend/patcher/vae.py.

SDXL: DiffusionEngine.encode_first_stage / decode_first_stage (returns scaled latent tensor).
SD 1.5 / 2.x: LatentDiffusion.encode_first_stage / decode_first_stage (returns DiagonalGaussianDistribution).

VAE_ALWAYS_TILED -> always 512px/64px 3-pass tiled; else full batch then OOM tiled fallback.
Bypasses multidiffusion VAEHook via encoder/decoder original_forward when present.
"""

from __future__ import annotations

import contextlib
import itertools
from typing import Callable, Optional

import torch
from tqdm import tqdm

from modules import devices
from modules.shared import state

DOWNSCALE_RATIO = 8
LATENT_CHANNELS = 4
PIXEL_CHANNELS = 3

# Forge SDXL memory heuristics (backend/patcher/vae.py)
MEMORY_USED_ENCODE = lambda h, w, dtype: (1767 * h * w) * _dtype_size(dtype)
MEMORY_USED_DECODE = lambda h, w, w_lat, dtype: (2178 * w_lat * h * 64) * _dtype_size(dtype)


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _get_free_memory(device: torch.device) -> int:
    if torch.cuda.is_available() and device.type == "cuda":
        dev = device.index if device.index is not None else devices.get_cuda_device_id()
        free, _ = torch.cuda.mem_get_info(dev)
        return int(free)
    return 2**62


def _oom_exceptions() -> tuple:
    errs = [torch.cuda.OutOfMemoryError]
    oom = getattr(torch, "OutOfMemoryError", None)
    if oom is not None:
        errs.append(oom)
    return tuple(errs)


OOM_EXCEPTIONS = _oom_exceptions()

# Forge memory_management.VAE_ALWAYS_TILED — set from multidiffusion "Enable Tiled VAE" UI.
VAE_ALWAYS_TILED = False

# Base tile sizes for Forge 3-pass encode/decode (UI caps legacy VAEHook megatile sizes).
_ENCODE_TILE_BASE = 512
_DECODE_TILE_BASE = 64
_ENCODE_OVERLAP_RATIO = 0.125  # Forge: 64 / 512
_DECODE_OVERLAP_RATIO = 0.25   # Forge: 16 / 64

# Set at apply_* time; sd_hijack CondFunc replaces class methods with lambdas so
# __forge_tiled_vae__ on the class attribute is not a reliable runtime marker.
_DIFFUSION_ENGINE_PATCH_APPLIED = False
_LATENT_DIFFUSION_PATCH_APPLIED = False
_patched_latent_diffusion_classes: set[type] = set()


def set_vae_always_tiled(enabled: bool) -> None:
    global VAE_ALWAYS_TILED
    VAE_ALWAYS_TILED = bool(enabled)


def set_vae_tile_sizes(encoder_tile: int = 512, decoder_tile: int = 64) -> None:
    """Multidiffusion UI tile sliders — Forge 3-pass uses modest bases (not 3072 VAEHook tiles)."""
    global _ENCODE_TILE_BASE, _DECODE_TILE_BASE
    _ENCODE_TILE_BASE = max(128, min(512, int(encoder_tile)))
    _DECODE_TILE_BASE = max(32, min(256, int(decoder_tile)))


def get_encode_tile_base() -> int:
    return _ENCODE_TILE_BASE


def get_decode_tile_base() -> int:
    return _DECODE_TILE_BASE


def is_vae_always_tiled() -> bool:
    return VAE_ALWAYS_TILED


def _tiled_accum_device():
    """Forge intermediate_device() — keep full tiled buffers off GPU."""
    return devices.cpu


def _encode_overlap_for_base(base: int) -> int:
    return max(16, min(base // 2, round(base * _ENCODE_OVERLAP_RATIO)))


def _decode_overlap_for_base(base: int) -> int:
    return max(8, min(base // 2, round(base * _DECODE_OVERLAP_RATIO)))


def _effective_encode_base(height: int, width: int) -> int:
    """Smaller tiles on large upscales to cap per-tile VAE encoder VRAM."""
    base = _ENCODE_TILE_BASE
    pixels = height * width
    if pixels >= 4_000_000:
        base = min(base, 192)
    elif pixels >= 2_500_000:
        base = min(base, 256)
    return max(128, base)


def _encode_passes(base: int, overlap: int):
    return (
        ((base, base), overlap),
        ((base * 2, max(base // 2, 128)), overlap),
        ((max(base // 2, 128), base * 2), overlap),
    )


def _decode_passes(base: int, overlap: int):
    half = max(base // 2, 16)
    double = min(base * 2, 256)
    return (
        ((half, double), overlap),
        ((double, half), overlap),
        ((base, base), overlap),
    )


@contextlib.contextmanager
def _bypass_vae_hooks(vae):
    """Skip multidiffusion tilevae.VAEHook on encoder/decoder during Forge-style passes."""
    enc = getattr(vae, "encoder", None)
    dec = getattr(vae, "decoder", None)
    enc_saved = dec_saved = None
    try:
        if enc is not None and hasattr(enc, "original_forward") and enc.forward is not enc.original_forward:
            enc_saved = enc.forward
            enc.forward = enc.original_forward
        if dec is not None and hasattr(dec, "original_forward") and dec.forward is not dec.original_forward:
            dec_saved = dec.forward
            dec.forward = dec.original_forward
        yield
    finally:
        if enc_saved is not None:
            enc.forward = enc_saved
        if dec_saved is not None:
            dec.forward = dec_saved


@torch.no_grad()
def tiled_scale_multidim(
    samples,
    function: Callable,
    tile=(64, 64),
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    downscale=False,
    index_formulas=None,
    pbar: Optional[tqdm] = None,
):
    """ComfyUI / Forge tiled_scale_multidim (2D)."""
    dims = len(tile)

    if not isinstance(upscale_amount, (tuple, list)):
        upscale_amount = [upscale_amount] * dims
    if not isinstance(overlap, (tuple, list)):
        overlap = [overlap] * dims
    if index_formulas is None:
        index_formulas = upscale_amount
    if not isinstance(index_formulas, (tuple, list)):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        return up(val) if callable(up) else up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        return up(val) if callable(up) else val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        return up(val) if callable(up) else up * val

    def get_downscale_pos(dim, val):
        up = index_formulas[dim]
        return up(val) if callable(up) else val / up

    get_scale = get_downscale if downscale else get_upscale
    get_pos = get_downscale_pos if downscale else get_upscale_pos

    def mult_list_upscale(a):
        return [round(get_scale(i, a[i])) for i in range(len(a))]

    output = torch.empty(
        [samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]),
        device=output_device,
    )

    for b in range(samples.shape[0]):
        if state.interrupted:
            break
        s = samples[b : b + 1]

        if all(s.shape[d + 2] <= tile[d] for d in range(dims)):
            output[b : b + 1] = function(s).to(output_device)
            if pbar is not None:
                pbar.update(1)
            continue

        out = torch.zeros(
            [s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]),
            device=output_device,
        )
        out_div = torch.zeros(
            [s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]),
            device=output_device,
        )

        positions = [
            range(0, s.shape[d + 2] - overlap[d], tile[d] - overlap[d]) if s.shape[d + 2] > tile[d] else [0]
            for d in range(dims)
        ]

        for it in itertools.product(*positions):
            if state.interrupted:
                break
            s_in = s
            upscaled = []
            last_axis = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                length = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, length)
                upscaled.append(round(get_pos(d, pos)))
                last_axis.append(pos + length >= s.shape[d + 2])

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)

            # End-align the last tile when VAE floor() leaves the output edge
            # uncovered (e.g. pixel width 2325 -> latent 291: last tile fills
            # 286-289 only, index 290 stays out_div==0 -> 0/0 = NaN).
            for d in range(dims):
                if last_axis[d] and upscaled[d] + mask.shape[d + 2] < out.shape[d + 2]:
                    upscaled[d] = max(0, out.shape[d + 2] - mask.shape[d + 2])

            for d in range(2, dims + 2):
                feather = round(get_scale(d - 2, overlap[d - 2]))
                if feather >= mask.shape[d]:
                    continue
                for t in range(feather):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o.add_(ps * mask)
            o_d.add_(mask)

            if pbar is not None:
                pbar.update(1)

        output[b : b + 1] = out / out_div

    return output


def tiled_scale(
    samples,
    function: Callable,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    downscale=False,
    pbar: Optional[tqdm] = None,
):
    return tiled_scale_multidim(
        samples,
        function,
        (tile_y, tile_x),
        overlap=overlap,
        upscale_amount=upscale_amount,
        out_channels=out_channels,
        output_device=output_device,
        downscale=downscale,
        pbar=pbar,
    )


def _vae_device_dtype(vae):
    device = getattr(vae, "device", None) or devices.device
    dtype = devices.dtype_vae
    return device, dtype


def _tile_position_count(dim: int, tile: int, overlap: int) -> int:
    if dim <= tile:
        return 1
    stride = tile - overlap
    if stride <= 0:
        return 1
    return len(range(0, dim - overlap, stride))


def _count_tiled_scale_steps(shape, tile, overlap) -> int:
    """Progress steps for one tiled_scale_multidim invocation (all batch items)."""
    if not isinstance(tile, (tuple, list)):
        tile = (tile, tile)
    if not isinstance(overlap, (tuple, list)):
        overlap = (overlap, overlap)
    dims = len(tile)
    total = 0
    for b in range(shape[0]):
        spatial = shape[2:]
        if all(spatial[d] <= tile[d] for d in range(dims)):
            total += 1
            continue
        n = 1
        for d in range(dims):
            n *= _tile_position_count(int(spatial[d]), int(tile[d]), int(overlap[d]))
        total += n
    return total


def _vae_progress_bar(is_decoder: bool, total: int):
    if total <= 0:
        return None
    return tqdm(
        total=total,
        desc=f"[Tiled VAE]: Executing {'Decoder' if is_decoder else 'Encoder'} Task Queue: ",
    )


def _log_encode_tile_grid(pixel_samples: torch.Tensor) -> None:
    _, _, h, w = pixel_samples.shape
    base = _effective_encode_base(h, w)
    overlap = _encode_overlap_for_base(base)
    passes = _encode_passes(base, overlap)
    grids = [
        (
            _tile_position_count(h, passes[0][0][0], overlap),
            _tile_position_count(w, passes[0][0][1], overlap),
        ),
        (
            _tile_position_count(h, passes[1][0][0], overlap),
            _tile_position_count(w, passes[1][0][1], overlap),
        ),
        (
            _tile_position_count(h, passes[2][0][0], overlap),
            _tile_position_count(w, passes[2][0][1], overlap),
        ),
    ]
    total = sum(g[0] * g[1] for g in grids)
    print(
        f"[Forge VAE] Tiled encode {h}x{w}px - tile grids (3-pass): "
        f"{grids[0][0]}x{grids[0][1]}, {grids[1][0]}x{grids[1][1]}, {grids[2][0]}x{grids[2][1]} "
        f"({passes[0][0][0]}x{passes[0][0][1]} / {passes[1][0][0]}x{passes[1][0][1]} / "
        f"{passes[2][0][0]}x{passes[2][0][1]}px, overlap {overlap}, ~{total} steps; "
        f"base {base}px, accum on CPU)"
    )


def _log_decode_tile_grid(latent_samples: torch.Tensor) -> None:
    _, _, h, w = latent_samples.shape
    base = _DECODE_TILE_BASE
    overlap = _decode_overlap_for_base(base)
    passes = _decode_passes(base, overlap)
    grids = [
        (
            _tile_position_count(h, passes[0][0][0], overlap),
            _tile_position_count(w, passes[0][0][1], overlap),
        ),
        (
            _tile_position_count(h, passes[1][0][0], overlap),
            _tile_position_count(w, passes[1][0][1], overlap),
        ),
        (
            _tile_position_count(h, passes[2][0][0], overlap),
            _tile_position_count(w, passes[2][0][1], overlap),
        ),
    ]
    lh, lw = h * DOWNSCALE_RATIO, w * DOWNSCALE_RATIO
    total = sum(g[0] * g[1] for g in grids)
    print(
        f"[Forge VAE] Tiled decode latent {h}x{w} -> ~{lh}x{lw}px - tile grids (3-pass): "
        f"{grids[0][0]}x{grids[0][1]}, {grids[1][0]}x{grids[1][1]}, {grids[2][0]}x{grids[2][1]} "
        f"({base}px base, overlap {overlap}, ~{total} steps, accum on CPU)"
    )


def _vae_encode_latent_tensor(vae, x: torch.Tensor) -> torch.Tensor:
    """VAE.encode -> BCHW latent mean (SD1.5 DiagonalGaussianDistribution or SDXL tensor)."""
    encoded = vae.encode(x)
    if hasattr(encoded, "mode"):
        return encoded.mode().float()
    return encoded.float()


def _posterior_from_latent_mode(z_mode: torch.Tensor):
    from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

    logvar = torch.zeros_like(z_mode)
    parameters = torch.cat([z_mode, logvar], dim=1)
    return DiagonalGaussianDistribution(parameters, deterministic=True)


def is_diffusion_engine_patched() -> bool:
    return _DIFFUSION_ENGINE_PATCH_APPLIED


def is_latent_diffusion_patched() -> bool:
    return _LATENT_DIFFUSION_PATCH_APPLIED


def is_patch_applied() -> bool:
    return is_diffusion_engine_patched() or is_latent_diffusion_patched()


def applies_to_model(sd_model) -> bool:
    """True when Forge tiled VAE patch is active for this checkpoint family."""
    if sd_model is None:
        return False
    if getattr(sd_model, "is_sdxl", False):
        return is_diffusion_engine_patched()
    if getattr(sd_model, "is_sd3", False):
        return False
    return is_latent_diffusion_patched() and hasattr(sd_model, "first_stage_model")


def _forge_3pass_tiled(
    samples: torch.Tensor,
    fn: Callable,
    passes,
    upscale_amount,
    out_channels: int,
    output_device,
    downscale: bool,
    pbar: Optional[tqdm],
) -> torch.Tensor:
    """Forge encode_tiled_ / decode_tiled_ 3-pass average — no in-place += on tiled tensors."""
    acc = None
    for idx, ((tile_y, tile_x), overlap) in enumerate(passes):
        part = tiled_scale(
            samples,
            fn,
            tile_x,
            tile_y,
            overlap,
            upscale_amount=upscale_amount,
            out_channels=out_channels,
            output_device=output_device,
            downscale=downscale,
            pbar=pbar,
        )
        nan_count = int(torch.isnan(part).sum().item())
        inf_count = int(torch.isinf(part).sum().item())
        absmax = float(part.abs().max().item()) if nan_count == 0 else float('nan')
        print(f'[MD-DIAG] 3pass[{idx}] tile=({tile_y},{tile_x}) ov={overlap} '
              f'part shape={tuple(part.shape)} nan={nan_count} inf={inf_count} absmax={absmax:.4g}')
        acc = part if acc is None else acc + part
    return acc / 3.0


def _encode_tiled(vae, pixel_samples: torch.Tensor, dtype, device) -> torch.Tensor:
    orig_device = pixel_samples.device
    accum_device = _tiled_accum_device()
    if orig_device != accum_device:
        pixel_samples = pixel_samples.to(device=accum_device)
    _log_encode_tile_grid(pixel_samples)
    _, _, h, w = pixel_samples.shape
    base = _effective_encode_base(h, w)
    overlap = _encode_overlap_for_base(base)
    encode_passes = _encode_passes(base, overlap)
    output_device = _tiled_accum_device()

    def encode_fn(a):
        a = a.to(dtype=dtype, device=device)
        try:
            return _vae_encode_latent_tensor(vae, a)
        finally:
            devices.torch_gc()

    upscale = 1.0 / DOWNSCALE_RATIO
    total_steps = sum(
        _count_tiled_scale_steps(pixel_samples.shape, tile, ov)
        for tile, ov in encode_passes
    )
    pbar = _vae_progress_bar(is_decoder=False, total=total_steps)
    try:
        samples = _forge_3pass_tiled(
            pixel_samples,
            encode_fn,
            encode_passes,
            upscale_amount=upscale,
            out_channels=LATENT_CHANNELS,
            output_device=output_device,
            downscale=False,
            pbar=pbar,
        )
    finally:
        if pbar is not None:
            pbar.close()
    devices.torch_gc()
    return samples.to(device=orig_device)


def _decode_tiled(vae, latent_samples: torch.Tensor, dtype, device) -> torch.Tensor:
    orig_device = latent_samples.device
    accum_device = _tiled_accum_device()
    if orig_device != accum_device:
        latent_samples = latent_samples.to(device=accum_device)
    _log_decode_tile_grid(latent_samples)
    base = _DECODE_TILE_BASE
    overlap = _decode_overlap_for_base(base)
    decode_passes = _decode_passes(base, overlap)
    output_device = _tiled_accum_device()

    def decode_fn(a):
        a = a.to(dtype=dtype, device=device)
        try:
            return vae.decode(a).float()
        finally:
            devices.torch_gc()

    upscale = DOWNSCALE_RATIO
    total_steps = sum(
        _count_tiled_scale_steps(latent_samples.shape, tile, ov)
        for tile, ov in decode_passes
    )
    pbar = _vae_progress_bar(is_decoder=True, total=total_steps)
    try:
        output = _forge_3pass_tiled(
            latent_samples,
            decode_fn,
            decode_passes,
            upscale_amount=upscale,
            out_channels=PIXEL_CHANNELS,
            output_device=output_device,
            downscale=False,
            pbar=pbar,
        )
    finally:
        if pbar is not None:
            pbar.close()
    devices.torch_gc()
    return output.to(device=orig_device)


def _encode_full(vae, pixel_samples: torch.Tensor, dtype, device) -> torch.Tensor:
    memory_used = MEMORY_USED_ENCODE(pixel_samples.shape[2], pixel_samples.shape[3], dtype)
    free = _get_free_memory(device)
    batch_number = max(1, int(free / max(1, memory_used)))

    out = None
    for start in range(0, pixel_samples.shape[0], batch_number):
        chunk = pixel_samples[start : start + batch_number].to(dtype=dtype, device=device)
        encoded = _vae_encode_latent_tensor(vae, chunk)
        if out is None:
            out = torch.empty(
                (pixel_samples.shape[0],) + tuple(encoded.shape[1:]),
                device=encoded.device,
                dtype=encoded.dtype,
            )
        out[start : start + batch_number] = encoded
    return out.to(pixel_samples.device)


def _decode_full(vae, latent_samples: torch.Tensor, dtype, device) -> torch.Tensor:
    _, _, h, w = latent_samples.shape
    memory_used = MEMORY_USED_DECODE(h, w, w, dtype)
    free = _get_free_memory(device)
    batch_number = max(1, int(free / max(1, memory_used)))

    out = None
    for start in range(0, latent_samples.shape[0], batch_number):
        chunk = latent_samples[start : start + batch_number].to(dtype=dtype, device=device)
        decoded = vae.decode(chunk).float()
        if out is None:
            out = torch.empty(
                (latent_samples.shape[0],) + tuple(decoded.shape[1:]),
                device=decoded.device,
                dtype=decoded.dtype,
            )
        out[start : start + batch_number] = decoded
    return out.to(latent_samples.device)


def encode_pixels(vae, pixel_samples: torch.Tensor) -> torch.Tensor:
    """Forge VAE.encode() — BCHW pixels in [-1, 1] (A1111 encode_first_stage input)."""
    device, dtype = _vae_device_dtype(vae)

    with _bypass_vae_hooks(vae):
        if VAE_ALWAYS_TILED:
            devices.torch_gc()
            return _encode_tiled(vae, pixel_samples, dtype, device)

        try:
            return _encode_full(vae, pixel_samples, dtype, device)
        except OOM_EXCEPTIONS:
            print(
                "Warning: Encountered Out of Memory during VAE Encoding; "
                "Retrying with Tiled VAE Encoding..."
            )
            devices.torch_gc()
            return _encode_tiled(vae, pixel_samples, dtype, device)


def decode_latent(vae, latent_samples: torch.Tensor) -> torch.Tensor:
    """Forge VAE.decode() — BCHW latents (A1111 decode_first_stage; no Forge process_output clamp)."""
    device, dtype = _vae_device_dtype(vae)

    with _bypass_vae_hooks(vae):
        if VAE_ALWAYS_TILED:
            devices.torch_gc()
            return _decode_tiled(vae, latent_samples, dtype, device)

        try:
            return _decode_full(vae, latent_samples, dtype, device)
        except OOM_EXCEPTIONS:
            print(
                "Warning: Encountered Out of Memory during VAE decoding; "
                "Retrying with Tiled VAE Decoding..."
            )
            devices.torch_gc()
            return _decode_tiled(vae, latent_samples, dtype, device)


@torch.no_grad()
def forge_encode_first_stage(self, x):
    disable = getattr(self, "disable_first_stage_autocast", False)
    with torch.autocast("cuda", enabled=not disable):
        z = encode_pixels(self.first_stage_model, x)
    return self.scale_factor * z


@torch.no_grad()
def forge_decode_first_stage(self, z):
    z = 1.0 / self.scale_factor * z
    disable = getattr(self, "disable_first_stage_autocast", False)
    with torch.autocast("cuda", enabled=not disable):
        out = decode_latent(self.first_stage_model, z)
    return out


@torch.no_grad()
def forge_ldm_encode_first_stage(self, x):
    if hasattr(self, "split_input_params") and self.split_input_params.get("patch_distributed_vq"):
        return self._forge_encode_first_stage_original(x)

    vae = self.first_stage_model
    device, dtype = _vae_device_dtype(vae)
    disable = getattr(self, "disable_first_stage_autocast", False)

    with _bypass_vae_hooks(vae):
        with torch.autocast("cuda", enabled=not disable):
            if VAE_ALWAYS_TILED:
                z_mode = _encode_tiled(vae, x, dtype, device)
                devices.torch_gc()
                return _posterior_from_latent_mode(z_mode)
            try:
                return vae.encode(x)
            except OOM_EXCEPTIONS:
                print(
                    "Warning: Encountered Out of Memory during VAE Encoding; "
                    "Retrying with Tiled VAE Encoding..."
                )
                devices.torch_gc()
                z_mode = _encode_tiled(vae, x, dtype, device)
                devices.torch_gc()
                return _posterior_from_latent_mode(z_mode)


@torch.no_grad()
def forge_ldm_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
    if hasattr(self, "split_input_params") and self.split_input_params.get("patch_distributed_vq"):
        return self._forge_decode_first_stage_original(
            z, predict_cids=predict_cids, force_not_quantize=force_not_quantize
        )

    if predict_cids:
        if z.dim() == 4:
            z = torch.argmax(z.exp(), dim=1).long()
        z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
        from einops import rearrange

        z = rearrange(z, "b h w c -> b c h w").contiguous()

    z = 1.0 / self.scale_factor * z
    disable = getattr(self, "disable_first_stage_autocast", False)
    with torch.autocast("cuda", enabled=not disable):
        vae = self.first_stage_model
        try:
            from ldm.models.autoencoder import VQModelInterface

            if isinstance(vae, VQModelInterface):
                return vae.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        except Exception:
            pass
        return decode_latent(vae, z)


def _patch_latent_diffusion_class(ld_class) -> None:
    global _LATENT_DIFFUSION_PATCH_APPLIED, _patched_latent_diffusion_classes
    if ld_class in _patched_latent_diffusion_classes:
        return

    ld_class._forge_encode_first_stage_original = ld_class.encode_first_stage
    ld_class._forge_decode_first_stage_original = ld_class.decode_first_stage
    ld_class.encode_first_stage = forge_ldm_encode_first_stage
    ld_class.decode_first_stage = forge_ldm_decode_first_stage
    ld_class.encode_first_stage.__forge_tiled_vae__ = True
    ld_class.decode_first_stage.__forge_tiled_vae__ = True
    _patched_latent_diffusion_classes.add(ld_class)


def apply_latent_diffusion_vae_patch() -> None:
    global _LATENT_DIFFUSION_PATCH_APPLIED
    import ldm.models.diffusion.ddpm as ldm_ddpm

    _patch_latent_diffusion_class(ldm_ddpm.LatentDiffusion)
    try:
        import modules.models.diffusion.ddpm_edit as ddpm_edit

        _patch_latent_diffusion_class(ddpm_edit.LatentDiffusion)
    except Exception:
        pass
    _LATENT_DIFFUSION_PATCH_APPLIED = len(_patched_latent_diffusion_classes) > 0


def apply_diffusion_engine_vae_patch() -> None:
    global _DIFFUSION_ENGINE_PATCH_APPLIED
    import sgm.models.diffusion as diffusion_module

    de = diffusion_module.DiffusionEngine
    if _DIFFUSION_ENGINE_PATCH_APPLIED:
        return

    de._forge_encode_first_stage_original = de.encode_first_stage
    de._forge_decode_first_stage_original = de.decode_first_stage
    de.encode_first_stage = forge_encode_first_stage
    de.decode_first_stage = forge_decode_first_stage
    de.encode_first_stage.__forge_tiled_vae__ = True
    de.decode_first_stage.__forge_tiled_vae__ = True
    _DIFFUSION_ENGINE_PATCH_APPLIED = True


def apply_all_vae_patches() -> None:
    apply_diffusion_engine_vae_patch()
    apply_latent_diffusion_vae_patch()

```

### 4-2. `modules/sd_models_xl.py` (added lines)

```python
from modules import devices, forge_tiled_vae, shared, prompt_parser
```

```python
# Forge tiled VAE at encode_first_stage / decode_first_stage (SD1.5 LatentDiffusion + SDXL DiffusionEngine).
forge_tiled_vae.apply_all_vae_patches()
```

**Meaning:** Patches `DiffusionEngine` and `LatentDiffusion` at import so Forge tiled VAE applies when UI enables Tiled VAE.

---

### 4-3. `tile_utils/utils.py` (added functions)

```python
def pixel_to_latent_h(px: int) -> int:
    """VAE latent height for SD1.x (e.g. 1945px -> 243, not 244)."""
    return int(px) // opt_f


def pixel_to_latent_w(px: int) -> int:
    """VAE latent width when pixels are not a multiple of 8 (e.g. 2325px -> 291)."""
    return (int(px) + opt_f - 1) // opt_f
```

**Meaning:** Canvas `self.h`/`self.w` match VAE latent geometry. Width ceil fixes 2325→291.

---

### 4-4. `scripts/tilevae.py` (added/changed in `process` and `postprocess`)

When Tiled VAE is disabled, undo Forge always-tiled and any legacy VAEHook:

```python
        if not enabled:
            try:
                from modules import forge_tiled_vae
                if forge_tiled_vae.is_patch_applied():
                    forge_tiled_vae.set_vae_always_tiled(False)
            except Exception:
                pass
            if self.hooked:
                if isinstance(encoder.forward, VAEHook):
                    encoder.forward.net = None
                    encoder.forward = encoder.original_forward
                if isinstance(decoder.forward, VAEHook):
                    decoder.forward.net = None
                    decoder.forward = decoder.original_forward
                self.hooked = False
            return
```

VRAM caps from output pixel count, then Forge path (skip VAEHook):

```python
        w = int(getattr(p, 'width', 0) or 0)
        h = int(getattr(p, 'height', 0) or 0)
        if w <= 0 or h <= 0:
            init_images = getattr(p, 'init_images', None) or []
            if init_images:
                w = int(getattr(init_images[0], 'width', 0) or 0)
                h = int(getattr(init_images[0], 'height', 0) or 0)
        pixels = w * h
        hook_enc_tsize = int(encoder_tile_size)
        hook_dec_tsize = int(decoder_tile_size)
        if pixels >= 4_000_000:
            hook_enc_tsize = min(hook_enc_tsize, 192)
        elif pixels >= 2_500_000:
            hook_enc_tsize = min(hook_enc_tsize, 256)
        else:
            hook_enc_tsize = min(hook_enc_tsize, 512)
        hook_dec_tsize = min(hook_dec_tsize, 256)
        if hook_enc_tsize != encoder_tile_size or hook_dec_tsize != decoder_tile_size:
            print(
                f"[Tiled VAE] VRAM cap for {w}x{h}: encoder tile {encoder_tile_size} -> {hook_enc_tsize}, "
                f"decoder tile {decoder_tile_size} -> {hook_dec_tsize}"
            )

        try:
            from modules import forge_tiled_vae
            if forge_tiled_vae.applies_to_model(p.sd_model):
                forge_tiled_vae.set_vae_tile_sizes(hook_enc_tsize, hook_dec_tsize)
                forge_tiled_vae.set_vae_always_tiled(True)
                print(
                    "[Forge VAE] SD1.5/2.x Forge encode/decode patch active — skipping multidiffusion VAEHook "
                    f"(encoder {hook_enc_tsize}px / decoder {hook_dec_tsize}px, 3-pass, accum CPU)."
                )
                return
        except Exception:
            pass

        # Fallback: legacy VAEHook when Forge class patch is unavailable.
        kwargs = {
            'fast_decoder': fast_decoder,
            'fast_encoder': fast_encoder,
            'color_fix':    color_fix,
            'to_gpu':       vae_to_gpu,
        }

        if not hasattr(encoder, 'original_forward'): setattr(encoder, 'original_forward', encoder.forward)
        if not hasattr(decoder, 'original_forward'): setattr(decoder, 'original_forward', decoder.forward)

        self.hooked = True

        encoder.forward = VAEHook(encoder, hook_enc_tsize, is_decoder=False, **kwargs)
        decoder.forward = VAEHook(decoder, hook_dec_tsize, is_decoder=True,  **kwargs)
```

Postprocess clears Forge flag or restores VAEHook originals:

```python
    def postprocess(self, p:Processing, processed, enabled:bool, *args):
        if not enabled: return

        try:
            from modules import forge_tiled_vae
            if forge_tiled_vae.applies_to_model(p.sd_model):
                forge_tiled_vae.set_vae_always_tiled(False)
                devices.torch_gc()
                return
        except Exception:
            pass

        vae = p.sd_model.first_stage_model
        encoder = vae.encoder
        decoder = vae.decoder
        if isinstance(encoder.forward, VAEHook):
            encoder.forward.net = None
            encoder.forward = encoder.original_forward
        if isinstance(decoder.forward, VAEHook):
            decoder.forward.net = None
            decoder.forward = decoder.original_forward
```

**Meaning:** Large images cap encoder tile size (192/256/512). If Forge patch applies to SD1.5/2.x or SDXL, set Forge tile sizes and `VAE_ALWAYS_TILED=True`, then **return without VAEHook**. Postprocess clears always-tiled flag or unhooks legacy VAEHook.

---

### 4-5. `tile_methods/abstractdiffusion.py`

#### `_align_latent_to_canvas` (new)

```python
def _align_latent_to_canvas(t: Tensor, lh: int, lw: int) -> Tensor:
    """Resize a latent tensor `t` (shape [..., H, W]) to (lh, lw) by edge-row/col replication
    when growing, or center-crop when shrinking. Designed for the ceil/floor mismatch between
    A1111's floor(width/8) noise tensors and Forge tiled VAE's ceil-rounded init_latent.
    No statistical drift: replicated rows/cols come from the existing noise distribution."""
    h_old, w_old = t.shape[-2], t.shape[-1]
    if (h_old, w_old) == (lh, lw):
        return t
    h_take = min(h_old, lh)
    w_take = min(w_old, lw)
    out = torch.empty(*t.shape[:-2], lh, lw, dtype=t.dtype, device=t.device)
    out[..., :h_take, :w_take] = t[..., :h_take, :w_take]
    if lw > w_old:
        out[..., :h_take, w_old:lw] = t[..., :h_take, w_old - 1:w_old]
    if lh > h_old:
        out[..., h_old:lh, :] = out[..., h_old - 1:h_old, :]
    return out
```

**Meaning:** Grows `noise`/`x` to match `init_latent` by replicating last row/column — no new random values, no mean shift.

#### Canvas init (changed)

```python
        self.w: int = pixel_to_latent_w(self.p.width)
        self.h: int = pixel_to_latent_h(self.p.height)
```

#### `_rebuild_latent_canvas` (new)

```python
    def _rebuild_latent_canvas(self, h: int, w: int) -> bool:
        if self.h == h and self.w == w:
            return False
        old_h, old_w = self.h, self.w
        self.h, self.w = h, w
        self.weights = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
        if self.enable_grid_bbox and self._grid_tile_w_cfg is not None:
            tile_w = min(self._grid_tile_w_cfg, self.w)
            tile_h = min(self._grid_tile_h_cfg, self.h)
            overlap = max(0, min(self._grid_overlap, min(tile_w, tile_h) - 4))
            bboxes, weights = split_bboxes(self.w, self.h, tile_w, tile_h, overlap, self.get_tile_weights())
            self.weights = weights
            self.num_tiles = len(bboxes)
            self.num_batches = math.ceil(self.num_tiles / self._grid_tile_bs_cfg)
            self.tile_bs = math.ceil(len(bboxes) / self.num_batches)
            self.tile_w = tile_w
            self.tile_h = tile_h
            self.batched_bboxes = [bboxes[i * self.tile_bs:(i + 1) * self.tile_bs] for i in range(self.num_batches)]
        if self.enable_custom_bbox and old_h > 0 and old_w > 0:
            scale_h = self.h / old_h
            scale_w = self.w / old_w
            for bbox in self.custom_bboxes:
                bbox.x = int(round(bbox.x * scale_w))
                bbox.y = int(round(bbox.y * scale_h))
                bbox.w = max(1, int(round(bbox.w * scale_w)))
                bbox.h = max(1, int(round(bbox.h * scale_h)))
                bbox.w = min(bbox.w, self.w - bbox.x)
                bbox.h = min(bbox.h, self.h - bbox.y)
                bbox.box = [bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h]
                bbox.slicer = slice(None), slice(None), slice(bbox.y, bbox.y + bbox.h), slice(bbox.x, bbox.x + bbox.w)
                if bbox.feather_mask is not None:
                    bbox.feather_mask = feather_mask(bbox.w, bbox.h, bbox.feather_ratio)
        print(f'[Tiled Diffusion] Realign latent canvas {old_h}x{old_w} -> {self.h}x{self.w}')
        return True
```

**Meaning:** When `init_latent` size differs from UI canvas, rebuild weights and tile grid; rescale custom bboxes proportionally.

#### `init_grid_bbox` (stores config for rebuild)

```python
        self._grid_tile_w_cfg = tile_w
        self._grid_tile_h_cfg = tile_h
        self._grid_overlap = overlap
        self._grid_tile_bs_cfg = tile_bs
```

#### ControlNet helpers (new)

```python
    def _hint_pixel_size_from_x_spatial(self, h: int, w: int) -> tuple[int, int]:
        """Map x_in spatial dims to ControlNet hint pixels (never overshoot canvas)."""
        px_h, px_w = int(self.p.height), int(self.p.width)
        if h > self.h * 2 or w > self.w * 2:
            return min(max(1, h), px_h), min(max(1, w), px_w)
        return h * opt_f, w * opt_f

    def set_controlnet_tensors_for_size(self, h_latent:int, w_latent:int):
        '''Crop ControlNet hint to match latent tile spatial size.'''
        if not self.enable_controlnet: return
        if self.org_control_tensor_batch is None: return

        target_h, target_w = self._hint_pixel_size_from_x_spatial(h_latent, w_latent)
        print(
            f'[Tiled Diffusion] Crop ControlNet hint to {target_h}x{target_w} '
            f'(latent {h_latent}x{w_latent}, canvas {int(self.p.height)}x{int(self.p.width)})'
        )
        for param_id in range(len(self.control_params)):
            param = self.control_params[param_id]
            full_hint = self.org_control_tensor_batch[param_id]
            _, _, fh, fw = full_hint.shape
            crop_h = min(fh, target_h)
            crop_w = min(fw, target_w)
            cropped = full_hint[:, :, :crop_h, :crop_w]
            if crop_h < target_h or crop_w < target_w:
                cropped = torch.nn.functional.interpolate(
                    cropped, size=(target_h, target_w), mode='bilinear', align_corners=False,
                )
            param.hint_cond = cropped.to(devices.device)
            if isinstance(param.hr_hint_cond, torch.Tensor):
                _, _, h_hr, w_hr = param.hr_hint_cond.shape
                crop_h_hr = min(h_hr, target_h)
                crop_w_hr = min(w_hr, target_w)
                cropped_hr = param.hr_hint_cond[:, :, :crop_h_hr, :crop_w_hr]
                if crop_h_hr < target_h or crop_w_hr < target_w:
                    cropped_hr = torch.nn.functional.interpolate(
                        cropped_hr, size=(target_h, target_w), mode='bilinear', align_corners=False,
                    )
                param.hr_hint_cond = cropped_hr.to(devices.device)

    def _crop_controlnet_tile(self, control_tensor: Tensor, bbox: BBox) -> Tensor:
        """Crop ControlNet hint for one latent tile; clip to hint bounds; uniform pixel size."""
        if control_tensor.ndim == 3:
            control_tensor = control_tensor.unsqueeze(0)

        th = bbox[3] - bbox[1]
        tw = bbox[2] - bbox[0]
        target_h, target_w = th * opt_f, tw * opt_f

        _, _, fh, fw = control_tensor.shape
        y0 = max(0, min(bbox[1] * opt_f, fh))
        y1 = max(0, min(bbox[3] * opt_f, fh))
        x0 = max(0, min(bbox[0] * opt_f, fw))
        x1 = max(0, min(bbox[2] * opt_f, fw))

        if y1 > y0 and x1 > x0:
            control_tile = control_tensor[:, :, y0:y1, x0:x1]
        else:
            control_tile = control_tensor[:, :, : min(target_h, fh), : min(target_w, fw)]

        if control_tile.shape[-2] != target_h or control_tile.shape[-1] != target_w:
            control_tile = torch.nn.functional.interpolate(
                control_tile, size=(target_h, target_w), mode='bilinear', align_corners=False,
            )
        return control_tile
```

`prepare_controlnet_tensors` now uses `_crop_controlnet_tile` per bbox instead of raw latent slicing only:

```python
    def prepare_controlnet_tensors(self, refresh:bool=False):
        ''' Crop the control tensor into tiles and cache them '''

        if not refresh:
            if self.control_tensor_batch is not None or self.control_params is not None: return

        if not self.enable_controlnet or self.controlnet_script is None: return

        latest_network = self.controlnet_script.latest_network
        if latest_network is None or not hasattr(latest_network, 'control_params'): return

        self.control_params = latest_network.control_params
        tensors = [param.hint_cond for param in latest_network.control_params]
        self.org_control_tensor_batch = tensors

        if len(tensors) == 0: return

        self.control_tensor_batch = []
        for i in range(len(tensors)):
            control_tile_list = []
            control_tensor = tensors[i]
            for bboxes in self.batched_bboxes:
                single_batch_tensors = []
                for bbox in bboxes:
                    single_batch_tensors.append(self._crop_controlnet_tile(control_tensor, bbox))
                control_tile = torch.cat(single_batch_tensors, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                control_tile_list.append(control_tile)
            self.control_tensor_batch.append(control_tile_list)

            if len(self.custom_bboxes) > 0:
                custom_control_tile_list = []
                for bbox in self.custom_bboxes:
                    control_tile = self._crop_controlnet_tile(control_tensor, bbox)
                    if self.control_tensor_cpu:
                        control_tile = control_tile.cpu()
                    custom_control_tile_list.append(control_tile)
                self.control_tensor_custom.append(custom_control_tile_list)
```

#### `switch_controlnet_tensors` (tile_offset)

```python
    def switch_controlnet_tensors(self, batch_id:int, x_batch_size:int, tile_batch_size:int, is_denoise=False, tile_offset:int=0):
        if not self.enable_controlnet: return
        if self.control_tensor_batch is None: return

        for param_id in range(len(self.control_params)):
            batch_tiles = self.control_tensor_batch[param_id][batch_id]
            control_tile = batch_tiles[tile_offset:tile_offset + tile_batch_size]
            if self.is_kdiff:
                all_control_tile = []
                for i in range(tile_batch_size):
                    this_control_tile = [control_tile[i].unsqueeze(0)] * x_batch_size
                    all_control_tile.append(torch.cat(this_control_tile, dim=0))
                control_tile = torch.cat(all_control_tile, dim=0)
            else:
                control_tile = control_tile.repeat([x_batch_size if is_denoise else x_batch_size * 2, 1, 1, 1])
            self.control_params[param_id].hint_cond = control_tile.to(devices.device)
```

#### `sample_img2img` prologue (alignment + rebuild + diag)

```python
        _, _, _lh, _lw = p.init_latent.shape
        if noise.shape[-2:] != (_lh, _lw):
            noise = _align_latent_to_canvas(noise, _lh, _lw)
        if x.shape[-2:] != (_lh, _lw):
            x = _align_latent_to_canvas(x, _lh, _lw)
```

```python
        x = self.p.init_latent
        _, _, lh, lw = x.shape
        if (lh, lw) != (self.h, self.w):
            self._rebuild_latent_canvas(lh, lw)
            if self.enable_controlnet:
                self.set_controlnet_tensors_for_size(lh, lw)
        print(f'[MD-DIAG] init_latent shape={tuple(x.shape)} dtype={x.dtype} '
              f'nan={torch.isnan(x).any().item()} inf={torch.isinf(x).any().item()} '
              f'min={float(x.min().item()):.4g} max={float(x.max().item()):.4g} '
              f'absmax={float(x.abs().max().item()):.4g}')
```

**Meaning:** Align noise before renoise mask; after inversion setup, rebuild canvas to encoded latent size and refresh ControlNet hints.

---

### 4-6. `tile_methods/multidiffusion.py`

#### `repeat_func` — one tile per forward (kdiff and default)

K-Diffusion path (`kdiff_forward`):

```python
        def repeat_func(x_tile:Tensor, bboxes:List[CustomBBox]) -> Tensor:
            # Process one tile at a time to cap VRAM peak (ControlNet + tiled latent).
            outs = []
            batch_per_tile = x_tile.shape[0] // len(bboxes)
            for i, bbox in enumerate(bboxes):
                xt = x_tile[i * batch_per_tile:(i + 1) * batch_per_tile]
                sigma_tile = self.repeat_tensor(sigma_in, 1)
                cond_tile = self.repeat_cond_dict(cond, [bbox])
                outs.append(self.sampler_forward(xt, sigma_tile, cond=cond_tile))
            return torch.cat(outs, dim=0)
```

DDIM / timestep path (`ddim_forward`):

```python
        def repeat_func(x_tile:Tensor, bboxes:List[CustomBBox]) -> Tuple[Tensor, Tensor]:
            outs = []
            batch_per_tile = x_tile.shape[0] // len(bboxes)
            for i, bbox in enumerate(bboxes):
                xt = x_tile[i * batch_per_tile:(i + 1) * batch_per_tile]
                ts_tile = self.repeat_tensor(ts_in, 1)
                if isinstance(cond, dict):
                    cond_tile = self.repeat_cond_dict(cond, [bbox])
                else:
                    cond_tile = self.repeat_tensor(cond, 1)
                outs.append(self.sampler_forward(xt, ts_tile, cond=cond_tile))
            return torch.cat(outs, dim=0)
```

#### `_pixel_slicer` (new)

```python
    def _pixel_slicer(self, bbox:BBox) -> tuple:
        '''latent-space bbox -> pixel-space slicer for ControlNet hint (VAE downscale=8)'''
        return (
            slice(None),
            slice(None),
            slice(bbox.y * 8, (bbox.y + bbox.h) * 8),
            slice(bbox.x * 8, (bbox.x + bbox.w) * 8),
        )
```

#### `_slice_icond_for_bboxes` (new)

```python
    def _slice_icond_for_bboxes(self, icond:Tensor, bboxes:List[CustomBBox]) -> Tensor:
        if icond.shape[2:] == (self.h, self.w):
            return torch.cat([icond[bbox.slicer] for bbox in bboxes], dim=0)
        if icond.shape[2:] == (self.h * 8, self.w * 8):
            return torch.cat([icond[self._pixel_slicer(bbox)] for bbox in bboxes], dim=0)
        return self.repeat_tensor(icond, len(bboxes))
```

#### `repeat_cond_dict` (icond path changed)

```python
    def repeat_cond_dict(self, cond_in:CondDict, bboxes:List[CustomBBox]) -> CondDict:
        ''' repeat all tensors in cond_dict on it's first dim (for a batch of tiles), returns a new object '''
        # n_repeat
        n_rep = len(bboxes)
        # txt cond
        tcond = self.get_tcond(cond_in)           # [B=1, L, D] => [B*N, L, D]
        tcond = self.repeat_tensor(tcond, n_rep)
        # img cond (ControlNet hint)
        icond = self._slice_icond_for_bboxes(self.get_icond(cond_in), bboxes)
        # vec cond (SDXL)
        vcond = self.get_vcond(cond_in)           # [B=1, D]
        if vcond is not None:
            vcond = self.repeat_tensor(vcond, n_rep)  # [B*N, D]
        return self.make_cond_dict(cond_in, tcond, icond, vcond)
```

#### `get_noise` — per-tile `apply_model` with `_slice_icond_for_bboxes`

```python
        def repeat_func(x_tile:Tensor, bboxes:List[CustomBBox]):
            outs = []
            batch_per_tile = x_tile.shape[0] // len(bboxes)
            for i, bbox in enumerate(bboxes):
                xt = x_tile[i * batch_per_tile:(i + 1) * batch_per_tile]
                cond_out = self.repeat_cond_dict(cond_in_original, [bbox])
                outs.append(shared.sd_model.apply_model(xt, sigma_in, cond=cond_out))
            return torch.cat(outs, dim=0)

        def custom_func(x:Tensor, bbox_id:int, bbox:CustomBBox):
            tcond = Condition.reconstruct_cond(bbox.cond, step).unsqueeze_(0)
            icond = self._slice_icond_for_bboxes(self.get_icond(cond_in_original), [bbox])
            cond_out = self.make_cond_dict(cond_in, tcond, icond)
            return shared.sd_model.apply_model(x, sigma_in, cond=cond_out)
```

#### `sample_one_step` — rebuild instead of org_func

```python
        if (H, W) != (self.h, self.w):
            self._rebuild_latent_canvas(H, W)
            if self.enable_controlnet:
                self.set_controlnet_tensors_for_size(H, W)
```

#### Micro-batch with ControlNet

```python
                tb = len(bboxes)
                if self.enable_controlnet:
                    micro_plan = [1] * tb
                elif tb == 6:
                    micro_plan = [3, 3]
                elif tb >= 4:
                    micro_plan = [2] * (tb // 2)
                    if tb % 2:
                        micro_plan.append(1)
                else:
                    micro_plan = [tb]

                if micro_plan == [tb]:
                    self.switch_controlnet_tensors(batch_id, N, tb, tile_offset=0)
                    x_tile_out = repeat_func(x_tile, bboxes)
                else:
                    outs = []
                    k = 0
                    for m in micro_plan:
                        bb = bboxes[k:k+m]
                        xt = x_tile[k * N:(k + m) * N, :, :, :]
                        self.switch_controlnet_tensors(batch_id, N, m, tile_offset=k)
                        outs.append(repeat_func(xt, bb))
                        k += m
                    x_tile_out = torch.cat(outs, dim=0)
```

#### NaN diagnostic

```python
        if torch.isnan(x_out).any() or torch.isinf(x_out).any():
            nan_buf = torch.isnan(self.x_buffer).any().item()
            inf_buf = torch.isinf(self.x_buffer).any().item()
            w_min = float(self.weights.min().item())
            w_zero = int((self.weights == 0).sum().item())
            print(f'[MD-NaN] step={state.sampling_step} shape={tuple(x_out.shape)} '
                  f'buf_nan={nan_buf} buf_inf={inf_buf} w_min={w_min} w_zero={w_zero}')
```

**Meaning:** Never fall back to full UNet; tile one at a time with ControlNet; log NaN source hints.

---

### 4-7. `tile_methods/demofusion.py` (`repeat_cond_dict` — icond handling)

Inside `repeat_cond_dict`, after `icond = self.get_icond(cond_in)`:

```python
        # img cond (ControlNet hint or latent mask)
        icond = self.get_icond(cond_in)
        if icond.shape[2:] == (self.h, self.w):   # latent-space mask
            if mode == 0:
                if self.p.random_jitter:
                    jitter_range = self.jitter_range
                    icond = F.pad(icond,(jitter_range, jitter_range, jitter_range, jitter_range),'constant',value=0)
                icond = torch.cat([icond[bbox.slicer] for bbox in bboxes], dim=0)
            else:
                icond = torch.cat([icond[:,:,bbox[1]::self.p.current_scale_num,bbox[0]::self.p.current_scale_num] for bbox in bboxes], dim=0)
        elif icond.shape[2:] == (self.h * 8, self.w * 8):  # pixel-space ControlNet hint
            if mode == 0:
                if self.p.random_jitter:
                    jitter_range = self.jitter_range
                    icond = F.pad(icond,(jitter_range, jitter_range, jitter_range, jitter_range),'constant',value=0)
                icond = torch.cat([
                    icond[:, :, bbox.y * 8:(bbox.y + bbox.h) * 8, bbox.x * 8:(bbox.x + bbox.w) * 8]
                    for bbox in bboxes
                ], dim=0)
            else:
                icond = torch.cat([icond[:,:,bbox[1]::self.p.current_scale_num,bbox[0]::self.p.current_scale_num] for bbox in bboxes], dim=0)
        else:                                     # txt2img dummy hint
            icond = self.repeat_tensor(icond, n_rep)
```

**Meaning:** DemoFusion batches tiles with `repeat_cond_dict`. Latent masks use `bbox.slicer` (or stride subsampling in scale mode). **New:** when ControlNet hint is full pixel resolution `(h*8, w*8)`, slice each tile with `bbox.y/x * 8` in pixel space — same geometry as MultiDiffusion, with jitter padding preserved for mode 0.

---

### 4-8. `tile_methods/mixtureofdiffusers.py` (pixel hint branch)

Grid bbox batching — inside `sample_one_step`, when building `icond_tile_list` per tile:

```python
                        # icond: might be dummy for txt2img, latent-space or pixel-space ControlNet hint
                        icond = self.get_icond(c_in)
                        if icond.shape[2:] == (self.h, self.w):
                            icond = icond[bbox.slicer]
                        elif icond.shape[2:] == (self.h * 8, self.w * 8):
                            icond = icond[
                                :, :,
                                bbox.y * 8:(bbox.y + bbox.h) * 8,
                                bbox.x * 8:(bbox.x + bbox.w) * 8
                            ]
                        icond_tile_list.append(icond)
```

Custom bbox region — noise-inversion branch when `noise_inverse_step >= 0`:

```python
                    tcond = Condition.reconstruct_cond(bbox.cond, noise_inverse_step)
                    icond = self.get_icond(c_in)
                    if icond.shape[2:] == (self.h, self.w):
                        icond = icond[bbox.slicer]
                    elif icond.shape[2:] == (self.h * 8, self.w * 8):
                        icond = icond[
                            :, :,
                            bbox.y * 8:(bbox.y + bbox.h) * 8,
                            bbox.x * 8:(bbox.x + bbox.w) * 8
                        ]
                    vcond = self.get_vcond(c_in)
                    c_out = self.make_cond_dict(c_in, tcond, icond, vcond)
                    x_tile_out = shared.sd_model.apply_model(x_tile, t_in, cond=c_out)
```

**Meaning:** Mixture-of-Diffusers batches tiles differently from MultiDiffusion, but the same rule applies: latent hints use `bbox.slicer`; full-pixel hints (`h*8`, `w*8`) are cropped in pixel space with `bbox.y/x * 8` so each tile’s `icond` matches its latent footprint.

---

## 5. Meaning of the integration (how the pieces connect)

1. **Startup:** `sd_models_xl.py` patches VAE entry points on SDXL `DiffusionEngine` and SD1.5 `LatentDiffusion`.
2. **User enables Tiled VAE:** `tilevae.py` sets Forge tile sizes from image pixels, sets `VAE_ALWAYS_TILED=True`, skips VAEHook.
3. **Encode:** `forge_tiled_vae.encode_pixels` runs 3-pass tiled encode with CPU accumulation; end-align fix prevents NaN on non-multiple-of-8 widths; returns latent at **ceil/floor** correct size.
4. **Canvas:** `AbstractDiffusion` initializes with `pixel_to_latent_*`; if `init_latent` differs, `_rebuild_latent_canvas` resizes tile grid and weights.
5. **ControlNet:** Hints cropped to `init_latent` spatial size × 8; per-tile crops use `_crop_controlnet_tile` with interpolate fallback.
6. **UNet steps:** `multidiffusion.sample_one_step` never calls `org_func` on mismatch; processes **one tile per forward** when ControlNet enabled.
7. **Noise Inversion:** `_align_latent_to_canvas` on `noise`/`x` before renoise mask so sizes match `init_latent`.
8. **Decode:** Same Forge 3-pass tiled decode path; postprocess clears `VAE_ALWAYS_TILED`.

---

## 6. Key log lines (sanity check)

| Log | Meaning |
|-----|---------|
| `[Forge VAE] SD1.5/2.x Forge encode/decode patch active — skipping multidiffusion VAEHook` | Forge path active; no VAEHook |
| `[Forge VAE] Tiled encode … (3-pass)` | 3-pass encode running |
| `[Tiled Diffusion] Realign latent canvas …` | Canvas rebuilt to match `init_latent` |
| `[Tiled Diffusion] Crop ControlNet hint to …` | Hint matched to latent tile size |
| `[MD-DIAG] init_latent shape=…` | Latent stats after encode |
| `[MD-NaN] step=…` | NaN in blended output — check VAE edge or weights |

---

*End of integration document.*
