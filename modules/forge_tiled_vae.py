"""
Forge (ComfyUI-style) tiled VAE for SDXL — ported from Forge backend/patcher/vae.py.

Forge backend/patcher/vae.py parity for SDXL DiffusionEngine encode/decode:
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


def set_vae_always_tiled(enabled: bool) -> None:
    global VAE_ALWAYS_TILED
    VAE_ALWAYS_TILED = bool(enabled)


def is_vae_always_tiled() -> bool:
    return VAE_ALWAYS_TILED


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


@torch.inference_mode()
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

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                length = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, length)
                upscaled.append(round(get_pos(d, pos)))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)

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
    overlap = 64
    grids = [
        (_tile_position_count(h, 512, overlap), _tile_position_count(w, 512, overlap)),
        (_tile_position_count(h, 1024, overlap), _tile_position_count(w, 256, overlap)),
        (_tile_position_count(h, 256, overlap), _tile_position_count(w, 1024, overlap)),
    ]
    print(
        f"[Forge VAE] Tiled encode {h}x{w}px - tile grids (3-pass): "
        f"{grids[0][0]}x{grids[0][1]}, {grids[1][0]}x{grids[1][1]}, {grids[2][0]}x{grids[2][1]} "
        f"(512/1024/256px tiles, overlap {overlap}; not 1x1 @3072)"
    )


def _log_decode_tile_grid(latent_samples: torch.Tensor) -> None:
    _, _, h, w = latent_samples.shape
    overlap = 16
    grids = [
        (_tile_position_count(h, 32, overlap), _tile_position_count(w, 128, overlap)),
        (_tile_position_count(h, 128, overlap), _tile_position_count(w, 32, overlap)),
        (_tile_position_count(h, 64, overlap), _tile_position_count(w, 64, overlap)),
    ]
    lh, lw = h * DOWNSCALE_RATIO, w * DOWNSCALE_RATIO
    print(
        f"[Forge VAE] Tiled decode latent {h}x{w} -> ~{lh}x{lw}px - tile grids (3-pass): "
        f"{grids[0][0]}x{grids[0][1]}, {grids[1][0]}x{grids[1][1]}, {grids[2][0]}x{grids[2][1]} "
        f"(64px base, overlap {overlap})"
    )


def is_patch_applied() -> bool:
    try:
        import sgm.models.diffusion as diffusion_module

        return getattr(diffusion_module.DiffusionEngine.encode_first_stage, "__forge_tiled_vae__", False)
    except Exception:
        return False


def _encode_tiled(vae, pixel_samples: torch.Tensor, dtype, device) -> torch.Tensor:
    _log_encode_tile_grid(pixel_samples)
    output_device = device

    def encode_fn(a):
        a = a.to(dtype=dtype, device=device)
        return vae.encode(a).float()

    upscale = 1.0 / DOWNSCALE_RATIO
    encode_passes = (
        ((512, 512), 64),
        ((512 * 2, 512 // 2), 64),
        ((512 // 2, 512 * 2), 64),
    )
    total_steps = sum(
        _count_tiled_scale_steps(pixel_samples.shape, tile, overlap)
        for tile, overlap in encode_passes
    )
    pbar = _vae_progress_bar(is_decoder=False, total=total_steps)
    try:
        samples = tiled_scale(
            pixel_samples,
            encode_fn,
            512,
            512,
            64,
            upscale_amount=upscale,
            out_channels=LATENT_CHANNELS,
            output_device=output_device,
            downscale=True,
            pbar=pbar,
        )
        samples += tiled_scale(
            pixel_samples,
            encode_fn,
            512 * 2,
            512 // 2,
            64,
            upscale_amount=upscale,
            out_channels=LATENT_CHANNELS,
            output_device=output_device,
            downscale=True,
            pbar=pbar,
        )
        samples += tiled_scale(
            pixel_samples,
            encode_fn,
            512 // 2,
            512 * 2,
            64,
            upscale_amount=upscale,
            out_channels=LATENT_CHANNELS,
            output_device=output_device,
            downscale=True,
            pbar=pbar,
        )
    finally:
        if pbar is not None:
            pbar.close()
    samples /= 3.0
    return samples


def _decode_tiled(vae, latent_samples: torch.Tensor, dtype, device) -> torch.Tensor:
    _log_decode_tile_grid(latent_samples)
    output_device = device

    def decode_fn(a):
        a = a.to(dtype=dtype, device=device)
        return vae.decode(a).float()

    upscale = DOWNSCALE_RATIO
    decode_passes = (
        ((64 // 2, 64 * 2), 16),
        ((64 * 2, 64 // 2), 16),
        ((64, 64), 16),
    )
    total_steps = sum(
        _count_tiled_scale_steps(latent_samples.shape, tile, overlap)
        for tile, overlap in decode_passes
    )
    pbar = _vae_progress_bar(is_decoder=True, total=total_steps)
    try:
        output = (
            tiled_scale(
                latent_samples,
                decode_fn,
                64 // 2,
                64 * 2,
                16,
                upscale_amount=upscale,
                out_channels=PIXEL_CHANNELS,
                output_device=output_device,
                pbar=pbar,
            )
            + tiled_scale(
                latent_samples,
                decode_fn,
                64 * 2,
                64 // 2,
                16,
                upscale_amount=upscale,
                out_channels=PIXEL_CHANNELS,
                output_device=output_device,
                pbar=pbar,
            )
            + tiled_scale(
                latent_samples,
                decode_fn,
                64,
                64,
                16,
                upscale_amount=upscale,
                out_channels=PIXEL_CHANNELS,
                output_device=output_device,
                pbar=pbar,
            )
        ) / 3.0
    finally:
        if pbar is not None:
            pbar.close()
    return output


def _encode_full(vae, pixel_samples: torch.Tensor, dtype, device) -> torch.Tensor:
    memory_used = MEMORY_USED_ENCODE(pixel_samples.shape[2], pixel_samples.shape[3], dtype)
    free = _get_free_memory(device)
    batch_number = max(1, int(free / max(1, memory_used)))

    out = None
    for start in range(0, pixel_samples.shape[0], batch_number):
        chunk = pixel_samples[start : start + batch_number].to(dtype=dtype, device=device)
        encoded = vae.encode(chunk).float()
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


def apply_diffusion_engine_vae_patch():
    import sgm.models.diffusion as diffusion_module

    de = diffusion_module.DiffusionEngine
    if getattr(de.encode_first_stage, "__forge_tiled_vae__", False):
        return

    de._forge_encode_first_stage_original = de.encode_first_stage
    de._forge_decode_first_stage_original = de.decode_first_stage
    de.encode_first_stage = forge_encode_first_stage
    de.decode_first_stage = forge_decode_first_stage
    de.encode_first_stage.__forge_tiled_vae__ = True
    de.decode_first_stage.__forge_tiled_vae__ = True
