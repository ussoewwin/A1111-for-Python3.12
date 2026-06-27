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
        # SD1.5/SDXL VAE encoder uses 3x asymmetric-padded stride-2 conv (pad=(0,1,0,1)).
        # Inputs with H or W not divisible by 8 produce floor(N/8)-sized latent, but
        # tiled_scale's output buffer is sized round(N/8). For a single tile spanning
        # the full input axis (e.g. pass[1] tile_y=1024 with image H=966), the
        # end-align branch shifts the (120-row) latent down by 1 row, leaving row 0
        # as out_div==0 -> NaN. Pad to next multiple of 8 with edge replication so the
        # VAE output matches the round-based buffer size.
        _, _, ah, aw = a.shape
        pad_h = (-ah) % 8
        pad_w = (-aw) % 8
        if pad_h or pad_w:
            a = torch.nn.functional.pad(a, (0, pad_w, 0, pad_h), mode="replicate")
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
