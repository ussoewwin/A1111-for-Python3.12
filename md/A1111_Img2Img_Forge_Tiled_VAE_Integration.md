# A1111-for-Python3.12 — img2img 16GB VRAM Stability: Forge Tiled VAE Integration

**Date:** 2026-06-26
**Scope:** `modules/forge_tiled_vae.py`, `extensions-builtin/multidiffusion-upscaler-for-automatic1111/`, `modules/sd_models_xl.py`
**Commit range:** `278fd712..24aefab9` (3 commits)
**Repository:** `ussoewwin/A1111-for-Python3.12`

---

## 1. Baseline (`278fd712`) — What Actually Existed

`278fd712` is a README-only commit. At that point, **the following held true**.

### 1-1. What Did Not Exist

- `modules/forge_tiled_vae.py` (**not created**)
- Class-level `encode_first_stage` / `decode_first_stage` patches on `LatentDiffusion` / `DiffusionEngine`

### 1-2. What Did Exist

- `extensions-builtin/multidiffusion-upscaler-for-automatic1111/scripts/tilevae.py` + legacy `VAEHook`
- MultiDiffusion tiled UNet (96×96, overlap 48, batch 4)
- ControlNet tile_resample (processor_res 1948, control_v11f1e_sd15_tile)
- Noise Inversion (Euler forced)

### 1-3. img2img Flow (Baseline)

```
[init_image] --full-res VAE encode--> [init_latent]
                                      |
                         [MultiDiffusion tiled UNet]
                         [ControlNet tile]
                         [Noise Inversion]
                                      |
                      [full-res VAE decode] --> [output_image]
```

**The problem was at the bookends, not in the middle.**

- **Start**: full-resolution VAE encode → VRAM spike and latency
- **End**: full-resolution VAE decode → same

The middle pipeline (MultiDiffusion + ControlNet + Noise Inversion) worked with the existing UI settings.

### 1-4. Latent Mismatch in the Baseline

In `abstractdiffusion.py`, the latent canvas was initialized with **both height and width using `// 8` (floor)**.

```python
self.w = int(self.p.width  // opt_f)   # floor
self.h = int(self.p.height // opt_f)   # floor
```

For SD VAE, pixel → latent conversion behaves as:

- **Height**: floor(`height / 8`)
- **Width**: not floor but effectively **ceil(`width / 8`)** (the right edge when width is not a multiple of 8 is included in latent space)

Therefore, when **width is not a multiple of 8**, e.g. 2325px → latent W = 291, the canvas was initialized as 290, creating a one-column drift from actual VAE output.

Also in `multidiffusion.py`, when input latent shape `(H, W)` did not match canvas `(self.h, self.w)`:

```python
if (H, W) != (self.h, self.w):
    # We don't tile highres, let's just use the original org_func
    self.reset_controlnet_tensors()
    return org_func(x_in)
```

**It fell back to a full UNet forward.** That always carried a VRAM explosion risk.

---

## 2. Role of the Three Commits

```
278fd712  (README only)
    |
e9ab00ae  feat: Forge-parity SDXL tiled VAE encode/decode with progress bar
    |
88d89476  fix: MultiDiffusion Noise Inversion OOM and Forge tiled VAE encode scale
    |
24aefab9  fix: MultiDiffusion latent canvas alignment and Forge VAE tile NaN
```

### 2-1. `e9ab00ae` — Bookend VRAM Mitigation

Main changes:

- **New file** `modules/forge_tiled_vae.py` (545 lines)
- `modules/sd_models_xl.py`: call `forge_tiled_vae.apply_diffusion_engine_vae_patch()`
- `tilevae.py`: when Tiled VAE is ON in the UI and the Forge patch is active, skip legacy `VAEHook`

#### Core Functions Added

| Function | Role |
|----------|------|
| `forge_encode_first_stage()` | Replace SDXL `DiffusionEngine.encode_first_stage` |
| `forge_decode_first_stage()` | Replace SDXL `DiffusionEngine.decode_first_stage` |
| `forge_ldm_encode_first_stage()` | Replace SD1.5/2.x `LatentDiffusion.encode_first_stage` |
| `forge_ldm_decode_first_stage()` | Replace SD1.5/2.x `LatentDiffusion.decode_first_stage` |
| `encode_pixels()` / `decode_latent()` | VAE call wrappers |
| `_encode_tiled()` / `_decode_tiled()` | Forge-compatible 3-pass tiled processing |
| `tiled_scale_multidim()` | Multi-dimensional tile blending (from ComfyUI/Forge) |

#### Patching Method

Methods are replaced at the class level.

```python
# modules/forge_tiled_vae.py
def apply_diffusion_engine_vae_patch() -> None:
    de = diffusion_module.DiffusionEngine
    de._forge_encode_first_stage_original = de.encode_first_stage
    de._forge_decode_first_stage_original = de.decode_first_stage
    de.encode_first_stage = forge_encode_first_stage
    de.decode_first_stage = forge_decode_first_stage
```

SD1.5/2.x `LatentDiffusion` is patched the same way.

#### 3-Pass Tiled Processing

`_encode_tiled()` / `_decode_tiled()` average three tile orientations.

```python
encode_passes = (
    ((512, 512), 64),
    ((1024, 256), 64),
    ((256, 1024), 64),
)
```

This avoids full-resolution VAE and caps VRAM per tile.

### 2-2. `88d89476` — Integration Fix (Part 1)

Main changes:

- Large rewrite of `forge_tiled_vae.py` (419-line diff)
  - Fixed tile passes → **dynamic tile passes**
  - Added `_ENCODE_TILE_BASE`, `_DECODE_TILE_BASE`, `_encode_passes()`, `_decode_passes()`
  - UI slider values applied via `set_vae_tile_sizes()`
- `tilevae.py`: formalized legacy `VAEHook` bypass when Forge patch is active
- `abstractdiffusion.py`: align canvas / ControlNet hint to input latent size for Noise Inversion
- `multidiffusion.py`: ControlNet hint size alignment

#### Critical Fix: encode `downscale` Bug

In `e9ab00ae`, the encode path used `downscale=True`.

That made tile positions return `h/8, w/8` when mapping pixel → latent space, which **shifted H/W relative to the MultiDiffusion latent canvas**.

`88d89476` changed it to `downscale=False`.

```python
# before
tiled_scale(..., downscale=True)

# after
tiled_scale(..., downscale=False)
```

#### Problem That Remained

At this commit, the `org_func` fallback on size mismatch in `multidiffusion.py` **was still present**.

```python
if (H, W) != (self.h, self.w):
    out = org_func(x_in)
    self.reset_controlnet_tensors()
    return out
```

With Noise Inversion this could trigger OOM. `24aefab9` fixed that at the root.

### 2-3. `24aefab9` — Integration Fix (Part 2, Core)

Main changes:

- `tile_utils/utils.py`: added `pixel_to_latent_h()` / `pixel_to_latent_w()`
- `abstractdiffusion.py`: canvas init uses `pixel_to_latent_*`, added `_rebuild_latent_canvas()`
- `multidiffusion.py`: removed `org_func` fallback on mismatch; rebuild canvas instead
- `forge_tiled_vae.py`: edge-tile NaN fix (end-align), debug logging

#### Asymmetric Latent Dimensions

```python
def pixel_to_latent_h(px: int) -> int:
    return int(px) // opt_f          # floor

def pixel_to_latent_w(px: int) -> int:
    return (int(px) + opt_f - 1) // opt_f   # ceil
```

Canvas initialization changed to:

```python
self.w = pixel_to_latent_w(self.p.width)
self.h = pixel_to_latent_h(self.p.height)
```

Non-multiple-of-8 widths (e.g. 2325px → 291 latent) now match actual VAE output.

#### Canvas Rebuild

```python
def _rebuild_latent_canvas(self, h: int, w: int) -> bool:
    if self.h == h and self.w == w:
        return False
    old_h, old_w = self.h, self.w
    self.h, self.w = h, w
    self.weights = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
    # ... rebuild buffer / bboxes
```

In `multidiffusion.py`, on size mismatch, rebuild the canvas and continue tiling instead of falling back.

```python
if (H, W) != (self.h, self.w):
    self._rebuild_latent_canvas(H, W)
    if self.enable_controlnet:
        self.set_controlnet_tensors_for_size(H, W)
```

#### Forge VAE Edge-Tile NaN Fix

When width is not a multiple of 8, the last tile may not cover the output edge; `out_div` becomes 0 and `0/0 = NaN`.

```python
# End-align the last tile when VAE floor() leaves the output edge
# uncovered (e.g. pixel width 2325 -> latent 291: last tile fills
# 286-289 only, index 290 stays out_div==0 -> 0/0 = NaN).
for d in range(dims):
    if last_axis[d] and upscaled[d] + mask.shape[d + 2] < out.shape[d + 2]:
        upscaled[d] = max(0, out.shape[d + 2] - mask.shape[d + 2])
```

This was the direct cause of “first image OK, second image NG” and resolution-dependent failures.

---

## 3. UI Settings Unchanged

All fixes are **code-only**. The user's existing UI stack was preserved.

| Setting | Value |
|---------|-------|
| MultiDiffusion tile | 96×96 |
| overlap | 48 |
| batch | 4 |
| ControlNet | tile_resample, processor_res 1948, control_v11f1e_sd15_tile |
| Noise Inversion | Euler |
| Forge Tiled VAE | encoder/decoder ~192px, 3-pass, CPU accum |
| DemoFusion | enabled (delegates to MultiDiffusion) |

`tilevae.py` reads these UI values and passes tile sizes to the Forge path.

---

## 4. Changed Files

```
 .gitignore                                         |   1 +
 extensions-builtin/multidiffusion-upscaler-for-automatic1111/scripts/tilevae.py                             |   66 +-
 extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py              |  152 +++-
 extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/demofusion.py                     |   17 +-
 extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/mixtureofdiffusers.py             |   14 +-
 extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/multidiffusion.py                 |  102 ++-
 extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_utils/utils.py                            |   10 +
 modules/forge_tiled_vae.py                         | 768 +++++++++++++++++++++
 modules/sd_models_xl.py                            |   4 +-
 9 files changed, 1072 insertions(+), 62 deletions(-)
```

---

## 5. Correction: Earlier Wrong Explanation

An earlier response claimed “the baseline MultiDiffusion math was broken.” That was **wrong**.

Correct summary:

- **Baseline**: middle pipeline worked. Problems were **full VAE at bookends** and latent mismatch for non-multiple-of-8 widths.
- **`e9ab00ae`**: added Forge-compatible tiled VAE for bookend VRAM.
- **`88d89476`**: fixed downscale bug and Noise Inversion OOM from integration.
- **`24aefab9`**: fixed non-multiple-of-8 latent dimensions and edge-tile NaN.

---

## 6. Remaining Risks and Next Steps

### 6-1. Cases to Verify

- More unusual resolutions (height also not a multiple of 8)
- Same alignment on the SDXL path
- Other tile size / overlap combinations

### 6-2. Debug Logs to Remove

Added in `24aefab9`; remove after stability is confirmed.

- `modules/forge_tiled_vae.py`: `[MD-DIAG] 3pass[...] tile=...`
- `extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/multidiffusion.py`: `[MD-NaN] step=...`

### 6-3. Cautions When Changing Code

- `pixel_to_latent_h()` / `pixel_to_latent_w()` asymmetry matches **VAE behavior**. Do not “symmetrize” or canvas mismatch returns.
- Skipping `_rebuild_latent_canvas()` and restoring `org_func` can bring back OOM with Noise Inversion.
- Changing Forge VAE `downscale` will desync latent size from MultiDiffusion.
