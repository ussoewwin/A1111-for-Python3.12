# A1111-for-Python3.12 — img2img 16GB VRAM Stability: Forge Tiled VAE Integration

**Date:** 2026-06-26
**Scope:** `modules/forge_tiled_vae.py`, `extensions-builtin/multidiffusion-upscaler-for-automatic1111/`, `modules/sd_models_xl.py`
**Repository:** `ussoewwin/A1111-for-Python3.12`

### Comparison anchor (mandatory baseline)

All “before” behavior in this document is taken from:

| Role | Commit | Message |
|------|--------|---------|
| **Baseline (previous)** | `278fd71238a10da6a8b55e5b08a657c0ce97fc20` | Update README.md |
| Integration tip (canvas + VAE) | `24aefab9` | fix: MultiDiffusion latent canvas alignment and Forge VAE tile NaN |
| **Current tip (includes Noise Inversion broadcast fix)** | `fd306900` | fix(tiled-diffusion): align noise/x to init_latent canvas in Noise Inversion |

Intermediate commits on `278fd712..fd306900` (code path):

- `e9ab00ae` — feat: Forge-parity SDXL tiled VAE encode/decode with progress bar
- `88d89476` — fix: MultiDiffusion Noise Inversion OOM and Forge tiled VAE encode scale
- `24aefab9` — fix: MultiDiffusion latent canvas alignment and Forge VAE tile NaN
- `fd306900` — fix: Noise Inversion `renoise_mask` broadcast mismatch (231 vs 232 latent width)

To reproduce sources locally:

```bash
git show 278fd71238a10da6a8b55e5b08a657c0ce97fc20:extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
git diff 278fd71238a10da6a8b55e5b08a657c0ce97fc20..24aefab9 -- modules/ extensions-builtin/multidiffusion-upscaler-for-automatic1111/
git diff 24aefab9..fd306900 -- extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
```

---

## 1. Baseline (`278fd71238a10da6a8b55e5b08a657c0ce97fc20`) — What Actually Existed

`278fd71238a10da6a8b55e5b08a657c0ce97fc20` is a README-only commit. At that point, **the following held true**.

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

## 2. Role of the Commits

```
278fd712  (README only)
    |
e9ab00ae  feat: Forge-parity SDXL tiled VAE encode/decode with progress bar
    |
88d89476  fix: MultiDiffusion Noise Inversion OOM and Forge tiled VAE encode scale
    |
24aefab9  fix: MultiDiffusion latent canvas alignment and Forge VAE tile NaN
    |
fd306900  fix: align noise/x to init_latent in Noise Inversion sample_img2img (231 vs 232)
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

### 2-4. `fd306900` — Noise Inversion broadcast fix (231 vs 232)

**Observed failure (2026-06-26):** img2img with MultiDiffusion + Noise Inversion + ControlNet tile + Forge tiled VAE encode. Image **1853×1254** px. Noise Inversion completed; crash on `combined_noise` blend:

```
RuntimeError: The size of tensor a (231) must match the size of tensor b (232) at non-singleton dimension 3
```

Location: `abstractdiffusion.py` — `combined_noise = ((1 - renoise_mask) * inverse_noise + renoise_mask * noise) / ...`

#### Root cause (two different latent width rules)

| Tensor | How size is chosen | 1853×1254 example |
|--------|-------------------|-------------------|
| `p.init_latent` | Forge tiled VAE encode → latent width uses **ceil** (`pixel_to_latent_w`) | **157×232** |
| `noise`, `x` (sampler args) | A1111 `create_random_tensors` uses **floor** (`width//8`, `height//8`) | **156×231** |
| `self.h`, `self.w` (canvas) | After `_rebuild_latent_canvas`, matches `init_latent` | **157×232** |
| `renoise_mask` | `F.interpolate(..., size=noise.shape[-2:])` | **156×231** |
| `inverse_noise` | `latent - p.init_latent / sigmas[0]` from inversion on full canvas | **157×232** |

`24aefab9` fixed canvas and ControlNet alignment to `init_latent`, but **did not resize** the `noise` and `x` tensors passed into `sample_img2img`. After inversion, `inverse_noise` is full-canvas size while `renoise_mask` and `noise` stayed floor-sized → broadcast error on width **231 vs 232**.

#### Fix

Module-level helper `_align_latent_to_canvas(t, lh, lw)`:

- If `t` is smaller: **replicate** the last row/column (edge padding), not new random noise.
- If `t` is larger: copy the top-left `min(h,w)` region (crop).

At the **start** of `sample_img2img`, before building `renoise_mask`:

```python
_, _, _lh, _lw = p.init_latent.shape
if noise.shape[-2:] != (_lh, _lw):
    noise = _align_latent_to_canvas(noise, _lh, _lw)
if x.shape[-2:] != (_lh, _lw):
    x = _align_latent_to_canvas(x, _lh, _lw)
```

Then `renoise_mask` interpolates to `noise.shape[-2:]` which matches `inverse_noise`. No change to UI settings or Forge VAE tile sizes.

**Why edge replication:** The extra latent column/row corresponds to partial pixel coverage at the image boundary (width not divisible by 8). Duplicating the boundary noise/latent values preserves the distribution of the existing tensor and avoids injecting fresh randomness into one column.

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

## 4. Integration code touched (scope)

This document covers **only** the eight integration paths in the table below (`278fd712` → `fd306900`). Use the **scoped** `git diff` commands in section 7-7 and the appendices to reproduce line-accurate patches — not a repository-wide `git diff`.

| File | Role |
|------|------|
| `modules/forge_tiled_vae.py` | **New.** Forge-parity tiled encode/decode (3-pass orientations, CPU accumulation). |
| `modules/sd_models_xl.py` | Invoke `forge_tiled_vae.apply_all_vae_patches()` when SDXL loads. |
| `extensions-builtin/.../scripts/tilevae.py` | When Tiled VAE is enabled, prefer Forge patch over legacy `VAEHook`. |
| `extensions-builtin/.../tile_methods/abstractdiffusion.py` | Latent canvas sizing, ControlNet crops, Noise Inversion hooks; `fd306900` aligns `noise`/`x` to `init_latent`. |
| `extensions-builtin/.../tile_methods/multidiffusion.py` | Per-tile UNet forwards, canvas rebuild, NaN diagnostics. |
| `extensions-builtin/.../tile_methods/demofusion.py` | Pixel-space ControlNet hint slicing (DemoFusion path). |
| `extensions-builtin/.../tile_methods/mixtureofdiffusers.py` | Same hint slicing for Mixture-of-Diffusers path. |
| `extensions-builtin/.../tile_utils/utils.py` | `pixel_to_latent_h` / `pixel_to_latent_w` (floor height, ceil width). |

**Line counts (code paths only):**

- `278fd712..24aefab9`: 8 files, **+1071 / −62** (`forge_tiled_vae.py` is +768).
- `278fd712..fd306900`: 8 files, **+1105 / −62** (+34 in `abstractdiffusion.py` for Noise Inversion broadcast fix; Appendix C).

---

## 5. Remaining Risks and Next Steps

### 5-1. Cases to Verify

- More unusual resolutions (height also not a multiple of 8) — `fd306900` handles the common width-ceil / noise-floor case; retest height-only edge cases on SDXL
- Same alignment on the SDXL path
- Other tile size / overlap combinations
- End-to-end rerun of **1853×1254** + Noise Inversion stack after `fd306900` (log showed inversion OK; sampling step was where it failed before)

### 5-2. Debug Logs to Remove

Added in `24aefab9`; remove after stability is confirmed.

- `modules/forge_tiled_vae.py`: `[MD-DIAG] 3pass[...] tile=...`
- `extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/multidiffusion.py`: `[MD-NaN] step=...`

### 5-3. Cautions When Changing Code

- `pixel_to_latent_h()` / `pixel_to_latent_w()` asymmetry matches **VAE behavior**. Do not “symmetrize” or canvas mismatch returns.
- Skipping `_rebuild_latent_canvas()` and restoring `org_func` can bring back OOM with Noise Inversion.
- Changing Forge VAE `downscale` will desync latent size from MultiDiffusion.
- Removing the `fd306900` alignment block in `sample_img2img` brings back **231 vs 232** (or similar) broadcast errors whenever `init_latent.shape[-1] != p.width//8`.

---

## 6. Before vs After — Side-by-Side at Key Points

Every “before” snippet is from `278fd71238a10da6a8b55e5b08a657c0ce97fc20`. Integration snippets labeled `24aefab9`; Noise Inversion noise alignment is `fd306900`.

### 6-1. Latent canvas size (`abstractdiffusion.py`)

**Before (`278fd712`):**

```python
self.w: int = int(self.p.width  // opt_f)       # latent size
self.h: int = int(self.p.height // opt_f)
```

**After (`24aefab9`):**

```python
self.w: int = pixel_to_latent_w(self.p.width)
self.h: int = pixel_to_latent_h(self.p.height)
```

**Meaning:** Width uses ceil division to match VAE latent width when pixels are not a multiple of 8. Height stays floor. Example: 2325×1945 px → latent 291×243 (not 290×243).

### 6-2. Size mismatch handling (`multidiffusion.py` `sample_one_step`)

**Before (`278fd712`):**

```python
N, C, H, W = x_in.shape
if (H, W) != (self.h, self.w):
    # We don't tile highres, let's just use the original org_func
    self.reset_controlnet_tensors()
    return org_func(x_in)
```

**After (`24aefab9`):**

```python
N, C, H, W = x_in.shape
if (H, W) != (self.h, self.w):
    self._rebuild_latent_canvas(H, W)
    if self.enable_controlnet:
        self.set_controlnet_tensors_for_size(H, W)
```

**Meaning:** Noise Inversion and other paths that pass a latent whose H/W differs from the UI-derived canvas no longer trigger a **full UNet forward** (OOM). The tiled canvas and ControlNet hints are rebuilt to match `x_in`.

### 6-3. VAE bookends (`278fd712` vs `24aefab9`)

| | `278fd712` | `24aefab9` |
|---|------------|------------|
| `modules/forge_tiled_vae.py` | **missing** | **768 lines**, class-level encode/decode patch |
| `sd_models_xl.py` | no Forge hook | `forge_tiled_vae.apply_all_vae_patches()` at import |
| `tilevae.py` | always `VAEHook` on encoder/decoder | Forge path when `applies_to_model()`; legacy hook only as fallback |

### 6-4. Forge encode `downscale` (`forge_tiled_vae.py`, introduced in `e9ab00ae`, fixed in `88d89476`)

**Before (`e9ab00ae` only):** `tiled_scale(..., downscale=True)` on encode → tile indices divided by 8 again → latent H/W drift vs MultiDiffusion.

**After (`88d89476`+):** `downscale=False` on encode; pixel tile positions map 1:1 into the latent accumulation buffer.

### 6-5. Noise Inversion `noise` / `x` vs `init_latent` (`abstractdiffusion.py` `sample_img2img`)

**Before (`24aefab9` only — without `fd306900`):**

```python
def sample_img2img(self, sampler, p, x, noise, conditioning, ...):
    # noise inverse sampling - renoise mask
    renoise_mask = None
    if self.noise_inverse_renoise_strength > 0:
        ...
        renoise_mask = F.interpolate(..., size=noise.shape[-2:], ...)
    ...
    inverse_noise = latent - (p.init_latent / sigmas[0])
    combined_noise = ((1 - renoise_mask) * inverse_noise + renoise_mask * noise) / ...
```

`noise` enters at **floor** `(H//8, W//8)`; `inverse_noise` at **ceil** `init_latent` size → crash when width mod 8 ≠ 0.

**After (`fd306900`):**

```python
def _align_latent_to_canvas(t, lh, lw):
    # replicate last row/col when growing; crop when shrinking
    ...

def sample_img2img(self, sampler, p, x, noise, conditioning, ...):
    _, _, _lh, _lw = p.init_latent.shape
    if noise.shape[-2:] != (_lh, _lw):
        noise = _align_latent_to_canvas(noise, _lh, _lw)
    if x.shape[-2:] != (_lh, _lw):
        x = _align_latent_to_canvas(x, _lh, _lw)
    # then renoise_mask uses aligned noise.shape[-2:]
```

**Meaning:** Closes the last gap between A1111’s img2img noise allocation and Forge/MultiDiffusion’s ceil-width `init_latent`. Required for Noise Inversion after `24aefab9` canvas fixes.

---

## 7. Per-file patch summary

Summaries below match commits `278fd712` → `24aefab9` (canvas + Forge VAE) and `24aefab9` → `fd306900` (Noise Inversion alignment). Full sources and diffs: **Appendix A–C** (`git show` / scoped `git diff` only — no pasted repo-wide diff).

### 7-1. `modules/sd_models_xl.py`

```diff
-from modules import devices, shared, prompt_parser
+from modules import devices, forge_tiled_vae, shared, prompt_parser
 ...
+# Forge tiled VAE at encode_first_stage / decode_first_stage (SD1.5 LatentDiffusion + SDXL DiffusionEngine).
+forge_tiled_vae.apply_all_vae_patches()
```

### 7-2. `tile_utils/utils.py` (added functions, full text)

```python
def pixel_to_latent_h(px: int) -> int:
    """VAE latent height for SD1.x (e.g. 1945px -> 243, not 244)."""
    return int(px) // opt_f


def pixel_to_latent_w(px: int) -> int:
    """VAE latent width when pixels are not a multiple of 8 (e.g. 2325px -> 291)."""
    return (int(px) + opt_f - 1) // opt_f
```

### 7-3. `abstractdiffusion.py` — new / changed methods (at `24aefab9`)

See scoped `git diff` in section 7-7 and Appendix B. Functionally added or replaced:

- Canvas init: `pixel_to_latent_w` / `pixel_to_latent_h`
- `_rebuild_latent_canvas(h, w)` — reallocates `weights`, grid bboxes, scales custom bboxes
- Grid config cache: `_grid_tile_w_cfg`, `_grid_tile_h_cfg`, `_grid_overlap`, `_grid_tile_bs_cfg`
- `_hint_pixel_size_from_x_spatial`, `set_controlnet_tensors_for_size`, `_crop_controlnet_tile`
- `switch_controlnet_tensors(..., tile_offset=0)` — slice pre-cached tiles per micro-batch
- Noise Inversion: align canvas to `init_latent` shape before inversion loop; `[MD-DIAG]` prints
- **`fd306900`:** `_align_latent_to_canvas()`; align `noise`/`x` to `p.init_latent` at start of `sample_img2img`

### 7-4. `multidiffusion.py` — new helpers and behavior (full text at `24aefab9`)

Added:

- `_pixel_slicer(bbox)` — latent bbox → pixel `slice` for ControlNet hints
- `_slice_icond_for_bboxes(icond, bboxes)` — latent-space, pixel-space (`h*8`, `w*8`), or txt2img dummy

Changed:

- `repeat_func` in kdiff / non-kdiff / `get_noise`: **one tile per forward** (VRAM cap with ControlNet)
- `sample_one_step`: rebuild canvas instead of `org_func` fallback; per-tile ControlNet via `tile_offset`
- `[MD-NaN]` diagnostic when output contains NaN/Inf

### 7-5. `tilevae.py` — Forge routing (full added block)

When Tiled VAE is enabled in UI:

1. Resolve `w×h` from `p` or `init_images`
2. VRAM-cap encoder/decoder tile sizes (e.g. 192px encoder above 4M pixels)
3. If `forge_tiled_vae.applies_to_model(p.sd_model)`: set tile sizes, `set_vae_always_tiled(True)`, **return** (skip `VAEHook`)
4. On disable / postprocess: `set_vae_always_tiled(False)` and restore originals if legacy hook was used

### 7-6. `demofusion.py` / `mixtureofdiffusers.py`

Both gain a branch for ControlNet hints in **pixel space** `(self.h * 8, self.w * 8)` with bbox slicing `bbox.y * 8` … — same convention as MultiDiffusion after `88d89476`.

### 7-7. Reproduce patches (scoped paths only)

```bash
# Main integration — eight code paths only (not repo-wide)
git diff 278fd71238a10da6a8b55e5b08a657c0ce97fc20..24aefab9 -- modules/ extensions-builtin/multidiffusion-upscaler-for-automatic1111/
git diff --stat 278fd71238a10da6a8b55e5b08a657c0ce97fc20..24aefab9 -- modules/ extensions-builtin/multidiffusion-upscaler-for-automatic1111/

# Noise Inversion noise/x alignment only
git diff 24aefab9..fd306900 -- extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
```

Stat through `24aefab9`: **8 files, +1071 / −62 lines** (`forge_tiled_vae.py` is +768 of those).

Stat for `fd306900`: **1 file, +34 lines** (`_align_latent_to_canvas` + `sample_img2img` prologue). See Appendix C.

---

## 8. Detailed Semantics (what each layer does)

### 8-1. Why bookend tiling matters

MultiDiffusion already tiles the **UNet** in latent space (96×96 tiles, overlap 48). ControlNet and Noise Inversion run inside that loop. Without Forge patches, **VAE encode** (img2img start) and **VAE decode** (end) still run at full image resolution. On 16 GB GPUs that produces the largest VRAM spikes. `forge_tiled_vae.py` patches `encode_first_stage` / `decode_first_stage` on both `LatentDiffusion` (SD1.5) and `DiffusionEngine` (SDXL) so bookends use the same 3-pass tiled algorithm as Forge, with accumulation on CPU.

### 8-2. Why `pixel_to_latent_w` is ceil

SD1.x VAE decoding/encoding maps pixel columns such that a width of 2325 px occupies **291** latent columns, not 290 (`floor(2325/8)`). MultiDiffusion’s buffer, weights, and ControlNet latent masks were sized with `width//8`, so the rightmost latent column was never written. That desync triggered the `org_func` fallback (full UNet) or wrong ControlNet crops. Ceil on width only matches observed VAE behavior; height remains floor.

### 8-3. `_rebuild_latent_canvas`

Called when `x_in` spatial size differs from `(self.h, self.w)` — common in Noise Inversion after the first step. Rebuilds:

- `self.weights` tensor
- Grid bboxes from saved UI tile size / overlap / batch
- Custom bbox coordinates scaled proportionally

Then tiling continues instead of aborting to `org_func`.

### 8-4. ControlNet hint paths

Hints may be:

1. Latent-sized `(self.h, self.w)` — slice with `bbox.slicer`
2. Pixel-sized `(self.h*8, self.w*8)` — slice with `_pixel_slicer` or `_crop_controlnet_tile`
3. txt2img dummy `(1,1)` — `repeat_tensor`

`set_controlnet_tensors_for_size` crops the full hint to the processing resolution when the whole canvas is not yet tiled (Noise Inversion alignment).

### 8-5. Per-tile `repeat_func` and VRAM

With ControlNet enabled, `micro_plan = [1] * num_tiles`: each tile gets its own `apply_model` / sampler forward with matching `cond_tile`. That trades speed for peak VRAM — required when tile ControlNet hints are large.

### 8-6. Forge `tiled_scale_multidim` end-align

On the last tile along an axis, if `upscaled[d] + mask.shape[d+2] < out.shape[d+2]`, the tile origin is shifted so the tile covers the final output indices. Otherwise `out_div` stays zero on uncovered cells → `0/0` → NaN in the blended output. This matches “first image OK, second fails” when width mod 8 ≠ 0.

### 8-7. Interaction with UI (unchanged)

`tilevae.py` still reads MultiDiffusion Tiled VAE checkboxes and tile size sliders. Forge path uses those sizes (with VRAM caps) for `_encode_passes()` / `_decode_passes()` triple orientations. No new Gradio controls were added.

### 8-8. Noise Inversion tensor size chain (`fd306900`)

End-to-end for img2img + Forge tiled VAE + Noise Inversion:

1. **Encode:** `forge_ldm_encode_first_stage` / tiled encode produces `p.init_latent` with spatial `(pixel_to_latent_h(H), pixel_to_latent_w(W))`.
2. **Canvas:** `AbstractDiffusion.__init__` uses the same `pixel_to_latent_*` for `self.h`, `self.w`; `_rebuild_latent_canvas` may resize buffers if a later tensor differs.
3. **Sampler entry:** A1111 still allocates `noise` and `x` with `create_random_tensors(..., (C, H//8, W//8))` — **floor** on both axes.
4. **Noise Inversion hook:** `sample_img2img` runs inversion → `inverse_noise` on `init_latent` grid.
5. **Blend:** `renoise_mask` is sized to `noise`; without `fd306900`, step 3 and steps 4–5 disagree when `pixel_to_latent_w(W) != W//8`.

The fix only touches step 3’s tensors **inside** the tiled-diffusion hook, immediately before mask construction. It does not change global `processing.py` or Forge VAE.

---

## Appendix A — `forge_tiled_vae.py` (new at `24aefab9`)

768 lines. Retrieve from git (not pasted inline):

```bash
git show 24aefab9:modules/forge_tiled_vae.py
git show fd306900:modules/forge_tiled_vae.py
```

---

## Appendix B — Scoped diff `278fd712` → `24aefab9` (eight integration paths)

```bash
git diff 278fd71238a10da6a8b55e5b08a657c0ce97fc20..24aefab9 -- modules/ extensions-builtin/multidiffusion-upscaler-for-automatic1111/
git diff --stat 278fd71238a10da6a8b55e5b08a657c0ce97fc20..24aefab9 -- modules/ extensions-builtin/multidiffusion-upscaler-for-automatic1111/
```

---

## Appendix C — Scoped diff `24aefab9` → `fd306900` (`abstractdiffusion.py`)

```bash
git diff 24aefab9..fd306900 -- extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
git diff --stat 24aefab9..fd306900 -- extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
```

Semantics of `_align_latent_to_canvas`: section 8-8.
