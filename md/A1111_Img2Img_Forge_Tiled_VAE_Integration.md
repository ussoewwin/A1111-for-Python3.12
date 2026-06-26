# A1111-for-Python3.12 — img2img 16GB VRAM Stability: Forge Tiled VAE Integration

**Date:** 2026-06-26
**Scope:** `modules/forge_tiled_vae.py`, `extensions-builtin/multidiffusion-upscaler-for-automatic1111/`, `modules/sd_models_xl.py`
**Repository:** `ussoewwin/A1111-for-Python3.12`

### Comparison anchor (mandatory baseline)

All “before” behavior in this document is taken from:

| Role | Commit | Message |
|------|--------|---------|
| **Baseline (前回)** | `278fd71238a10da6a8b55e5b08a657c0ce97fc20` | Update README.md |
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
git diff 278fd71238a10da6a8b55e5b08a657c0ce97fc20..24aefab9
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
- **`fd306900`**: fixed Noise Inversion crash when `init_latent` is ceil-sized but A1111 `noise`/`x` are floor-sized (e.g. 231 vs 232).

---

## 6. Remaining Risks and Next Steps

### 6-1. Cases to Verify

- More unusual resolutions (height also not a multiple of 8) — `fd306900` handles the common width-ceil / noise-floor case; retest height-only edge cases on SDXL
- Same alignment on the SDXL path
- Other tile size / overlap combinations
- End-to-end rerun of **1853×1254** + Noise Inversion stack after `fd306900` (log showed inversion OK; sampling step was where it failed before)

### 6-2. Debug Logs to Remove

Added in `24aefab9`; remove after stability is confirmed.

- `modules/forge_tiled_vae.py`: `[MD-DIAG] 3pass[...] tile=...`
- `extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/multidiffusion.py`: `[MD-NaN] step=...`

### 6-3. Cautions When Changing Code

- `pixel_to_latent_h()` / `pixel_to_latent_w()` asymmetry matches **VAE behavior**. Do not “symmetrize” or canvas mismatch returns.
- Skipping `_rebuild_latent_canvas()` and restoring `org_func` can bring back OOM with Noise Inversion.
- Changing Forge VAE `downscale` will desync latent size from MultiDiffusion.
- Removing the `fd306900` alignment block in `sample_img2img` brings back **231 vs 232** (or similar) broadcast errors whenever `init_latent.shape[-1] != p.width//8`.

---

## 7. Before vs After — Side-by-Side at Key Points

Every “before” snippet is from `278fd71238a10da6a8b55e5b08a657c0ce97fc20`. Integration snippets labeled `24aefab9`; Noise Inversion noise alignment is `fd306900`.

### 7-1. Latent canvas size (`abstractdiffusion.py`)

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

### 7-2. Size mismatch handling (`multidiffusion.py` `sample_one_step`)

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

### 7-3. VAE bookends (`278fd712` vs `24aefab9`)

| | `278fd712` | `24aefab9` |
|---|------------|------------|
| `modules/forge_tiled_vae.py` | **missing** | **768 lines**, class-level encode/decode patch |
| `sd_models_xl.py` | no Forge hook | `forge_tiled_vae.apply_all_vae_patches()` at import |
| `tilevae.py` | always `VAEHook` on encoder/decoder | Forge path when `applies_to_model()`; legacy hook only as fallback |

### 7-4. Forge encode `downscale` (`forge_tiled_vae.py`, introduced in `e9ab00ae`, fixed in `88d89476`)

**Before (`e9ab00ae` only):** `tiled_scale(..., downscale=True)` on encode → tile indices divided by 8 again → latent H/W drift vs MultiDiffusion.

**After (`88d89476`+):** `downscale=False` on encode; pixel tile positions map 1:1 into the latent accumulation buffer.

### 7-5. Noise Inversion `noise` / `x` vs `init_latent` (`abstractdiffusion.py` `sample_img2img`)

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

## 8. Full Text of Modified Files (except new module)

Below is the **complete unified diff** from baseline `278fd71238a10da6a8b55e5b08a657c0ce97fc20` to `24aefab9` for all changed paths except `modules/forge_tiled_vae.py` (new file; see Appendix A). The follow-up patch **`24aefab9` → `fd306900`** (`abstractdiffusion.py` only) is in **Appendix C**.

### 8-1. `modules/sd_models_xl.py`

```diff
-from modules import devices, shared, prompt_parser
+from modules import devices, forge_tiled_vae, shared, prompt_parser
 ...
+# Forge tiled VAE at encode_first_stage / decode_first_stage (SD1.5 LatentDiffusion + SDXL DiffusionEngine).
+forge_tiled_vae.apply_all_vae_patches()
```

### 8-2. `tile_utils/utils.py` (added functions, full text)

```python
def pixel_to_latent_h(px: int) -> int:
    """VAE latent height for SD1.x (e.g. 1945px -> 243, not 244)."""
    return int(px) // opt_f


def pixel_to_latent_w(px: int) -> int:
    """VAE latent width when pixels are not a multiple of 8 (e.g. 2325px -> 291)."""
    return (int(px) + opt_f - 1) // opt_f
```

### 8-3. `abstractdiffusion.py` — new / changed methods (full text at `24aefab9`)

See git diff in section 8-8 for line-accurate patch. Functionally added or replaced:

- Canvas init: `pixel_to_latent_w` / `pixel_to_latent_h`
- `_rebuild_latent_canvas(h, w)` — reallocates `weights`, grid bboxes, scales custom bboxes
- Grid config cache: `_grid_tile_w_cfg`, `_grid_tile_h_cfg`, `_grid_overlap`, `_grid_tile_bs_cfg`
- `_hint_pixel_size_from_x_spatial`, `set_controlnet_tensors_for_size`, `_crop_controlnet_tile`
- `switch_controlnet_tensors(..., tile_offset=0)` — slice pre-cached tiles per micro-batch
- Noise Inversion: align canvas to `init_latent` shape before inversion loop; `[MD-DIAG]` prints
- **`fd306900`:** `_align_latent_to_canvas()`; align `noise`/`x` to `p.init_latent` at start of `sample_img2img`

### 8-4. `multidiffusion.py` — new helpers and behavior (full text at `24aefab9`)

Added:

- `_pixel_slicer(bbox)` — latent bbox → pixel `slice` for ControlNet hints
- `_slice_icond_for_bboxes(icond, bboxes)` — latent-space, pixel-space (`h*8`, `w*8`), or txt2img dummy

Changed:

- `repeat_func` in kdiff / non-kdiff / `get_noise`: **one tile per forward** (VRAM cap with ControlNet)
- `sample_one_step`: rebuild canvas instead of `org_func` fallback; per-tile ControlNet via `tile_offset`
- `[MD-NaN]` diagnostic when output contains NaN/Inf

### 8-5. `tilevae.py` — Forge routing (full added block)

When Tiled VAE is enabled in UI:

1. Resolve `w×h` from `p` or `init_images`
2. VRAM-cap encoder/decoder tile sizes (e.g. 192px encoder above 4M pixels)
3. If `forge_tiled_vae.applies_to_model(p.sd_model)`: set tile sizes, `set_vae_always_tiled(True)`, **return** (skip `VAEHook`)
4. On disable / postprocess: `set_vae_always_tiled(False)` and restore originals if legacy hook was used

### 8-6. `demofusion.py` / `mixtureofdiffusers.py`

Both gain a branch for ControlNet hints in **pixel space** `(self.h * 8, self.w * 8)` with bbox slicing `bbox.y * 8` … — same convention as MultiDiffusion after `88d89476`.

### 8-7. `.gitignore`

Unrelated to VAE: adds `/謝罪文/`.

### 8-8. Complete diff command output

Run on the repo to get every changed line:

```bash
# Main integration (through canvas + Forge VAE)
git diff 278fd71238a10da6a8b55e5b08a657c0ce97fc20..24aefab9

# Noise Inversion noise/x alignment only
git diff 24aefab9..fd306900 -- extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
```

Stat through `24aefab9`: **9 files, +1072 / −62 lines** (`forge_tiled_vae.py` is +768 of those).

Stat for `fd306900`: **1 file, +34 lines** (`_align_latent_to_canvas` + `sample_img2img` prologue). See Appendix C.

---

## 9. Detailed Semantics (what each layer does)

### 9-1. Why bookend tiling matters

MultiDiffusion already tiles the **UNet** in latent space (96×96 tiles, overlap 48). ControlNet and Noise Inversion run inside that loop. Without Forge patches, **VAE encode** (img2img start) and **VAE decode** (end) still run at full image resolution. On 16 GB GPUs that produces the largest VRAM spikes. `forge_tiled_vae.py` patches `encode_first_stage` / `decode_first_stage` on both `LatentDiffusion` (SD1.5) and `DiffusionEngine` (SDXL) so bookends use the same 3-pass tiled algorithm as Forge, with accumulation on CPU.

### 9-2. Why `pixel_to_latent_w` is ceil

SD1.x VAE decoding/encoding maps pixel columns such that a width of 2325 px occupies **291** latent columns, not 290 (`floor(2325/8)`). MultiDiffusion’s buffer, weights, and ControlNet latent masks were sized with `width//8`, so the rightmost latent column was never written. That desync triggered the `org_func` fallback (full UNet) or wrong ControlNet crops. Ceil on width only matches observed VAE behavior; height remains floor.

### 9-3. `_rebuild_latent_canvas`

Called when `x_in` spatial size differs from `(self.h, self.w)` — common in Noise Inversion after the first step. Rebuilds:

- `self.weights` tensor
- Grid bboxes from saved UI tile size / overlap / batch
- Custom bbox coordinates scaled proportionally

Then tiling continues instead of aborting to `org_func`.

### 9-4. ControlNet hint paths

Hints may be:

1. Latent-sized `(self.h, self.w)` — slice with `bbox.slicer`
2. Pixel-sized `(self.h*8, self.w*8)` — slice with `_pixel_slicer` or `_crop_controlnet_tile`
3. txt2img dummy `(1,1)` — `repeat_tensor`

`set_controlnet_tensors_for_size` crops the full hint to the processing resolution when the whole canvas is not yet tiled (Noise Inversion alignment).

### 9-5. Per-tile `repeat_func` and VRAM

With ControlNet enabled, `micro_plan = [1] * num_tiles`: each tile gets its own `apply_model` / sampler forward with matching `cond_tile`. That trades speed for peak VRAM — required when tile ControlNet hints are large.

### 9-6. Forge `tiled_scale_multidim` end-align

On the last tile along an axis, if `upscaled[d] + mask.shape[d+2] < out.shape[d+2]`, the tile origin is shifted so the tile covers the final output indices. Otherwise `out_div` stays zero on uncovered cells → `0/0` → NaN in the blended output. This matches “first image OK, second fails” when width mod 8 ≠ 0.

### 9-7. Interaction with UI (unchanged)

`tilevae.py` still reads MultiDiffusion Tiled VAE checkboxes and tile size sliders. Forge path uses those sizes (with VRAM caps) for `_encode_passes()` / `_decode_passes()` triple orientations. No new Gradio controls were added.

### 9-8. Noise Inversion tensor size chain (`fd306900`)

End-to-end for img2img + Forge tiled VAE + Noise Inversion:

1. **Encode:** `forge_ldm_encode_first_stage` / tiled encode produces `p.init_latent` with spatial `(pixel_to_latent_h(H), pixel_to_latent_w(W))`.
2. **Canvas:** `AbstractDiffusion.__init__` uses the same `pixel_to_latent_*` for `self.h`, `self.w`; `_rebuild_latent_canvas` may resize buffers if a later tensor differs.
3. **Sampler entry:** A1111 still allocates `noise` and `x` with `create_random_tensors(..., (C, H//8, W//8))` — **floor** on both axes.
4. **Noise Inversion hook:** `sample_img2img` runs inversion → `inverse_noise` on `init_latent` grid.
5. **Blend:** `renoise_mask` is sized to `noise`; without `fd306900`, step 3 and steps 4–5 disagree when `pixel_to_latent_w(W) != W//8`.

The fix only touches step 3’s tensors **inside** the tiled-diffusion hook, immediately before mask construction. It does not change global `processing.py` or Forge VAE.

---

## Appendix A — Full source: `modules/forge_tiled_vae.py` at `24aefab9`

New file at `e9ab00ae`, finalized at `24aefab9`. **768 lines.** Snapshot command:

```bash
git show 24aefab9:modules/forge_tiled_vae.py
```


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

`


---

## Appendix B — Full unified diff (278fd712 → 24aefab9, extension + hook files)

`diff
diff --git a/.gitignore b/.gitignore
index 6f4380d2..5051136c 100644
--- a/.gitignore
+++ b/.gitignore
@@ -51,3 +51,4 @@ trace.json
 /test_scipy2.whl
 /my test/
 /scratch/
+/謝罪文/
diff --git a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/scripts/tilevae.py b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/scripts/tilevae.py
index 0cc459ef..dc6dc194 100644
--- a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/scripts/tilevae.py
+++ b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/scripts/tilevae.py
@@ -711,6 +711,12 @@ class Script(scripts.Script):
 
         # undo hijack if disabled (in cases last time crashed)
         if not enabled:
+            try:
+                from modules import forge_tiled_vae
+                if forge_tiled_vae.is_patch_applied():
+                    forge_tiled_vae.set_vae_always_tiled(False)
+            except Exception:
+                pass
             if self.hooked:
                 if isinstance(encoder.forward, VAEHook):
                     encoder.forward.net = None
@@ -724,26 +730,70 @@ class Script(scripts.Script):
         if devices.get_optimal_device_name().startswith('cuda') and vae.device == devices.cpu and not vae_to_gpu:
             print("[Tiled VAE] warn: VAE is not on GPU, check 'Move VAE to GPU' if possible.")
 
-        # do hijack
+        w = int(getattr(p, 'width', 0) or 0)
+        h = int(getattr(p, 'height', 0) or 0)
+        if w <= 0 or h <= 0:
+            init_images = getattr(p, 'init_images', None) or []
+            if init_images:
+                w = int(getattr(init_images[0], 'width', 0) or 0)
+                h = int(getattr(init_images[0], 'height', 0) or 0)
+        pixels = w * h
+        hook_enc_tsize = int(encoder_tile_size)
+        hook_dec_tsize = int(decoder_tile_size)
+        if pixels >= 4_000_000:
+            hook_enc_tsize = min(hook_enc_tsize, 192)
+        elif pixels >= 2_500_000:
+            hook_enc_tsize = min(hook_enc_tsize, 256)
+        else:
+            hook_enc_tsize = min(hook_enc_tsize, 512)
+        hook_dec_tsize = min(hook_dec_tsize, 256)
+        if hook_enc_tsize != encoder_tile_size or hook_dec_tsize != decoder_tile_size:
+            print(
+                f"[Tiled VAE] VRAM cap for {w}x{h}: encoder tile {encoder_tile_size} -> {hook_enc_tsize}, "
+                f"decoder tile {decoder_tile_size} -> {hook_dec_tsize}"
+            )
+
+        try:
+            from modules import forge_tiled_vae
+            if forge_tiled_vae.applies_to_model(p.sd_model):
+                forge_tiled_vae.set_vae_tile_sizes(hook_enc_tsize, hook_dec_tsize)
+                forge_tiled_vae.set_vae_always_tiled(True)
+                print(
+                    "[Forge VAE] SD1.5/2.x Forge encode/decode patch active — skipping multidiffusion VAEHook "
+                    f"(encoder {hook_enc_tsize}px / decoder {hook_dec_tsize}px, 3-pass, accum CPU)."
+                )
+                return
+        except Exception:
+            pass
+
+        # Fallback: legacy VAEHook when Forge class patch is unavailable.
         kwargs = {
-            'fast_decoder': fast_decoder, 
-            'fast_encoder': fast_encoder, 
-            'color_fix':    color_fix, 
+            'fast_decoder': fast_decoder,
+            'fast_encoder': fast_encoder,
+            'color_fix':    color_fix,
             'to_gpu':       vae_to_gpu,
         }
 
-        # save original forward (only once)
         if not hasattr(encoder, 'original_forward'): setattr(encoder, 'original_forward', encoder.forward)
         if not hasattr(decoder, 'original_forward'): setattr(decoder, 'original_forward', decoder.forward)
 
         self.hooked = True
-        
-        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, **kwargs)
-        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True,  **kwargs)
+
+        encoder.forward = VAEHook(encoder, hook_enc_tsize, is_decoder=False, **kwargs)
+        decoder.forward = VAEHook(decoder, hook_dec_tsize, is_decoder=True,  **kwargs)
 
     def postprocess(self, p:Processing, processed, enabled:bool, *args):
         if not enabled: return
 
+        try:
+            from modules import forge_tiled_vae
+            if forge_tiled_vae.applies_to_model(p.sd_model):
+                forge_tiled_vae.set_vae_always_tiled(False)
+                devices.torch_gc()
+                return
+        except Exception:
+            pass
+
         vae = p.sd_model.first_stage_model
         encoder = vae.encoder
         decoder = vae.decoder
diff --git a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
index 88916f38..0eb044f3 100644
--- a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
+++ b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
@@ -22,8 +22,8 @@ class AbstractDiffusion:
         # cache. final result of current sampling step, [B, C=4, H//8, W//8]
         # avoiding overhead of creating new tensors and weight summing
         self.x_buffer: Tensor = None
-        self.w: int = int(self.p.width  // opt_f)       # latent size
-        self.h: int = int(self.p.height // opt_f)
+        self.w: int = pixel_to_latent_w(self.p.width)
+        self.h: int = pixel_to_latent_h(self.p.height)
         # weights for background & grid bboxes
         self.weights: Tensor = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
 
@@ -40,6 +40,10 @@ class AbstractDiffusion:
         self.num_tiles: int = None
         self.num_batches: int = None
         self.batched_bboxes: List[List[BBox]] = []
+        self._grid_tile_w_cfg: int = None
+        self._grid_tile_h_cfg: int = None
+        self._grid_overlap: int = 0
+        self._grid_tile_bs_cfg: int = 1
 
         # ext. Region Prompt Control (custom bbox)
         self.enable_custom_bbox: bool = False
@@ -169,9 +173,48 @@ class AbstractDiffusion:
 
     ''' ↓↓↓ extensive functionality ↓↓↓ '''
 
+    def _rebuild_latent_canvas(self, h: int, w: int) -> bool:
+        if self.h == h and self.w == w:
+            return False
+        old_h, old_w = self.h, self.w
+        self.h, self.w = h, w
+        self.weights = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
+        if self.enable_grid_bbox and self._grid_tile_w_cfg is not None:
+            tile_w = min(self._grid_tile_w_cfg, self.w)
+            tile_h = min(self._grid_tile_h_cfg, self.h)
+            overlap = max(0, min(self._grid_overlap, min(tile_w, tile_h) - 4))
+            bboxes, weights = split_bboxes(self.w, self.h, tile_w, tile_h, overlap, self.get_tile_weights())
+            self.weights = weights
+            self.num_tiles = len(bboxes)
+            self.num_batches = math.ceil(self.num_tiles / self._grid_tile_bs_cfg)
+            self.tile_bs = math.ceil(len(bboxes) / self.num_batches)
+            self.tile_w = tile_w
+            self.tile_h = tile_h
+            self.batched_bboxes = [bboxes[i * self.tile_bs:(i + 1) * self.tile_bs] for i in range(self.num_batches)]
+        if self.enable_custom_bbox and old_h > 0 and old_w > 0:
+            scale_h = self.h / old_h
+            scale_w = self.w / old_w
+            for bbox in self.custom_bboxes:
+                bbox.x = int(round(bbox.x * scale_w))
+                bbox.y = int(round(bbox.y * scale_h))
+                bbox.w = max(1, int(round(bbox.w * scale_w)))
+                bbox.h = max(1, int(round(bbox.h * scale_h)))
+                bbox.w = min(bbox.w, self.w - bbox.x)
+                bbox.h = min(bbox.h, self.h - bbox.y)
+                bbox.box = [bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h]
+                bbox.slicer = slice(None), slice(None), slice(bbox.y, bbox.y + bbox.h), slice(bbox.x, bbox.x + bbox.w)
+                if bbox.feather_mask is not None:
+                    bbox.feather_mask = feather_mask(bbox.w, bbox.h, bbox.feather_ratio)
+        print(f'[Tiled Diffusion] Realign latent canvas {old_h}x{old_w} -> {self.h}x{self.w}')
+        return True
+
     @grid_bbox
     def init_grid_bbox(self, tile_w:int, tile_h:int, overlap:int, tile_bs:int):
         self.enable_grid_bbox = True
+        self._grid_tile_w_cfg = tile_w
+        self._grid_tile_h_cfg = tile_h
+        self._grid_overlap = overlap
+        self._grid_tile_bs_cfg = tile_bs
 
         self.tile_w = min(tile_w, self.w)
         self.tile_h = min(tile_h, self.h)
@@ -471,6 +514,75 @@ class AbstractDiffusion:
         for param_id in range(len(self.control_params)):
             self.control_params[param_id].hint_cond = self.org_control_tensor_batch[param_id]
 
+    @controlnet
+    def _hint_pixel_size_from_x_spatial(self, h: int, w: int) -> tuple[int, int]:
+        """Map x_in spatial dims to ControlNet hint pixels (never overshoot canvas)."""
+        px_h, px_w = int(self.p.height), int(self.p.width)
+        if h > self.h * 2 or w > self.w * 2:
+            return min(max(1, h), px_h), min(max(1, w), px_w)
+        return h * opt_f, w * opt_f
+
+    @controlnet
+    def set_controlnet_tensors_for_size(self, h_latent:int, w_latent:int):
+        '''Crop ControlNet hint to match latent tile spatial size.'''
+        if not self.enable_controlnet: return
+        if self.org_control_tensor_batch is None: return
+
+        target_h, target_w = self._hint_pixel_size_from_x_spatial(h_latent, w_latent)
+        print(
+            f'[Tiled Diffusion] Crop ControlNet hint to {target_h}x{target_w} '
+            f'(latent {h_latent}x{w_latent}, canvas {int(self.p.height)}x{int(self.p.width)})'
+        )
+        for param_id in range(len(self.control_params)):
+            param = self.control_params[param_id]
+            full_hint = self.org_control_tensor_batch[param_id]
+            _, _, fh, fw = full_hint.shape
+            crop_h = min(fh, target_h)
+            crop_w = min(fw, target_w)
+            cropped = full_hint[:, :, :crop_h, :crop_w]
+            if crop_h < target_h or crop_w < target_w:
+                cropped = torch.nn.functional.interpolate(
+                    cropped, size=(target_h, target_w), mode='bilinear', align_corners=False,
+                )
+            param.hint_cond = cropped.to(devices.device)
+            if isinstance(param.hr_hint_cond, torch.Tensor):
+                _, _, h_hr, w_hr = param.hr_hint_cond.shape
+                crop_h_hr = min(h_hr, target_h)
+                crop_w_hr = min(w_hr, target_w)
+                cropped_hr = param.hr_hint_cond[:, :, :crop_h_hr, :crop_w_hr]
+                if crop_h_hr < target_h or crop_w_hr < target_w:
+                    cropped_hr = torch.nn.functional.interpolate(
+                        cropped_hr, size=(target_h, target_w), mode='bilinear', align_corners=False,
+                    )
+                param.hr_hint_cond = cropped_hr.to(devices.device)
+
+    @controlnet
+    def _crop_controlnet_tile(self, control_tensor: Tensor, bbox: BBox) -> Tensor:
+        """Crop ControlNet hint for one latent tile; clip to hint bounds; uniform pixel size."""
+        if control_tensor.ndim == 3:
+            control_tensor = control_tensor.unsqueeze(0)
+
+        th = bbox[3] - bbox[1]
+        tw = bbox[2] - bbox[0]
+        target_h, target_w = th * opt_f, tw * opt_f
+
+        _, _, fh, fw = control_tensor.shape
+        y0 = max(0, min(bbox[1] * opt_f, fh))
+        y1 = max(0, min(bbox[3] * opt_f, fh))
+        x0 = max(0, min(bbox[0] * opt_f, fw))
+        x1 = max(0, min(bbox[2] * opt_f, fw))
+
+        if y1 > y0 and x1 > x0:
+            control_tile = control_tensor[:, :, y0:y1, x0:x1]
+        else:
+            control_tile = control_tensor[:, :, : min(target_h, fh), : min(target_w, fw)]
+
+        if control_tile.shape[-2] != target_h or control_tile.shape[-1] != target_w:
+            control_tile = torch.nn.functional.interpolate(
+                control_tile, size=(target_h, target_w), mode='bilinear', align_corners=False,
+            )
+        return control_tile
+
     @controlnet
     def prepare_controlnet_tensors(self, refresh:bool=False):
         ''' Crop the control tensor into tiles and cache them '''
@@ -496,10 +608,7 @@ class AbstractDiffusion:
             for bboxes in self.batched_bboxes:
                 single_batch_tensors = []
                 for bbox in bboxes:
-                    if len(control_tensor.shape) == 3:
-                        control_tensor.unsqueeze_(0)
-                    control_tile = control_tensor[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f]
-                    single_batch_tensors.append(control_tile)
+                    single_batch_tensors.append(self._crop_controlnet_tile(control_tensor, bbox))
                 control_tile = torch.cat(single_batch_tensors, dim=0)
                 if self.control_tensor_cpu:
                     control_tile = control_tile.cpu()
@@ -509,21 +618,20 @@ class AbstractDiffusion:
             if len(self.custom_bboxes) > 0:
                 custom_control_tile_list = []
                 for bbox in self.custom_bboxes:
-                    if len(control_tensor.shape) == 3:
-                        control_tensor.unsqueeze_(0)
-                    control_tile = control_tensor[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f]
+                    control_tile = self._crop_controlnet_tile(control_tensor, bbox)
                     if self.control_tensor_cpu:
                         control_tile = control_tile.cpu()
                     custom_control_tile_list.append(control_tile)
                 self.control_tensor_custom.append(custom_control_tile_list)
 
     @controlnet
-    def switch_controlnet_tensors(self, batch_id:int, x_batch_size:int, tile_batch_size:int, is_denoise=False):
+    def switch_controlnet_tensors(self, batch_id:int, x_batch_size:int, tile_batch_size:int, is_denoise=False, tile_offset:int=0):
         if not self.enable_controlnet: return
         if self.control_tensor_batch is None: return
 
         for param_id in range(len(self.control_params)):
-            control_tile = self.control_tensor_batch[param_id][batch_id]
+            batch_tiles = self.control_tensor_batch[param_id][batch_id]
+            control_tile = batch_tiles[tile_offset:tile_offset + tile_batch_size]
             if self.is_kdiff:
                 all_control_tile = []
                 for i in range(tile_batch_size):
@@ -689,6 +797,15 @@ class AbstractDiffusion:
         assert self.p.sampler_name == 'Euler'
 
         x = self.p.init_latent
+        _, _, lh, lw = x.shape
+        if (lh, lw) != (self.h, self.w):
+            self._rebuild_latent_canvas(lh, lw)
+            if self.enable_controlnet:
+                self.set_controlnet_tensors_for_size(lh, lw)
+        print(f'[MD-DIAG] init_latent shape={tuple(x.shape)} dtype={x.dtype} '
+              f'nan={torch.isnan(x).any().item()} inf={torch.isinf(x).any().item()} '
+              f'min={float(x.min().item()):.4g} max={float(x.max().item()):.4g} '
+              f'absmax={float(x.abs().max().item()):.4g}')
         s_in = x.new_ones([x.shape[0]])
         skip = 1 if shared.sd_model.parameterization == "v" else 0
         sigmas = dnw.get_sigmas(steps).flip(0)
@@ -722,7 +839,18 @@ class AbstractDiffusion:
             t = dnw.sigma_to_t(sigma_in)
             t = t / self.noise_inverse_retouch
 
-            eps = self.get_noise(x_in * c_in, t, cond_in, steps - i)
+            x_in_scaled = x_in * c_in
+            if i == 1:
+                print(f'[MD-DIAG] NI i=1 x_in_scaled nan={torch.isnan(x_in_scaled).any().item()} '
+                      f'absmax={float(x_in_scaled.abs().max().item()):.4g} '
+                      f'sigma={float(sigma_in[0].item()):.4g} t={float(t[0].item()):.4g} '
+                      f'c_in={float(c_in.flatten()[0].item()):.4g} c_out={float(c_out.flatten()[0].item()):.4g}')
+            eps = self.get_noise(x_in_scaled, t, cond_in, steps - i)
+            if i == 1:
+                eps_nan = torch.isnan(eps).any().item()
+                eps_inf = torch.isinf(eps).any().item()
+                eps_abs = float(eps.abs().max().item()) if not eps_nan else float('nan')
+                print(f'[MD-DIAG] NI i=1 eps nan={eps_nan} inf={eps_inf} absmax={eps_abs:.4g}')
             denoised = x_in + eps * c_out
 
             # Euler method:
diff --git a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/demofusion.py b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/demofusion.py
index 758ccfe0..ebfe8271 100644
--- a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/demofusion.py
+++ b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/demofusion.py
@@ -64,9 +64,9 @@ class DemoFusion(AbstractDiffusion):
         # txt cond
         tcond = self.get_tcond(cond_in)           # [B=1, L, D] => [B*N, L, D]
         tcond = self.repeat_tensor(tcond, n_rep)
-        # img cond
+        # img cond (ControlNet hint or latent mask)
         icond = self.get_icond(cond_in)
-        if icond.shape[2:] == (self.h, self.w):   # img2img, [B=1, C, H, W]
+        if icond.shape[2:] == (self.h, self.w):   # latent-space mask
             if mode == 0:
                 if self.p.random_jitter:
                     jitter_range = self.jitter_range
@@ -74,7 +74,18 @@ class DemoFusion(AbstractDiffusion):
                 icond = torch.cat([icond[bbox.slicer] for bbox in bboxes], dim=0)
             else:
                 icond = torch.cat([icond[:,:,bbox[1]::self.p.current_scale_num,bbox[0]::self.p.current_scale_num] for bbox in bboxes], dim=0)
-        else:                                     # txt2img, [B=1, C=5, H=1, W=1]
+        elif icond.shape[2:] == (self.h * 8, self.w * 8):  # pixel-space ControlNet hint
+            if mode == 0:
+                if self.p.random_jitter:
+                    jitter_range = self.jitter_range
+                    icond = F.pad(icond,(jitter_range, jitter_range, jitter_range, jitter_range),'constant',value=0)
+                icond = torch.cat([
+                    icond[:, :, bbox.y * 8:(bbox.y + bbox.h) * 8, bbox.x * 8:(bbox.x + bbox.w) * 8]
+                    for bbox in bboxes
+                ], dim=0)
+            else:
+                icond = torch.cat([icond[:,:,bbox[1]::self.p.current_scale_num,bbox[0]::self.p.current_scale_num] for bbox in bboxes], dim=0)
+        else:                                     # txt2img dummy hint
             icond = self.repeat_tensor(icond, n_rep)
 
         # vec cond (SDXL)
diff --git a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/mixtureofdiffusers.py b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/mixtureofdiffusers.py
index a7203897..c1464dec 100644
--- a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/mixtureofdiffusers.py
+++ b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/mixtureofdiffusers.py
@@ -91,10 +91,16 @@ class MixtureOfDiffusers(AbstractDiffusion):
                         # tcond
                         tcond_tile = self.get_tcond(c_in)      # cond, [1, 77, 768]
                         tcond_tile_list.append(tcond_tile)
-                        # icond: might be dummy for txt2img, latent mask for img2img
+                        # icond: might be dummy for txt2img, latent-space or pixel-space ControlNet hint
                         icond = self.get_icond(c_in)
                         if icond.shape[2:] == (self.h, self.w):
                             icond = icond[bbox.slicer]
+                        elif icond.shape[2:] == (self.h * 8, self.w * 8):
+                            icond = icond[
+                                :, :,
+                                bbox.y * 8:(bbox.y + bbox.h) * 8,
+                                bbox.x * 8:(bbox.x + bbox.w) * 8
+                            ]
                         icond_tile_list.append(icond)
                         # vcond:
                         vcond = self.get_vcond(c_in)
@@ -145,6 +151,12 @@ class MixtureOfDiffusers(AbstractDiffusion):
                     icond = self.get_icond(c_in)
                     if icond.shape[2:] == (self.h, self.w):
                         icond = icond[bbox.slicer]
+                    elif icond.shape[2:] == (self.h * 8, self.w * 8):
+                        icond = icond[
+                            :, :,
+                            bbox.y * 8:(bbox.y + bbox.h) * 8,
+                            bbox.x * 8:(bbox.x + bbox.w) * 8
+                        ]
                     vcond = self.get_vcond(c_in)
                     c_out = self.make_cond_dict(c_in, tcond, icond, vcond)
                     x_tile_out = shared.sd_model.apply_model(x_tile, t_in, cond=c_out)
diff --git a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/multidiffusion.py b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/multidiffusion.py
index 193beea1..cef47052 100644
--- a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/multidiffusion.py
+++ b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/multidiffusion.py
@@ -57,12 +57,15 @@ class MultiDiffusion(AbstractDiffusion):
             return self.sampler_forward(x, sigma_in, cond=cond)
 
         def repeat_func(x_tile:Tensor, bboxes:List[CustomBBox]) -> Tensor:
-            # For kdiff sampler, the dim 0 of input x_in is:
-            #   = batch_size * (num_AND + 1)   if not an edit model
-            #   = batch_size * (num_AND + 2)   otherwise
-            sigma_tile = self.repeat_tensor(sigma_in, len(bboxes))
-            cond_tile = self.repeat_cond_dict(cond, bboxes)
-            return self.sampler_forward(x_tile, sigma_tile, cond=cond_tile)
+            # Process one tile at a time to cap VRAM peak (ControlNet + tiled latent).
+            outs = []
+            batch_per_tile = x_tile.shape[0] // len(bboxes)
+            for i, bbox in enumerate(bboxes):
+                xt = x_tile[i * batch_per_tile:(i + 1) * batch_per_tile]
+                sigma_tile = self.repeat_tensor(sigma_in, 1)
+                cond_tile = self.repeat_cond_dict(cond, [bbox])
+                outs.append(self.sampler_forward(xt, sigma_tile, cond=cond_tile))
+            return torch.cat(outs, dim=0)
 
         def custom_func(x:Tensor, bbox_id:int, bbox:CustomBBox) -> Tensor:
             return self.kdiff_custom_forward(x, sigma_in, cond, bbox_id, bbox, self.sampler_forward)
@@ -79,13 +82,17 @@ class MultiDiffusion(AbstractDiffusion):
             return self.sampler_forward(x, ts_in, cond=cond)
 
         def repeat_func(x_tile:Tensor, bboxes:List[CustomBBox]) -> Tuple[Tensor, Tensor]:
-            n_rep = len(bboxes)
-            ts_tile = self.repeat_tensor(ts_in, n_rep)
-            if isinstance(cond, dict):   # FIXME: when will enter this branch?
-                cond_tile = self.repeat_cond_dict(cond, bboxes)
-            else:
-                cond_tile = self.repeat_tensor(cond, n_rep)
-            return self.sampler_forward(x_tile, ts_tile, cond=cond_tile)
+            outs = []
+            batch_per_tile = x_tile.shape[0] // len(bboxes)
+            for i, bbox in enumerate(bboxes):
+                xt = x_tile[i * batch_per_tile:(i + 1) * batch_per_tile]
+                ts_tile = self.repeat_tensor(ts_in, 1)
+                if isinstance(cond, dict):
+                    cond_tile = self.repeat_cond_dict(cond, [bbox])
+                else:
+                    cond_tile = self.repeat_tensor(cond, 1)
+                outs.append(self.sampler_forward(xt, ts_tile, cond=cond_tile))
+            return torch.cat(outs, dim=0)
 
         def custom_func(x:Tensor, bbox_id:int, bbox:CustomBBox) -> Tensor:
             # before the final forward, we can set the control tensor
@@ -109,6 +116,25 @@ class MultiDiffusion(AbstractDiffusion):
             shape = [n] + [1] * r_dims      # [N, 1, ...]
             return x.repeat(shape)
 
+    def _pixel_slicer(self, bbox:BBox) -> tuple:
+        '''latent-space bbox -> pixel-space slicer for ControlNet hint (VAE downscale=8)'''
+        return (
+            slice(None),
+            slice(None),
+            slice(bbox.y * 8, (bbox.y + bbox.h) * 8),
+            slice(bbox.x * 8, (bbox.x + bbox.w) * 8),
+        )
+
+    def _slice_icond_for_bboxes(self, icond:Tensor, bboxes:List[CustomBBox]) -> Tensor:
+        '''Tile a ControlNet hint to match latent bboxes. Handles latent-space hints,
+        pixel-space hints (downscale 8), and txt2img dummy hints.'''
+        if icond.shape[2:] == (self.h, self.w):                 # already latent-space
+            return torch.cat([icond[bbox.slicer] for bbox in bboxes], dim=0)
+        if icond.shape[2:] == (self.h * 8, self.w * 8):         # pixel-space hint
+            return torch.cat([icond[self._pixel_slicer(bbox)] for bbox in bboxes], dim=0)
+        # txt2img dummy hint [B, C, 1, 1] etc.
+        return self.repeat_tensor(icond, len(bboxes))
+
     def repeat_cond_dict(self, cond_in:CondDict, bboxes:List[CustomBBox]) -> CondDict:
         ''' repeat all tensors in cond_dict on it's first dim (for a batch of tiles), returns a new object '''
         # n_repeat
@@ -116,12 +142,8 @@ class MultiDiffusion(AbstractDiffusion):
         # txt cond
         tcond = self.get_tcond(cond_in)           # [B=1, L, D] => [B*N, L, D]
         tcond = self.repeat_tensor(tcond, n_rep)
-        # img cond
-        icond = self.get_icond(cond_in)
-        if icond.shape[2:] == (self.h, self.w):   # img2img, [B=1, C, H, W]
-            icond = torch.cat([icond[bbox.slicer] for bbox in bboxes], dim=0)
-        else:                                     # txt2img, [B=1, C=5, H=1, W=1]
-            icond = self.repeat_tensor(icond, n_rep)
+        # img cond (ControlNet hint)
+        icond = self._slice_icond_for_bboxes(self.get_icond(cond_in), bboxes)
         # vec cond (SDXL)
         vcond = self.get_vcond(cond_in)           # [B=1, D]
         if vcond is not None:
@@ -139,9 +161,9 @@ class MultiDiffusion(AbstractDiffusion):
 
         N, C, H, W = x_in.shape
         if (H, W) != (self.h, self.w):
-            # We don't tile highres, let's just use the original org_func
-            self.reset_controlnet_tensors()
-            return org_func(x_in)
+            self._rebuild_latent_canvas(H, W)
+            if self.enable_controlnet:
+                self.set_controlnet_tensors_for_size(H, W)
 
         # clear buffer canvas
         self.reset_buffer(x_in)
@@ -154,16 +176,14 @@ class MultiDiffusion(AbstractDiffusion):
                 # batching
                 x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)   # [TB, C, TH, TW]
 
-                # controlnet tiling
-                # FIXME: is_denoise is default to False, however it is set to True in case of MixtureOfDiffusers, why?
-                self.switch_controlnet_tensors(batch_id, N, len(bboxes))
-
                 # stablesr tiling
                 self.switch_stablesr_tensors(batch_id)
 
                 # compute tiles with micro-batch plan to avoid VRAM spikes
                 tb = len(bboxes)
-                if tb == 6:
+                if self.enable_controlnet:
+                    micro_plan = [1] * tb
+                elif tb == 6:
                     micro_plan = [3, 3]
                 elif tb >= 4:
                     micro_plan = [2] * (tb // 2)
@@ -174,6 +194,7 @@ class MultiDiffusion(AbstractDiffusion):
                     micro_plan = [tb]
 
                 if micro_plan == [tb]:
+                    self.switch_controlnet_tensors(batch_id, N, tb, tile_offset=0)
                     x_tile_out = repeat_func(x_tile, bboxes)
                 else:
                     outs = []
@@ -181,9 +202,7 @@ class MultiDiffusion(AbstractDiffusion):
                     for m in micro_plan:
                         bb = bboxes[k:k+m]
                         xt = x_tile[k * N:(k + m) * N, :, :, :]
-                        # Adjust ControlNet tensors only when micro-batch size differs from full batch
-                        if m != tb:
-                            self.switch_controlnet_tensors(batch_id, N, len(bb))
+                        self.switch_controlnet_tensors(batch_id, N, m, tile_offset=k)
                         outs.append(repeat_func(xt, bb))
                         k += m
                     x_tile_out = torch.cat(outs, dim=0)
@@ -239,6 +258,14 @@ class MultiDiffusion(AbstractDiffusion):
             # Weighted average with original x_buffer
             x_out = torch.where(x_feather_count > 0, x_out * (1 - x_feather_mask) + x_feather_buffer * x_feather_mask, x_out)
 
+        if torch.isnan(x_out).any() or torch.isinf(x_out).any():
+            nan_buf = torch.isnan(self.x_buffer).any().item()
+            inf_buf = torch.isinf(self.x_buffer).any().item()
+            w_min = float(self.weights.min().item())
+            w_zero = int((self.weights == 0).sum().item())
+            print(f'[MD-NaN] step={state.sampling_step} shape={tuple(x_out.shape)} '
+                  f'buf_nan={nan_buf} buf_inf={inf_buf} w_min={w_min} w_zero={w_zero}')
+
         return x_out
 
     def get_noise(self, x_in:Tensor, sigma_in:Tensor, cond_in:Dict[str, Tensor], step:int) -> Tensor:
@@ -249,18 +276,19 @@ class MultiDiffusion(AbstractDiffusion):
             return shared.sd_model.apply_model(x, sigma_in, cond=cond_in_original)
 
         def repeat_func(x_tile:Tensor, bboxes:List[CustomBBox]):
-            sigma_in_tile = sigma_in.repeat(len(bboxes))
-            cond_out = self.repeat_cond_dict(cond_in_original, bboxes)
-            x_tile_out = shared.sd_model.apply_model(x_tile, sigma_in_tile, cond=cond_out)
-            return x_tile_out
+            outs = []
+            batch_per_tile = x_tile.shape[0] // len(bboxes)
+            for i, bbox in enumerate(bboxes):
+                xt = x_tile[i * batch_per_tile:(i + 1) * batch_per_tile]
+                cond_out = self.repeat_cond_dict(cond_in_original, [bbox])
+                outs.append(shared.sd_model.apply_model(xt, sigma_in, cond=cond_out))
+            return torch.cat(outs, dim=0)
 
         def custom_func(x:Tensor, bbox_id:int, bbox:CustomBBox):
             # The negative prompt in custom bbox should not be used for noise inversion
             # otherwise the result will be astonishingly bad.
             tcond = Condition.reconstruct_cond(bbox.cond, step).unsqueeze_(0)
-            icond = self.get_icond(cond_in_original)
-            if icond.shape[2:] == (self.h, self.w):
-                icond = icond[bbox.slicer]
+            icond = self._slice_icond_for_bboxes(self.get_icond(cond_in_original), [bbox])
             cond_out = self.make_cond_dict(cond_in, tcond, icond)
             return shared.sd_model.apply_model(x, sigma_in, cond=cond_out)
 
diff --git a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_utils/utils.py b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_utils/utils.py
index ef8497ee..395f8143 100644
--- a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_utils/utils.py
+++ b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_utils/utils.py
@@ -15,6 +15,16 @@ from modules.processing import opt_f
 
 from tile_utils.typing import *
 
+
+def pixel_to_latent_h(px: int) -> int:
+    """VAE latent height for SD1.x (e.g. 1945px -> 243, not 244)."""
+    return int(px) // opt_f
+
+
+def pixel_to_latent_w(px: int) -> int:
+    """VAE latent width when pixels are not a multiple of 8 (e.g. 2325px -> 291)."""
+    return (int(px) + opt_f - 1) // opt_f
+
 state: State
 
 
diff --git a/modules/sd_models_xl.py b/modules/sd_models_xl.py
index 227ea520..e1e073f7 100644
--- a/modules/sd_models_xl.py
+++ b/modules/sd_models_xl.py
@@ -5,7 +5,7 @@ import torch
 import sgm.models.diffusion
 import sgm.modules.diffusionmodules.denoiser_scaling
 import sgm.modules.diffusionmodules.discretizer
-from modules import devices, shared, prompt_parser
+from modules import devices, forge_tiled_vae, shared, prompt_parser
 from modules import torch_utils
 
 
@@ -51,6 +51,8 @@ sgm.models.diffusion.DiffusionEngine.get_learned_conditioning = get_learned_cond
 sgm.models.diffusion.DiffusionEngine.apply_model = apply_model
 sgm.models.diffusion.DiffusionEngine.get_first_stage_encoding = get_first_stage_encoding
 
+# Forge tiled VAE at encode_first_stage / decode_first_stage (SD1.5 LatentDiffusion + SDXL DiffusionEngine).
+forge_tiled_vae.apply_all_vae_patches()
 
 def encode_embedding_init_text(self: sgm.modules.GeneralConditioner, init_text, nvpt):
     res = []

```

---

## Appendix C — Diff `24aefab9` → `fd306900` (`abstractdiffusion.py` only)

Commit: `fd306900` — fix(tiled-diffusion): align noise/x to init_latent canvas in Noise Inversion.

Reproduce:

```bash
git show fd306900:extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
git diff 24aefab9..fd306900 -- extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
```

Full patch:

```diff
diff --git a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
index 0eb044f3..0b7cf80f 100644
--- a/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
+++ b/extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py
@@ -1,6 +1,28 @@
 from tile_utils.utils import *
 
 
+def _align_latent_to_canvas(t: Tensor, lh: int, lw: int) -> Tensor:
+    """Resize a latent tensor `t` (shape [..., H, W]) to (lh, lw) by edge-row/col replication
+    when growing, or center-crop when shrinking. Designed for the ceil/floor mismatch between
+    A1111's floor(width/8) noise tensors and Forge tiled VAE's ceil-rounded init_latent.
+    No statistical drift: replicated rows/cols come from the existing noise distribution."""
+    h_old, w_old = t.shape[-2], t.shape[-1]
+    if (h_old, w_old) == (lh, lw):
+        return t
+    # crop if larger
+    h_take = min(h_old, lh)
+    w_take = min(w_old, lw)
+    out = torch.empty(*t.shape[:-2], lh, lw, dtype=t.dtype, device=t.device)
+    out[..., :h_take, :w_take] = t[..., :h_take, :w_take]
+    # pad with last column (right side) by replication
+    if lw > w_old:
+        out[..., :h_take, w_old:lw] = t[..., :h_take, w_old - 1:w_old]
+    # pad with last row (bottom side) by replication, including newly-filled right columns
+    if lh > h_old:
+        out[..., h_old:lh, :] = out[..., h_old - 1:h_old, :]
+    return out
+
+
 class AbstractDiffusion:
 
     def __init__(self, p: Processing, sampler: Sampler):
@@ -714,6 +736,18 @@ class AbstractDiffusion:
     def sample_img2img(self, sampler: KDiffusionSampler, p:ProcessingImg2Img, 
                        x:Tensor, noise:Tensor, conditioning, unconditional_conditioning,
                        steps=None, image_conditioning=None):
+        # Forge-compat: A1111 builds `x`/`noise` from floor(W/8, H/8) via create_random_tensors,
+        # but Forge tiled VAE encode can produce a ceil-rounded `init_latent` (e.g. 1853x1254 ->
+        # latent 232x157 instead of 231x156). _rebuild_latent_canvas later realigns
+        # self.h/self.w to init_latent, but the `noise` / `x` arguments are still old-sized,
+        # causing the renoise_mask broadcast at line ~782 to fail with a (231 vs 232) mismatch.
+        # Align them to init_latent canvas here by replicating the last row/col (no statistical drift).
+        _, _, _lh, _lw = p.init_latent.shape
+        if noise.shape[-2:] != (_lh, _lw):
+            noise = _align_latent_to_canvas(noise, _lh, _lw)
+        if x.shape[-2:] != (_lh, _lw):
+            x = _align_latent_to_canvas(x, _lh, _lw)
+
         # noise inverse sampling - renoise mask
         import torch.nn.functional as F
         renoise_mask = None
```

**Document revision:** 2026-06-26 — includes `fd306900` (Noise Inversion broadcast fix). Baseline remains `278fd712`; integration through `24aefab9`; current tip `fd306900`.
