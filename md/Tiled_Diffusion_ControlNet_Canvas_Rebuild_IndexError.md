# Tiled Diffusion + ControlNet — Canvas Rebuild `IndexError` (Full Explanation)

> **Repository:** `ussoewwin/A1111-for-Python3.12` (workspace `D:\USERFILES\A1111`)  
> **Commit:** `1cf51f90` — `fix(tiled-diffusion): rebuild ControlNet per-tile cache on canvas rebuild`  
> **File changed:** `extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_methods/abstractdiffusion.py` only (+44 lines)  
> **Related integration guide:** [A1111_Img2Img_Forge_Tiled_VAE_Integration.md](./A1111_Img2Img_Forge_Tiled_VAE_Integration.md) (sections 1-2, 1-3, 1-6, 1-7, §4-7)

---

## Table of contents

1. [Error content (symptoms, log, traceback)](#1-error-content-symptoms-log-traceback)
2. [Essential root cause](#2-essential-root-cause)
3. [Countermeasure (fix design)](#3-countermeasure-fix-design)
4. [Full added/changed code (no omissions)](#4-full-addedchanged-code-no-omissions)
5. [Meaning of each change](#5-meaning-of-each-change)
6. [Related code you must understand](#6-related-code-you-must-understand)
7. [Timeline: why 240×252 becomes 241×252 and batches 5→7](#7-timeline-why-240252-becomes-241252-and-batches-57)
8. [Why `set_controlnet_tensors_for_size` alone was not enough](#8-why-set_controlnet_tensors_for_size-alone-was-not-enough)
9. [Verification checklist](#9-verification-checklist)
10. [Approaches that do not fix this bug](#10-approaches-that-do-not-fix-this-bug)

---

## 1. Error content (symptoms, log, traceback)

### 1-1. When it happens

| Item | Typical value |
|------|----------------|
| Pipeline | **img2img upscale** with **MultiDiffusion** (Tiled Diffusion) |
| ControlNet | Enabled (e.g. **tile_resample** / tile preprocessor) |
| Sampler extras | **Noise Inversion** (Euler or other k-diff path) |
| VAE | **Forge Tiled VAE** enabled (tiled encode at job start) |
| VRAM target | ~16 GB — other fixes (bookend VAE, per-tile UNet) already applied |

The job often **survives tiled VAE encode** and prints MultiDiffusion progress. The crash appears during **Noise Inversion** or the **first tiled UNet step** after the latent canvas is realigned to match `init_latent`.

### 1-2. Console log sequence (representative)

Before the exception, logs typically show:

```text
[Tiled VAE]: encode ... (Forge 3-pass tiled encode completes)
[Tiled Diffusion]: Realign latent canvas 240x252 -> 241x252
```

Interpretation:

- **240×252** — latent canvas size derived from UI / `AbstractDiffusion` init (`pixel_to_latent_*`, grid bbox split).
- **241×252** — actual `init_latent` spatial size after Forge tiled encode (ceil rounding on width or height).
- After realign, **`batched_bboxes` is recomputed** → more tiles and **more batch groups** (e.g. **5 → 7**).

Then Python raises:

```text
IndexError: list index out of range
```

### 1-3. Stack trace (essential frames)

```text
File ".../tile_methods/multidiffusion.py", line 197, in sample_one_step
    self.switch_controlnet_tensors(batch_id, N, tb, tile_offset=0)
File ".../tile_methods/abstractdiffusion.py", line 699, in switch_controlnet_tensors
    batch_tiles = self.control_tensor_batch[param_id][batch_id]
IndexError: list index out of range
```

**Failing expression:** `self.control_tensor_batch[param_id][batch_id]`

- `param_id` — index over ControlNet units (0 … number of enabled ControlNets − 1). Usually valid.
- **`batch_id`** — index into the **per-batch tile cache**, one entry per element of `self.batched_bboxes`.
- **Failure mode:** loop runs `for batch_id, bboxes in enumerate(self.batched_bboxes)` with `batch_id ∈ {0,…,6}` but `control_tensor_batch[param_id]` still has length **5** (built at init time against the old grid).

So this is **not** a ControlNet model bug or CUDA OOM. It is a **stale Python list** whose length no longer matches the rebuilt tile grid.

---

## 2. Essential root cause

### 2-1. Three parallel data structures

MultiDiffusion + ControlNet maintains **three** related structures:

| Structure | Role | When built / updated (before fix) |
|-----------|------|-----------------------------------|
| **`org_control_tensor_batch`** | Full-resolution ControlNet hint tensor(s) copied from `control_params[].hint_cond` at init | **`prepare_controlnet_tensors()`** once at `init_controlnet` |
| **`batched_bboxes`** | Latent-space tile grid: list of batches, each batch a list of `BBox` | **`init_grid_bbox()`** at setup; **rebuilt** in **`_rebuild_latent_canvas()`** when `(H,W)` changes |
| **`control_tensor_batch`** | Pre-cropped hint **per tile batch**, indexed by `[param_id][batch_id]` | **`prepare_controlnet_tensors()`** only — uses **`batched_bboxes` at init time** |

**Invariant required for correctness:**

```text
len(control_tensor_batch[param_id]) == len(batched_bboxes)   for every param_id
```

**Before commit `1cf51f90`:** `_rebuild_latent_canvas()` updated `self.h`, `self.w`, `weights`, and **`batched_bboxes`**, but **did not** rebuild **`control_tensor_batch`** or **`control_tensor_custom`**.

`org_control_tensor_batch` remained valid (full hint unchanged). Only the **derived per-tile cache** went stale.

### 2-2. Why canvas size changes after init

Forge tiled VAE encode returns `init_latent` whose H/W can differ by **±1** from the UI-derived canvas:

- Integration uses **`pixel_to_latent_w` = ceil(width/8)** for width and floor for height in places; Forge encode may **ceil** latent dimensions.
- Example: pixel width maps to latent W **291** on encode, while init canvas used **290** → after realign, **240 → 241** in latent rows/cols depending on image geometry and tile grid.

Any **+1** change on an axis can add **extra tiles** along that axis → **`num_batches` increases**.

### 2-3. Call chain at crash

```text
multidiffusion.sample_one_step(x_in, ...)
  N, C, H, W = x_in.shape
  if (H, W) != (self.h, self.w):
      self._rebuild_latent_canvas(H, W)          # batched_bboxes grows (e.g. 5 → 7 batches)
      if self.enable_controlnet:
          self.set_controlnet_tensors_for_size(H, W)   # updates global hint_cond only
  ...
  for batch_id, bboxes in enumerate(self.batched_bboxes):   # batch_id = 5, 6, ...
      self.switch_controlnet_tensors(batch_id, ...)
          self.control_tensor_batch[param_id][batch_id]   # IndexError: only 5 entries
```

**Essence in one sentence:** the **tile iteration space** was refreshed, but the **ControlNet tile cache** was still sized for the **old** iteration space.

---

## 3. Countermeasure (fix design)

### 3-1. Goal

After every **`_rebuild_latent_canvas()`**, recompute **`control_tensor_batch`** and **`control_tensor_custom`** from the unchanged source of truth **`org_control_tensor_batch`**, using the **current** `batched_bboxes` / `custom_bboxes`, with the **same cropping logic** as **`prepare_controlnet_tensors()`**.

### 3-2. Why not call `prepare_controlnet_tensors(refresh=True)`?

`prepare_controlnet_tensors()` early-returns if caches already exist unless `refresh=True`, and it re-reads hints from **`latest_network.control_params`**. Rebuilding from **`org_control_tensor_batch`** is safer because:

- Canvas rebuild can happen **mid-job**; network hooks may have mutated `hint_cond` via `set_controlnet_tensors_for_size`.
- **`org_control_tensor_batch`** is explicitly kept as the **full-resolution snapshot** at init.
- Logic is **duplicated intentionally** to mirror init cropping (`_crop_controlnet_tile`) without side effects on ControlNet global state.

### 3-3. Implementation summary

1. Add **`_rebuild_controlnet_tile_cache()`** — same nested loops as `prepare_controlnet_tensors` body, source = `org_control_tensor_batch`.
2. Call it at the **end** of **`_rebuild_latent_canvas()`**, after `batched_bboxes` / custom bbox scaling is final.
3. No-op if ControlNet disabled or `org_control_tensor_batch` empty.

This restores the invariant `len(control_tensor_batch[i]) == len(batched_bboxes)`.

---

## 4. Full added/changed code (no omissions)

### 4-1. Hook inside `_rebuild_latent_canvas` (lines 230–235)

These lines are inserted **after** the realign print and **before** `return True`:

```python
        print(f'[Tiled Diffusion] Realign latent canvas {old_h}x{old_w} -> {self.h}x{self.w}')
        # batched_bboxes / custom_bboxes have changed; rebuild the per-tile ControlNet
        # cache so switch_controlnet_tensors does not IndexError when the new grid has
        # more batches than the cache built at init_controlnet time.
        self._rebuild_controlnet_tile_cache()
        return True
```

### 4-2. New method `_rebuild_controlnet_tile_cache` (lines 237–275)

Full method as committed:

```python
    def _rebuild_controlnet_tile_cache(self) -> None:
        """Rebuild control_tensor_batch / control_tensor_custom for the current
        batched_bboxes / custom_bboxes, using org_control_tensor_batch (preserved
        full-resolution hint from init_controlnet) as the source of truth.

        Required when _rebuild_latent_canvas changes the tile grid (e.g. Forge
        tiled encode returns a ceil-rounded latent that is one row/col larger
        than the UI canvas). Without this, switch_controlnet_tensors indexes
        past the original control_tensor_batch length and raises IndexError."""
        if not getattr(self, 'enable_controlnet', False):
            return
        org_tensors = getattr(self, 'org_control_tensor_batch', None)
        if not org_tensors:
            return
        self.control_tensor_batch = []
        # control_tensor_custom is a per-param list of per-bbox tile lists; reset
        # only when we actually rebuild, so init_controlnet's empty list survives
        # the no-ControlNet case.
        self.control_tensor_custom = []
        for i in range(len(org_tensors)):
            control_tile_list = []
            control_tensor = org_tensors[i]
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

**Total:** 44 lines added in commit `1cf51f90` (comments + method + one call).

---

## 5. Meaning of each change

### 5-1. Call site in `_rebuild_latent_canvas`

| Lines | Meaning |
|-------|---------|
| Comment block | Documents **why** canvas rebuild must refresh ControlNet cache — prevents repeat investigation. |
| `self._rebuild_controlnet_tile_cache()` | Keeps **`control_tensor_batch`** in sync with **`batched_bboxes`** whenever latent H/W changes. Without this, `switch_controlnet_tensors` is undefined behavior once `batch_id ≥ old_num_batches`. |

`_rebuild_latent_canvas` already:

- Recomputes **`split_bboxes`** → new **`batched_bboxes`**, **`num_tiles`**, **`num_batches`**, **`tile_bs`**, **`weights`**.
- Scales **custom bboxes** proportionally when enabled.

The new call completes the job for ControlNet: **any structure indexed by `batch_id` must be rebuilt here**.

### 5-2. `_rebuild_controlnet_tile_cache` guards

| Code | Meaning |
|------|---------|
| `if not getattr(self, 'enable_controlnet', False): return` | No ControlNet → no cache → no work. Avoids touching lists on txt2img paths without ControlNet. |
| `org_tensors = getattr(self, 'org_control_tensor_batch', None)` | Use **init snapshot**, not live `hint_cond` (may be cropped per-step by `set_controlnet_tensors_for_size`). |
| `if not org_tensors: return` | Init did not run or no hints — same as `prepare_controlnet_tensors` bail-out. |

### 5-3. Rebuild loops (mirror of `prepare_controlnet_tensors`)

| Code block | Meaning |
|------------|---------|
| `self.control_tensor_batch = []` | Drop stale batch list entirely; length must match new `len(batched_bboxes)`. |
| `self.control_tensor_custom = []` | Custom bbox hints are also tile-indexed; custom bbox **coordinates** were already scaled in `_rebuild_latent_canvas`, so hints must be **re-cropped** to new boxes. |
| `for i in range(len(org_tensors))` | One ControlNet model / param group per `i` (multi-ControlNet). |
| `for bboxes in self.batched_bboxes` | One **`control_tile_list`** entry per **batch** (same as `batch_id` in `sample_one_step`). |
| `for bbox in bboxes:` + `_crop_controlnet_tile` | Crop full hint to latent tile, map to pixel space, clip, uniform size — **identical** to init path. |
| `torch.cat(single_batch_tensors, dim=0)` | Batch dimension = number of tiles in this MultiDiffusion batch (matches UNet tile batch). |
| `if self.control_tensor_cpu: control_tile = control_tile.cpu()` | Preserve **CPU offload** behavior from extension settings. |
| Custom bbox loop | Populates **`control_tensor_custom[param_id][bbox_id]`** used by **`set_custom_controlnet_tensors`** during custom-region / Noise Inversion paths. |

**Design principle:** treat **`org_control_tensor_batch` + current bboxes** as a pure function; re-derive caches whenever bboxes change.

---

## 6. Related code you must understand

### 6-1. Where caches are first built — `prepare_controlnet_tensors`

```python
    @controlnet
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

**Note:** `_rebuild_controlnet_tile_cache` is intentionally parallel to the inner loops here, but reads **`org_control_tensor_batch`** instead of re-fetching from `latest_network`.

### 6-2. Where the crash happens — `switch_controlnet_tensors`

```python
    @controlnet
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

**`batch_id`** must satisfy `0 <= batch_id < len(self.control_tensor_batch[param_id])`. After fix, that length equals **`len(self.batched_bboxes)`**.

### 6-3. Where canvas rebuild is triggered — `multidiffusion.sample_one_step`

```python
        N, C, H, W = x_in.shape
        if (H, W) != (self.h, self.w):
            self._rebuild_latent_canvas(H, W)
            if self.enable_controlnet:
                self.set_controlnet_tensors_for_size(H, W)

        # clear buffer canvas
        self.reset_buffer(x_in)

        # Background sampling (grid bbox)
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                ...
                if micro_plan == [tb]:
                    self.switch_controlnet_tensors(batch_id, N, tb, tile_offset=0)
                    x_tile_out = repeat_func(x_tile, bboxes)
                else:
                    ...
                        self.switch_controlnet_tensors(batch_id, N, m, tile_offset=k)
```

First UNet call (or Noise Inversion step) passes **`x_in`** whose spatial size matches **`init_latent`**. If that size differs from init canvas → rebuild → **then** tile loop uses new batch count.

---

## 7. Timeline: why 240×252 becomes 241×252 and batches 5→7

### 7-1. Phase A — Job setup (before encode)

1. User sets img2img size (pixels). MultiDiffusion **`init_grid_bbox`** stores tile 96×96, overlap 48, batch 4.
2. **`AbstractDiffusion`** sets canvas **`self.h`, `self.w`** from latent formulas (e.g. W = 240, H = 252 for a particular upscale).
3. **`split_bboxes`** produces **`batched_bboxes`** with **`num_batches = 5`** (example).
4. **`prepare_controlnet_tensors()`** builds **`control_tensor_batch[param_id]`** with **5** entries.

### 7-2. Phase B — Forge tiled VAE encode

1. **`encode_first_stage`** (Forge hook) encodes image in tiles; latent width/height may use **ceil** per axis.
2. **`p.init_latent`** becomes **241×252** (one extra latent row or column vs canvas).

Earlier fixes (`fd306900`, `_align_latent_to_canvas` in `sample_img2img`) align **noise** and **x** to `init_latent`. MultiDiffusion must align **its internal canvas and tile grid** too — that is **`_rebuild_latent_canvas`**.

### 7-3. Phase C — First `sample_one_step` (failure before fix)

1. `x_in.shape[-2:]` = **(241, 252)** ≠ **(240, 252)** → `_rebuild_latent_canvas(241, 252)`.
2. New **`split_bboxes`** → **`num_batches = 7`** (example).
3. **Before fix:** `control_tensor_batch` still length **5**.
4. Loop `batch_id = 0..6` → at **`batch_id == 5`** → **IndexError**.

After fix, step 2 is followed by **`_rebuild_controlnet_tile_cache()`** → cache length **7** → loop completes.

---

## 8. Why `set_controlnet_tensors_for_size` alone was not enough

`set_controlnet_tensors_for_size(H, W)` (called right after `_rebuild_latent_canvas` in `sample_one_step`):

- Computes target hint pixel size from latent tile spatial dims.
- Crops/interpolates **`param.hint_cond`** (and HR hint) from **`org_control_tensor_batch[param_id]`**.
- Updates **global** ControlNet conditioning for the **current forward**.

It does **not**:

- Rebuild **`control_tensor_batch`** pre-split per **batch_id**.
- Update **`control_tensor_custom`**.

`switch_controlnet_tensors` never reads the global hint for tiling; it reads **`control_tensor_batch[param_id][batch_id]`**. So global hint can be correct while **per-batch cache** is still stale — exactly the bug.

**Division of responsibility after full fix set:**

| Function | Updates |
|----------|---------|
| `set_controlnet_tensors_for_size` | Live **`hint_cond`** for current latent spatial size (compatibility with non-tiled hooks) |
| `_rebuild_controlnet_tile_cache` | **`control_tensor_batch`**, **`control_tensor_custom`** aligned to **`batched_bboxes`** |
| `switch_controlnet_tensors` | Selects slice for current micro-batch and assigns **`hint_cond`** during tiled loop |

---

## 9. Verification checklist

After deploying commit `1cf51f90`:

1. **Settings:** MultiDiffusion on, ControlNet tile model on, Noise Inversion on, Forge Tiled VAE on, ~16 GB GPU.
2. **Run** img2img upscale where pixel width is **not** a multiple of 8 (forces ceil/floor divergence).
3. **Expect in log:**
   - `[Tiled Diffusion] Realign latent canvas AxB -> CxD` with **C≠A or D≠B** possible.
   - **No** `IndexError` at `switch_controlnet_tensors`.
4. **Expect job to continue** through Noise Inversion steps and into normal denoise tiling.
5. **Regression:** txt2img/img2img **without** ControlNet — no behavior change (guard returns immediately).
6. **Regression:** canvas size unchanged (`_rebuild_latent_canvas` returns False) — **`_rebuild_controlnet_tile_cache` not called**.

Optional debug (temporary): after realign, assert:

```python
assert len(self.control_tensor_batch[0]) == len(self.batched_bboxes)
```

---

## 10. Approaches that do not fix this bug

| Approach | Why it fails |
|----------|----------------|
| Only fix VAE ceil/floor in UI canvas init | Reduces how often realign happens; **whenever** encode still differs by 1, bug returns. |
| Only `set_controlnet_tensors_for_size` | Updates global hint, **not** `control_tensor_batch[param_id][batch_id]`. |
| Catch `IndexError` and skip ControlNet | Silent wrong images or broken conditioning. |
| Disable Noise Inversion | Avoids path that often hits realign; does not fix structural inconsistency. |
| Call full `org_func` on mismatch | Old behavior — **full UNet OOM** on 16 GB (why `_rebuild_latent_canvas` exists). |

---

## Summary

| Topic | Conclusion |
|-------|------------|
| **Error** | `IndexError` on `control_tensor_batch[param_id][batch_id]` after `[Tiled Diffusion] Realign latent canvas …` |
| **Root cause** | `_rebuild_latent_canvas` refreshed **`batched_bboxes`** but not **`control_tensor_batch`** |
| **Fix** | `_rebuild_controlnet_tile_cache()` from **`org_control_tensor_batch`**, called at end of **`_rebuild_latent_canvas`** |
| **Commit** | `1cf51f9086107aae8fe68b35e127f90b5c9489d3` |

This fix is **orthogonal** to Forge Tiled VAE bookends, per-tile UNet micro-batching, and Noise Inversion noise alignment — it closes the last **ControlNet cache vs tile grid** gap once latent canvas size changes mid-pipeline.
