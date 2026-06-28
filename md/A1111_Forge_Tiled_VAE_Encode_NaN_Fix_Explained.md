# Forge Tiled VAE Encode — NaN / narrow Crash Fix (v2.3.2)

**Repository:** `ussoewwin/A1111-for-Python3.12`  
**File:** `modules/forge_tiled_vae.py`  
**Scope:** Forge 3-pass tiled **encode** only (`tiled_scale_multidim` + `encode_fn`). Does **not** cover ControlNet canvas rebuild (see separate doc for that).

**Typical failing pipeline:** img2img upscale, MultiDiffusion + ControlNet tile + Noise Inversion, Forge Tiled VAE enabled, ~16 GB VRAM. Example input: **1086×954** px (latent **136×120** after encode).

---

## 1. What Failed (Error Symptoms)

### 1-1. Primary failure: NaNs in UNet

After Forge tiled VAE encode, sampling aborts at step 0:

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

or (more often, after device fix):

```text
NansException: A tensor with NaNs was produced in Unet
```

With diagnostic logging enabled (`[MD-DIAG]` in `_forge_3pass_tiled`), the NaNs appear **inside the encode passes**, not in the final average:

```text
[MD-DIAG] 3pass[0] tile=(192,192) ov=24 part shape=(1,4,120,136) nan=0 inf=0 absmax=...
[MD-DIAG] 3pass[1] tile=(384,128) ov=24 part shape=(1,4,120,136) nan=624 inf=0 absmax=...
[MD-DIAG] 3pass[2] tile=(128,384) ov=24 part shape=(1,4,120,136) nan=0 inf=0 absmax=...
[MD-DIAG] init_latent ... nan=True ...
```

**624 NaNs** on pass `[1]` (wide tile **384×128** with overlap 24) was the smoking gun for image **H=966** (and similarly for **H=954** with different pass geometry).

### 1-2. Secondary failure (superseded intermediate fix)

After commit `019e419d` (replicate-pad **encoder input** to multiple of 8), some heights that already satisfied `floor(H/8) == round(H/8)` **overshot** the latent buffer and crashed during accumulation:

```text
RuntimeError: The size of tensor a (121) must match the size of tensor b (120) at non-singleton dimension 3
```

or PyTorch **narrow** errors when slicing the output tensor. Example: **H=954** → latent **119×120** expected, but padded encode produced **120×120** latent rows.

---

## 2. Root Cause

Forge tiled encode runs three `tiled_scale_multidim(..., downscale=False)` passes over the same pixel image, then averages the three latent results (`acc / 3.0`). Each pass uses different tile shapes from `_encode_passes`:

```python
((base, base), overlap),
((base * 2, max(base // 2, 128)), overlap),   # pass[1]: wide
((max(base // 2, 128), base * 2), overlap),   # pass[2]: tall
```

Inside `tiled_scale_multidim`, each tile is VAE-encoded and **accumulated into a fixed output buffer** `out`, then normalized:

```python
out[b, :, ...] += mask * ps
out_div[b, :, ...] += mask
# ...
output[b] = out / out_div
```

### 2-1. `floor(N/8)` vs `round(N/8)` mismatch

SD1.5 / SDXL VAE encoders use stacked stride-2 convolutions with asymmetric padding (`pad=(0,1,0,1)` on some layers). For pixel height **H**:

| Quantity | Formula | H=966 | H=954 |
|----------|---------|-------|-------|
| VAE latent rows (actual) | `floor(H/8)` | **120** | **119** |
| `tiled_scale` buffer rows | `round(H/8)` | **121** | **119** |

The output tensor is allocated with **`round`**:

```python
output = torch.empty(
    [samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]),
    ...
)
# mult_list_upscale uses round(get_pos(d, pos)) with upscale_amount=1/8
```

Each tile's VAE output `ps` has size based on **`floor`** of that tile's pixel extent. When a **last-axis tile** does not fill the remaining buffer slots, some latent indices never receive `out_div > 0`. After division:

```python
0 / 0  →  NaN
```

### 2-2. Why pass `[1]` produced 624 NaNs (H=966)

For **H=966**, pass `[1]` uses tile **(384, 128)**. Because **966 < 384** on the Y axis, the tile loop runs **once** at `y=0` with `length=966` — a **single tile covers the full image height**.

- Buffer latent height: `round(966/8) = 121`
- VAE output height: `floor(966/8) = 120`
- **Row 0** of the buffer slot is never written → **121 NaNs per channel × 4 channels × ~156 cols ≈ 624** (matches log).

Pass `[0]` and `[2]` use smaller tiles on at least one axis, so multiple tile positions overlap and mask-weighted blending covers the buffer. Pass `[1]` is the problematic **single-tile full-axis** case.

### 2-3. Why end-align (`24aefab9`) was insufficient

Commit `24aefab9` added **end-align**: shift `upscaled[d]` so the last tile's output sits at the **bottom/right** of the buffer:

```python
for d in range(dims):
    if last_axis[d] and upscaled[d] + mask.shape[d + 2] < out.shape[d + 2]:
        upscaled[d] = max(0, out.shape[d + 2] - mask.shape[d + 2])
```

This fixes **multi-tile** last rows (e.g. width 2325 → latent 291, last tile covers 286–289, index 290 empty). It **fails** when **one tile spans the entire axis**: shifting the 120-row output to index **1..120** leaves **index 0** uncovered → same `0/0` NaN.

### 2-4. Why encoder-input padding (`019e419d`) was wrong place

Padding the **encoder input** to the next multiple of 8 forces `floor(H'/8) == round(H'/8)` for padded H'. When **already** `floor(H/8) == round(H/8)` (e.g. **H=954** → 119), padding adds 8 px → **H'=962** → latent **120×120**, but the buffer remains **119×120** → **narrow / size mismatch** crash.

The fix must pad the **VAE output tensor `ps`** to fill the buffer slot, not inflate the encoder input when sizes already align on one axis.

---

## 3. What Was Wrong at the v2.3.1 Milestone (Before `1e2f758a`)

| Stage | Commit | State |
|-------|--------|--------|
| v2.3 shipped | `24aefab9` | Forge tiled VAE + **end-align** in `tiled_scale_multidim`. Multi-tile edge gaps fixed; **single-tile full-axis pass[1] still NaN**. |
| v2.3.1 — ControlNet cache (`1cf51f90`) | `1cf51f90` | ControlNet tile cache rebuild after canvas realign. **No change** to VAE NaN logic. NaN still reproduced on 1086×954 img2img. |
| v2.3.1 — NaN attempt (`019e419d`) | `019e419d` | Replicate-pad **encoder input** in `encode_fn`. Fixed some H (966) but ** broke H=954** with narrow/size errors. End-align still present. |
| v2.3.1 — NaN shipped (`1e2f758a`) | `1e2f758a` | Remove encoder-input pad; remove end-align; **replicate-pad trailing edge of `ps`**. |

At **v2.3.1** (after `1cf51f90`, before `1e2f758a`), the code still had **end-align + encoder-input padding** — the worst combination for mixed image heights.

---

## 4. Countermeasure (Final Fix in `1e2f758a`)

1. **Delete** end-align loop in `tiled_scale_multidim` (shifting `upscaled[d]`).
2. **Delete** replicate padding in `encode_fn` (encoder input).
3. **Add** after `ps = function(s_in)`: compute per-axis **gap** between buffer slot end and `(upscaled[d] + ps.shape[d+2])` for `last_axis[d]` tiles; **replicate-pad `ps` on the trailing edge only** so it exactly fills the slot before mask blending.

Properties:

- **Leading** latent rows stay covered when one tile spans full height (pass[1] / H=966).
- No encoder input inflation → **H=954** stays 119×120 with no overshoot.
- Replicate mode repeats the last latent row/col — neutral for edge pixels, avoids NaN.

Decode path is unchanged (decode uses `downscale=True` and different pass sizes; this bug was encode-specific).

---

## 5. Added / Changed Code (Full Text from `1e2f758a`)

### 5-1. `tiled_scale_multidim` — removed end-align, added trailing replicate pad

**Removed** (was between `ps = function(...)` and `mask = torch.ones_like(ps)`):

```python
            # End-align the last tile when VAE floor() leaves the output edge
            # uncovered (e.g. pixel width 2325 -> latent 291: last tile fills
            # 286-289 only, index 290 stays out_div==0 -> 0/0 = NaN).
            for d in range(dims):
                if last_axis[d] and upscaled[d] + mask.shape[d + 2] < out.shape[d + 2]:
                    upscaled[d] = max(0, out.shape[d + 2] - mask.shape[d + 2])
```

**Added** (current production code):

```python
            ps = function(s_in).to(output_device)

            # Replicate-pad the trailing edge when the last tile's output is
            # smaller than the buffer slot. VAE returns floor(N/8) but the
            # buffer is sized round(N/8); for tiles whose last_axis fills to
            # the image edge, this leaves a 1-row/col gap (out_div==0 -> NaN).
            # Padding (not shifting via end-align) keeps the leading row
            # covered, which matters when a single tile spans the full axis
            # (e.g. pass[1] tile_y=1024 with image H=966): shifting would
            # leave row 0 as 0/0=NaN. F.pad expects (left_lastdim, right_lastdim,
            # ..., left_firstdim, right_firstdim), so build from the last spatial
            # dim backward.
            pad_amounts = []
            for d in range(dims - 1, -1, -1):
                if last_axis[d]:
                    gap = out.shape[d + 2] - (upscaled[d] + ps.shape[d + 2])
                    pad_amounts.extend([0, max(0, gap)])
                else:
                    pad_amounts.extend([0, 0])
            if any(p > 0 for p in pad_amounts):
                ps = torch.nn.functional.pad(ps, pad_amounts, mode="replicate")

            mask = torch.ones_like(ps)
```

### 5-2. `_encode_tiled` → `encode_fn` — removed encoder-input padding

**Removed** from inside `encode_fn` (after `a = a.to(dtype=..., device=...)`):

```python
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
```

**Current `encode_fn`** (unchanged aside from removal):

```python
    def encode_fn(a):
        a = a.to(dtype=dtype, device=device)
        try:
            return _vae_encode_latent_tensor(vae, a)
        finally:
            devices.torch_gc()
```

---

## 6. Code Meaning (Detailed Walkthrough)

### 6-1. Tile loop variables (context)

For each tile position `it`:

| Variable | Meaning |
|----------|---------|
| `pos`, `length` | Pixel slice `[pos, pos+length)` along each spatial dim |
| `upscaled[d]` | Latent index where this tile's output **starts** (`round(pos/8)`) |
| `last_axis[d]` | `True` if this tile touches the **far edge** of the image on axis `d` |
| `ps` | VAE encode output for tile `s_in`, shape `[1, 4, h_lat, w_lat]` |
| `out`, `out_div` | Accumulators sized to **full-image** latent `round(H/8) × round(W/8)` |

### 6-2. Gap calculation

```python
gap = out.shape[d + 2] - (upscaled[d] + ps.shape[d + 2])
```

- `out.shape[d+2]`: total latent size on axis `d` (round-based).
- `upscaled[d] + ps.shape[d+2]`: end index (exclusive) of this tile's output if placed at `upscaled[d]`.
- If `gap > 0`, the tile output is **shorter** than the remaining buffer slot → uncovered indices would get `out_div==0`.

Only **`last_axis[d]`** tiles need this fix: interior tiles are followed by overlapping tiles that fill the gap.

### 6-3. `F.pad` axis order

PyTorch `pad` tuple is **last dimension first**: for 2D latent `(N, C, H, W)`, order is `(left_W, right_W, left_H, right_H)`.

The loop `for d in range(dims - 1, -1, -1)` appends pairs from **W then H**, producing the correct tuple. Only **trailing** (right/bottom) padding is applied: `[0, max(0, gap)]` per axis.

`mode="replicate"` extends the last row/column of `ps` — standard edge handling, avoids NaN without shifting coverage away from row 0.

### 6-4. Why `mask` comes **after** pad

`mask = torch.ones_like(ps)` must match **padded** `ps` dimensions so accumulation indices align:

```python
out[b, :, upscaled[0]:..., upscaled[1]:...] += mask * ps
out_div[b, :, ...] += mask
```

If mask were created before pad, shapes would mismatch or only the unpadded region would be weighted.

### 6-5. Numeric example (H=966, pass[1])

| Step | Value |
|------|-------|
| Pixel H | 966 |
| Buffer latent H | `round(966/8) = 121` |
| Tile pixel H | 966 (single tile) |
| `ps` latent H | `floor(966/8) = 120` |
| `upscaled[0]` | 0 |
| `gap` | `121 - (0 + 120) = 1` |
| After pad | `ps` height **121** (row 120 replicated from row 119) |
| Row 0 coverage | **Yes** — no `0/0` |

### 6-6. Numeric example (H=954, why input pad was removed)

| Step | Value |
|------|-------|
| Pixel H | 954 |
| Buffer latent H | `round(954/8) = round(119.25) = 119` |
| Without input pad | `ps` H = `floor(954/8) = 119`, gap often **0** on Y |
| With input pad (+6→960) | VAE sees 960 → latent **120**, buffer still **119** → **crash** |

Trailing pad on `ps` only runs when `gap > 0`; no spurious inflation.

---

## 7. Verification

1. Restart WebUI (reload `modules/forge_tiled_vae.py`).
2. Same img2img: MultiDiffusion + ControlNet tile + Noise Inversion + Forge Tiled VAE, ~1086×954 input.
3. Expect:
   - `[MD-DIAG] 3pass[0..2] ... nan=0` (if diagnostics still enabled)
   - No `NansException` at step 0
   - Full sampling completes

**Commit on `main`:** `1e2f758a` (included in `37570af3` push with CHANGELOG v2.3.1 NaN bullet).

---

## 8. Summary

| Item | Detail |
|------|--------|
| **Bug** | `tiled_scale_multidim` left latent cells at `out_div==0` → NaN after normalize |
| **Trigger** | Encode pass `[1]` wide tile; image height not divisible by 8; single tile covers full H |
| **Wrong fixes** | End-align (uncovers row 0); encoder-input pad (OOB when floor==round already) |
| **Correct fix** | Replicate-pad **trailing edge of `ps`** for last-axis tiles only |
| **File** | `modules/forge_tiled_vae.py` only |
| **Commit** | `1e2f758a` |
