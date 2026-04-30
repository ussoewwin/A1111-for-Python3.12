# Multidiffusion Tiled VAE OOM Fix — Root Cause & Patch Explanation

**Date:** 2026-04-30  
**Scope:** `extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_utils/attn.py`  
**Commit context:** Fork-specific fixes atop xformers removal (`71616a8`) and A1111 OOM prevention promotion (`283c56e`).

---

## 1. Symptoms — What Failed

When running **Tiled VAE** (via the `multidiffusion-upscaler-for-automatic1111` extension), inference repeatedly crashed with **CUDA Out-Of-Memory (OOM)** errors during the attention forward pass.

Typical traceback sequence:

```
RuntimeError: CUDA error: out of memory
  File "tile_utils/attn.py", line XXX, in flash_attention_attnblock_forward
    out = flash_attn_func(q_f, k_f, v_f, ...)
  ...
  File "tile_utils/attn.py", line YYY, in flash_attention_attnblock_forward
    out = torch.nn.functional.scaled_dot_product_attention(...)
  ...
  File "tile_utils/attn.py", line ZZZ, in cross_attention_attnblock_forward
    q1 = self.q(h_)
```

Key observations from the logs:

1. **Flash-Attention** failed first with `head_dim > 256` (AttnBlock channels exceeded the FA2 hard limit).
2. The **SDPA** (`scaled_dot_product_attention`) fallback then OOM'd immediately.
3. Even the **sub-quadratic** and **cross-attention** fallbacks subsequently failed — often at `self.q(h_)`, a simple `Conv2d` that should not OOM on its own.
4. `get_available_vram()` itself threw exceptions in the OOMed CUDA context, returning a conservative **256 MB** fallback, which caused `cross_attention_attnblock_forward` to attempt absurdly small slice sizes on a poisoned CUDA context.

The user noted: *"It used to finish perfectly before."* This indicates a **regression**, not an inherent hardware limitation.

---

## 2. Root Cause — Why It Broke

### 2.1 Historical Context: The xformers Era (Worked)

Before commit `71616a8` (2025-11-18), the system had **xformers** installed and enabled.

- The WebUI's `optimization_method` was `"xformers"`.
- `tile_utils/attn.py::get_attn_func()` dispatched to `xformers_attnblock_forward`.
- That function's fallback chain was:

```
xformers memory_efficient_attention
    ↓ (NotImplementedError)
sub_quad_attention (chunked)
    ↓ (Exception)
cross_attention_attnblock_forward (manually sliced)
```

**SDPA was never invoked.** The system consistently fell back through chunked, OOM-resilient paths and completed successfully.

### 2.2 The Breaking Change: xformers Removal (2025-11-18)

Commit `71616a8` removed xformers from requirements and deleted the `xformers_fix/` compatibility shims. Consequently:

- `optimization_method` switched from `"xformers"` to `"flash-attention"` (or `"sdp"`).
- `get_attn_func()` began returning **`flash_attention_attnblock_forward`**.

That function's fallback chain **contained SDPA**:

```
flash_attn_func
    ↓ (head_dim > 256 or any error)
scaled_dot_product_attention   ← BAD
    ↓ (OOM)
sub_quad_attention
    ↓ (OOM / sticky CUDA error)
cross_attention_attnblock_forward
```

### 2.3 Why SDPA Is Fatal for Tiled VAE

Tiled VAE's `AttnBlock` operates on tensors where `seq_len = h * w` (height × width of a tile or aggregated latent). In practice this reaches **tens of thousands to hundreds of thousands**.

- **SDPA** materializes the full attention score matrix of shape `(batch, seq_len, seq_len)`.
- For `seq_len = 65,536`, that is a **17 GB float16 tensor** — guaranteed OOM on any consumer GPU.
- SDPA is efficient for *moderate* sequence lengths (the kernel fuses operations), but it is **not chunked**; it cannot handle Tiled VAE scales.

### 2.4 The Sticky CUDA Error Cascade

When a CUDA kernel (e.g., SDPA's `matmul`) fails with OOM, the CUDA context enters a **sticky error state**. Subsequent CUDA operations — even innocuous ones like `self.q(h_)` or `torch.cuda.empty_cache()` — fail with the *same* opaque `CUDA error: out of memory` until the error is explicitly cleared.

The original `flash_attention_attnblock_forward` did **not** clear this error between fallback attempts. After SDPA poisoned the context:

- `sub_quad_attention` failed on its first `matmul`.
- `cross_attention_attnblock_forward` failed at `self.q(h_)`.
- `get_available_vram()` threw inside `torch.cuda.memory_stats()`, masking the real memory picture with a 256 MB default.

### 2.5 The Final Straw: get_available_vram() Exception Handling

Commit `283c56e` (2026-04-23) added a `try/except` around `get_available_vram()` to prevent crashes during OOM. While well-intentioned, it returned **256 MB** whenever the CUDA context was poisoned. This caused `cross_attention_attnblock_forward` to calculate `steps = 1` (since `mem_required` was often less than 256 MB on paper), allocate a giant tensor anyway, and crash again — perpetuating the failure loop.

---

## 3. Remediation — What Was Fixed

### 3.1 Design Goal

Restore the **xformers-era fallback semantics** for the `flash-attention` dispatch path:

```
Flash-Attention (if head_dim ≤ 256)
    ↓ (any error)
sub_quad_attention (chunked, OOM-safe)
    ↓ (any error)
cross_attention_attnblock_forward (manually sliced)
```

**SDPA is intentionally excluded** from this chain.

### 3.2 Cross-Cutting Mechanism: CUDA Recovery

A helper was already present; its **usage** was expanded:

```python
def _recover_cuda_after_oom():
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
```

- `torch.cuda.synchronize()` forces pending operations to complete, surfacing (and thereby clearing) any sticky asynchronous error.
- `torch.cuda.empty_cache()` releases cached-but-idle allocations from PyTorch's CUDA allocator, giving subsequent allocations a clean memory pool.

This is now called **between every fallback transition** and at the start of `cross_attention_attnblock_forward`.

---

## 4. Code Changes — Line-by-Line Explanation

### 4.1 New Constant: `SDP_ATTNBLOCK_MAX_SEQ`

```python
# Sequence-length threshold: above this, SDP allocates the full
# (seq_len, seq_len) attention matrix and OOMs on Tiled VAE workloads.
# Mirrors modules.sd_hijack_optimizations.SDP_ATTNBLOCK_MAX_SEQ.
SDP_ATTNBLOCK_MAX_SEQ = 4096
```

**Meaning:** This threshold mirrors the one added to `modules/sd_hijack_optimizations.py` in commit `283c56e`. When a user explicitly selects `--opt-sdp-attention`, the `sdp_attnblock_forward` path still exists. If `seq_len > 4096`, we **bypass SDPA entirely** and route directly to `sub_quad_attention`. This prevents the guaranteed OOM before it happens.

---

### 4.2 `flash_attention_attnblock_forward` — Complete Rewrite

**Before (broken):**
```
flash_attn_func
    ↓
scaled_dot_product_attention   ← OOM here
    ↓
sub_quad_attention
    ↓
cross_attention_attnblock_forward
```

**After (fixed):**
```python
def flash_attention_attnblock_forward(self, h_):
    """Direct Flash-Attention for AttnBlock without xformers.

    Fallback order (SDPA is intentionally excluded — it materializes the
    full (seq_len, seq_len) attention matrix and OOMs at the huge sequence
    lengths Tiled VAE produces):

        Flash (only if head_dim <= 256)  →  sub_quad (chunked)  →  cross_attention (chunked)

    Between every fallback we clear the sticky CUDA error state and free
    intermediate tensors so the next path starts clean.
    """
```

**Docstring meaning:** Explicitly documents why SDPA is excluded. Future maintainers (including the LLM itself) cannot re-insert SDPA without confronting this documented rationale.

#### Step 1: Compute shared tensors once

```python
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape

    # Precompute shared rearranged tensors once for the flash + sub_quad
    # paths; cross_attention needs different shapes so it recomputes its own.
    q_r = rearrange(q, 'b c h w -> b (h w) c').contiguous()
    k_r = rearrange(k, 'b c h w -> b (h w) c').contiguous()
    v_r = rearrange(v, 'b c h w -> b (h w) c').contiguous()
    original_dtype = q_r.dtype
```

**Why:** `rearrange(...)` and `.contiguous()` allocate new tensors. By computing `q_r`, `k_r`, `v_r` once, the `sub_quad` fallback avoids recomputing them from `h_`. The `cross_attention` fallback uses different reshapes, so it does **not** reuse these — we delete them before handing off.

#### Step 2: Flash-Attention with early rejection

```python
    if c <= 256 and HAS_FLASH_ATTN:
        try:
            q_f = q_r.reshape(b, h * w, 1, c).contiguous()
            k_f = k_r.reshape(b, h * w, 1, c).contiguous()
            v_f = v_r.reshape(b, h * w, 1, c).contiguous()

            if q_f.dtype not in [torch.float16, torch.bfloat16]:
                q_f = q_f.to(torch.float16)
                k_f = k_f.to(torch.float16)
                v_f = v_f.to(torch.float16)

            out = flash_attn_func(q_f, k_f, v_f, dropout_p=0.0, causal=False)
            ...
            return out
        except Exception as e:
            print(f"[Tiled VAE] Flash-Attention direct failed: {e}")
            try:
                del q_f, k_f, v_f
            except UnboundLocalError:
                pass
            _recover_cuda_after_oom()
```

**`c <= 256` check:** Flash-Attention 2 has a hard `head_dim ≤ 256` limit. In `AttnBlock`, `nheads=1`, so `head_dim == channels`. If `c > 256`, we **skip allocation entirely** rather than allocating `q_f/k_f/v_f` just to watch them fail and fragment memory.

**`del q_f, k_f, v_f`:** Frees Flash-Attention-specific allocations before falling back. The `UnboundLocalError` guard handles the case where the exception occurred *before* those variables were bound (e.g., during `reshape`).

**`_recover_cuda_after_oom()`:** Clears the sticky CUDA error so the next fallback starts with a valid context.

#### Step 3: sub_quad fallback (primary safe path)

```python
    try:
        out = sub_quad_attention(
            q_r, k_r, v_r,
            q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size,
            kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size,
            chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold,
            use_checkpoint=False,
        )
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return out
    except Exception as e3:
        print(f"[Tiled VAE] sub_quad failed: {e3}")
        _recover_cuda_after_oom()
```

**Why this is the primary safe path:** `sub_quad_attention` (from `modules.sd_hijack_optimizations`) processes queries and key-values in configurable chunks. It never materializes the full `(seq_len, seq_len)` matrix. For Tiled VAE scales, this is the workhorse that actually completes.

**Recovery call:** If even `sub_quad` fails (e.g., because the CUDA context was already poisoned by a previous *different* operation), we clear errors before the final fallback.

#### Step 4: Final fallback to manually chunked cross attention

```python
    try:
        del q_r, k_r, v_r, q, k, v
    except UnboundLocalError:
        pass
    _recover_cuda_after_oom()
    return cross_attention_attnblock_forward(self, h_)
```

**Memory hygiene:** `cross_attention_attnblock_forward` allocates its own `q1, k1, v` from `h_`. We aggressively delete the shared tensors (`q_r`, `k_r`, `v_r`) and the original conv outputs (`q`, `k`, `v`) to maximize free memory before entering the final fallback.

**Recovery call:** One last CUDA clear before `cross_attention` starts.

---

### 4.3 `sdp_attnblock_forward` — Hardening for Explicit SDPA Selection

Even though the `flash_attention` path no longer calls SDPA, a user may still explicitly set `--opt-sdp-attention`. We hardened that path too.

```python
def sdp_attnblock_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    seq_len = q.shape[1]
    # Skip SDPA entirely for huge sequence lengths — it materializes the
    # full attention matrix and reliably OOMs on Tiled VAE outputs.
    if seq_len > SDP_ATTNBLOCK_MAX_SEQ:
        out = sub_quad_attention(
            q, k, v,
            q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size,
            kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size,
            chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold,
            use_checkpoint=False,
        )
        out = out.to(dtype)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return out
```

**Early bypass:** If `seq_len > 4096`, we do **not** even attempt `scaled_dot_product_attention`. This avoids the guaranteed OOM.

**Try-except wrapper:** If `seq_len ≤ 4096` but SDPA still fails (e.g., due to memory fragmentation from other extensions), we catch the exception, recover CUDA, and fall back to `sub_quad_attention`:

```python
    try:
        out = torch.nn.functional.scaled_dot_product_attention(...)
        ...
        return out
    except Exception as e:
        print(f"[Tiled VAE] SDPA failed (seq_len={seq_len}): {e}")
        _recover_cuda_after_oom()
        out = sub_quad_attention(q, k, v, ...)
        ...
        return out
```

**Meaning:** The explicit SDPA path is now *defensive* rather than fatal. It attempts the fast path for small tiles, but never lets Tiled VAE crash on large tiles.

---

### 4.4 `sdp_no_mem_attnblock_forward` — Unchanged Semantics

```python
def sdp_no_mem_attnblock_forward(self, x):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return sdp_attnblock_forward(self, x)
```

This simply forces PyTorch to avoid the memory-efficient SDPA kernel (which can still OOM) and delegates to the hardened `sdp_attnblock_forward`. The `SDP_ATTNBLOCK_MAX_SEQ` guard inside `sdp_attnblock_forward` protects this path as well.

---

### 4.5 `cross_attention_attnblock_forward` — Already Fixed in Prior Iteration

This function was rewritten in an earlier patch (before 2026-04-30) and was **not modified again** today. Its key OOM-resilient properties are:

- Calls `_recover_cuda_after_oom()` at entry.
- Avoids `torch.zeros_like(k)` (a full `(b, c, hw)` upfront allocation).
- Instead accumulates output slices in a Python list and `torch.cat`s them at the end.
- Computes `steps` from `get_available_vram()` and slices the sequence accordingly.
- Uses `del` aggressively after every intermediate tensor (`q1`, `q2`, `w1`, `w2`, etc.).

---

## 5. Summary Table

| Component | Before (Broken) | After (Fixed) | Rationale |
|---|---|---|---|
| **Fallback order** | Flash → **SDPA** → sub_quad → cross_attn | Flash → sub_quad → cross_attn | SDPA materializes `(seq, seq)` matrix; OOMs at Tiled VAE scale |
| **CUDA error clearing** | None between fallbacks | `_recover_cuda_after_oom()` between every step | Sticky CUDA errors from OOM poison subsequent ops |
| **Flash gate** | Allocated then failed | `c <= 256` check before allocation | Avoids wasted allocation + fragmentation |
| **Explicit SDPA path** | Unconditional SDPA | `seq_len > 4096` → skip to sub_quad | Prevents guaranteed OOM even if user selects `--opt-sdp-attention` |
| **Memory hygiene** | Leaked intermediate tensors | `del` + `UnboundLocalError` guards | Maximizes free memory for next fallback |
| **`get_available_vram()`** | Threw, returned 256 MB (poisoned context) | Still throws-safe, but now used on *recovered* context only | 256 MB default is harmless if CUDA is clean; fatal if sticky error persists |

---

## 6. Why This Restores "It Used to Work"

The xformers-era path (`xformers_attnblock_forward`) was:

```
xformers → sub_quad → cross_attention
```

Our fixed `flash_attention_attnblock_forward` now follows the **same structural shape**:

```
flash (fast path) → sub_quad (chunked) → cross_attention (manually sliced)
```

SDPA — the single component that was **not** present in the working era — has been removed from the chain. The system once again falls back through progressively more memory-conservative, chunked implementations until one succeeds.

---

## 7. Files Modified

- `extensions-builtin/multidiffusion-upscaler-for-automatic1111/tile_utils/attn.py`
  - Added `SDP_ATTNBLOCK_MAX_SEQ = 4096`
  - Rewrote `flash_attention_attnblock_forward` fallback chain
  - Hardened `sdp_attnblock_forward` with early bypass and try-except fallback

No changes to:
- `modules/sd_hijack_optimizations.py` (the upstream already had `SDP_ATTNBLOCK_MAX_SEQ`)
- `cross_attention_attnblock_forward` (already OOM-resilient from prior patch)
- Any other extension or core module

---

*End of document.*
