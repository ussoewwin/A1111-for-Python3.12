# MultiDiffusion / Tiled VAE CUDA OOM Fix — Technical Explanation

## 1. What Was Broken (Symptoms)

When using **img2img with MultiDiffusion + Tiled VAE enabled**, the WebUI crashes with a **CUDA out-of-memory (OOM) error**. The traceback typically points to one of the following locations:

- `torch.bmm` inside `sub_quadratic_attention.py`
- `torch.cuda.mem_get_info` inside `modules/memmon.py`
- `mid.attn_1` inside the VAE encoder

After the first OOM, the GPU enters a **sticky error state**: subsequent calls to `torch.cuda.empty_cache()` or `mem_get_info()` throw the *same* CUDA error, making the session unusable until a full restart.

## 2. Root Cause

The OOM is **not** caused by insufficient VRAM in an absolute sense. It is caused by a **chain of heuristic shortcuts** that decide — incorrectly — that a huge attention matrix "fits" in memory, and then allocate it all at once.

### 2.1 The Tiled VAE "tiny skip" heuristic (`tilevae.py`)

`VAEHook.__call__` contained a guard:

```python
if self.is_decoder and max(H, W) <= self.pad * 2 + self.tile_size:
    return self.net.original_forward(x)
```

For the **encoder** path (`is_decoder = False`), the guard was already disabled, but for the **decoder** path it still existed. More importantly, when the UI sets a very large encoder tile size (e.g. 1536–3072), the condition `max(H,W) <= pad*2 + tile_size` becomes **true for normal-resolution images** (768–2048 px). When this happens, Tiled VAE skips tiling entirely and falls back to the **non-tiled VAE forward**, which runs the VAE `mid.attn_1` block with **global attention** on the full feature map.

### 2.2 The global attention path (`sdp_attnblock_forward`)

The non-tiled VAE encoder hits `AttnBlock` with `seq_len = H * W`. For a 1024×1024 image this can be **4096 tokens or more**.

The code flow is:
1. `sdp_attnblock_forward` is invoked.
2. If `seq_len >= SDP_ATTNBLOCK_MAX_SEQ` (4096), it bypasses `torch.nn.functional.scaled_dot_product_attention` and calls `sub_quad_attention` directly.
3. Otherwise it tries SDPA first; on OOM it falls back to `sub_quad_attention`.

### 2.3 The sub-quadratic "fits VRAM" shortcut (`sub_quad_attention`)

Inside `sub_quad_attention` (in `modules/sub_quadratic_attention.py`), the logic checks whether the KV tensor "fits" in available VRAM:

```python
chunk_threshold_bytes = chunk_threshold  # default = some large value from cmd_opts
if chunk_threshold_bytes is None or chunk_threshold_bytes <= 0:
    # heuristic disabled → always use chunked path
else:
    # if (k_tokens * head_dim * element_size) < chunk_threshold_bytes:
    #     kv_chunk_size = k_tokens   ← NON-CHUNKED fast path
    # else:
    #     kv_chunk_size = ...        ← chunked safe path
```

When `chunk_threshold` comes from `shared.cmd_opts.sub_quad_chunk_threshold`, it is usually a **very large positive number** (e.g. several GB). This means the heuristic says *"yes, the KV tensor fits"*, sets `kv_chunk_size = k_tokens`, and routes to `_get_attention_scores_no_kv_chunking`, which calls `torch.bmm(q, k)` on the **full `(seq_len, seq_len)` attention matrix**.

For `seq_len = 4096` and FP16, that matrix is:
- `4096 × 4096 × 2 bytes ≈ 32 MB` per head
- multiplied by batch and channels → **hundreds of MB to several GB**

During **img2img encode_first_stage**, the GPU is already heavily loaded with the UNet and other buffers. This extra allocation pushes it over the edge → **OOM inside `torch.bmm`**.

### 2.4 MemMon amplification (`memmon.py`)

`MemUsageMonitor` polls `torch.cuda.mem_get_info()` in a background thread. After the OOM, CUDA is in an error state. The next call to `mem_get_info()` throws the same CUDA error, which **kills the MemMon thread** (or at least makes it spam exceptions). Without MemMon, the WebUI loses memory-pressure feedback and becomes even more likely to OOM again.

## 3. All Code Changes

### 3.1 `extensions-builtin/.../scripts/tilevae.py`

**Before:**
```python
            B, C, H, W = x.shape
            if self.is_decoder and max(H, W) <= self.pad * 2 + self.tile_size:
                print("[Tiled VAE]: the input size is tiny and unnecessary to tile.")
                return self.net.original_forward(x)
            return self.vae_tile_forward(x)
```

**After:**
```python
            # Never call original_forward here: large UI tile_size makes "tiny skip" true for
            # full-res encoder inputs → global VAE AttnBlock + OOM (img2img encode_first_stage).
            return self.vae_tile_forward(x)
```

**Meaning:** Remove the `original_forward` short-circuit completely. Tiled VAE **always** tiles, even for inputs that look "tiny" relative to the UI tile size. This prevents the non-tiled global-attention path from ever being reached inside `VAEHook`.

---

### 3.2 `modules/sd_hijack_optimizations.py` — `sdp_attnblock_forward`

**Before:**
```python
    use_sub_quad_direct = seq_len >= SDP_ATTNBLOCK_MAX_SEQ
    _sub_quad_attnblock_kw = dict(
        q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size,
        kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size,
        chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold,
        use_checkpoint=self.training,
    )
```

**After:**
```python
    use_sub_quad_direct = seq_len >= SDP_ATTNBLOCK_MAX_SEQ
    # chunk_threshold=0 disables sub_quad_attention's "fits VRAM → kv_chunk=k_tokens" shortcut
    # that routes to _get_attention_scores_no_kv_chunking and OOMs on large seq_len.
    _sub_quad_attnblock_kw = dict(
        q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size,
        kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size,
        chunk_threshold=0,
        use_checkpoint=self.training,
    )
```

**Meaning:** Force `chunk_threshold=0`. In the `sub_quad_attention` implementation, `chunk_threshold_bytes` is the numeric limit for the "fits" heuristic. Passing `0` makes the condition `chunk_threshold_bytes <= 0` true, which **forces the chunked KV path** (`kv_chunk_size < k_tokens`) and prevents the `_get_attention_scores_no_kv_chunking` / `torch.bmm` OOM.

---

### 3.3 `extensions-builtin/.../tile_utils/attn.py`

**Before (all 5 call sites):**
```python
chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold,
```

**After (all 5 call sites):**
```python
chunk_threshold=0,
```

**Meaning:** The Tiled VAE extension has its own copies of `sdp_attnblock_forward`, `sub_quad_attnblock_forward`, and `flash_attention_attnblock_forward`. Each of them calls `sub_quad_attention` with `chunk_threshold`. We replace every occurrence with `0` so the extension and the main module behave identically — **always chunked, never the "fits" fast path**.

---

### 3.4 `modules/memmon.py`

**Before:**
```python
    def cuda_mem_get_info(self):
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(index)

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            torch.cuda.reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = self.cuda_mem_get_info()[0]

            while self.run_flag.is_set():
                free, total = self.cuda_mem_get_info()
                self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.opts.memmon_poll_rate)

    def read(self):
        if not self.disabled:
            free, total = self.cuda_mem_get_info()
            self.data["free"] = free
            self.data["total"] = total

            torch_stats = torch.cuda.memory_stats(self.device)
            self.data["active"] = torch_stats["active.all.current"]
            self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
            self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
            self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data
```

**After:**
```python
    def cuda_mem_get_info(self):
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(index)

    def cuda_mem_get_info_safe(self):
        try:
            return self.cuda_mem_get_info()
        except Exception:
            return None

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            torch.cuda.reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            mi0 = self.cuda_mem_get_info_safe()
            self.data["min_free"] = mi0[0] if mi0 is not None else 0

            while self.run_flag.is_set():
                mi = self.cuda_mem_get_info_safe()
                if mi is not None:
                    free, _total = mi
                    self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.opts.memmon_poll_rate)

    def read(self):
        if not self.disabled:
            mi = self.cuda_mem_get_info_safe()
            if mi is not None:
                free, total = mi
                self.data["free"] = free
                self.data["total"] = total
                self.data["system_peak"] = total - self.data["min_free"]

            try:
                torch_stats = torch.cuda.memory_stats(self.device)
                self.data["active"] = torch_stats["active.all.current"]
                self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
                self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
                self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            except Exception:
                pass

        return self.data
```

**Meaning:** Wrap every CUDA API call (`mem_get_info` and `memory_stats`) in `try/except`. After an OOM, the CUDA runtime stays in an error state for the **current stream/context**. If MemMon’s background thread hits that error unchecked, it dies or loops on exceptions. By catching the error and returning `None`, MemMon **survives** the sticky error state and continues reporting whatever metrics are still readable. `read()` no longer crashes the caller either.

## 4. Why Each Fix Is Necessary

| # | File | Change | Why it fixes the bug |
|---|------|--------|----------------------|
| 1 | `tilevae.py` | Delete `original_forward` short-circuit | Prevents the non-tiled VAE path from running global attention on full-resolution feature maps. |
| 2 | `sd_hijack_optimizations.py` | `chunk_threshold=0` in `sdp_attnblock_forward` | Forces `sub_quad_attention` to always use the chunked KV path, avoiding the `(seq_len, seq_len)` `torch.bmm` that OOMs. |
| 3 | `tile_utils/attn.py` | `chunk_threshold=0` in all `sub_quad_attention` calls | The extension has its own attention wrappers; they must use the same safe setting as the main module. |
| 4 | `memmon.py` | `try/except` around `mem_get_info` and `memory_stats` | Keeps the memory monitor alive after a CUDA OOM sticky error, preserving memory-pressure feedback for the UI. |

## 5. Summary

The OOM chain is:

```
img2img encode_first_stage
  → VAEHook decides image is "tiny" (large UI tile_size)
    → skips tiling, calls original_forward
      → VAE encoder mid.attn_1 runs with global seq_len
        → sdp_attnblock_forward → sub_quad_attention
          → sub_quad "fits VRAM" heuristic says yes
            → non-chunked path → torch.bmm(seq_len, seq_len) → OOM
              → sticky CUDA error → MemMon dies → no memory feedback
```

The fix breaks this chain at **three independent points**:

1. **Never skip tiling** (`tilevae.py`).
2. **Never trust the "fits VRAM" heuristic for large attention** (`chunk_threshold=0` everywhere).
3. **Make MemMon survive the sticky error** (`memmon.py`).

All three are required. Removing only the tiny skip would still leave the sub-quadratic shortcut vulnerable. Removing only the shortcut would still leave MemMon dead. The fixes are **orthogonal and additive**.
