# PCM / Turbo SDXL Speed LoRA Mapping Issue in A1111 — Complete Explanation

> **Base commit:** `1fc455d` (v2.0, after revert of incorrect fix)
>
> **Fix commit:** `53be7f5`

---

## Overview

When loading PCM (Phased Consistency Model) or Turbo-style SDXL speed LoRAs (e.g. `pcm_sdxl_normalcfg_16step_converted.safetensors`) in A1111, a warning `3/2364 unmatched keys` appeared and some keys were not applied correctly. These LoRAs work correctly in Forge and ComfyUI, but A1111 had not been updated for them because they were released after A1111 development ended.

This document explains the technical background, root cause, and fix in full detail.

---

## Symptoms

When loading a PCM LoRA, the console prints:

```
WARNING:root:[LORA] Loading pcm_sdxl_normalcfg_16step_converted.safetensors for OpenAIWrapper with 3/2364 unmatched keys
```

Debug output showed that the mismatched keys were a single module (3 sub-keys):

```
lora_key: lora_unet_up_blocks_0_upsamplers_0_conv.alpha
lora_key: lora_unet_up_blocks_0_upsamplers_0_conv.lora_down.weight
lora_key: lora_unet_up_blocks_0_upsamplers_0_conv.lora_up.weight
  -> compvis_key: diffusion_model_output_blocks_2_1_conv
```

Only the upsampler in `up_blocks_0` failed to map to an A1111 UNet module.

---

## Root cause: wrong upsampler type index

### How A1111 LoRA key mapping works

PCM LoRAs use Kohaku-ss format (Diffusers-compatible) key names:

```
lora_unet_up_blocks_0_upsamplers_0_conv
```

A1111's UNet uses CompVis-style module names, so key names must be converted. That conversion is handled by `convert_diffusers_name_to_compvis()` in `extensions-builtin/Lora/networks.py`.

### Buggy code

```python
# Before fix (buggy)
if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
    return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"
```

For `up_blocks_0` (`m[0]=0`):
- Index: `2 + 0 * 3 = 2`
- Type: `1` (because `m[0] > 0` is false)
- Result: `diffusion_model_output_blocks_2_1_conv`

### Why this is wrong

Structure of A1111 SDXL UNet output blocks:

| output_blocks index | Content | type |
|---|---|---|
| 0 | up_0 attention_0 (10 transformer blocks) | 1 |
| 1 | up_0 attention_1 (10 transformer blocks) | 1 |
| 2 | up_0 attention_2 (10 transformer blocks) | 1 |
| **2** | **up_0 upsampler** | **2** |
| 3 | up_1 attention_0 (2 transformer blocks) | 1 |
| ... | ... | ... |

`output_blocks_2` contains **two modules**:

1. `output_blocks_2_1_*`: attention_2 (type=1)
2. `output_blocks_2_2_conv`: upsampler (type=2)

The buggy code assigned **type=1** to the upsampler, producing `output_blocks_2_1_conv`. However, type=1 positions are for attention; there is no `conv` attribute there. Lookup in `network_layer_mapping` therefore failed → unmatched.

### Why only `up_blocks_0` was affected

The condition `2 if m[0]>0 else 1` yields:
- `up_blocks_1` (m[0]=1): type=2 → `output_blocks_5_2_conv` ✅ correct
- `up_blocks_0` (m[0]=0): type=1 → `output_blocks_2_1_conv` ❌ wrong

In SD 1.x/2.x UNet layouts, `up_blocks_0` upsamplers may have lived at type=1 positions, which likely left this branch in place. For SDXL, type=2 is always correct.

### Reference: why Forge / ComfyUI worked

- The **`unet_to_diffusers` map** (`unet_diffusers_map.py`) correctly maps `up_blocks.0.upsamplers.0.conv` → `output_blocks.2.2.conv`
- `_register_diffusers_unet_aliases` uses that map to create correct aliases
- **However**, for special keys such as upsampler `conv_shortcut` or `embed`, `network_layer_mapping` lookup can return `None`, so alias registration in `_register_diffusers_unet_aliases` is skipped
- The fallback `convert_diffusers_name_to_compvis` is then used, applying the buggy type logic

---

## Fix

### File changed

`extensions-builtin/Lora/networks.py`

### Before

```python
if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
    return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"
```

### After

```python
if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
    return f"diffusion_model_output_blocks_{2 + m[0] * 3}_2_conv"
```

### What the fix does

- Removes the branch `2 if m[0]>0 else 1` and always uses **type=2**
- In SDXL UNet, upsamplers always sit after attention blocks at type=2
- Index calculation `2 + m[0] * 3` is unchanged (already correct)

---

## Verification

### Before fix

```
[LORA DEBUG] Unmatched keys for pcm_sdxl_normalcfg_16step_converted.safetensors:
  -> compvis_key: diffusion_model_output_blocks_2_1_conv  (3 keys)
```

### After fix

User confirmation: **warning gone**. Reported as fixed. All 2364 keys map correctly.

### Scope of impact

- **Affected:** Kohaku-ss SDXL LoRA keys `up_blocks_0_upsamplers_0_conv`
- **Backward compatibility:** SD 1.x/2.x has no such upsampler key in `up_blocks_0`, or uses other mapping — no impact
- `up_blocks_1` and above already used type=2 correctly — unchanged

---

## LoRAs involved

| LoRA file | Key count | Fix needed |
|---|---|---|
| `pcm_sdxl_normalcfg_16step_converted.safetensors` | 2364 | ✅ Yes |
| `pcm_sdxl_smallcfg_16step_converted.safetensors` | 2364 | ✅ Yes (same structure) |
| `pcm_sd15_normalcfg_16step_converted.safetensors` | 834 | ❌ No (SD1.5) |
| `pcm_sd15_smallcfg_16step_converted.safetensors` | 834 | ❌ No (SD1.5) |

---

## Environment

| Item | Value |
|---|---|
| A1111 version | v2.0 |
| Python | 3.12.10 |
| PyTorch | 2.12.1+cu132 |
| CUDA | 13.2 |
| Flash-Attention | 2.9.1 |
| open_clip | 3.1.0 |
| Repository | `ussoewwin/A1111-for-Python3.12` |

---

## Files changed

| File | Change |
|---|---|
| `extensions-builtin/Lora/networks.py` | `convert_diffusers_name_to_compvis()`: upsampler type index always 2 |

---

## Investigation timeline

### Wrong approach (reverted)

Initially we assumed SDXL UNet block module order differed between Diffusers (attentions → resnets) and CompVis (resnets → attentions), and added an offset to resnet indices. That made things worse (3 unmatched → 54 unmatched) and was reverted immediately.

Lesson: A1111's CompVis UNet interleaves resnets and attentions in pairs within each block; they share the same block index and are distinguished by type (0 vs 1). That internal layout is fundamentally different from Diffusers' attentions → resnets ordering.

### Correct approach

1. Add debug logging in `load_network()` to identify actual unmatched keys
2. Confirm only the 3 sub-keys of `lora_unet_up_blocks_0_upsamplers_0_conv` were unmatched
3. Find `compvis_key` was `diffusion_model_output_blocks_2_1_conv` (type=1)
4. Compare with correct mapping in `unet_to_diffusers` (type=2) and pinpoint the bug in `convert_diffusers_name_to_compvis` type logic
5. Remove the branch and always use type=2

---

## Summary

The core issue was a **type index branch bug** in upsampler mapping inside `convert_diffusers_name_to_compvis()`.

In SDXL UNet, upsamplers always sit at type=2, but legacy code `2 if m[0]>0 else 1` remained, so only `up_blocks_0` was wrongly mapped to type=1 (attention slot).

The fix is removing one conditional branch. Impact is very narrow and fully backward compatible.
