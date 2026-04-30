'''
    This file is modified from the sd_hijack_optimizations.py to remove the residual and norm part,
    So that the Tiled VAE can support other types of attention.
'''
import math
import torch

from modules import shared, sd_hijack
from einops import rearrange
from modules.sd_hijack_optimizations import get_available_vram, get_xformers_flash_attention_op, sub_quad_attention

try:
    import xformers
    import xformers.ops
except ImportError:
    pass

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def get_attn_func():
    method = sd_hijack.model_hijack.optimization_method
    if method is None:
        return attn_forward
    method = method.lower()
    # The method should be one of the following:
    # ['none', 'sdp-no-mem', 'sdp', 'xformers', ''sub-quadratic', 'v1', 'invokeai', 'doggettx', 'flash-attention']
    
    # If xformers is available and method is unknown (like 'flash-attention'), use xformers
    if method not in ['none', 'sdp-no-mem', 'sdp', 'xformers', 'sub-quadratic', 'v1', 'invokeai', 'doggettx', 'flash-attention']:
        print(f"[Tiled VAE] Warning: Unknown attention optimization method {method}. Please try to update the extension.")
        # Try to use xformers if available
        try:
            import xformers.ops
            print(f"[Tiled VAE] Using xformers for method '{method}'")
            return xformers_attnblock_forward
        except ImportError:
            return attn_forward
    
    if method == 'none':
        return attn_forward
    elif method == 'xformers':
        return xformers_attnblock_forward
    elif method == 'flash-attention':
        # Direct Flash-Attention without xformers
        if HAS_FLASH_ATTN:
            print("[Tiled VAE] Using direct Flash-Attention (without xformers)")
            return flash_attention_attnblock_forward
        else:
            print("[Tiled VAE] Flash-Attention not available, falling back to sub_quad_attention")
            return sub_quad_attnblock_forward
    elif method == 'sdp-no-mem':
        return sdp_no_mem_attnblock_forward
    elif method == 'sdp':
        return sdp_attnblock_forward
    elif method == 'sub-quadratic':
        return sub_quad_attnblock_forward
    elif method == 'doggettx':
        return cross_attention_attnblock_forward
    
    return attn_forward


# The following functions are all copied from modules.sd_hijack_optimizations
# However, the residual & normalization are removed and computed separately.

def attn_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q.shape
    q = q.reshape(b, c, h*w)
    q = q.permute(0, 2, 1)   # b,hw,c
    k = k.reshape(b, c, h*w)  # b,c,hw
    w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.reshape(b, c, h*w)
    w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
    # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w[b,i,j]
    h_ = torch.bmm(v, w_)
    h_ = h_.reshape(b, c, h, w)

    h_ = self.proj_out(h_)

    return h_

def xformers_attnblock_forward(self, h_):
    try:
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
        out = xformers.ops.memory_efficient_attention(q, k, v, op=get_xformers_flash_attention_op(q, k, v))
        out = out.to(dtype)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return out
    except NotImplementedError as e:
        print(f"[Tiled VAE] xformers failed: {e}")
        print("[Tiled VAE] Fallback: sub_quad_attention")
        try:
            # Fallback to sub_quad for memory efficiency
            q = self.q(h_)
            k = self.k(h_)
            v = self.v(h_)
            b, c, h, w = q.shape
            q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            out = sub_quad_attention(q, k, v, q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size, kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size, chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold, use_checkpoint=False)
            out = rearrange(out, 'b (h w) c -> b c h w', h=h)
            out = self.proj_out(out)
            return out
        except Exception as e2:
            print(f"[Tiled VAE] sub_quad also failed: {e2}")
            print("[Tiled VAE] Fallback: cross_attention_attnblock_forward")
            return cross_attention_attnblock_forward(self, h_)

def _recover_cuda_after_oom():
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def cross_attention_attnblock_forward(self, h_):
    # Recover from any sticky CUDA error left by previous failed kernels
    _recover_cuda_after_oom()

    q1 = self.q(h_)
    k1 = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q1.shape

    q2 = q1.reshape(b, c, h*w)
    del q1

    q = q2.permute(0, 2, 1)   # b,hw,c
    del q2

    k = k1.reshape(b, c, h*w) # b,c,hw
    del k1

    # OOM-resilient: avoid torch.zeros_like(k) which allocates a full
    # (b, c, hw) tensor up-front. Instead accumulate slices and cat.
    slices = []

    mem_free_total = get_available_vram()

    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    mem_required = tensor_size * 2.5
    steps = 1

    if mem_required > mem_free_total:
        steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

    slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size

        w1 = torch.bmm(q[:, i:end], k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w2 = w1 * (int(c)**(-0.5))
        del w1
        w3 = torch.nn.functional.softmax(w2, dim=2, dtype=q.dtype)
        del w2

        # attend to values
        v1 = v.reshape(b, c, h*w)
        w4 = w3.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        del w3

        slice_out = torch.bmm(v1, w4)     # b, c, hw_slice
        del v1, w4
        slices.append(slice_out)

    # Reconstruct full tensor; cat needs one extra allocation but the
    # intermediate slices can be freed immediately after.
    h_ = torch.cat(slices, dim=2)
    del slices

    h2 = h_.reshape(b, c, h, w)
    del h_

    h3 = self.proj_out(h2)
    del h2

    return h3

# Sequence-length threshold: above this, SDP allocates the full
# (seq_len, seq_len) attention matrix and OOMs on Tiled VAE workloads.
# Mirrors modules.sd_hijack_optimizations.SDP_ATTNBLOCK_MAX_SEQ.
SDP_ATTNBLOCK_MAX_SEQ = 4096


def sdp_no_mem_attnblock_forward(self, x):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return sdp_attnblock_forward(self, x)

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

    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.to(dtype)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return out
    except Exception as e:
        print(f"[Tiled VAE] SDPA failed (seq_len={seq_len}): {e}")
        _recover_cuda_after_oom()
        # Drop SDP-shaped tensors and retry with sub_quad on the same q/k/v.
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

def sub_quad_attnblock_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = sub_quad_attention(q, k, v, q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size, kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size, chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold, use_checkpoint=self.training)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return out

def flash_attention_attnblock_forward(self, h_):
    """Direct Flash-Attention for AttnBlock without xformers.

    Fallback order (SDPA is intentionally excluded — it materializes the
    full (seq_len, seq_len) attention matrix and OOMs at the huge sequence
    lengths Tiled VAE produces):

        Flash (only if head_dim <= 256)  →  sub_quad (chunked)  →  cross_attention (chunked)

    Between every fallback we clear the sticky CUDA error state and free
    intermediate tensors so the next path starts clean.
    """
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

    # FlashAttention head_dim hard limit is 256.
    # In single-head mode (nheads=1) head_dim equals channels.
    # Skip flash entirely if channels exceed the limit so we don't even
    # allocate tensors that would fail immediately and fragment memory.
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

            if out.dtype != original_dtype:
                out = out.to(original_dtype)

            out = out.reshape(b, h * w, c)
            out = rearrange(out, 'b (h w) c -> b c h w', h=h)
            out = self.proj_out(out)
            return out
        except Exception as e:
            print(f"[Tiled VAE] Flash-Attention direct failed: {e}")
            try:
                del q_f, k_f, v_f
            except UnboundLocalError:
                pass
            _recover_cuda_after_oom()

    # sub_quad fallback (chunked, memory-efficient).
    # This is the primary safe path for huge sequence lengths.
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

    # Final fallback: manually chunked cross attention.
    # Purge shared tensors so cross_attention can allocate its own buffers
    # in the freed memory.
    try:
        del q_r, k_r, v_r, q, k, v
    except UnboundLocalError:
        pass
    _recover_cuda_after_oom()
    return cross_attention_attnblock_forward(self, h_)
