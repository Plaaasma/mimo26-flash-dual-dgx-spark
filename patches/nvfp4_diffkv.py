# SPDX-License-Identifier: Apache-2.0
"""NVFP4 KV cache for the Triton DiffKV backend (mimo26 kit).

Packed row per (page, kv head, token), all uint8:

    [ K fp4 nibbles: hqk/2 B | K e4m3 scales: hqk/16 B | V fp4 nibbles: hv/2 B | V e4m3 scales: hv/16 B ]

Byte i of a nibble region holds elements (2i, 2i+1): low nibble = even element, high nibble = odd element.
Each scale covers 16 consecutive elements (= 8 bytes); scale = amax(group) / 6 in E4M3 (NVFP4 convention, global
scale folded to 1.0). e2m1 codes: sign bit 3, magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.

MiMo-V2.6 (hqk 192 / hv 128): 96 + 12 + 64 + 8 = 180 B per token per head, vs 320 B in fp8 and 640 B in bf16.

The attention kernel (triton_unified_attention_diffkv.py, NVFP4_KV_CACHE=True) reads the even and odd halves of K
and V as two half-width tiles: Q.K = Q_even.K_lo + Q_odd.K_hi, and P.V lands in two accumulators stored to the even
and odd output columns, so no nibble interleave is ever materialised.
"""
import torch
import triton
import triton.language as tl


def nvfp4_row_bytes(head_size_qk: int, head_size_v: int) -> int:
    return head_size_qk // 2 + head_size_qk // 16 + head_size_v // 2 + head_size_v // 16


def nvfp4_offsets(head_size_qk: int, head_size_v: int) -> tuple[int, int, int, int]:
    """(K_DATA_OFF, K_SCALE_OFF, V_DATA_OFF, V_SCALE_OFF) byte offsets inside one packed row."""
    k_data = 0
    k_scale = head_size_qk // 2
    v_data = k_scale + head_size_qk // 16
    v_scale = v_data + head_size_v // 2
    return k_data, k_scale, v_data, v_scale


@triton.jit
def _e2m1_encode(x):
    """float -> 4-bit e2m1 code (round to nearest of {0,.5,1,1.5,2,3,4,6}); x already divided by the scale."""
    a = tl.abs(x)
    m = tl.where(
        a < 0.25, 0,
        tl.where(a < 0.75, 1,
        tl.where(a < 1.25, 2,
        tl.where(a < 1.75, 3,
        tl.where(a < 2.5, 4,
        tl.where(a < 3.5, 5,
        tl.where(a < 5.0, 6, 7)))))))
    return (m | tl.where(x < 0, 8, 0)).to(tl.uint8)


@triton.jit
def _reshape_and_cache_nvfp4_kernel(
    key_ptr, value_ptr, cache_ptr, slot_ptr,
    k_stride0, k_stride1, v_stride0, v_stride1,
    c_stride0, c_stride1, c_stride2,
    BLOCK_SIZE: tl.constexpr,
    NG_K: tl.constexpr, NG_V: tl.constexpr, NG_K_PAD: tl.constexpr, NG_V_PAD: tl.constexpr,
    K_DATA_OFF: tl.constexpr, K_SCALE_OFF: tl.constexpr, V_DATA_OFF: tl.constexpr, V_SCALE_OFF: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_ptr + tok).to(tl.int64)
    valid = slot >= 0
    slot = tl.where(valid, slot, 0)
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    row = blk * c_stride0 + head * c_stride1 + off * c_stride2
    p = tl.arange(0, 8)
    # ---- K: NG_K groups of 16 elements -> 8 bytes + 1 e4m3 scale each
    gk = tl.arange(0, NG_K_PAD)
    gk_ok = gk < NG_K
    k_src = key_ptr + tok * k_stride0 + head * k_stride1
    k_lo = tl.load(k_src + gk[:, None] * 16 + 2 * p[None, :], mask=gk_ok[:, None], other=0.0).to(tl.float32)
    k_hi = tl.load(k_src + gk[:, None] * 16 + 2 * p[None, :] + 1, mask=gk_ok[:, None], other=0.0).to(tl.float32)
    k_amax = tl.maximum(tl.max(tl.abs(k_lo), axis=1), tl.max(tl.abs(k_hi), axis=1))
    k_s8 = tl.maximum(k_amax / 6.0, 1e-8).to(tl.float8e4nv)
    k_s = k_s8.to(tl.float32)
    k_lo_c = _e2m1_encode(k_lo / k_s[:, None])
    k_hi_c = _e2m1_encode(k_hi / k_s[:, None])
    tl.store(cache_ptr + row + K_SCALE_OFF + gk, k_s8.to(tl.uint8, bitcast=True), mask=valid & gk_ok)
    tl.store(cache_ptr + row + K_DATA_OFF + gk[:, None] * 8 + p[None, :], k_lo_c | (k_hi_c << 4),
             mask=valid & gk_ok[:, None] & (p[None, :] >= 0))
    # ---- V: NG_V groups
    gv = tl.arange(0, NG_V_PAD)
    gv_ok = gv < NG_V
    v_src = value_ptr + tok * v_stride0 + head * v_stride1
    v_lo = tl.load(v_src + gv[:, None] * 16 + 2 * p[None, :], mask=gv_ok[:, None], other=0.0).to(tl.float32)
    v_hi = tl.load(v_src + gv[:, None] * 16 + 2 * p[None, :] + 1, mask=gv_ok[:, None], other=0.0).to(tl.float32)
    v_amax = tl.maximum(tl.max(tl.abs(v_lo), axis=1), tl.max(tl.abs(v_hi), axis=1))
    v_s8 = tl.maximum(v_amax / 6.0, 1e-8).to(tl.float8e4nv)
    v_s = v_s8.to(tl.float32)
    v_lo_c = _e2m1_encode(v_lo / v_s[:, None])
    v_hi_c = _e2m1_encode(v_hi / v_s[:, None])
    tl.store(cache_ptr + row + V_SCALE_OFF + gv, v_s8.to(tl.uint8, bitcast=True), mask=valid & gv_ok)
    tl.store(cache_ptr + row + V_DATA_OFF + gv[:, None] * 8 + p[None, :], v_lo_c | (v_hi_c << 4),
             mask=valid & gv_ok[:, None] & (p[None, :] >= 0))


def reshape_and_cache_nvfp4_diffkv(
    key: torch.Tensor,        # [num_tokens, num_kv_heads, head_size_qk] bf16/fp16
    value: torch.Tensor,      # [num_tokens, num_kv_heads, head_size_v]
    kv_cache: torch.Tensor,   # [num_blocks, num_kv_heads, block_size, ROW] uint8 (vLLM's allocation layout)
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens, num_kv_heads, hqk = key.shape
    hv = value.shape[2]
    assert hqk % 16 == 0 and hv % 16 == 0, "NVFP4 needs head sizes that are multiples of 16"
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[3] == nvfp4_row_bytes(hqk, hv), kv_cache.shape
    if num_tokens == 0:
        return
    kd, ks, vd, vs = nvfp4_offsets(hqk, hv)
    _reshape_and_cache_nvfp4_kernel[(num_tokens, num_kv_heads)](
        key, value, kv_cache, slot_mapping,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        BLOCK_SIZE=kv_cache.shape[2], NG_K=hqk // 16, NG_V=hv // 16,
        NG_K_PAD=triton.next_power_of_2(hqk // 16), NG_V_PAD=triton.next_power_of_2(hv // 16),
        K_DATA_OFF=kd, K_SCALE_OFF=ks, V_DATA_OFF=vd, V_SCALE_OFF=vs,
        num_warps=2,
    )


# ---------------------------------------------------------------------------
# Reference (torch) dequantization, used by the unit tests only.
_E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_nvfp4_cache(kv_cache: torch.Tensor, head_size_qk: int, head_size_v: int) -> torch.Tensor:
    """[num_blocks, num_kv_heads, block_size, ROW] uint8 -> [num_blocks, num_kv_heads, block_size, hqk+hv] float32."""
    kd, ks, vd, vs = nvfp4_offsets(head_size_qk, head_size_v)
    table = _E2M1_TABLE.to(kv_cache.device)

    def side(data: torch.Tensor, scale: torch.Tensor, n: int) -> torch.Tensor:
        lo = data & 0xF
        hi = data >> 4
        def dec(c):
            v = table[(c & 7).long()]
            return torch.where((c & 8) != 0, -v, v)
        x = torch.stack([dec(lo), dec(hi)], dim=-1).reshape(*data.shape[:-1], n)  # interleave even/odd
        s = scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
        return x * s

    k = side(kv_cache[..., kd:ks], kv_cache[..., ks:vd], head_size_qk)
    v = side(kv_cache[..., vd:vs], kv_cache[..., vs:], head_size_v)
    return torch.cat([k, v], dim=-1)
