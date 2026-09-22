// Purpose-built chunked-prefill attention for MiMo-V2.6 global layers on SM120 (TP2 shape).
//   Q  [nq, HQ=32, DK=192] bf16 (varlen: cu_seqlens_q), packed paged KV cache [pages, HKV=2, PS, DK+DV] fp8-e4m3 or bf16,
//   block_table [B, max_pages] int32, seq_lens [B] int32 (context + new), causal w/ prefix. Out [nq, HQ, DV=128] bf16.
// CTA = (8 query tokens x 16 heads of one KV head) = 128 rows; warp w owns token w (16 head rows), so its causal boundary is one
// key position. K/V stream through a 2-stage smem ring as fp16 (converted on load), 64 keys per tile, fragments via ldmatrix,
// mma.m16n8k16 f16 -> f32, FA2 online softmax. Per-tensor K/V descales fold into the softmax scale and the output.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int HQ = 32, HKV = 2, GQA = HQ / HKV, DK = 192, DV = 128, KVROW = DK + DV;
constexpr int TOK = 8, ROWS = TOK * GQA;       // 128 rows per CTA, 8 warps x 16 rows
constexpr int TK = 64;                          // keys per tile
constexpr int KS = DK + 8, VS = DV + 8;         // smem row strides (halves): 400 B / 272 B -> conflict-free ldmatrix
constexpr int STAGES = 2;
constexpr int SMEM_K = TK * KS * 2, SMEM_V = TK * VS * 2, SMEM_STAGE = SMEM_K + SMEM_V;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_f16x2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b); return *reinterpret_cast<uint32_t*>(&h);
}
// 8 fp8 (uint2) -> 8 fp16 (uint4) ; 8 bf16 (uint4) -> 8 fp16 (uint4)
__device__ __forceinline__ uint4 fp8x8_to_f16x8(uint2 v) {
  uint4 o;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(o.x) : "h"((unsigned short)(v.x & 0xFFFF)));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(o.y) : "h"((unsigned short)(v.x >> 16)));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(o.z) : "h"((unsigned short)(v.y & 0xFFFF)));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(o.w) : "h"((unsigned short)(v.y >> 16)));
  return o;
}
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t v) {
  __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&v);
  float2 f = __bfloat1622float2(b);
  return pack_f16x2(f.x, f.y);
}

// Load one 64-key tile (K 192 + V 128 per key) from the paged cache into registers: per thread 5 x 16 fp8 (or 5 x 8 bf16).
// Chunking: each key row = 320 elems = 20 chunks of 16 (fp8) -> 64*20 = 1280 chunks / 256 threads = 5.
template <bool FP8>
struct TileLoader {
  static constexpr int CH = 16;                       // elems per chunk
  static constexpr int CPR = KVROW / CH;              // 20 chunks per key row
  static constexpr int NCH = TK * CPR;                // 1280
  static constexpr int PER_T = NCH / 256;             // 5
  uint4 reg[PER_T][FP8 ? 1 : 2];                      // fp8: 16 B per chunk; bf16: 32 B per chunk

  __device__ __forceinline__ void load(const void* __restrict__ cache, const int32_t* __restrict__ bt, int page_size,
                                       int key0, int seq_len, int tid) {
#pragma unroll
    for (int i = 0; i < PER_T; ++i) {
      const int c = tid + 256 * i, row = c / CPR, ch = c - row * CPR;
      const int key = key0 + row;
      if (key < seq_len) {
        const int page = bt[key / page_size], off = key - (key / page_size) * page_size;
        const size_t base = ((size_t)page * HKV + blockIdx.y) * page_size + off;  // row index in [pages*HKV*PS]
        if (FP8) {
          reg[i][0] = __ldg(reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(cache) + base * KVROW + ch * CH));
        } else {
          const uint4* p = reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(cache) + base * KVROW + ch * CH);
          reg[i][0] = __ldg(p); reg[i][1] = __ldg(p + 1);
        }
      } else {
        reg[i][0] = make_uint4(0, 0, 0, 0);
        if (!FP8) reg[i][1] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  __device__ __forceinline__ void store(uint8_t* stage, int tid) {
    __half* ks = reinterpret_cast<__half*>(stage);
    __half* vs = reinterpret_cast<__half*>(stage + SMEM_K);
#pragma unroll
    for (int i = 0; i < PER_T; ++i) {
      const int c = tid + 256 * i, row = c / CPR, ch = c - row * CPR;
      uint4 lo, hi;  // 16 fp16 = 2 x uint4
      if (FP8) {
        lo = fp8x8_to_f16x8(make_uint2(reg[i][0].x, reg[i][0].y));
        hi = fp8x8_to_f16x8(make_uint2(reg[i][0].z, reg[i][0].w));
      } else {
        lo = make_uint4(bf16x2_to_f16x2(reg[i][0].x), bf16x2_to_f16x2(reg[i][0].y), bf16x2_to_f16x2(reg[i][0].z), bf16x2_to_f16x2(reg[i][0].w));
        hi = make_uint4(bf16x2_to_f16x2(reg[i][1].x), bf16x2_to_f16x2(reg[i][1].y), bf16x2_to_f16x2(reg[i][1].z), bf16x2_to_f16x2(reg[i][1].w));
      }
      const int e0 = ch * CH;  // element offset within the 320-wide row
      if (e0 < DK) {
        uint4* d = reinterpret_cast<uint4*>(ks + row * KS + e0);
        d[0] = lo; d[1] = hi;
      } else {
        uint4* d = reinterpret_cast<uint4*>(vs + row * VS + (e0 - DK));
        d[0] = lo; d[1] = hi;
      }
    }
  }
};

template <bool FP8>
__global__ void __launch_bounds__(256, 1) prefill_attn_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ cache, const int32_t* __restrict__ block_table, int bt_stride,
    int page_size, const int32_t* __restrict__ cu_seqlens_q, const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ qblk_seq, const int32_t* __restrict__ qblk_start,  // per q-block: seq idx, first q row (global)
    float scale_log2, float v_descale, __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
  const int qb = blockIdx.x, kvh = blockIdx.y;
  const int seq = qblk_seq[qb];
  const int q_start = cu_seqlens_q[seq], q_end = cu_seqlens_q[seq + 1];
  const int row0 = qblk_start[qb];                       // first global q row of this block
  const int seq_len = seq_lens[seq], ctx = seq_len - (q_end - q_start);
  const int my_tok = row0 + warp;                        // this warp's global q row (token)
  const bool tok_valid = my_tok < q_end;
  const int my_pos = ctx + (my_tok - q_start);           // absolute position of this token (keys <= my_pos allowed)
  const int32_t* bt = block_table + (size_t)seq * bt_stride;
  // keys this CTA needs: [0, ctx + (last valid token local idx) + 1)
  const int last_tok = min(row0 + TOK - 1, q_end - 1);
  const int n_keys = ctx + (last_tok - q_start) + 1;
  const int n_tiles = (n_keys + TK - 1) / TK;

  // ---- Q fragments: warp rows = 16 heads of token my_tok; A[row=head][k=d]. 12 k16 steps x 4 regs (fp16).
  uint32_t qa[DK / 16][4];
  {
    const __nv_bfloat16* qp = tok_valid ? q + ((size_t)my_tok * HQ + kvh * GQA) * DK : q;  // head rows contiguous (stride DK)
#pragma unroll
    for (int ks = 0; ks < DK / 16; ++ks) {
      // a0,a1: row g, k 2c..; a2,a3: row g+8; a4,a5: row g, k 2c+8; a6,a7: row g+8, k 2c+8
      const __nv_bfloat16* r0 = qp + (size_t)g * DK + ks * 16 + 2 * c;
      const __nv_bfloat16* r8 = qp + (size_t)(g + 8) * DK + ks * 16 + 2 * c;
      uint32_t v0 = *reinterpret_cast<const uint32_t*>(r0), v1 = *reinterpret_cast<const uint32_t*>(r8);
      uint32_t v2 = *reinterpret_cast<const uint32_t*>(r0 + 8), v3 = *reinterpret_cast<const uint32_t*>(r8 + 8);
      qa[ks][0] = bf16x2_to_f16x2(v0); qa[ks][1] = bf16x2_to_f16x2(v1); qa[ks][2] = bf16x2_to_f16x2(v2); qa[ks][3] = bf16x2_to_f16x2(v3);
      if (!tok_valid) { qa[ks][0] = qa[ks][1] = qa[ks][2] = qa[ks][3] = 0u; }
    }
  }
  float o[DV / 8][4];
#pragma unroll
  for (int n = 0; n < DV / 8; ++n) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
  float m_r[2] = {-1e30f, -1e30f}, l_r[2] = {0.f, 0.f};  // rows g, g+8

  TileLoader<FP8> ld;
  ld.load(cache, bt, page_size, 0, n_keys, tid);
  ld.store(smem, tid);
  __syncthreads();

  for (int t = 0; t < n_tiles; ++t) {
    uint8_t* cur = smem + (t & 1) * SMEM_STAGE;
    uint8_t* nxt = smem + ((t + 1) & 1) * SMEM_STAGE;
    if (t + 1 < n_tiles) ld.load(cache, bt, page_size, (t + 1) * TK, n_keys, tid);  // in flight during compute
    const __half* ks = reinterpret_cast<const __half*>(cur);
    const __half* vs = reinterpret_cast<const __half*>(cur + SMEM_K);
    const int key0 = t * TK;

    // ---- S = Q K^T : 16 rows x 64 keys = 8 n-tiles
    float s[TK / 8][4];
#pragma unroll
    for (int n = 0; n < TK / 8; ++n) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
#pragma unroll
    for (int kk = 0; kk < DK / 16; ++kk) {
#pragma unroll
      for (int np = 0; np < TK / 16; ++np) {  // pairs of n8 tiles via one ldmatrix.x4 (16 keys x 16 d)
        // ldmatrix x4 non-trans on K[key][d]: matrices (keys 0-7,d 0-7),(keys 8-15,d 0-7),(keys 0-7,d 8-15),(keys 8-15,d 8-15)
        const int key = np * 16 + (lane & 15), dd = kk * 16 + (lane >> 4) * 8;
        uint32_t r[4];
        ldmatrix_x4(r, smem_u32(ks + key * KS + dd));
        mma16816(s[2 * np], qa[kk], r[0], r[2]);
        mma16816(s[2 * np + 1], qa[kk], r[1], r[3]);
      }
    }
    // ---- scale + mask (only tiles reaching past my_pos need masking; also keys >= n_keys are zero rows -> mask)
    const bool need_mask = (key0 + TK - 1 > my_pos) || !tok_valid;
#pragma unroll
    for (int n = 0; n < TK / 8; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v = s[n][e] * scale_log2;
        if (need_mask) {
          const int key = key0 + n * 8 + 2 * c + (e & 1);
          if (key > my_pos || !tok_valid) v = -1e30f;
        }
        s[n][e] = v;
      }
    }
    // ---- online softmax (base 2)
    float mx[2] = {m_r[0], m_r[1]};
#pragma unroll
    for (int n = 0; n < TK / 8; ++n) { mx[0] = fmaxf(mx[0], fmaxf(s[n][0], s[n][1])); mx[1] = fmaxf(mx[1], fmaxf(s[n][2], s[n][3])); }
#pragma unroll
    for (int r = 0; r < 2; ++r) { mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1)); mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2)); }
    float alpha[2] = {exp2f(m_r[0] - mx[0]), exp2f(m_r[1] - mx[1])};
    float rs[2] = {0.f, 0.f};
    uint32_t pa[TK / 16][4];  // P as A fragments for PV (k16 = 16 keys)
#pragma unroll
    for (int n = 0; n < TK / 8; ++n) {
      const float p0 = exp2f(s[n][0] - mx[0]), p1 = exp2f(s[n][1] - mx[0]);
      const float p2 = exp2f(s[n][2] - mx[1]), p3 = exp2f(s[n][3] - mx[1]);
      rs[0] += p0 + p1; rs[1] += p2 + p3;
      const int kp = n >> 1, hi = n & 1;  // n8 tile n -> k16 step n/2, low/high half
      pa[kp][hi ? 2 : 0] = pack_f16x2(p0, p1);
      pa[kp][hi ? 3 : 1] = pack_f16x2(p2, p3);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) { rs[r] += __shfl_xor_sync(0xffffffff, rs[r], 1); rs[r] += __shfl_xor_sync(0xffffffff, rs[r], 2); }
    l_r[0] = l_r[0] * alpha[0] + rs[0]; l_r[1] = l_r[1] * alpha[1] + rs[1];
    m_r[0] = mx[0]; m_r[1] = mx[1];
#pragma unroll
    for (int n = 0; n < DV / 8; ++n) { o[n][0] *= alpha[0]; o[n][1] *= alpha[0]; o[n][2] *= alpha[1]; o[n][3] *= alpha[1]; }
    // ---- O += P V : k = 64 keys (4 k16 steps), n = 128 d (16 n8 tiles); V[key][d] row-major -> ldmatrix.trans
#pragma unroll
    for (int kp = 0; kp < TK / 16; ++kp) {
#pragma unroll
      for (int np = 0; np < DV / 16; ++np) {
        // x4.trans on V[key][d]: matrices (keys 0-7, d 0-7),(keys 8-15, d 0-7),(keys 0-7, d 8-15),(keys 8-15, d 8-15)
        const int key = kp * 16 + (lane & 15), dd = np * 16 + (lane >> 4) * 8;
        uint32_t r[4];
        ldmatrix_x4_trans(r, smem_u32(vs + key * VS + dd));
        mma16816(o[2 * np], pa[kp], r[0], r[1]);
        mma16816(o[2 * np + 1], pa[kp], r[2], r[3]);
      }
    }
    __syncthreads();                       // everyone done reading `cur` before it becomes `nxt` next iteration
    if (t + 1 < n_tiles) ld.store(nxt, tid);
    __syncthreads();
  }
  // ---- epilogue: O / l, times v_descale; rows g / g+8 = heads kvh*16 + g / +8 of token my_tok
  if (!tok_valid) return;
  const float inv0 = v_descale / l_r[0], inv1 = v_descale / l_r[1];
  __nv_bfloat16* op = out + ((size_t)my_tok * HQ + kvh * GQA) * DV;
#pragma unroll
  for (int n = 0; n < DV / 8; ++n) {
    const int dd = n * 8 + 2 * c;
    *reinterpret_cast<__nv_bfloat162*>(op + (size_t)g * DV + dd) = __floats2bfloat162_rn(o[n][0] * inv0, o[n][1] * inv0);
    *reinterpret_cast<__nv_bfloat162*>(op + (size_t)(g + 8) * DV + dd) = __floats2bfloat162_rn(o[n][2] * inv1, o[n][3] * inv1);
  }
}

}  // namespace

// qblk_seq / qblk_start: per q-block metadata built by the caller (static shape: cdiv(nq, 8) + batch entries).
void prefill_attn(torch::Tensor q, torch::Tensor kv_cache, torch::Tensor block_table, torch::Tensor cu_seqlens_q,
                  torch::Tensor seq_lens, torch::Tensor qblk_seq, torch::Tensor qblk_start, double scale, double k_descale,
                  double v_descale, torch::Tensor out) {
  TORCH_CHECK(q.size(1) == HQ && q.size(2) == DK && kv_cache.size(1) == HKV && kv_cache.size(3) == KVROW, "shape");
  const bool fp8 = kv_cache.dtype() == torch::kFloat8_e4m3fn;
  const int page = kv_cache.size(2), nblk = qblk_seq.size(0);
  auto stream = c10::cuda::getCurrentCUDAStream();
  const float scale_log2 = (float)(scale * k_descale * 1.4426950408889634);
  dim3 grid(nblk, HKV);
  const size_t smem = STAGES * SMEM_STAGE;
  auto launch = [&](auto fp8c) {
    constexpr bool F = decltype(fp8c)::value;
    cudaFuncSetAttribute(prefill_attn_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    prefill_attn_kernel<F><<<grid, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), kv_cache.data_ptr(), block_table.data_ptr<int32_t>(),
        (int)block_table.stride(0), page, cu_seqlens_q.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
        qblk_seq.data_ptr<int32_t>(), qblk_start.data_ptr<int32_t>(), scale_log2, (float)v_descale,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  };
  if (fp8) launch(std::true_type{}); else launch(std::false_type{});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("prefill_attn", &prefill_attn); }
