# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attention backend with different K/V head dimensions (DiffKV).

The KV cache layout is identical to ``FlashAttentionDiffKVBackend``: K and V
are packed along the last dim in the logical shape
``[num_blocks, num_kv_heads, block_size, head_size_qk + head_size_v]``.
"""

import os
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)
from vllm.v1.attention.ops.nvfp4_diffkv import (
    dequant_nvfp4_blocks,
    nvfp4_row_bytes,
    reshape_and_cache_nvfp4_diffkv,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# mimo26: NVFP4 prefill on global layers dequantizes the touched blocks once (bf16) and runs the bf16 kernel path
# when the partition has at least this many query tokens (0 = never; in-kernel decode for everything). The bf16
# scratch is capped at MIMO26_NVFP4_DQ_MAX_MB; larger requests use the in-kernel decode.
_NVFP4_DQ_MIN_Q = int(os.environ.get("MIMO26_NVFP4_DQ_MIN_Q", "64"))
_NVFP4_DQ_MAX_BYTES = int(float(os.environ.get("MIMO26_NVFP4_DQ_MAX_MB", "1536")) * 2**20)

# thor patch: purpose-built prefill attention for the global (non-SWA, no-sink) layers, TP2 shape 32/2 heads, 192/128.
_PF_MIN_Q = int(os.environ.get("VLLM_DIFFKV_CUSTOM_PREFILL_MIN_Q", "64"))  # 0 = disabled
_pf_ext = None
_pf_meta_cache: dict = {}


def _pf_load():
    global _pf_ext
    if _pf_ext is None:
        from torch.utils.cpp_extension import load

        build_dir = os.environ.get("ATTN_CUSTOM_BUILD", "/opt/mimo26/attn-build")
        os.makedirs(build_dir, exist_ok=True)  # torch.cpp_extension.load() does not create it
        _pf_ext = load(name="prefill_attn_ext", sources=[os.environ.get("ATTN_CUSTOM_SRC", "/opt/mimo26/kernels/attention/prefill_attn.cu")],
                       extra_cuda_cflags=["-O3", "--use_fast_math"] + os.environ.get("ATTN_CUSTOM_GENCODE", "-gencode=arch=compute_121a,code=sm_121a").split(),
                       build_directory=build_dir, verbose=False)
        logger.info_once("custom diffkv prefill attention kernel loaded (min_q=%d)", _PF_MIN_Q)
    return _pf_ext


def _pf_qblocks(query_start_loc_cpu: torch.Tensor, num_reqs: int, device: torch.device):
    """Per 8-token q-block: (seq idx, first global q row). Built on host from the CPU offsets (prefill only, not graphed)."""
    seqs, starts = [], []
    qsl = query_start_loc_cpu.tolist()
    for i in range(num_reqs):
        for r in range(qsl[i], qsl[i + 1], 8):
            seqs.append(i); starts.append(r)
    t = torch.tensor([seqs, starts], dtype=torch.int32).to(device, non_blocking=True)
    return t[0], t[1]


@dataclass
class TritonAttentionDiffKVMetadata(TritonAttentionMetadata):
    # thor port of local-inference-lab/vllm #848: empty for a single launch, otherwise
    # (decode prefix, prefill suffix) so decode/verify rows keep split-KV in mixed batches.
    partitions: tuple[TritonAttentionMetadata, ...] = ()
    pf_qsl_cpu: torch.Tensor | None = None
    pf_num_reqs: int = -1


class TritonAttentionDiffKVMetadataBuilder(TritonAttentionMetadataBuilder):
    # The partition boundary is host metadata; mixed batches must not replay a full graph
    # with stale partitions (we run FULL_DECODE_ONLY, so mixed batches were never captured).
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_varlen_decode_cudagraph: ClassVar[bool] = True
    """Override the parent's softmax buffer last-dim to head_size_v.

    The parent allocates ``softmax_segm_output`` with last-dim sized to
    ``next_power_of_2(head_size)`` (== Q/K head size).  For DiffKV the
    accumulator and per-segment partial outputs are V-shaped, so we
    re-allocate with ``next_power_of_2(head_size_v)`` instead.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        # thor patch: full-attention groups get more split-KV segments. The spec-verify launch covers a whole
        # request per program (BLOCK_M = q_len x GQA group), so the segment axis must supply the CTAs:
        # 64 segments = 1.3-1.5x (q=4) / 1.8-2.6x (q=8) per layer vs stock 16 on 46K-180K contexts.
        # SWA groups keep the stock count (their loops are window-bounded).
        full_attn_segments = int(os.environ.get("VLLM_DIFFKV_FULL_ATTN_SEGMENTS", "64"))
        if (
            getattr(kv_cache_spec, "sliding_window", None) is None
            and full_attn_segments > self.num_par_softmax_segments
        ):
            self.num_par_softmax_segments = full_attn_segments
            self.softmax_segm_max = torch.empty(
                (self.seq_threshold_3D, self.num_heads_q, full_attn_segments),
                dtype=torch.float32,
                device=device,
            )
            self.softmax_segm_expsum = torch.empty(
                (self.seq_threshold_3D, self.num_heads_q, full_attn_segments),
                dtype=torch.float32,
                device=device,
            )

        head_size_v = TritonAttentionDiffKVBackend.head_size_v
        head_size_v_padded = next_power_of_2(head_size_v)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                head_size_v_padded,
            ),
            dtype=torch.float32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionDiffKVMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata = TritonAttentionDiffKVMetadata(**vars(base))
        metadata.pf_qsl_cpu = common_attn_metadata.query_start_loc_cpu
        capacity = min(
            self.seq_threshold_3D,
            self.softmax_segm_output.shape[0],
            self.softmax_segm_max.shape[0],
            self.softmax_segm_expsum.shape[0],
        )
        assert self.reorder_batch_threshold is not None
        if (
            base.num_actual_tokens <= capacity
            or base.max_query_len <= self.reorder_batch_threshold
        ):
            return metadata
        num_decodes, _, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.reorder_batch_threshold
        )
        if not (0 < num_decode_tokens <= capacity and num_prefill_tokens > 0):
            return metadata
        decode = replace(
            base,
            num_actual_tokens=num_decode_tokens,
            max_query_len=int(
                (
                    common_attn_metadata.query_start_loc_cpu[1 : num_decodes + 1]
                    - common_attn_metadata.query_start_loc_cpu[:num_decodes]
                ).max()
            ),
            query_start_loc=base.query_start_loc[: num_decodes + 1],
            seq_lens=base.seq_lens[:num_decodes],
            block_table=base.block_table[:num_decodes],
            slot_mapping=base.slot_mapping[:num_decode_tokens],
        )
        prefill = replace(
            base,
            num_actual_tokens=num_prefill_tokens,
            query_start_loc=base.query_start_loc[num_decodes:] - num_decode_tokens,
            seq_lens=base.seq_lens[num_decodes:],
            block_table=base.block_table[num_decodes:],
            slot_mapping=base.slot_mapping[num_decode_tokens:],
        )
        metadata.partitions = (decode, prefill)
        metadata.pf_qsl_cpu = common_attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
        metadata.pf_num_reqs = int(common_attn_metadata.query_start_loc_cpu.numel() - 1 - num_decodes)
        return metadata


class TritonAttentionDiffKVBackend(TritonAttentionBackend):
    # V head dim — set per layer via ``set_head_size_v`` before instantiation.
    head_size_v: int = 128

    # Packed DiffKV supports unquantized cache and per-tensor E4M3 scales (thor port of fork #830).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "nvfp4",
    ]

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        cls.head_size_v = head_size_v

    @classmethod
    def customize_spec(cls, spec):
        # mimo26 NVFP4: one packed slot per KV head holding K and V nibbles + e4m3 block scales
        # (hqk/2 + hqk/16 + hv/2 + hv/16 bytes; 180 for MiMo's 192/128 vs 320 in fp8).
        if spec.state_content_bytes is not None or not getattr(spec.kv_quant_mode, "is_nvfp4", False):
            return spec
        from dataclasses import replace as _replace
        return _replace(
            spec,
            num_head_slots=spec.num_kv_heads,
            state_content_bytes=nvfp4_row_bytes(spec.head_size, spec.head_size_v),
        )

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionDiffKVImpl"]:
        return TritonAttentionDiffKVImpl

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionDiffKVMetadataBuilder"]:
        return TritonAttentionDiffKVMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # DiffKV K head sizes (e.g. 192 for MiMo-V2.5) need to be allowed.
        return head_size >= 32

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        # DiffKV only implements decoder self-attention.  Unlike the parent
        # TritonAttentionBackend (which advertises all types), encoder
        # attention is not supported, so gate it here at backend selection.
        return attn_type == AttentionType.DECODER


class TritonAttentionDiffKVImpl(TritonAttentionImpl):
    """Triton attention impl for the DiffKV packed KV cache layout."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # DiffKV dequantizes K/V to the query dtype before its dot products.
        # Inheriting the parent's CUDA query-quantization flag would instead
        # quantize Q without supplying a query descale to this implementation.
        self.supports_quant_query_input = False
        self.is_nvfp4 = str(self.kv_cache_dtype).startswith("nvfp4")
        if is_quantized_kv_cache(self.kv_cache_dtype) and self.kv_cache_dtype not in (
            "fp8",
            "fp8_e4m3",
            "nvfp4",
        ):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend only supports per-tensor E4M3 "
                f"quantized KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )
        if self._is_per_token_head_quant:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support per-token-head "
                "quantization."
            )
        if self.chunk_lookback > -1:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support chunked "
                "attention with lookback."
            )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.is_nvfp4:
            reshape_and_cache_nvfp4_diffkv(key, value, kv_cache, slot_mapping)
            return
        # Cache is logical (B, H, N, C); the diffkv reshape kernel expects
        # (B, N, H, C).
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache.transpose(1, 2),
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        # The fused rope+cache path assumes the standard 2-tensor layout.
        return False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Shapes:
            query:    [num_tokens, num_heads, head_size_qk]
            key:      [num_tokens, num_kv_heads, head_size_qk]
            value:    [num_tokens, num_kv_heads, head_size_v]
            kv_cache: [num_blocks, num_kv_heads, block_size,
                       head_size_qk + head_size_v]
            output:   [num_tokens, num_heads, head_size_v]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for "
                "TritonAttentionDiffKVImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False, (
            "Cascade attention not supported for TritonAttentionDiffKVImpl"
        )

        head_size_qk = self.head_size
        head_size_v = TritonAttentionDiffKVBackend.head_size_v

        # Triton DiffKV kernels consume (B, N, H, D) cache views.
        kv_cache = kv_cache.transpose(1, 2)
        if self.is_nvfp4:
            nvfp4_bhnc = kv_cache.transpose(1, 2)  # vLLM layout (B, H, N, ROW) for the prefill dequant
            nvfp4_view = kv_cache  # (B, N, H, ROW) uint8; the kernel decodes nibbles + scales itself
            kv_packed = kv_cache
            key_cache = kv_cache
            value_cache = kv_cache
        else:
            nvfp4_view = None
            if is_quantized_kv_cache(self.kv_cache_dtype):
                kv_cache = kv_cache.view(self.fp8_dtype)
            kv_packed = kv_cache.transpose(1, 2)  # back to [pages, kv_heads, page, hqk+hv] for the custom prefill kernel
            key_cache = kv_cache[..., :head_size_qk]
            value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]

        partitions = getattr(attn_metadata, "partitions", ()) or (attn_metadata,)
        token_start = 0
        for partition in partitions:
            token_end = token_start + partition.num_actual_tokens
            if (
                _PF_MIN_Q > 0
                and not self.is_nvfp4
                and partition.max_query_len >= _PF_MIN_Q
                and self.sliding_window == (-1, -1)
                and self.sinks is None
                and self.num_heads == 32 and self.num_kv_heads == 2
                and head_size_qk == 192 and head_size_v == 128
                and self.alibi_slopes is None and self.logits_soft_cap == 0
                and (partition is attn_metadata or partition is partitions[-1])
            ):
                qsl_cpu = attn_metadata.pf_qsl_cpu
                num_reqs = attn_metadata.pf_num_reqs if partition is not attn_metadata else int(qsl_cpu.numel() - 1)
                kd = float(layer._k_scale_float) if is_quantized_kv_cache(self.kv_cache_dtype) else 1.0
                vd = float(layer._v_scale_float) if is_quantized_kv_cache(self.kv_cache_dtype) else 1.0
                qb_seq, qb_start = _pf_qblocks(qsl_cpu[: num_reqs + 1], num_reqs, query.device)
                _pf_load().prefill_attn(
                    query[token_start:token_end], kv_packed, partition.block_table, partition.query_start_loc,
                    partition.seq_lens, qb_seq, qb_start, float(self.scale), kd, vd, output[token_start:token_end])
                token_start = token_end
                continue
            if (
                self.is_nvfp4
                and _NVFP4_DQ_MIN_Q > 0
                and partition.max_query_len >= _NVFP4_DQ_MIN_Q
                and self.sliding_window == (-1, -1)
            ):
                blk = nvfp4_bhnc.shape[2]
                n_used = -(-int(partition.max_seq_len) // blk)
                used = partition.block_table[:, :n_used]
                need = used.numel() * blk * nvfp4_bhnc.shape[1] * (head_size_qk + head_size_v) * 2
                if need <= _NVFP4_DQ_MAX_BYTES:
                    scratch = dequant_nvfp4_blocks(
                        nvfp4_bhnc, used.reshape(-1).clamp_min(0), head_size_qk, head_size_v
                    )
                    scratch_bt = torch.arange(
                        used.numel(), device=used.device, dtype=torch.int32
                    ).view(used.shape)
                    unified_attention_diffkv(
                        q=query[token_start:token_end],
                        k=scratch[..., :head_size_qk],
                        v=scratch[..., head_size_qk:],
                        out=output[token_start:token_end],
                        cu_seqlens_q=partition.query_start_loc,
                        seqused_k=partition.seq_lens,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        use_alibi_sqrt=self.use_alibi_sqrt,
                        window_size=self.sliding_window,
                        block_table=scratch_bt,
                        softcap=self.logits_soft_cap,
                        sinks=self.sinks,
                        max_seqlen_q=partition.max_query_len,
                        seq_threshold_3D=partition.seq_threshold_3D,
                        num_par_softmax_segments=partition.num_par_softmax_segments,
                        softmax_segm_output=partition.softmax_segm_output,
                        softmax_segm_max=partition.softmax_segm_max,
                        softmax_segm_expsum=partition.softmax_segm_expsum,
                    )
                    del scratch
                    token_start = token_end
                    continue
            unified_attention_diffkv(
                q=query[token_start:token_end],
                k=key_cache,
                v=value_cache,
                out=output[token_start:token_end],
                cu_seqlens_q=partition.query_start_loc,
                seqused_k=partition.seq_lens,
                softmax_scale=self.scale,
                k_descale=None if self.is_nvfp4 else layer._k_scale,
                v_descale=None if self.is_nvfp4 else layer._v_scale,
                nvfp4_cache=nvfp4_view,
                head_size_v=head_size_v if self.is_nvfp4 else None,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                use_alibi_sqrt=self.use_alibi_sqrt,
                window_size=self.sliding_window,
                block_table=partition.block_table,
                softcap=self.logits_soft_cap,
                sinks=self.sinks,
                max_seqlen_q=partition.max_query_len,
                seq_threshold_3D=partition.seq_threshold_3D,
                num_par_softmax_segments=partition.num_par_softmax_segments,
                softmax_segm_output=partition.softmax_segm_output,
                softmax_segm_max=partition.softmax_segm_max,
                softmax_segm_expsum=partition.softmax_segm_expsum,
            )
            token_start = token_end
        return output
