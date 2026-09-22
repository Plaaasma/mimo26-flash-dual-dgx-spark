# MiMo-V2.6-Flash-RL serving image for 2x DGX Spark (GB10 / sm_121), vLLM TP=2 over CX7 RoCE.
#
# Base: NVIDIA/vLLM's day-0 MiMo-V2.6 image (arm64, CUDA 13.0, torch 2.13, vLLM 0.29.1rc1.dev449+geb8798058). The cu129 variant
# compiles some _C kernels only as arch-specific sm_120a, which an sm_121 GB10 cannot run ("no kernel image", boot 3);
# the cu130 build ships generic sm_120 SASS for everything:
# has the MiMo-V2 loader (mxfp4 experts + bf16 router, fused-qkv sharding, SupportsEagle3 on the Omni class).
# On top of it (patches/, from the diffbot RTX PRO 6000 recipe; the originals are byte-identical to this image):
#   mimo_v2.py                          cache_config reaches the attention layers (fp8 KV actually applies;
#                                       global layers stay global), opt-in FP8 o_proj (VLLM_MIMO_OPROJ_FP8=1)
#   triton_attn_diffkv.py               fp8 KV enable + descales, 64 split-KV segments on full-attention groups,
#                                       mixed-batch partition, dispatch to the custom prefill kernel
#   triton_unified_attention_diffkv.py  split-KV for the spec-decode verify step (stock ran an 8-token verify on
#                                       18 CTAs), BLOCK_M 128 prefill tiles, per-tensor E4M3 K/V with fp16 dots
#   custom_all_reduce.py                VLLM_CA_MAX_SIZE_MB knob (SM12x has no entry in vLLM's table)
# The custom prefill kernel is prebuilt here for sm_121a so the serving container never JIT-compiles.
ARG BASE_IMAGE=vllm/vllm-openai:mimo-v26-aarch64-cu130
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
LABEL mimo26.base="${BASE_IMAGE}"
ARG VPY=/usr/local/lib/python3.12/dist-packages/vllm

COPY patches/mimo_v2.py                         ${VPY}/model_executor/models/mimo_v2.py
COPY patches/triton_attn_diffkv.py              ${VPY}/v1/attention/backends/triton_attn_diffkv.py
COPY patches/triton_unified_attention_diffkv.py ${VPY}/v1/attention/ops/triton_unified_attention_diffkv.py
COPY patches/custom_all_reduce.py               ${VPY}/distributed/device_communicators/custom_all_reduce.py
RUN find ${VPY}/model_executor/models/__pycache__ ${VPY}/v1/attention/backends/__pycache__ \
         ${VPY}/v1/attention/ops/__pycache__ ${VPY}/distributed/device_communicators/__pycache__ \
         -name '*.pyc' -delete 2>/dev/null || true

COPY kernels/attention/prefill_attn.cu /opt/mimo26/kernels/attention/prefill_attn.cu
COPY scripts/build-kernel.py          /opt/mimo26/build-kernel.py
ENV ATTN_CUSTOM_BUILD=/opt/mimo26/attn-build \
    ATTN_CUSTOM_SRC=/opt/mimo26/kernels/attention/prefill_attn.cu \
    ATTN_CUSTOM_GENCODE="-gencode=arch=compute_121a,code=sm_121a" \
    TORCH_CUDA_ARCH_LIST=12.1a
RUN python3 /opt/mimo26/build-kernel.py 2>&1 | tail -3 && ls -la /opt/mimo26/attn-build/*.so

# audio + video decode (the image lacks soundfile / PyAV); pins keep torch/numpy/triton/transformers as shipped
RUN pip freeze 2>/dev/null | grep -iE "^(torch|torchaudio|torchvision|numpy|triton|transformers)==" > /tmp/pins.txt \
 && pip install --no-cache-dir --root-user-action=ignore -c /tmp/pins.txt soundfile av \
 && python3 -c "import soundfile, av; print('audio/video libs ok')"

# amd-quark: pulls in vLLM's pure-torch MXFP4 MoE "emulation" backend (--moe-backend emulation), a numerical reference
# for bisecting kernel bugs on a new GPU; pure python, leaves torch/triton/transformers pins untouched (~300 MB).
RUN pip install --no-cache-dir --root-user-action=ignore -c /tmp/pins.txt amd-quark \
 && python3 -c "import quark; print('quark', quark.__version__)"

COPY overlay/ /opt/mimo26/overlay/
