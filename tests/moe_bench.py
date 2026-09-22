"""Single-layer MoE kernel bench at MiMo-V2.6-Flash TP2 shapes (E=256, H=4096, I/rank=1024, top-8, MXFP4 e8m0/32).
Backends on identical random MXFP4 weights + routing: marlin W4A16, marlin W4A8-FP8, b12x W4A8-MX.
Reports us/layer, distinct experts, effective weight GB/s (distinct x bytes/expert) and TFLOPS.
Usage: python3 moe_bench.py [backends=marlin,marlin_a8,b12x] [Ms=4,8,16,32,64,128,512,4096] [corr=0.0]"""
import os, sys, time, types
import torch

E, H, I, TOPK = 256, 4096, 1024, 8
dev = "cuda"
torch.manual_seed(0)
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
BACKENDS = args.get("backends", "marlin,marlin_a8,b12x").split(",")
MS = [int(x) for x in args.get("Ms", "4,8,16,32,64,128,512,4096").split(",")]
CORR = float(args.get("corr", "0.0"))  # fraction of each token's experts copied from the previous token (spec tokens share experts)
BYTES_PER_EXPERT = (2 * I * H + H * I) // 2 + (2 * I * H + H * I) // 32  # fp4 payload + e8m0 scales


def rand_fp4(shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev)


def rand_scales(shape):  # e8m0 exponents around 2^-6..2^-3 (bias 127)
    return torch.randint(121, 125, shape, dtype=torch.uint8, device=dev)


from vllm.config import VllmConfig, set_current_vllm_config
_VCFG = set_current_vllm_config(VllmConfig()); _VCFG.__enter__()  # marlin W4A8-FP8 input quant needs a vLLM config context

NL = int(args.get("layers", "8"))  # independent layers run back-to-back per timed step: defeats the 128 MB L2 like real decode
LAYERS = [(rand_fp4((E, 2 * I, H // 2)), rand_scales((E, 2 * I, H // 32)), rand_fp4((E, H, I // 2)), rand_scales((E, H, I // 32)))
          for _ in range(NL)]


def routing(m):
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(m)])
    if CORR > 0:
        keep = int(round(CORR * TOPK))
        for t in range(1, m):
            ids[t, :keep] = ids[t - 1, :keep]
            rest = [e for e in torch.randperm(E).tolist() if e not in ids[t, :keep].tolist()][: TOPK - keep]
            ids[t, keep:] = torch.tensor(rest, device=dev)
    w = torch.softmax(torch.randn(m, TOPK, device=dev), -1)
    return w.float(), ids.to(torch.int32)


GRAPH = args.get("graph", "1") == "1"  # time CUDA-graph replays (how vLLM runs decode); eager adds Python/launch overhead


def timeit(fn, n=20):
    fn(); torch.cuda.synchronize()
    if GRAPH:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2): fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    else:
        run = fn
    run(); torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(n): run()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n * 1e3  # us


def make_marlin(input_dtype, W13, W13S, W2, W2S):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_fp4_layer_for_marlin
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types
    layer = torch.nn.Module()
    layer.moe_config = types.SimpleNamespace(num_local_experts=E, hidden_dim=H, intermediate_size_per_partition=I)
    layer.params_dtype = torch.bfloat16
    layer.w13_weight = torch.nn.Parameter(W13.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(W2.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(W13S.clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(W2S.clone(), requires_grad=False)
    prepare_moe_fp4_layer_for_marlin(layer, input_dtype=input_dtype)
    qid = scalar_types.float4_e2m1f.id

    def run(x, w, ids):
        return fused_marlin_moe(x, layer.w13_weight, layer.w2_weight, None, None, layer.w13_weight_scale,
                                layer.w2_weight_scale, w, ids, quant_type_id=qid, global_num_experts=E,
                                input_dtype=input_dtype)
    return run


PATCH = set(filter(None, args.get("patch", "").split("+")))


def make_b12x(W13, W13S, W2, W2S):
    from vllm.utils.b12x import get_b12x_fused_moe
    fm = get_b12x_fused_moe()
    import b12x.moe.fused_moe._impl as bi
    if "direct0" in PATCH:  # group routes on SM120 too (b12x does this only for SM121 at E=256,n=1024)
        bi._DIRECT_ROUTING_MAX_ROUTED_ROWS = 0
    if "decode256" in PATCH:  # extend the prepared-W4A8 shared-input decode regime to 256 routed rows
        bi._W4A8_DECODE_MAX_ROUTED_ROWS = 256
    w1, w2 = W13.clone(), W2.clone()
    for p in (w1, w2):  # b12x canonical FP4 zero signs (as vLLM does)
        mag = p & 0x77; nz = (mag | (mag >> 1) | (mag >> 2)) & 0x11; p.bitwise_and_(0x77 | (nz << 3))
    ones = torch.ones(E, device=dev, dtype=torch.float32)
    plan = fm.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                           params_dtype=torch.bfloat16, num_experts=E, hidden_size=H, intermediate_size=I, w13_layout="w31")
    prep = fm.prepare_weights(plan=plan, w1_fp4=w1, w1_blockscale=W13S.clone(), w1_global_scale=ones, a1_gscale=ones,
                              w2_fp4=w2, w2_blockscale=W2S.clone(), w2_global_scale=ones, a2_gscale=ones,
                              params_dtype=torch.bfloat16)
    cache = {}

    def run(x, w, ids):
        m = x.shape[0]
        if m not in cache:
            ep = fm.plan(fm.Caps(max_tokens=m, num_topk=TOPK, device=x.device, weight_plan=prep.plan, core_token_counts=(m,),
                                 route_num_experts=0, quant_mode="w4a8_mx", apply_router_weight_on_input=False,
                                 swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None, frozen=True))  # as vLLM plans it
            spec = ep.scratch_specs()[0]
            cache[m] = (ep, torch.empty(spec.shape, dtype=spec.dtype, device=dev), torch.empty(m, H, dtype=torch.bfloat16, device=dev))
        ep, scratch, out = cache[m]
        b = fm.bind(ep, scratch=scratch, a=x, experts=prep, topk_weights=w, topk_ids=ids, output=out,
                    input_scales_static=True, unit_scale_contract=True)
        fm.run(binding=b)
        return out
    return run


makers = {"marlin": lambda L: make_marlin(None, *L), "marlin_a8": lambda L: make_marlin(torch.float8_e4m3fn, *L),
          "b12x": lambda L: make_b12x(*L)}
runs = {}
for b in BACKENDS:
    try:
        fns = [makers[b](L) for L in LAYERS]
        runs[b] = (lambda fns: (lambda x, w, ids: [f(x, w, ids) for f in fns][-1]))(fns)
    except Exception as ex:  # noqa: BLE001
        print(f"{b}: SETUP FAILED {type(ex).__name__}: {str(ex)[:200]}")
del LAYERS
torch.cuda.empty_cache()
print(f"[{NL} layers/step, graph={GRAPH}] MoE layer bench E={E} H={H} I/rank={I} top{TOPK} corr={CORR} patch={sorted(PATCH)} env={ {k:v for k,v in os.environ.items() if k.startswith('B12X_')} }; {BYTES_PER_EXPERT/2**20:.2f} MiB/expert")
outs = {}
for m in MS:
    x = torch.randn(m, H, device=dev, dtype=torch.bfloat16) * 0.5
    w, ids = routing(m)
    distinct = int(torch.unique(ids).numel())
    flops = 2 * m * TOPK * 3 * H * I
    line = f"M={m:5d} distinct={distinct:3d} |"
    ref = None
    for b, fn in runs.items():
        try:
            us = timeit(lambda: fn(x, w, ids), n=50 if m <= 128 else 10) / NL  # per layer
            o = fn(x, w, ids).float()
            if ref is None: ref = o
            rel = ((o - ref).norm() / (ref.norm() + 1e-9)).item()
            gbs = distinct * BYTES_PER_EXPERT / (us * 1e-6) / 1e9
            line += f" {b} {us:8.1f}us {gbs:6.0f}GB/s {flops/(us*1e-6)/1e12:6.1f}TF rel{rel:.3f} |"
        except Exception as ex:  # noqa: BLE001
            line += f" {b} ERR {type(ex).__name__}: {str(ex)[:80]} |"
    print(line, flush=True)
