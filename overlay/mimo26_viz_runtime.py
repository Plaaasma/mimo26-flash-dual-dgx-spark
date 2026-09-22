# SPDX-License-Identifier: MIT
"""Live activation telemetry for the Spark dashboard (MiMo-V2.6-Flash on vLLM, TP=2).

Recorded on TP rank 0 only, into preallocated device buffers (device-side writes only, so the hooks are safe inside
CUDA graph captures and replays):

  routing : routed expert ids of each token, per MoE layer (1..47)              [48, MAX_T, 8]  int32
  heads   : per-token, per-head L2 norm of the attention output, every layer   [48, MAX_T, 32] f32
            (rank 0 holds heads 0-31 of 64, for global and sliding-window layers alike)
  h3      : final hidden state at the sampled positions, fixed random 3-D projection   [MAX_T, 3] f32
  meta    : [forward counter, tokens in the forward]

Transfer: the model runner's per-step hook (set_batch: eager, main thread, never inside a capture) queues an async
device->pinned copy of the previous step's buffers at <= MIMO26_VIZ_HZ and hands the last completed copy to the
publisher thread as numpy arrays. The publisher makes no CUDA calls at all: it folds the arrays and sends UDP JSON
datagrams to the collector (MIMO26_VIZ_UDP):

  act        routing histogram [48, 256] u16, per-layer head means [48, 32] u8 (scaled by the layer's max),
             per-layer attention-output magnitude "ribbon" [48]
  act3d      per-token routing [47, T3, 8] u8 (layers 1..47), h3 [n, 3] f16, request spans, accepted counts
  act3h      per-token head norms [48, TA, 32] u8 (scaled by each layer's max)
  viz_status heartbeat every 2 s (armed / dead + reason, layers seen, errors, architecture)

Privacy: no token ids or text ever leave the engine; the dashboard gets per-request token COUNTS only (request ids
are cut to their last 8 characters). Gate: MIMO26_VIZ=1 (patch_viz_hooks.py installs nothing otherwise); any hook
error disarms the telemetry for the rest of the run instead of touching serving.
"""
import base64, json, os, queue, re, socket, threading, time
import numpy as np
import torch

_ON = os.environ.get("MIMO26_VIZ", "0") == "1"
_UDP = os.environ.get("MIMO26_VIZ_UDP", "127.0.0.1:9103")
_HZ = float(os.environ.get("MIMO26_VIZ_HZ", "10"))
MAX_T = int(os.environ.get("MIMO26_VIZ_MAX_T", "128"))      # tokens per forward that are recorded
N_LAYERS, N_EXPERTS, TOPK, HEADS = 48, 256, 8, 32
FULL_LAYERS = (0, 5, 11, 17, 23, 29, 35, 41, 47)            # global attention; the other 39 are sliding-window (128)
MAX_T3, MAX_TA = 32, 16                                      # per-token 3-D frames: routing/h3 tokens, head-volume tokens
SPEC_BLOCK = 8                                               # 1 + DFlash draft tokens: a request with <= this many tokens is decoding
ARCH = {"model": "mimo-v2.6-flash", "layers": N_LAYERS, "full": list(FULL_LAYERS), "moe_first": 1, "experts": N_EXPERTS,
        "topk": TOPK, "heads_rank": HEADS, "heads_total": 64, "swa_window": 128, "drafter": "dflash", "draft_layers": 5}
_LAYER_RX = re.compile(r"layers\.(\d+)\.")


def enabled() -> bool:
    return _ON


class _State:
    def __init__(self):
        self.ready = False
        self.off = False             # not TP rank 0: every hook returns at once
        self.dead = False
        self.reason = ""
        self.rank = -1
        self.lock = threading.Lock()
        self.moe_seen: set = set()
        self.attn_seen: set = set()
        self.batch: list = []        # [(req id tail, first token, token count)] of the step whose data is on the device
        self.sampled: list = []      # [(req id tail, tokens accepted)]: counts only
        self.h3_rows = 0
        self.steps = 0
        self.pending = None
        self.last_copy = 0.0
        self.q: "queue.Queue" = queue.Queue(maxsize=2)
        self.errors = 0
        self.last_error = ""
        self.frames = 0
        self.proj = None


S = _State()


def _disarm(where: str, exc: BaseException) -> None:
    if not S.dead:
        S.dead = True
        S.reason = f"{where}: {type(exc).__name__}: {str(exc)[:160]}"
        try:
            from vllm.logger import init_logger
            init_logger("vllm.mimo26_viz").warning("viz telemetry disabled after %s failed: %s", where, exc)
        except Exception:
            pass


def _idx(name) -> int:
    m = _LAYER_RX.search(str(name))
    return int(m.group(1)) if m else -1


def init(device) -> None:
    """Allocate the buffers (eager only: called after model load, else lazily from the first eager hook)."""
    if not _ON or S.ready or S.off or S.dead:
        return
    with S.lock:
        if S.ready or S.off:
            return
        try:
            try:
                from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
                S.rank = int(get_tensor_model_parallel_rank())
            except Exception:
                S.rank = 0
            if S.rank != 0:
                S.off = True
                return
            device = torch.device(device)
            with torch.inference_mode(False):
                dz = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
                S.d_routing = torch.full((N_LAYERS, MAX_T, TOPK), -1, dtype=torch.int32, device=device)
                S.d_heads = dz(N_LAYERS, MAX_T, HEADS)
                S.d_h3 = dz(MAX_T, 3)
                S.d_meta = dz(2, dt=torch.int32)
                S.d_one = torch.ones(1, dtype=torch.int32, device=device)
                pin = lambda t: torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                S.h_routing, S.h_heads, S.h_h3, S.h_meta = (pin(S.d_routing), pin(S.d_heads), pin(S.d_h3), pin(S.d_meta))
            S.ev = torch.cuda.Event()
            threading.Thread(target=_publisher, daemon=True, name="mimo26-viz").start()
            S.ready = True
        except Exception as e:
            _disarm("init", e)


def _armed(t: torch.Tensor) -> bool:
    if not _ON or S.dead or S.off:
        return False
    if not S.ready:
        if torch.cuda.is_current_stream_capturing():
            return False             # never allocate inside a capture
        init(t.device)
    return S.ready


# ---------------------------------------------------------------- hooks ----
def record_routing(layer_name, topk_ids: torch.Tensor) -> None:
    """MoE runner, after expert selection: topk_ids [T, 8] on device."""
    if not _armed(topk_ids):
        return
    try:
        i = _idx(layer_name)
        if not 0 <= i < N_LAYERS:
            return
        S.moe_seen.add(i)
        t = min(topk_ids.shape[0], MAX_T)
        k = min(topk_ids.shape[-1], TOPK)
        S.d_routing[i, :t, :k].copy_(topk_ids[:t, :k])
    except Exception as e:
        _disarm("record_routing", e)


def record_attn(layer, output: torch.Tensor) -> None:
    """DiffKV attention backend, end of forward: output [T, heads, 128] (this rank's heads)."""
    if not _armed(output):
        return
    try:
        i = _idx(getattr(layer, "layer_name", ""))
        if not 0 <= i < N_LAYERS:
            return                   # drafter layers are numbered 48+ (and use FlashInfer anyway)
        o = output if output.dim() == 3 else output.view(output.shape[0], -1, 128)
        t, h = min(o.shape[0], MAX_T), min(o.shape[1], HEADS)
        if i == 0:
            S.d_meta[0].add_(S.d_one[0])
            S.d_meta[1].fill_(t)
        S.attn_seen.add(i)
        S.d_heads[i, :t, :h].copy_(torch.linalg.vector_norm(o[:t, :h], dim=-1, dtype=torch.float32))
    except Exception as e:
        _disarm("record_attn", e)


def record_hidden3d(hidden: torch.Tensor) -> None:
    """compute_logits (eager): final hidden state at the sampled positions -> fixed random 3-D projection."""
    if not _armed(hidden):
        return
    try:
        if S.proj is None or S.proj.shape[0] != hidden.shape[-1]:
            if torch.cuda.is_current_stream_capturing():
                return
            g = torch.Generator(device="cpu").manual_seed(26)
            p = torch.randn(hidden.shape[-1], 3, generator=g)
            with torch.inference_mode(False):
                S.proj = (p / p.norm(dim=0, keepdim=True)).to(hidden.device)
        t = min(hidden.shape[0], MAX_T)
        S.d_h3[:t].copy_(hidden[:t].float() @ S.proj)
        S.h3_rows = t
    except Exception as e:
        _disarm("record_hidden3d", e)


def record_sampled(req_ids, sampled) -> None:
    """Tokens each request had accepted this step (1 + accepted drafts). Counts only, never the ids."""
    if not _ON or S.dead or S.off:
        return
    try:
        S.sampled = [(str(r)[-8:], len(ids)) for r, ids in zip(req_ids, sampled) if ids]
    except Exception as e:
        _disarm("record_sampled", e)


def set_batch(input_batch) -> None:
    """Model runner, every step before the forward is enqueued (eager, main thread)."""
    if not _ON or S.dead or S.off:
        return
    try:
        ids = list(input_batch.req_ids)[: input_batch.num_reqs]
        qsl = input_batch.query_start_loc_np
        batch = [(str(r)[-8:], int(qsl[i]), int(qsl[i + 1] - qsl[i])) for i, r in enumerate(ids)]
        if S.ready and not torch.cuda.is_current_stream_capturing():
            _pump(S.batch)           # the device buffers hold the previous step, i.e. S.batch
        S.batch = batch
        S.steps += 1
    except Exception as e:
        _disarm("set_batch", e)


def _pump(batch) -> None:
    """Deliver a completed copy to the publisher; queue the next one (rate-limited). Main thread only."""
    p = S.pending
    if p is not None:
        if not S.ev.query():
            return
        T, b, h3n, acc = p
        try:
            S.q.put_nowait({"meta": S.h_meta.numpy().copy(), "routing": S.h_routing.numpy()[:, :T].copy(),
                            "heads": S.h_heads.numpy()[:, :T].copy(), "h3": S.h_h3.numpy()[:h3n].copy(),
                            "T": T, "batch": b, "acc": acc, "ts": time.time()})
        except queue.Full:
            pass
        S.pending = None
    now = time.monotonic()
    if now - S.last_copy < 1.0 / _HZ:
        return
    T = min(sum(n for _, _, n in batch), MAX_T)
    if T <= 0:
        return
    S.last_copy = now
    for h, d in ((S.h_routing, S.d_routing), (S.h_heads, S.d_heads), (S.h_h3, S.d_h3), (S.h_meta, S.d_meta)):
        h.copy_(d, non_blocking=True)    # whole contiguous buffers: a true async DMA, no staging sync
    S.ev.record()
    S.pending = (T, list(batch), min(S.h3_rows, MAX_T), list(S.sampled))


# ------------------------------------------------------------ publisher ----
def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _u8_per_layer(x: np.ndarray) -> np.ndarray:
    """Scale each layer (axis 0) by its own max into 0..255."""
    mx = x.reshape(x.shape[0], -1).max(axis=1)
    mx = np.maximum(mx, 1e-6).reshape((-1,) + (1,) * (x.ndim - 1))
    return np.clip(x / mx * 255.0, 0, 255).astype(np.uint8)


def _spans(batch, limit):
    return [[r, s, min(n, limit - s)] for r, s, n in batch if s < limit]


def _h3_spans(batch, rows):
    """Sampled rows per request: every token of a decode/verify step, one row for a prefill chunk."""
    out, o = [], 0
    for r, _s, n in batch:
        k = n if n <= SPEC_BLOCK else 1
        if o >= rows:
            break
        out.append([r, o, min(k, rows - o)]); o += k
    return out


def fold(item) -> list:
    T = max(1, int(item["T"]))
    r = item["routing"][:, :T]                                   # [48, T, 8]
    hn = item["heads"][:, :T]                                    # [48, T, 32]
    step = int(item["meta"][0])
    valid = (r >= 0) & (r < N_EXPERTS)
    flat = (np.arange(N_LAYERS, dtype=np.int64)[:, None, None] * N_EXPERTS + r)[valid]
    hist = np.bincount(flat, minlength=N_LAYERS * N_EXPERTS).reshape(N_LAYERS, N_EXPERTS).astype(np.uint16)
    ribbon = np.sqrt((hn.astype(np.float64) ** 2).sum(-1)).mean(-1)          # per layer: mean per-token output norm
    ts = item["ts"]
    act = {"kind": "act", "ts": ts, "step": step, "T": T, "n_layers": N_LAYERS, "n_moe": len(S.moe_seen) or 47,
           "experts": N_EXPERTS, "topk": TOPK, "heads": HEADS, "routing": _b64(hist),
           "head_means": _b64(_u8_per_layer(hn.mean(axis=1))), "ribbon": [round(float(x), 3) for x in ribbon]}
    T3, TA = min(T, MAX_T3), min(T, MAX_TA)
    r3 = np.where((r[1:, :T3] >= 0) & (r[1:, :T3] < N_EXPERTS), r[1:, :T3], 255).astype(np.uint8)   # layers 1..47
    h3 = item["h3"][:MAX_T3].astype(np.float16)
    act3d = {"kind": "act3d", "ts": ts, "step": step, "T3": T3, "n_moe": N_LAYERS - 1, "moe_first": 1, "topk": TOPK,
             "r3dtype": "u8", "routing3d": _b64(r3), "h3": _b64(h3), "h3n": int(h3.shape[0]),
             "reqs": _spans(item["batch"], T3), "h3reqs": _h3_spans(item["batch"], int(h3.shape[0])),
             "acc": [[q, int(n)] for q, n in item["acc"]]}
    act3h = {"kind": "act3h", "ts": ts, "step": step, "TA": TA, "n_layers": N_LAYERS, "heads": HEADS,
             "heads3d": _b64(_u8_per_layer(hn[:, :TA])), "reqs": _spans(item["batch"], TA)}
    return [act, act3d, act3h]


def _publisher() -> None:
    host, port = _UDP.rsplit(":", 1)
    addr = (host, int(port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hb = 0.0
    last_step = -1
    while True:
        try:
            item = S.q.get(timeout=1.0)
        except queue.Empty:
            item = None
        now = time.time()
        if now - hb > 2.0:
            hb = now
            try:
                st = {"kind": "viz_status", "ts": now, "ready": S.ready, "dead": S.dead, "reason": S.reason, "rank": S.rank,
                      "n_moe": len(S.moe_seen), "n_attn": len(S.attn_seen), "n_mla": len(S.attn_seen), "last_step": last_step,
                      "frames": S.frames, "errors": S.errors, "last_error": S.last_error, "arch": ARCH}
                sock.sendto(json.dumps(st).encode(), addr)
            except Exception:
                pass
        if item is None:
            continue
        try:
            step = int(item["meta"][0])
            if step == last_step:
                continue             # no forward ran since the last frame
            last_step = step
            for f in fold(item):
                sock.sendto(json.dumps(f, separators=(",", ":")).encode(), addr)
            S.frames += 1
        except Exception as e:
            S.errors += 1
            S.last_error = f"{type(e).__name__}: {str(e)[:140]}"


# ----------------------------------------------------------- self-test ----
def synth_item(step: int, T: int = 16, rng=None) -> dict:
    """A plausible transfer item (numpy only): drifting hot experts, head patterns, a spiral in 3-D."""
    rng = rng or np.random.default_rng(step)
    routing = np.full((N_LAYERS, MAX_T, TOPK), -1, np.int32)
    hot = (np.arange(24) * 37 + step // 20) % N_EXPERTS
    for l in range(1, N_LAYERS):
        p = np.full(N_EXPERTS, 1.0); p[(hot + l * 11) % N_EXPERTS] = 16.0; p /= p.sum()
        for t in range(T):
            routing[l, t] = rng.choice(N_EXPERTS, TOPK, replace=False, p=p)
    heads = np.abs(rng.normal(1.0, 0.25, (N_LAYERS, MAX_T, HEADS))).astype(np.float32)
    for l in FULL_LAYERS:
        heads[l, :, (step // 5 + l) % HEADS] *= 4.0
    heads *= (1.0 + np.arange(N_LAYERS) / 24.0)[:, None, None]
    ang = step * 0.05 + np.arange(MAX_T) * 0.4
    h3 = np.stack([np.cos(ang) * 3, np.sin(ang) * 3, np.sin(2 * ang)], 1).astype(np.float32)
    half = T // 2
    batch = [("req-aa11", 0, half), ("req-bb22", half, T - half)]
    return {"meta": np.array([step, T], np.int32), "routing": routing[:, :T], "heads": heads[:, :T], "h3": h3[:T],
            "T": T, "batch": batch, "acc": [("req-aa11", 5), ("req-bb22", 3)], "ts": time.time()}


def synth(seconds: float = 30.0, hz: float = 10.0) -> None:
    """Send synthetic frames (no GPU, no vLLM) to MIMO26_VIZ_UDP so the dashboard can be exercised."""
    host, port = _UDP.rsplit(":", 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0, step = time.time(), 0
    while time.time() - t0 < seconds:
        step += 1
        for f in fold(synth_item(step)):
            sock.sendto(json.dumps(f, separators=(",", ":")).encode(), (host, int(port)))
        time.sleep(1.0 / hz)


if __name__ == "__main__":
    import sys
    synth(float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)
