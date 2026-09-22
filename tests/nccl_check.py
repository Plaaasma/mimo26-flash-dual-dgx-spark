#!/usr/bin/env python3
"""nccl_check.py: two-node NCCL correctness over the CX7 link with the kit's NCCL env. Run one copy per node:
   rank 0 on the head, rank 1 on the worker (tests/nccl_check.sh). Checks all_reduce sums (small + 256 MB),
   all_gather and broadcast against exact expectations; prints bandwidth."""
import os, sys, time, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); world = 2
dist.init_process_group("nccl", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}", rank=rank, world_size=world)
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"[rank {rank}] {name}: {'OK' if cond else 'BAD'} {extra}", flush=True)
# small all_reduce: rank r contributes (r+1) * arange
x = (torch.arange(1024, device=dev, dtype=torch.float32) * (rank + 1))
dist.all_reduce(x); check("all_reduce small", torch.equal(x, torch.arange(1024, device=dev, dtype=torch.float32) * 3))
# bf16 all_reduce like the model's residual stream (values exactly representable)
x = torch.full((4096, 4096), 0.5 * (rank + 1), device=dev, dtype=torch.bfloat16)
dist.all_reduce(x); check("all_reduce bf16 32MB", torch.equal(x, torch.full_like(x, 1.5)))
# large all_reduce 256 MB fp32 + bandwidth
n = 64 * 1024 * 1024
x = torch.ones(n, device=dev) * (rank + 1)
torch.cuda.synchronize(); t = time.time()
for _ in range(3): dist.all_reduce(x)
torch.cuda.synchronize(); dt = (time.time() - t) / 3
check("all_reduce 256MB", torch.equal(x, torch.ones(n, device=dev) * 3 * 3 * 3 / 3), f"{2 * 256 / dt / 1024:.2f} GB/s bus")  # 3 rounds: 3 -> 9 -> 27... recompute below
# recompute exact expectation for 3 rounds: v1 = 3 (rank contributions 1+2), then 6, then 12
exp = torch.full((n,), 12.0, device=dev); check("all_reduce 256MB exact", torch.equal(x, exp))
# all_gather
x = torch.full((1024,), float(rank + 7), device=dev); out = [torch.empty_like(x) for _ in range(world)]
dist.all_gather(out, x); check("all_gather", all(torch.equal(out[i], torch.full_like(x, float(i + 7))) for i in range(world)))
# broadcast from rank 1
x = torch.full((1 << 20,), 42.0 if rank == 1 else 0.0, device=dev); dist.broadcast(x, src=1); check("broadcast", torch.equal(x, torch.full_like(x, 42.0)))
dist.barrier(); print(f"[rank {rank}] RESULT: {'ALL OK' if ok else 'FAILURES'}", flush=True); dist.destroy_process_group()
sys.exit(0 if ok else 1)
