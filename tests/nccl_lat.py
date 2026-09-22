#!/usr/bin/env python3
"""nccl_lat.py: 2-node NCCL all-reduce latency for decode-sized messages (64 KB, 8 tokens x 4096 bf16) and 1 MB,
1000 iterations, reports mean us per all-reduce. Rank/env as in nccl_check.sh."""
import os, time, torch, torch.distributed as dist
rank = int(os.environ["RANK"])
dist.init_process_group("nccl", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}", rank=rank, world_size=2)
torch.cuda.set_device(0)
for n, label in ((8 * 4096, "64 KB (8 tok x 4096 bf16)"), (48 * 4096, "384 KB (48 tok)"), (512 * 1024, "1 MB")):
    x = torch.ones(n, device="cuda", dtype=torch.bfloat16)
    for _ in range(50): dist.all_reduce(x)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(1000): dist.all_reduce(x)
    torch.cuda.synchronize(); dt = (time.time() - t) / 1000
    if rank == 0: print(f"all_reduce {label:28s}: {dt * 1e6:7.1f} us  (PROTO={os.environ.get('NCCL_PROTO','auto')} ALGO={os.environ.get('NCCL_ALGO','auto')})", flush=True)
dist.barrier(); dist.destroy_process_group()
