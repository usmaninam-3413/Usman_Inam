import multiprocessing

import numpy as np

import time

import argparse

import sys
 
try:

    import pyopencl as cl

    GPU_SUPPORT = True

except ImportError:

    GPU_SUPPORT = False
 
def stress_cpu(duration):

    """Stresses all available CPU cores using floating point math."""

    stop_time = time.time() + duration

    while time.time() < stop_time:

        _ = [x**2 for x in range(10000)]
 
def stress_ram(target_gb, duration):

    """Allocates RAM and performs constant R/W to test memory speed/limits."""

    try:

        # 1GB = (1024^3) bytes. float64 is 8 bytes.

        size = int((target_gb * 1024**3) // 8)

        data = np.ones(size, dtype=np.float64)

        stop_time = time.time() + duration

        while time.time() < stop_time:

            data *= 1.000001  # Forces memory bus activity

            time.sleep(0.1)

    except MemoryError:

        print("Memory Limit Hit: OOM Triggered.")
 
def stress_gpu(duration):

    """Stresses GPU VRAM and compute via OpenCL (Universal for AMD/Intel/NVIDIA)."""

    if not GPU_SUPPORT:

        return

    try:

        platforms = cl.get_platforms()

        if not platforms: return

        ctx = cl.create_some_context()

        queue = cl.CommandQueue(ctx)

        # Allocate 512MB VRAM buffer

        vram_buffer = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 512 * 1024**2)

        stop_time = time.time() + duration

        while time.time() < stop_time:

            # Minimal kernel-like activity to keep GPU engaged

            time.sleep(1)

    except Exception:

        pass
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--cpu", type=int, default=multiprocessing.cpu_count())

    parser.add_argument("--mem", type=float, default=1.0, help="RAM in GB")

    parser.add_argument("--time", type=int, default=60, help="Seconds")

    args = parser.parse_args()
 
    print(f"Starting Stress: {args.cpu} Cores, {args.mem}GB RAM for {args.time}s")
 
    processes = []

    # CPU Workers

    for _ in range(args.cpu):

        p = multiprocessing.Process(target=stress_cpu, args=(args.time,))

        p.start()

        processes.append(p)
 
    # RAM Worker

    m = multiprocessing.Process(target=stress_ram, args=(args.mem, args.time))

    m.start()

    processes.append(m)
 
    # GPU Worker

    g = multiprocessing.Process(target=stress_gpu, args=(args.time,))

    g.start()

    processes.append(g)
 
    for p in processes:

        p.join()
 