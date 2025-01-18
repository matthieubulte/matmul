import time
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMPY_BACKEND"] = "accelerate"

import numpy as np


DIM = 1024

if __name__ == "__main__":
    A = np.random.rand(DIM, DIM).astype(dtype=np.float32)
    B = np.random.rand(DIM, DIM).astype(dtype=np.float32)
    C = np.zeros((DIM, DIM), dtype=np.float32)

    flops_ish = DIM * DIM * 2 * DIM
    while True:
        A = np.random.rand(DIM, DIM).astype(dtype=np.float32)
        B = np.random.rand(DIM, DIM).astype(dtype=np.float32)
        C = np.zeros((DIM, DIM), dtype=np.float32)

        t1 = time.monotonic()
        np.matmul(A, B, out=C)
        dt = time.monotonic() - t1
        print(f"{dt*1000:.2f}us \t {flops_ish/dt * 1e-9:.2f} GFLOP/S")
