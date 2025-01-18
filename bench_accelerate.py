import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import time

if __name__ == "__main__":
    DIM = 1024

    accelerate = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/Accelerate.framework/Accelerate"
    )

    a_sgemm = accelerate.cblas_sgemm
    a_sgemm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ndpointer(dtype=np.float32),
        ctypes.c_int,
        ndpointer(dtype=np.float32),
        ctypes.c_int,
        ctypes.c_float,
        ndpointer(dtype=np.float32),
        ctypes.c_int,
    ]

    # Constants
    CblasRowMajor = 101
    CblasColMajor = 102
    CblasNoTrans = 111
    CblasTrans = 112


    def matmul_accelerate(A, B, C):
        a_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            DIM,
            DIM,
            DIM,
            1.0,
            A,
            DIM,
            B,
            DIM,
            0.0,
            C,
            DIM,
        )
        return C


    A = np.random.rand(DIM, DIM).astype(dtype=np.float32)
    B = np.random.rand(DIM, DIM).astype(dtype=np.float32)
    C = np.zeros((DIM, DIM), dtype=np.float32)

    cA = np.ascontiguousarray(A, dtype=np.float32)
    cB = np.ascontiguousarray(B, dtype=np.float32)
    cC = np.zeros((DIM, DIM), dtype=np.float32)

    ######################################################################## EVAL


    flops_ish = DIM * DIM * 2 * DIM
    while True:
        A = np.random.rand(DIM, DIM).astype(dtype=np.float32)
        B = np.random.rand(DIM, DIM).astype(dtype=np.float32)
        C = np.zeros((DIM, DIM), dtype=np.float32)

        cA = np.ascontiguousarray(A, dtype=np.float32)
        cB = np.ascontiguousarray(B, dtype=np.float32)
        cC = np.zeros((DIM, DIM), dtype=np.float32)

        t1 = time.monotonic()
        matmul_accelerate(cA, cB, cC)
        dt = time.monotonic() - t1
        print(f"{dt*1000:.2f}us \t {flops_ish/dt * 1e-9:.2f} GFLOP/S")
