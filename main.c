// gcc -O3 -ffast-math -g -march=native main.c -o main
#include <math.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <assert.h>

// #define BENCH_BLAS

// This seemed to be the best size, and then I optimized with this assumption since
// it made it possible to use 16 registers for the C submatrix buffer
#define BS 8

#define LOAD_FROM_BUFF1(buff, i, j) vdupq_n_f32(buff[(i) * BS + (j)])
#define LOAD_FROM_BUFF4(buff, i, j) vld1q_f32(&buff[(i) * BS + (j)])

#define ADD_MUL(a, b, c) a = vfmaq_f32(a, b, c)

#define N 1024

// Load a block of A into the A buffer
inline void load_a_buff(float *A_buff, float *A, int row, int col)
{
    for (int i = 0; i < BS; i++)
    {
        vst1q_f32(&A_buff[i * BS], vld1q_f32(&A[(row + i) * N + (col)]));
        vst1q_f32(&A_buff[i * BS + 4], vld1q_f32(&A[(row + i) * N + (col + 4)]));
    }
}

// Load a block of B into the B buffer
inline void load_b_buff(float *B_buff, float *B, int row, int col)
{
    for (int i = 0; i < BS; i++)
    {
        vst1q_f32(&B_buff[i * BS], vld1q_f32(&B[(col + i) * N + row]));
        vst1q_f32(&B_buff[i * BS + 4], vld1q_f32(&B[(col + i) * N + row + 4]));
    }
}

// Reset the simd C registers
inline void zero_c_buff(float32x4_t *C_buff)
{
    for (int ii = 0; ii < BS; ii++)
    {
        C_buff[ii * BS] = vdupq_n_f32(0.0f);
        C_buff[ii * BS + 1] = vdupq_n_f32(0.0f);
    }
}

// Store the simd registers back into the C matrix
inline void store_c_buff(float32x4_t *C_buff, float *C, int i, int j)
{
    for (int ii = 0; ii < BS; ii++)
    {
        vst1q_f32(&C[(i + ii) * N + j], C_buff[ii * BS]);
        vst1q_f32(&C[(i + ii) * N + j + 4], C_buff[ii * BS + 1]);
    }
}

inline void block_mul(float *A_buff, float *B_buff, float32x4_t *C_buff)
{
    for (int k = 0; k < BS; k++)
    {
        // We unroll two levels of the loop to maximize the use of registers
        float32x4_t b0 = LOAD_FROM_BUFF4(B_buff, k, 0);
        float32x4_t b1 = LOAD_FROM_BUFF4(B_buff, k, 4);

        // Sadly can't find a way to force the loop to be unrolled
        // it got me 3GFLOPS though. The point here is to maximally use
        // the registers
        float32x4_t a0 = LOAD_FROM_BUFF1(A_buff, 0, k);
        float32x4_t a1 = LOAD_FROM_BUFF1(A_buff, 1, k);
        float32x4_t a2 = LOAD_FROM_BUFF1(A_buff, 2, k);
        float32x4_t a3 = LOAD_FROM_BUFF1(A_buff, 3, k);
        float32x4_t a4 = LOAD_FROM_BUFF1(A_buff, 4, k);
        float32x4_t a5 = LOAD_FROM_BUFF1(A_buff, 5, k);
        float32x4_t a6 = LOAD_FROM_BUFF1(A_buff, 6, k);
        float32x4_t a7 = LOAD_FROM_BUFF1(A_buff, 7, k);

        // TODO: this assumes that C_buff has (flattened) layout (8, 2) (8*8 floats)
        // should be agnostic to BS and be (BS, BS/4)
        ADD_MUL(C_buff[0], a0, b0);
        ADD_MUL(C_buff[1], a0, b1);

        ADD_MUL(C_buff[BS], a1, b0);
        ADD_MUL(C_buff[BS + 1], a1, b1);

        ADD_MUL(C_buff[2 * BS], a2, b0);
        ADD_MUL(C_buff[2 * BS + 1], a2, b1);

        ADD_MUL(C_buff[3 * BS], a3, b0);
        ADD_MUL(C_buff[3 * BS + 1], a3, b1);

        ADD_MUL(C_buff[4 * BS], a4, b0);
        ADD_MUL(C_buff[4 * BS + 1], a4, b1);

        ADD_MUL(C_buff[5 * BS], a5, b0);
        ADD_MUL(C_buff[5 * BS + 1], a5, b1);

        ADD_MUL(C_buff[6 * BS], a6, b0);
        ADD_MUL(C_buff[6 * BS + 1], a6, b1);

        ADD_MUL(C_buff[7 * BS], a7, b0);
        ADD_MUL(C_buff[7 * BS + 1], a7, b1);
    }
}

// numpy single threaded: 90 GFLOPS/S
// numpy multi threaded: 200 GFLOPS/S
// C openblas single threaded: 35 GFLOPS/S (weird ?)

// 98.16ms         21.8765 GFLOPS/S -- baseline with the right order of loops
// 44.50ms         48.2624 GFLOPS/S -- + vectorization
// 35.47ms         60.5471 GFLOPS/S -- + block multiplication
// 28.38ms         75.6662 GFLOPS/S -- + choosing the right block size (8)
// 27.21ms         78.9313 GFLOPS/S -- + loop unrolling for register control
// 31.08ms         69.0954 GFLOPS/S -- fix mistake in loading B
void multiply(float *A, float *B, float *C)
{
    // The submatrix buffers for A and B are just smaller 8x8 matrices
    static float A_buff[BS * BS] __attribute__((aligned(128)));
    static float B_buff[BS * BS] __attribute__((aligned(128)));

    // To optimize register usage, use 16 (out of 32) simd registers
    // for the C submatrix buffer. 16 register of 4 floats each = 8x8 matrix
    static float32x4_t C_buff[BS * BS / 4];

    // Simple algorighm that uses blocks of size BS
    // and tries to keep the same data as long as possible in registers
    for (int i = 0; i < N; i += BS)
    {
        for (int j = 0; j < N; j += BS)
        {
            zero_c_buff(C_buff);

            for (int k = 0; k < N; k += BS)
            {
                load_a_buff(A_buff, A, i, k);
                load_b_buff(B_buff, B, j, k);

                block_mul(A_buff, B_buff, C_buff);
            }

            store_c_buff(C_buff, C, i, j);
        }
    }
}

void benchmark(float *A, float *B, float *C);
int compare_matrices(const float *A, const float *B, float tol);
void fill_rand(float *A, int n);
void assert_correct(float *A, float *B, float *C, float *C1);
void assert_correct_meh(float *A, float *B, float *C, float *A1, float *B1, float *C1);

int main()
{
    static float A[N * N] __attribute__((aligned(128)));
    static float B[N * N] __attribute__((aligned(128)));
    static float C[N * N] __attribute__((aligned(128)));
    static float C1[N * N] __attribute__((aligned(128)));

    static float A1[N * N] __attribute__((aligned(128)));
    static float B1[N * N] __attribute__((aligned(128)));

    srand(time(NULL));

    assert_correct(A, B, C, C1);

    printf("\t Time (ours) \t|\t Perf (ours) \t|\n");
    printf("------------------------+-----------------------+\n");

    while (1)
    {
        fill_rand(A, N * N);
        fill_rand(B, N * N);
        fill_rand(A1, N * N);
        fill_rand(B1, N * N);

        benchmark(A, B, C);

        // calling this in-between will drop the perf by 10x
        // probably because of issues related to
        // instruction cache + cpu opts (predictions / prefecthing / ...)

        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        //             N, N, N, 1.0, A, N, B, N, 0.0, C, N);

        // This thesis is semi-verified by the fact that calling benchmark(...) again
        // with other matrices doesn't change the performance, although we're changing
        // what's in the cache
        // benchmark(A1, B1, C1);

        // assert_correct_meh(A, B, C, A1, B1, C1); // This is to make sure that A1,B1,C1 aren't optimized away
    }
    return 0;
}

// ############################################################
// #######################################  The boring stuff
// ############################################################

void assert_correct_meh(float *A, float *B, float *C, float *A1, float *B1, float *C1)
{

    for (int i = 0; i < N * N; i++)
    {
        if (fabsf(A[i] - A1[i]) + fabsf(B[i] - B1[i]) + fabsf(C[i] - C1[i]) + 1 < 1e-15)
        {
            printf("------------------------+-----------------------+\n");
            return;
        }
    }
}

void benchmark(float *A, float *B, float *C)
{
    clock_t start, end;
    float duration_ms;
    float flops = 2.0 * N * N * N;

    start = clock();

#ifdef BENCH_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C, N);
#else
    multiply(A, B, C);
#endif

    end = clock();
    duration_ms = ((float)(end - start) * 1000.0) / CLOCKS_PER_SEC;

    printf("\t %.2fms \t|\t %.1f GFLOPS/S \n", duration_ms, flops / duration_ms / 1e6);
}

int compare_matrices(const float *A, const float *B, float tol)
{
    for (int i = 0; i < N * N; i++)
    {
        if (fabsf(A[i] - B[i]) > tol)
        {
            return 0;
        }
    }
    return 1;
}

void fill_rand(float *A, int n)
{
    for (int i = 0; i < n; i++)
    {
        A[i] = rand() % 10;
    }
}

void assert_correct(float *A, float *B, float *C, float *C1)
{
    // Check correctness
    fill_rand(A, N * N);
    fill_rand(B, N * N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C1, N);
    benchmark(A, B, C);
    if (!compare_matrices(C, C1, 1e-5f))
    {
        printf("Results don't match - there might be an error in the implementation\n");
        assert(0);
    }
}
