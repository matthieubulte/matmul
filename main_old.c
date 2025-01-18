// gcc -O3 -ffast-math -g -march=native main.c -o main

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define BS 8

static float A[N * N] __attribute__((aligned(128)));
static float B[N * N] __attribute__((aligned(128)));
static float C[N * N] __attribute__((aligned(128)));

static float A_buff[BS * BS] __attribute__((aligned(128)));
static float B_buff[BS * BS] __attribute__((aligned(128)));

float32x4_t C_buff[BS][2];

inline void load_block(float *src, float *dst, int row, int col)
{
    for (int i = 0; i < BS; i++)
    {
        vst1q_f32(&dst[i * BS], vld1q_f32(&src[(row + i) * N + (col)]));
        vst1q_f32(&dst[i * BS + 4], vld1q_f32(&src[(row + i) * N + (col + 4)]));
    }
}

inline void load_c_buff(int i, int j)
{
    for (int ii = 0; ii < BS; ii++)
    {
        C_buff[ii][0] = vld1q_f32(&C[(i + ii) * N + j]);
        C_buff[ii][1] = vld1q_f32(&C[(i + ii) * N + j + 4]);
    }
}

inline void store_c_buff(int i, int j)
{
    for (int ii = 0; ii < BS; ii++)
    {
        vst1q_f32(&C[(i + ii) * N + j], C_buff[ii][0]);
        vst1q_f32(&C[(i + ii) * N + j + 4], C_buff[ii][1]);
    }
}

inline void block_mul()
{
    for (int k = 0; k < BS; k++)
    {
        float32x4_t b0 = vld1q_f32(&B_buff[k * BS]);
        float32x4_t b1 = vld1q_f32(&B_buff[k * BS + 4]);

        // Sadly can't find a way to force the loop to be unrolled
        // it got me 3GFLOPS though. The point here is to maximally use
        // the registers
        float32x4_t a0 = vdupq_n_f32(A_buff[0 * BS + k]);
        float32x4_t a1 = vdupq_n_f32(A_buff[1 * BS + k]);
        float32x4_t a2 = vdupq_n_f32(A_buff[2 * BS + k]);
        float32x4_t a3 = vdupq_n_f32(A_buff[3 * BS + k]);
        float32x4_t a4 = vdupq_n_f32(A_buff[4 * BS + k]);
        float32x4_t a5 = vdupq_n_f32(A_buff[5 * BS + k]);
        float32x4_t a6 = vdupq_n_f32(A_buff[6 * BS + k]);
        float32x4_t a7 = vdupq_n_f32(A_buff[7 * BS + k]);

        C_buff[0][0] = vfmaq_f32(C_buff[0][0], a0, b0);
        C_buff[0][1] = vfmaq_f32(C_buff[0][1], a0, b1);

        C_buff[1][0] = vfmaq_f32(C_buff[1][0], a1, b0);
        C_buff[1][1] = vfmaq_f32(C_buff[1][1], a1, b1);

        C_buff[2][0] = vfmaq_f32(C_buff[2][0], a2, b0);
        C_buff[2][1] = vfmaq_f32(C_buff[2][1], a2, b1);

        C_buff[3][0] = vfmaq_f32(C_buff[3][0], a3, b0);
        C_buff[3][1] = vfmaq_f32(C_buff[3][1], a3, b1);

        C_buff[4][0] = vfmaq_f32(C_buff[4][0], a4, b0);
        C_buff[4][1] = vfmaq_f32(C_buff[4][1], a4, b1);

        C_buff[5][0] = vfmaq_f32(C_buff[5][0], a5, b0);
        C_buff[5][1] = vfmaq_f32(C_buff[5][1], a5, b1);

        C_buff[6][0] = vfmaq_f32(C_buff[6][0], a6, b0);
        C_buff[6][1] = vfmaq_f32(C_buff[6][1], a6, b1);

        C_buff[7][0] = vfmaq_f32(C_buff[7][0], a7, b0);
        C_buff[7][1] = vfmaq_f32(C_buff[7][1], a7, b1);
    }
}

// numpy single threaded: 90 GFLOPS/S
// numpy multi threaded: 200 GFLOPS/S

// 98.16ms         21.8765 GFLOPS/S -- baseline with the right order of loops
// 44.50ms         48.2624 GFLOPS/S -- + vectorization
// 35.47ms         60.5471 GFLOPS/S -- + block multiplication
// 28.38ms         75.6662 GFLOPS/S -- + choosing the right lbock size (8)
// 27.21ms         78.9313 GFLOPS/S -- + loop unrolling for register control
void multiply()
{
    for (int i = 0; i < N; i += BS)
    {
        for (int j = 0; j < N; j += BS)
        {
            load_c_buff(i, j);

            for (int k = 0; k < N; k += BS)
            {
                load_block(A, A_buff, i, k);
                load_block(B, B_buff, j, k);

                block_mul();
            }

            store_c_buff(i, j);
        }
    }
}

void benchmark()
{
    clock_t start = clock();
    multiply();
    clock_t end = clock();

    float duration_ms = ((float)(end - start) * 1000.0) / CLOCKS_PER_SEC;
    float flops = 2.0 * N * N * N;
    printf("%.2fms \t%.4f GFLOPS/S\n", duration_ms, flops / duration_ms / 1e6);
}

int main()
{
    srand(time(NULL));

    while (1)
    {
        for (int i = 0; i < N; i++)
        {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
            C[i] = 0.0;
        }
        benchmark();
    }
    return 0;
}