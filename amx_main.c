#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>

#define N 1024
#define BS 16

#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
    __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr) + (btf)) : "memory")

#define AMX_OP_LDZ 4
#define AMX_OP_STZ 5

#define AMX_LDX(gpr) AMX_OP_GPR(0, gpr)
#define AMX_LDY(gpr) AMX_OP_GPR(1, gpr)
#define AMX_FMA32(gpr) AMX_OP_GPR(12, gpr)
#define AMX_SET() AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR() AMX_NOP_OP_IMM5(17, 1)

void ldz(float *z)
{
    for (int ridx0 = 0; ridx0 < BS; ridx0++)
    {
        AMX(AMX_OP_LDZ, (int *)(&z[ridx0 * BS]), (ridx0 * 4ull) << 56);
    }
}

void stz(float *z)
{
    for (int ridx0 = 0; ridx0 < BS; ridx0++)
    {
        AMX(AMX_OP_STZ, (int *)(&z[ridx0 * BS]), (ridx0 * 4ull) << 56);
    }
}

void extract_block(const float *matrix, int block_row, int block_col, float *block);
void extract_block_t(const float *matrix, int block_row, int block_col, float *block);
void insert_block(float *matrix, int block_row, int block_col, const float *block);

void transpose(float *A)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            float tmp = A[i * N + j];
            A[i * N + j] = A[j * N + i];
            A[j * N + i] = tmp;
        }
    }
}

void matmul16x16(float *A, float *B, float *C)
{
    for (int j = 0; j < BS; j++)
    {
        for (int i = 0; i < BS; i++)
        {
            C[i * BS + j] = 0;
            for (int k = 0; k < BS; k++)
            {
                C[i * BS + j] += A[i * BS + k] * B[k * BS + j];
            }
        }
    }
}

void matmul(float *A, float *B, float *C)
{
    // assert(n % BS == 0);
    // int num_blocks = n / BS;

    // static float blockA[BS * BS] __attribute__((aligned(128)));
    // static float blockB[BS * BS] __attribute__((aligned(128)));
    // static float blockC[BS * BS] __attribute__((aligned(128)));

    // for (int i = 0; i < num_blocks; i++)
    // {
    //     for (int j = 0; j < num_blocks; j++)
    //     {
    //         for (int k = 0; k < num_blocks; k++)
    //         {
    //             extract_block(A, i, k, blockA);
    //             extract_block(B, k, j, blockB);
    //             matmul16x16(blockA, blockB, blockC);
    //         }
    //         insert_block(C, i, j, blockC);
    //     }
    // }
    transpose(B);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[j * N + k];
            }
            C[i * N + j] = sum;
        }
    }
    transpose(B);
}

void matmul_amx(const float *A, const float *B, float *C)
{
    assert(N % BS == 0);
    int num_blocks = N / BS;

    static float blockAt[BS * BS] __attribute__((aligned(128)));
    static float blockB[BS * BS] __attribute__((aligned(128)));
    static float blockC[BS * BS] __attribute__((aligned(128)));

    for (int i = 0; i < num_blocks; i++)
    {
        for (int j = 0; j < num_blocks; j++)
        {
            memset(blockC, 0, BS * BS * sizeof(float));
            ldz(blockC);

            for (int k = 0; k < num_blocks; k++)
            {
                extract_block_t(A, i, k, blockAt);
                extract_block(B, k, j, blockB);

                for (int k = 0; k < BS; k++)
                {
                    AMX_LDX(&blockB[k * BS]);
                    AMX_LDY(&blockAt[k * BS]);
                    AMX_FMA32(0);
                }
            }
            stz(blockC);
            insert_block(C, i, j, blockC);
        }
    }
}

void cmp(float *A, float *B);
void fill_rand(float *A);

void benchmark(float *A, float *B, float *out_amx, float *out_ref)
{
    clock_t start, end;
    int reps = 1;
    float duration_ms = 0.0;
    float flops = reps * 2.0 * N * N * N;

    for (int i = 0; i < reps; i++)
    {
        fill_rand(A);
        fill_rand(B);
        memset(out_amx, 0, N * N * sizeof(float));
        // memset(out_ref, 0, N * N * sizeof(float));

        start = clock();

        // matmul_amx(A, B, out_amx);
        matmul(A, B, out_amx);

        end = clock();
        // matmul(A, B, out_ref, N);
        // cmp(out_amx, out_ref);
        duration_ms += ((float)(end - start) * 1000.0) / CLOCKS_PER_SEC;
    }

    printf("\t %.2fms \t|\t %.1f GFLOPS/S \n", duration_ms / reps, flops / duration_ms / 1e6);
}

int main()
{
    // AMX_SET();
    srand(time(NULL));

    static float x[N * N] __attribute__((aligned(128)));
    static float y[N * N] __attribute__((aligned(128)));
    static float out_amx[N * N] __attribute__((aligned(128)));
    static float out_ref[N * N] __attribute__((aligned(128)));

    printf("\t Time (ours) \t|\t Perf (ours) \t|\n");
    printf("------------------------+-----------------------+\n");

    for (int i = 0; i < 20; i++)
    {
        benchmark(x, y, out_amx, out_ref);
    }

    // AMX_CLR();
    return 0;
}

void extract_block(const float *matrix, int block_row, int block_col, float *block)
{
    for (int i = 0; i < BS; i++)
    {
        for (int j = 0; j < BS; j++)
        {
            block[i * BS + j] = matrix[(block_row * BS + i) * N + (block_col * BS + j)];
        }
    }
}

void extract_block_t(const float *matrix, int block_row, int block_col, float *block)
{
    for (int i = 0; i < BS; i++)
    {
        for (int j = 0; j < BS; j++)
        {
            block[j * BS + i] = matrix[(block_row * BS + i) * N + (block_col * BS + j)];
        }
    }
}

void insert_block(float *matrix, int block_row, int block_col, const float *block)
{
    for (int i = 0; i < BS; i++)
    {
        for (int j = 0; j < BS; j++)
        {
            matrix[(block_row * BS + i) * N + (block_col * BS + j)] = block[i * BS + j];
        }
    }
}

void cmp(float *A, float *B)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (A[i * N + j] != B[i * N + j])
            {
                printf("Error at %d %d\n", i, j);
                return;
            }
        }
    }
}

void fill_rand(float *A)
{
    for (int i = 0; i < N * N; i++)
    {
        A[i] = rand() % 10;
    }
}
