#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdio.h>

#define TILE_WIDTH 32
#define BLOCK_SIZE 256

__global__ void matmulNaiveKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmulTiledKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < M && t * TILE_WIDTH + tx < K) {
            tileA[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        if (t * TILE_WIDTH + ty < K && col < N) {
            tileB[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void matmulTransposeKernel(const float* A, const float* B, float* C, int M, int N, int K, bool transposeA, bool transposeB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float a_val = transposeA ? A[k * M + row] : A[row * K + k];
            float b_val = transposeB ? B[col * K + k] : B[k * N + col];
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

__global__ void batchedMatmulKernel(const float* A, const float* B, float* C, int batch_size, int M, int N, int K) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && row < M && col < N) {
        const float* A_batch = A + batch_idx * M * K;
        const float* B_batch = B + batch_idx * K * N;
        float* C_batch = C + batch_idx * M * N;
        
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A_batch[row * K + k] * B_batch[k * N + col];
        }
        C_batch[row * N + col] = sum;
    }
}

__global__ void gemmKernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

__global__ void transposeKernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1];
    
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_WIDTH + threadIdx.x;
    y = blockIdx.x * TILE_WIDTH + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void outerProductKernel(const float* a, const float* b, float* c, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        c[row * N + col] = a[row] * b[col];
    }
}

extern "C" {

void cuda_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    matmulNaiveKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

void cuda_matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmulTiledKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

void cuda_matmul_transpose(const float* A, const float* B, float* C, int M, int N, int K, bool transposeA, bool transposeB) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    matmulTransposeKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, transposeA, transposeB);
}

void cuda_batched_matmul(const float* A, const float* B, float* C, int batch_size, int M, int N, int K) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y, batch_size);
    batchedMatmulKernel<<<gridDim, blockDim>>>(A, B, C, batch_size, M, N, K);
}

void cuda_gemm(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    gemmKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}

void cuda_transpose(const float* input, float* output, int rows, int cols) {
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((cols + TILE_WIDTH - 1) / TILE_WIDTH, (rows + TILE_WIDTH - 1) / TILE_WIDTH);
    transposeKernel<<<gridDim, blockDim>>>(input, output, rows, cols);
}

void cuda_outer_product(const float* a, const float* b, float* c, int M, int N) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    outerProductKernel<<<gridDim, blockDim>>>(a, b, c, M, N);
}

}
