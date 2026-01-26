#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define BLOCK_SIZE 256
#define TILE_SIZE 32

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vectorSubKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void vectorMulKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void scalarMulKernel(const float* a, float scalar, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * scalar;
    }
}

__global__ void vectorDivKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / (b[idx] + 1e-8f);
    }
}

__global__ void vectorSqrtKernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = sqrtf(a[idx] + 1e-8f);
    }
}

__global__ void vectorSquareKernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * a[idx];
    }
}

__global__ void vectorExpKernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = expf(a[idx]);
    }
}

__global__ void vectorLogKernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = logf(a[idx] + 1e-8f);
    }
}

__global__ void vectorPowKernel(const float* a, float exponent, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = powf(a[idx], exponent);
    }
}

__global__ void vectorMaxKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmaxf(a[idx], b[idx]);
    }
}

__global__ void vectorMinKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fminf(a[idx], b[idx]);
    }
}

__global__ void vectorClampKernel(const float* a, float min_val, float max_val, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fminf(fmaxf(a[idx], min_val), max_val);
    }
}

__global__ void reduceSum(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void reduceMean(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0] / n);
    }
}

__global__ void reduceMax(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : -INFINITY;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax((int*)output, __float_as_int(sdata[0]));
    }
}

__global__ void batchNormForwardKernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int spatial_size,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        float mean = running_mean[c];
        float var = running_var[c];
        float std = sqrtf(var + epsilon);
        float normalized = (input[idx] - mean) / std;
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

__global__ void layerNormForwardKernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int feature_size,
    float epsilon
) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        const float* input_ptr = input + batch_idx * feature_size;
        float* output_ptr = output + batch_idx * feature_size;
        
        float mean = 0.0f;
        for (int i = 0; i < feature_size; i++) {
            mean += input_ptr[i];
        }
        mean /= feature_size;
        
        float variance = 0.0f;
        for (int i = 0; i < feature_size; i++) {
            float diff = input_ptr[i] - mean;
            variance += diff * diff;
        }
        variance /= feature_size;
        
        float std = sqrtf(variance + epsilon);
        
        for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
            float normalized = (input_ptr[i] - mean) / std;
            output_ptr[i] = gamma[i] * normalized + beta[i];
        }
    }
}

__global__ void dropoutForwardKernel(
    const float* input,
    float* output,
    const float* mask,
    float dropout_prob,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float scale = 1.0f / (1.0f - dropout_prob);
        output[idx] = mask[idx] > dropout_prob ? input[idx] * scale : 0.0f;
    }
}

__global__ void convolutionIm2ColKernel(
    const float* input,
    float* col,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int height_col = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int width_col = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;
    
    if (index < channels_col * height_col * width_col) {
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        int c_col = index / (width_col * height_col);
        int c_im = c_col / (kernel_h * kernel_w);
        int kh = (c_col / kernel_w) % kernel_h;
        int kw = c_col % kernel_w;
        
        int h_in = h_out * stride_h - pad_h + kh * dilation_h;
        int w_in = w_out * stride_w - pad_w + kw * dilation_w;
        
        col[index] = (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) ?
                     input[c_im * (height * width) + h_in * width + w_in] : 0.0f;
    }
}

__global__ void maxPooling2DForwardKernel(
    const float* input,
    float* output,
    int* indices,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int total_outputs = batch_size * channels * out_height * out_width;
    
    if (idx < total_outputs) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int n = idx / (out_width * out_height * channels);
        
        int h_start = h_out * stride_h - pad_h;
        int w_start = w_out * stride_w - pad_w;
        
        float max_val = -INFINITY;
        int max_idx = 0;
        
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h = h_start + kh;
                int w = w_start + kw;
                
                if (h >= 0 && h < height && w >= 0 && w < width) {
                    int input_idx = ((n * channels + c) * height + h) * width + w;
                    if (input[input_idx] > max_val) {
                        max_val = input[input_idx];
                        max_idx = input_idx;
                    }
                }
            }
        }
        
        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

__global__ void avgPooling2DForwardKernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int total_outputs = batch_size * channels * out_height * out_width;
    
    if (idx < total_outputs) {
        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int n = idx / (out_width * out_height * channels);
        
        int h_start = h_out * stride_h - pad_h;
        int w_start = w_out * stride_w - pad_w;
        
        float sum = 0.0f;
        int count = 0;
        
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h = h_start + kh;
                int w = w_start + kw;
                
                if (h >= 0 && h < height && w >= 0 && w < width) {
                    int input_idx = ((n * channels + c) * height + h) * width + w;
                    sum += input[input_idx];
                    count++;
                }
            }
        }
        
        output[idx] = sum / count;
    }
}

extern "C" {

void cuda_vector_add(const float* a, const float* b, float* c, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAddKernel<<<blocks, BLOCK_SIZE>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_vector_sub(const float* a, const float* b, float* c, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorSubKernel<<<blocks, BLOCK_SIZE>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_vector_mul(const float* a, const float* b, float* c, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorMulKernel<<<blocks, BLOCK_SIZE>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_scalar_mul(const float* a, float scalar, float* c, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scalarMulKernel<<<blocks, BLOCK_SIZE>>>(a, scalar, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_vector_div(const float* a, const float* b, float* c, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorDivKernel<<<blocks, BLOCK_SIZE>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_batch_norm_forward(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int spatial_size,
    float epsilon
) {
    int total_size = batch_size * channels * spatial_size;
    int blocks = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batchNormForwardKernel<<<blocks, BLOCK_SIZE>>>(
        input, gamma, beta, running_mean, running_var, output,
        batch_size, channels, spatial_size, epsilon
    );
    CUDA_CHECK(cudaGetLastError());
}

void cuda_layer_norm_forward(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int feature_size,
    float epsilon
) {
    layerNormForwardKernel<<<batch_size, BLOCK_SIZE>>>(
        input, gamma, beta, output, batch_size, feature_size, epsilon
    );
    CUDA_CHECK(cudaGetLastError());
}

void cuda_dropout_forward(
    const float* input,
    float* output,
    const float* mask,
    float dropout_prob,
    int n
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dropoutForwardKernel<<<blocks, BLOCK_SIZE>>>(input, output, mask, dropout_prob, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_max_pooling_2d_forward(
    const float* input,
    float* output,
    int* indices,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int total_outputs = batch_size * channels * out_height * out_width;
    int blocks = (total_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    maxPooling2DForwardKernel<<<blocks, BLOCK_SIZE>>>(
        input, output, indices, batch_size, channels, height, width,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
    );
    CUDA_CHECK(cudaGetLastError());
}

void cuda_avg_pooling_2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int total_outputs = batch_size * channels * out_height * out_width;
    int blocks = (total_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    avgPooling2DForwardKernel<<<blocks, BLOCK_SIZE>>>(
        input, output, batch_size, channels, height, width,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
    );
    CUDA_CHECK(cudaGetLastError());
}

}
