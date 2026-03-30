#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define BLOCK_SIZE 256

__device__ float sigmoid_device(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float tanh_device(float x) {
    return tanhf(x);
}

__device__ float relu_device(float x) {
    return fmaxf(0.0f, x);
}

__device__ float leaky_relu_device(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

__device__ float elu_device(float x, float alpha) {
    return x > 0.0f ? x : alpha * (expf(x) - 1.0f);
}

__device__ float selu_device(float x) {
    float alpha = 1.6732632423543772848170429916717f;
    float scale = 1.0507009873554804934193349852946f;
    return scale * (x > 0.0f ? x : alpha * (expf(x) - 1.0f));
}

__device__ float gelu_device(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__device__ float swish_device(float x) {
    return x * sigmoid_device(x);
}

__device__ float mish_device(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void reluForwardKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = relu_device(input[idx]);
    }
}

__global__ void reluBackwardKernel(const float* grad_output, const float* input, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

__global__ void sigmoidForwardKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sigmoid_device(input[idx]);
    }
}

__global__ void sigmoidBackwardKernel(const float* grad_output, const float* output, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = output[idx];
        grad_input[idx] = grad_output[idx] * s * (1.0f - s);
    }
}

__global__ void tanhForwardKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanh_device(input[idx]);
    }
}

__global__ void tanhBackwardKernel(const float* grad_output, const float* output, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = output[idx];
        grad_input[idx] = grad_output[idx] * (1.0f - t * t);
    }
}

__global__ void leakyReluForwardKernel(const float* input, float* output, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = leaky_relu_device(input[idx], alpha);
    }
}

__global__ void leakyReluBackwardKernel(const float* grad_output, const float* input, float* grad_input, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : alpha * grad_output[idx];
    }
}

__global__ void eluForwardKernel(const float* input, float* output, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = elu_device(input[idx], alpha);
    }
}

__global__ void eluBackwardKernel(const float* grad_output, const float* input, float* grad_input, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        grad_input[idx] = x > 0.0f ? grad_output[idx] : grad_output[idx] * alpha * expf(x);
    }
}

__global__ void seluForwardKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = selu_device(input[idx]);
    }
}

__global__ void geluForwardKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gelu_device(input[idx]);
    }
}

__global__ void geluBackwardKernel(const float* grad_output, const float* input, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        float pdf = 0.7978845608028654f * (1.0f + 0.134145f * x * x);
        grad_input[idx] = grad_output[idx] * (cdf + x * pdf * (1.0f - cdf * cdf));
    }
}

__global__ void swishForwardKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = swish_device(input[idx]);
    }
}

__global__ void swishBackwardKernel(const float* grad_output, const float* input, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float s = sigmoid_device(x);
        grad_input[idx] = grad_output[idx] * (s + x * s * (1.0f - s));
    }
}

__global__ void mishForwardKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = mish_device(input[idx]);
    }
}

__global__ void softmaxForwardKernel(const float* input, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        const float* input_ptr = input + batch_idx * dim;
        float* output_ptr = output + batch_idx * dim;
        
        float max_val = -INFINITY;
        for (int i = 0; i < dim; i++) {
            max_val = fmaxf(max_val, input_ptr[i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += expf(input_ptr[i] - max_val);
        }
        
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            output_ptr[i] = expf(input_ptr[i] - max_val) / sum;
        }
    }
}

__global__ void logSoftmaxForwardKernel(const float* input, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        const float* input_ptr = input + batch_idx * dim;
        float* output_ptr = output + batch_idx * dim;
        
        float max_val = -INFINITY;
        for (int i = 0; i < dim; i++) {
            max_val = fmaxf(max_val, input_ptr[i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += expf(input_ptr[i] - max_val);
        }
        float log_sum = logf(sum);
        
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            output_ptr[i] = input_ptr[i] - max_val - log_sum;
        }
    }
}

extern "C" {

void cuda_relu_forward(const float* input, float* output, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reluForwardKernel<<<blocks, BLOCK_SIZE>>>(input, output, n);
}

void cuda_relu_backward(const float* grad_output, const float* input, float* grad_input, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reluBackwardKernel<<<blocks, BLOCK_SIZE>>>(grad_output, input, grad_input, n);
}

void cuda_sigmoid_forward(const float* input, float* output, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoidForwardKernel<<<blocks, BLOCK_SIZE>>>(input, output, n);
}

void cuda_sigmoid_backward(const float* grad_output, const float* output, float* grad_input, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoidBackwardKernel<<<blocks, BLOCK_SIZE>>>(grad_output, output, grad_input, n);
}

void cuda_tanh_forward(const float* input, float* output, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tanhForwardKernel<<<blocks, BLOCK_SIZE>>>(input, output, n);
}

void cuda_tanh_backward(const float* grad_output, const float* output, float* grad_input, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tanhBackwardKernel<<<blocks, BLOCK_SIZE>>>(grad_output, output, grad_input, n);
}

void cuda_leaky_relu_forward(const float* input, float* output, float alpha, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    leakyReluForwardKernel<<<blocks, BLOCK_SIZE>>>(input, output, alpha, n);
}

void cuda_gelu_forward(const float* input, float* output, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    geluForwardKernel<<<blocks, BLOCK_SIZE>>>(input, output, n);
}

void cuda_gelu_backward(const float* grad_output, const float* input, float* grad_input, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    geluBackwardKernel<<<blocks, BLOCK_SIZE>>>(grad_output, input, grad_input, n);
}

void cuda_swish_forward(const float* input, float* output, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    swishForwardKernel<<<blocks, BLOCK_SIZE>>>(input, output, n);
}

void cuda_softmax_forward(const float* input, float* output, int batch_size, int dim) {
    softmaxForwardKernel<<<batch_size, BLOCK_SIZE>>>(input, output, batch_size, dim);
}

void cuda_log_softmax_forward(const float* input, float* output, int batch_size, int dim) {
    logSoftmaxForwardKernel<<<batch_size, BLOCK_SIZE>>>(input, output, batch_size, dim);
}

}
