#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void sgdUpdateKernel(
    float* params,
    const float* grads,
    float lr,
    float momentum,
    float* velocity,
    float weight_decay,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = grads[idx];
        if (weight_decay != 0.0f) {
            grad += weight_decay * params[idx];
        }
        
        if (momentum != 0.0f) {
            velocity[idx] = momentum * velocity[idx] + grad;
            params[idx] -= lr * velocity[idx];
        } else {
            params[idx] -= lr * grad;
        }
    }
}

__global__ void adamUpdateKernel(
    float* params,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = grads[idx];
        if (weight_decay != 0.0f) {
            grad += weight_decay * params[idx];
        }
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        params[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

__global__ void adamwUpdateKernel(
    float* params,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = grads[idx];
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        params[idx] -= lr * (m_hat / (sqrtf(v_hat) + epsilon) + weight_decay * params[idx]);
    }
}

__global__ void rmspropUpdateKernel(
    float* params,
    const float* grads,
    float* v,
    float lr,
    float alpha,
    float epsilon,
    float weight_decay,
    float momentum,
    float* buf,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = grads[idx];
        if (weight_decay != 0.0f) {
            grad += weight_decay * params[idx];
        }
        
        v[idx] = alpha * v[idx] + (1.0f - alpha) * grad * grad;
        
        if (momentum > 0.0f) {
            buf[idx] = momentum * buf[idx] + grad / (sqrtf(v[idx]) + epsilon);
            params[idx] -= lr * buf[idx];
        } else {
            params[idx] -= lr * grad / (sqrtf(v[idx]) + epsilon);
        }
    }
}

__global__ void adagradUpdateKernel(
    float* params,
    const float* grads,
    float* sum,
    float lr,
    float epsilon,
    float weight_decay,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = grads[idx];
        if (weight_decay != 0.0f) {
            grad += weight_decay * params[idx];
        }
        
        sum[idx] += grad * grad;
        params[idx] -= lr * grad / (sqrtf(sum[idx]) + epsilon);
    }
}

__global__ void adadeltaUpdateKernel(
    float* params,
    const float* grads,
    float* square_avg,
    float* acc_delta,
    float rho,
    float epsilon,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = grads[idx];
        
        square_avg[idx] = rho * square_avg[idx] + (1.0f - rho) * grad * grad;
        float std = sqrtf(square_avg[idx] + epsilon);
        float delta = sqrtf(acc_delta[idx] + epsilon) / std * grad;
        
        params[idx] -= delta;
        acc_delta[idx] = rho * acc_delta[idx] + (1.0f - rho) * delta * delta;
    }
}

__global__ void lambUpdateKernel(
    float* params,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = grads[idx];
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        float update = m_hat / (sqrtf(v_hat) + epsilon) + weight_decay * params[idx];
        
        float param_norm = sqrtf(params[idx] * params[idx]);
        float update_norm = sqrtf(update * update);
        float trust_ratio = (param_norm > 0.0f && update_norm > 0.0f) ? param_norm / update_norm : 1.0f;
        
        params[idx] -= lr * trust_ratio * update;
    }
}

__global__ void gradientClipByNormKernel(float* grads, float max_norm, float current_norm, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (current_norm > max_norm) {
            grads[idx] *= max_norm / (current_norm + 1e-6f);
        }
    }
}

__global__ void gradientClipByValueKernel(float* grads, float clip_value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grads[idx] = fminf(fmaxf(grads[idx], -clip_value), clip_value);
    }
}

__global__ void computeGradNormKernel(const float* grads, float* partial_norms, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    if (i < n) {
        sum = grads[i] * grads[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_norms[blockIdx.x] = sdata[0];
    }
}

extern "C" {

void cuda_sgd_update(
    float* params,
    const float* grads,
    float lr,
    float momentum,
    float* velocity,
    float weight_decay,
    int n
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sgdUpdateKernel<<<blocks, BLOCK_SIZE>>>(params, grads, lr, momentum, velocity, weight_decay, n);
}

void cuda_adam_update(
    float* params,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step,
    int n
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adamUpdateKernel<<<blocks, BLOCK_SIZE>>>(params, grads, m, v, lr, beta1, beta2, epsilon, weight_decay, step, n);
}

void cuda_adamw_update(
    float* params,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step,
    int n
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    adamwUpdateKernel<<<blocks, BLOCK_SIZE>>>(params, grads, m, v, lr, beta1, beta2, epsilon, weight_decay, step, n);
}

void cuda_rmsprop_update(
    float* params,
    const float* grads,
    float* v,
    float lr,
    float alpha,
    float epsilon,
    float weight_decay,
    float momentum,
    float* buf,
    int n
) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rmspropUpdateKernel<<<blocks, BLOCK_SIZE>>>(params, grads, v, lr, alpha, epsilon, weight_decay, momentum, buf, n);
}

void cuda_gradient_clip_by_norm(float* grads, float max_norm, float current_norm, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gradientClipByNormKernel<<<blocks, BLOCK_SIZE>>>(grads, max_norm, current_norm, n);
}

void cuda_gradient_clip_by_value(float* grads, float clip_value, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gradientClipByValueKernel<<<blocks, BLOCK_SIZE>>>(grads, clip_value, n);
}

}
