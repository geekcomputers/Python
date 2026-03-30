#ifndef CUDA_OPS_H
#define CUDA_OPS_H

extern "C" {

void cuda_vector_add(const float* a, const float* b, float* c, int n);
void cuda_vector_sub(const float* a, const float* b, float* c, int n);
void cuda_vector_mul(const float* a, const float* b, float* c, int n);
void cuda_scalar_mul(const float* a, float scalar, float* c, int n);
void cuda_vector_div(const float* a, const float* b, float* c, int n);

void cuda_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K);
void cuda_matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K);
void cuda_matmul_transpose(const float* A, const float* B, float* C, int M, int N, int K, bool transposeA, bool transposeB);
void cuda_batched_matmul(const float* A, const float* B, float* C, int batch_size, int M, int N, int K);
void cuda_gemm(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta);
void cuda_transpose(const float* input, float* output, int rows, int cols);

void cuda_relu_forward(const float* input, float* output, int n);
void cuda_relu_backward(const float* grad_output, const float* input, float* grad_input, int n);
void cuda_sigmoid_forward(const float* input, float* output, int n);
void cuda_sigmoid_backward(const float* grad_output, const float* output, float* grad_input, int n);
void cuda_tanh_forward(const float* input, float* output, int n);
void cuda_tanh_backward(const float* grad_output, const float* output, float* grad_input, int n);
void cuda_gelu_forward(const float* input, float* output, int n);
void cuda_gelu_backward(const float* grad_output, const float* input, float* grad_input, int n);
void cuda_softmax_forward(const float* input, float* output, int batch_size, int dim);

void cuda_sgd_update(float* params, const float* grads, float lr, float momentum, float* velocity, float weight_decay, int n);
void cuda_adam_update(float* params, const float* grads, float* m, float* v, float lr, float beta1, float beta2, float epsilon, float weight_decay, int step, int n);
void cuda_adamw_update(float* params, const float* grads, float* m, float* v, float lr, float beta1, float beta2, float epsilon, float weight_decay, int step, int n);

void cuda_batch_norm_forward(const float* input, const float* gamma, const float* beta, const float* running_mean, const float* running_var, float* output, int batch_size, int channels, int spatial_size, float epsilon);
void cuda_layer_norm_forward(const float* input, const float* gamma, const float* beta, float* output, int batch_size, int feature_size, float epsilon);

void cuda_max_pooling_2d_forward(const float* input, float* output, int* indices, int batch_size, int channels, int height, int width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
void cuda_avg_pooling_2d_forward(const float* input, float* output, int batch_size, int channels, int height, int width, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);

void cuda_gradient_clip_by_norm(float* grads, float max_norm, float current_norm, int n);
void cuda_gradient_clip_by_value(float* grads, float clip_value, int n);

}

#endif
