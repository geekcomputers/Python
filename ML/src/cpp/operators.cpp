#include <torch/extension.h>
#include <vector>
#include "include/cuda_ops.h"

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    
    cuda_vector_add(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
    
    return c;
}

torch::Tensor vector_mul_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    
    cuda_vector_mul(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
    
    return c;
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b, bool use_tiled) {
    TORCH_CHECK(a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(b.dim() == 2, "Matrix B must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions must match");
    
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    
    auto c = torch::empty({M, N}, a.options());
    
    if (use_tiled) {
        cuda_matmul_tiled(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            M, N, K
        );
    } else {
        cuda_matmul_naive(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            M, N, K
        );
    }
    
    return c;
}

torch::Tensor batched_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() == 3, "Tensor A must be 3D");
    TORCH_CHECK(b.dim() == 3, "Tensor B must be 3D");
    TORCH_CHECK(a.size(0) == b.size(0), "Batch sizes must match");
    TORCH_CHECK(a.size(2) == b.size(1), "Matrix dimensions must match");
    
    int batch_size = a.size(0);
    int M = a.size(1);
    int K = a.size(2);
    int N = b.size(2);
    
    auto c = torch::empty({batch_size, M, N}, a.options());
    
    cuda_batched_matmul(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        batch_size, M, N, K
    );
    
    return c;
}

torch::Tensor relu_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    
    cuda_relu_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}

torch::Tensor relu_backward_cuda(torch::Tensor grad_output, torch::Tensor input) {
    auto grad_input = torch::empty_like(input);
    int n = input.numel();
    
    cuda_relu_backward(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        n
    );
    
    return grad_input;
}

torch::Tensor sigmoid_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    
    cuda_sigmoid_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}

torch::Tensor gelu_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    
    cuda_gelu_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}

torch::Tensor gelu_backward_cuda(torch::Tensor grad_output, torch::Tensor input) {
    auto grad_input = torch::empty_like(input);
    int n = input.numel();
    
    cuda_gelu_backward(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        n
    );
    
    return grad_input;
}

torch::Tensor softmax_forward_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    
    int batch_size = input.size(0);
    int dim = input.size(1);
    
    auto output = torch::empty_like(input);
    
    cuda_softmax_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
    
    return output;
}

torch::Tensor batch_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float epsilon
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int spatial_size = 1;
    for (int i = 2; i < input.dim(); i++) {
        spatial_size *= input.size(i);
    }
    
    auto output = torch::empty_like(input);
    
    cuda_batch_norm_forward(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        epsilon
    );
    
    return output;
}

std::vector<torch::Tensor> max_pool2d_forward_cuda(
    torch::Tensor input,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    auto indices = torch::empty({batch_size, channels, out_height, out_width}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    
    cuda_max_pooling_2d_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        indices.data_ptr<int>(),
        batch_size, channels, height, width,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
    );
    
    return {output, indices};
}

void adam_update_cuda(
    torch::Tensor params,
    torch::Tensor grads,
    torch::Tensor m,
    torch::Tensor v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step
) {
    int n = params.numel();
    
    cuda_adam_update(
        params.data_ptr<float>(),
        grads.data_ptr<float>(),
        m.data_ptr<float>(),
        v.data_ptr<float>(),
        lr, beta1, beta2, epsilon, weight_decay, step, n
    );
}

void adamw_update_cuda(
    torch::Tensor params,
    torch::Tensor grads,
    torch::Tensor m,
    torch::Tensor v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int step
) {
    int n = params.numel();
    
    cuda_adamw_update(
        params.data_ptr<float>(),
        grads.data_ptr<float>(),
        m.data_ptr<float>(),
        v.data_ptr<float>(),
        lr, beta1, beta2, epsilon, weight_decay, step, n
    );
}
