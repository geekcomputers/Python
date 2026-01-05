#include <torch/extension.h>
#include "include/cuda_ops.h"

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor vector_mul_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b, bool use_tiled);
torch::Tensor batched_matmul_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor relu_forward_cuda(torch::Tensor input);
torch::Tensor relu_backward_cuda(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor sigmoid_forward_cuda(torch::Tensor input);
torch::Tensor gelu_forward_cuda(torch::Tensor input);
torch::Tensor gelu_backward_cuda(torch::Tensor grad_output, torch::Tensor input);
torch::Tensor softmax_forward_cuda(torch::Tensor input);
torch::Tensor batch_norm_forward_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor running_mean, torch::Tensor running_var, float epsilon);
std::vector<torch::Tensor> max_pool2d_forward_cuda(torch::Tensor input, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
void adam_update_cuda(torch::Tensor params, torch::Tensor grads, torch::Tensor m, torch::Tensor v, float lr, float beta1, float beta2, float epsilon, float weight_decay, int step);
void adamw_update_cuda(torch::Tensor params, torch::Tensor grads, torch::Tensor m, torch::Tensor v, float lr, float beta1, float beta2, float epsilon, float weight_decay, int step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "Vector addition (CUDA)");
    m.def("vector_mul", &vector_mul_cuda, "Vector multiplication (CUDA)");
    m.def("matmul", &matmul_cuda, "Matrix multiplication (CUDA)", py::arg("a"), py::arg("b"), py::arg("use_tiled") = true);
    m.def("batched_matmul", &batched_matmul_cuda, "Batched matrix multiplication (CUDA)");
    m.def("relu_forward", &relu_forward_cuda, "ReLU forward (CUDA)");
    m.def("relu_backward", &relu_backward_cuda, "ReLU backward (CUDA)");
    m.def("sigmoid_forward", &sigmoid_forward_cuda, "Sigmoid forward (CUDA)");
    m.def("gelu_forward", &gelu_forward_cuda, "GELU forward (CUDA)");
    m.def("gelu_backward", &gelu_backward_cuda, "GELU backward (CUDA)");
    m.def("softmax_forward", &softmax_forward_cuda, "Softmax forward (CUDA)");
    m.def("batch_norm_forward", &batch_norm_forward_cuda, "Batch normalization forward (CUDA)");
    m.def("max_pool2d_forward", &max_pool2d_forward_cuda, "Max pooling 2D forward (CUDA)");
    m.def("adam_update", &adam_update_cuda, "Adam optimizer update (CUDA)");
    m.def("adamw_update", &adamw_update_cuda, "AdamW optimizer update (CUDA)");
}
