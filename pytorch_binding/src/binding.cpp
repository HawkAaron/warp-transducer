#include <iostream>
#include <vector>

#include <numeric>

#include <torch/extension.h>
#include "rnnt.h"

#ifdef WARPRNNT_ENABLE_GPU
    #include "THC.h"
    #include "THCTensor.h"
    #include "detail/reduce.h"
    extern THCState* state;
#else
    #include "TH.h"
#endif

int cpu_rnnt(torch::Tensor trans_acts,
            torch::Tensor pred_acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor trans_grad,
            torch::Tensor pred_grad,
            int blank_label,
            int num_threads) {

    float *trans_acts_ptr = (float*) trans_acts.data_ptr();
    float *pred_acts_ptr = (float*) pred_acts.data_ptr();
    float *trans_grad_ptr = NULL; // this will trigger the score forward code path
    float *pred_grad_ptr = NULL;

    if (trans_grad.storage() && pred_grad.storage()) {
        trans_grad_ptr = (float*) trans_grad.data_ptr();
        pred_grad_ptr = (float*) pred_grad.data_ptr();
    }

    int *input_lengths_ptr = (int*) input_lengths.data_ptr();
    int *labels_ptr = (int*) labels.data_ptr();
    int *label_lengths_ptr = (int*) label_lengths.data_ptr();
    float *costs_ptr = (float*) costs.data_ptr();

    int maxT = trans_acts.size(1);
    int maxU = pred_acts.size(1);
    int minibatch_size = trans_acts.size(0);
    int alphabet_size = trans_acts.size(2);

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.loc = RNNT_CPU;
    options.num_threads = num_threads;
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes;
    get_workspace_size(maxT, maxU, minibatch_size,
                       false, &cpu_size_bytes);

    void* cpu_workspace = malloc(cpu_size_bytes);
    compute_rnnt_loss(trans_acts_ptr, pred_acts_ptr,
                     trans_grad_ptr, pred_grad_ptr,
                     labels_ptr, label_lengths_ptr,
                     input_lengths_ptr, alphabet_size,
                     minibatch_size, costs_ptr,
                     cpu_workspace, options);

    free(cpu_workspace);
    return 1;
}
#ifdef WARPRNNT_ENABLE_GPU
int gpu_rnnt(torch::Tensor trans_acts,
            torch::Tensor pred_acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor trans_grad,
            torch::Tensor pred_grad,
            int blank_label,
            int num_threads=0) { // not used in GPU version

    float *trans_acts_ptr = (float*) trans_acts.data_ptr();
    float *pred_acts_ptr = (float*) pred_acts.data_ptr();
    float *trans_grad_ptr = NULL; // this will trigger the score forward code path
    float *pred_grad_ptr = NULL;

    if (trans_grad.storage() && pred_grad.storage()) {
        trans_grad_ptr = (float*) trans_grad.data_ptr();
        pred_grad_ptr = (float*) pred_grad.data_ptr();
    }

    int *input_lengths_ptr = (int*) input_lengths.data_ptr();
    int *labels_ptr = (int*) labels.data_ptr();
    int *label_lengths_ptr = (int*) label_lengths.data_ptr();
    float *costs_ptr = (float*) costs.data_ptr();

    int maxT = trans_acts.size(1);
    int maxU = pred_acts.size(1);
    int minibatch_size = trans_acts.size(0);
    int alphabet_size = trans_acts.size(2);

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.loc = RNNT_GPU;
    options.stream = at::cuda::getCurrentCUDAStream();

    size_t gpu_size_bytes;
    get_workspace_size(maxT, maxU, minibatch_size,
                       true, &gpu_size_bytes);

    cudaSetDevice(trans_acts.get_device());

    void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);

    compute_rnnt_loss(trans_acts_ptr, pred_acts_ptr,
                     trans_grad_ptr, pred_grad_ptr,
                     labels_ptr, label_lengths_ptr,
                     input_lengths_ptr, alphabet_size,
                     minibatch_size, costs_ptr,
                     gpu_workspace, options);

    THCudaFree(state, gpu_workspace);
    return 1;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_rnnt", &cpu_rnnt, "RNNT CPU version");
#ifdef WARPRNNT_ENABLE_GPU
    m.def("gpu_rnnt", &gpu_rnnt, "RNNT GPU version");
#endif
}
