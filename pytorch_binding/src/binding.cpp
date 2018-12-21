#include <iostream>
#include <numeric>

#include <torch/extension.h>
#include "rnnt.h"

#ifdef WARPRNNT_ENABLE_GPU
    #include "THC.h"
    extern THCState* state;
#endif

int cpu_rnnt(torch::Tensor acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor grads,
            int blank_label,
            int num_threads) {

    float *acts_ptr = (float*) acts.data_ptr();
    float *grads_ptr = NULL; // this will trigger the score forward code path
    if (grads.storage())
        grads_ptr = (float*) grads.data_ptr();

    int *input_lengths_ptr = (int*) input_lengths.data_ptr();
    int *labels_ptr = (int*) labels.data_ptr();
    int *label_lengths_ptr = (int*) label_lengths.data_ptr();
    float *costs_ptr = (float*) costs.data_ptr();

    int maxT = acts.size(0);
    int maxU = acts.size(1);
    int minibatch_size = acts.size(2);
    int alphabet_size = acts.size(3);

	if (true) {
		minibatch_size = acts.size(0);
		maxT = acts.size(1);
		maxU = acts.size(2);
	}

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.batch_first = true;
    options.loc = RNNT_CPU;
    options.num_threads = num_threads;
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes;
    get_workspace_size(maxT, maxU, minibatch_size,
                       false, &cpu_size_bytes);

    float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];
    compute_rnnt_loss(acts_ptr, grads_ptr,
                     labels_ptr, label_lengths_ptr,
                     input_lengths_ptr, alphabet_size,
                     minibatch_size, costs_ptr,
                     cpu_workspace, options);

    delete cpu_workspace;
    return 1;
}
#ifdef WARPRNNT_ENABLE_GPU
int gpu_rnnt(torch::Tensor acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor grads,
            int blank_label,
            int num_threads) {

    float *acts_ptr = (float*) acts.data_ptr();
    float *grads_ptr = NULL; // this will trigger the score forward code path
    if (grads.storage())
        grads_ptr = (float*) grads.data_ptr();

    int *input_lengths_ptr = (int*) input_lengths.data_ptr();
    int *labels_ptr = (int*) labels.data_ptr();
    int *label_lengths_ptr = (int*) label_lengths.data_ptr();
    float *costs_ptr = (float*) costs.data_ptr();

    int minibatch_size = acts.size(0);
    int maxT = acts.size(1);
    int maxU = acts.size(2);
    int alphabet_size = acts.size(3);

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.loc = RNNT_GPU;
    options.stream = at::cuda::getCurrentCUDAStream();
    options.num_threads = num_threads;
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t gpu_size_bytes;
    get_workspace_size(maxT, maxU, minibatch_size,
                       true, &gpu_size_bytes);

    cudaSetDevice(acts.get_device());

    void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);

    compute_rnnt_loss(acts_ptr, grads_ptr,
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
