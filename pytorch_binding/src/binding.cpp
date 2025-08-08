#include <iostream>
#include <numeric>

#include <torch/extension.h>
#include "rnnt.h"

#ifdef WARPRNNT_ENABLE_GPU
    #include "c10/cuda/CUDACachingAllocator.h"
#endif

int cpu_rnnt(torch::Tensor acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor grads,
            int blank_label,
            float fastemit_lambda,
            int num_threads) {

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
    options.fastemit_lambda = fastemit_lambda;
    options.blank_label = blank_label;
    options.batch_first = true;
    options.loc = RNNT_CPU;
    options.num_threads = num_threads;
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes = 0;
    switch (acts.scalar_type()) {
      case torch::kFloat:
        {
        get_workspace_size(maxT, maxU, minibatch_size,
                           false, &cpu_size_bytes);

        float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];
        compute_rnnt_loss(acts.data_ptr<float>(), grads.data_ptr<float>(),
                         labels.data_ptr<int>(), label_lengths.data_ptr<int>(),
                         input_lengths.data_ptr<int>(), alphabet_size,
                         minibatch_size, costs.data_ptr<float>(),
                         cpu_workspace, options);

        delete cpu_workspace;
        return 0;
        }
      case torch::kDouble:
        {
        get_workspace_size(maxT, maxU, minibatch_size,
                           false, &cpu_size_bytes,
                           sizeof(double));

        double* cpu_workspace = (double*) new unsigned char[cpu_size_bytes];
        compute_rnnt_loss_fp64(acts.data_ptr<double>(), grads.data_ptr<double>(),
                         labels.data_ptr<int>(), label_lengths.data_ptr<int>(),
                         input_lengths.data_ptr<int>(), alphabet_size,
                         minibatch_size, costs.data_ptr<double>(),
                         cpu_workspace, options);

        delete cpu_workspace;
        return 0;
        }
      default:
        std::cerr << __FILE__ << ':' << __LINE__ << ": " << "unsupported data type" << std::endl;
    }
    return -1;
}
#ifdef WARPRNNT_ENABLE_GPU
int gpu_rnnt(torch::Tensor acts,
            torch::Tensor labels,
            torch::Tensor input_lengths,
            torch::Tensor label_lengths,
            torch::Tensor costs,
            torch::Tensor grads,
            int blank_label,
            float fastemit_lambda,
            int num_threads) {

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
    options.fastemit_lambda = fastemit_lambda;
    options.num_threads = num_threads;
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    switch (acts.scalar_type()) {
      case torch::kFloat:
        {
        size_t gpu_size_bytes;
        get_workspace_size(maxT, maxU, minibatch_size,
                           true, &gpu_size_bytes);

        cudaSetDevice(acts.get_device());

        void* gpu_workspace = c10::cuda::CUDACachingAllocator::raw_alloc(gpu_size_bytes);
        if (gpu_workspace == nullptr) {
            std::cerr << __FILE__ << ':' << __LINE__ << ": " << "failed to allocate GPU workspace" << std::endl;
            return -1;
        }

        compute_rnnt_loss(acts.data_ptr<float>(), grads.data_ptr<float>(),
                         labels.data_ptr<int>(), label_lengths.data_ptr<int>(),
                         input_lengths.data_ptr<int>(), alphabet_size,
                         minibatch_size, costs.data_ptr<float>(),
                         gpu_workspace, options);

        c10::cuda::CUDACachingAllocator::raw_delete(gpu_workspace);
        return 0;
        }
      case torch::kDouble:
        {
        size_t gpu_size_bytes;
        get_workspace_size(maxT, maxU, minibatch_size,
                           true, &gpu_size_bytes);

        cudaSetDevice(acts.get_device());

        void* gpu_workspace = c10::cuda::CUDACachingAllocator::raw_alloc(gpu_size_bytes);
        if (gpu_workspace == nullptr) {
            std::cerr << __FILE__ << ':' << __LINE__ << ": " << "failed to allocate GPU workspace" << std::endl;
            return -1;
        }

        compute_rnnt_loss_fp64(acts.data_ptr<double>(), grads.data_ptr<double>(),
                         labels.data_ptr<int>(), label_lengths.data_ptr<int>(),
                         input_lengths.data_ptr<int>(), alphabet_size,
                         minibatch_size, costs.data_ptr<double>(),
                         gpu_workspace, options);

        c10::cuda::CUDACachingAllocator::raw_delete(gpu_workspace);
        return 0;
        }
      default:
        std::cerr << __FILE__ << ':' << __LINE__ << ": " << "unsupported data type" << std::endl;
    }
    return -1;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_rnnt", &cpu_rnnt, "RNNT CPU version");
#ifdef WARPRNNT_ENABLE_GPU
    m.def("gpu_rnnt", &gpu_rnnt, "RNNT GPU version");
#endif
}
