#include <iostream>
#include <vector>

#include <numeric>

#include "rnnt.h"

#ifdef WARPRNNT_ENABLE_GPU
    #include "THC.h"
    #include "THCTensor.h"
    #include "detail/reduce.h"
    extern THCState* state;
#else
    #include "TH.h"
#endif

extern "C" int cpu_rnnt(THFloatTensor *acts,
                        THIntTensor *labels,
                        THIntTensor *input_lengths,
                        THIntTensor *label_lengths,
                        THFloatTensor *costs,
                        THFloatTensor *grads,
                        int blank_label,
                        int num_threads) {

    float *acts_ptr = THFloatTensor_data(acts);
    float *grads_ptr = NULL; // this will trigger the score forward code path
    if (grads->storage) 
        grads_ptr = THFloatTensor_data(grads);

    int *input_lengths_ptr = THIntTensor_data(input_lengths);
    int *labels_ptr = THIntTensor_data(labels);
    int *label_lengths_ptr = THIntTensor_data(label_lengths);
    float *costs_ptr = THFloatTensor_data(costs);

    int maxT = acts->size[0];
    int maxU = acts->size[1];
    int minibatch_size = acts->size[2];
    int alphabet_size = acts->size[3];

	if (true) {
		minibatch_size = acts->size[0];
		maxT = acts->size[1];
		maxU = acts->size[2];
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
extern "C" int gpu_rnnt(THCudaTensor *acts,
                        THCudaIntTensor *labels,
                        THCudaIntTensor *input_lengths,
                        THCudaIntTensor *label_lengths,
                        THFloatTensor *costs,
                        THCudaTensor *grads,
                        int blank_label,
                        int num_threads) {

    float *acts_ptr = THCudaTensor_data(state, acts);
    float *grads_ptr = NULL; // this will trigger the score forward code path
    if (grads->storage) 
        grads_ptr = THCudaTensor_data(state, grads);

    int *input_lengths_ptr = THCudaIntTensor_data(state, input_lengths);
    int *labels_ptr = THCudaIntTensor_data(state, labels);
    int *label_lengths_ptr = THCudaIntTensor_data(state, label_lengths);
    float *costs_ptr = THFloatTensor_data(costs);

    int minibatch_size = acts->size[0];
    int maxT = acts->size[1];
    int maxU = acts->size[2];
    int alphabet_size = acts->size[3];

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.loc = RNNT_GPU;
    options.stream = THCState_getCurrentStream(state);
    options.num_threads = num_threads;
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t gpu_size_bytes;
    get_workspace_size(maxT, maxU, minibatch_size,
                       true, &gpu_size_bytes);

    void* gpu_workspace;
    THCudaMalloc(state, &gpu_workspace, gpu_size_bytes);

    compute_rnnt_loss(acts_ptr, grads_ptr,
                     labels_ptr, label_lengths_ptr,
                     input_lengths_ptr, alphabet_size,
                     minibatch_size, costs_ptr,
                     gpu_workspace, options);

    THCudaFree(state, gpu_workspace);
    return 1;
}
#endif