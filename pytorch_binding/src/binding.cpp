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

extern "C" int cpu_rnnt(THFloatTensor *trans_acts,
                        THFloatTensor *pred_acts,
                        THIntTensor *labels,
                        THIntTensor *input_lengths,
                        THIntTensor *label_lengths,
                        THFloatTensor *costs,
                        THFloatTensor *trans_grad,
                        THFloatTensor *pred_grad,
                        int blank_label,
                        int num_threads) {

    float *trans_acts_ptr = THFloatTensor_data(trans_acts);
    float *pred_acts_ptr = THFloatTensor_data(pred_acts);
    float *trans_grad_ptr = NULL; // this will trigger the score forward code path
    float *pred_grad_ptr = NULL;

    if (trans_grad->storage && pred_grad->storage) {
        trans_grad_ptr = THFloatTensor_data(trans_grad);
        pred_grad_ptr = THFloatTensor_data(pred_grad);
    }

    int *input_lengths_ptr = THIntTensor_data(input_lengths);
    int *labels_ptr = THIntTensor_data(labels);
    int *label_lengths_ptr = THIntTensor_data(label_lengths);
    float *costs_ptr = THFloatTensor_data(costs);

    int maxT = trans_acts->size[1];
    int maxU = pred_acts->size[1];
    int minibatch_size = trans_acts->size[0];
    int alphabet_size = trans_acts->size[2];

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
    get_workspace_size(maxT, maxU, minibatch_size, alphabet_size,
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
extern "C" int gpu_rnnt(THCudaTensor *trans_acts,
                        THCudaTensor *pred_acts,
                        THIntTensor *labels,
                        THIntTensor *input_lengths,
                        THIntTensor *label_lengths,
                        THFloatTensor *costs,
                        THCudaTensor *trans_grad,
                        THCudaTensor *pred_grad,
                        int blank_label,
                        int num_threads=0) { // not used in GPU version

    float *trans_acts_ptr = THCudaTensor_data(state, trans_acts);
    float *pred_acts_ptr = THCudaTensor_data(state, pred_acts);
    float *trans_grad_ptr = NULL; // this will trigger the score forward code path
    float *pred_grad_ptr = NULL;

    if (trans_grad->storage && pred_grad->storage) {
        trans_grad_ptr = THCudaTensor_data(state, trans_grad);
        pred_grad_ptr = THCudaTensor_data(state, pred_grad);
    }

    int *input_lengths_ptr = THIntTensor_data(input_lengths);
    int *labels_ptr = THIntTensor_data(labels);
    int *label_lengths_ptr = THIntTensor_data(label_lengths);
    float *costs_ptr = THFloatTensor_data(costs);

    int maxT = trans_acts->size[1];
    int maxU = pred_acts->size[1];
    int minibatch_size = trans_acts->size[0];
    int alphabet_size = trans_acts->size[2];

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.loc = RNNT_GPU;
    options.stream = THCState_getCurrentStream(state);

    size_t gpu_size_bytes;
    get_workspace_size(maxT, maxU, minibatch_size, alphabet_size,
                       true, &gpu_size_bytes);

    void* gpu_workspace;
    THCudaMalloc(state, &gpu_workspace, gpu_size_bytes);

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