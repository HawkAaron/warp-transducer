#include <cstddef>
#include <iostream>
#include <algorithm>

#include <rnnt.h>

#include "detail/cpu_rnnt.h"

extern "C" {

int get_warprnnt_version() {
    return 1;
}

const char* rnntGetStatusString(rnntStatus_t status) {
    switch (status) {
    case RNNT_STATUS_SUCCESS:
        return "no error";
    case RNNT_STATUS_MEMOPS_FAILED:
        return "cuda memcpy or memset failed";
    case RNNT_STATUS_INVALID_VALUE:
        return "invalid value";
    case RNNT_STATUS_EXECUTION_FAILED:
        return "execution failed";

    case RNNT_STATUS_UNKNOWN_ERROR:
    default:
        return "unknown error";

    }

}


rnntStatus_t compute_rnnt_loss(float* const activations, //BTUV
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             rnntOptions options) {

    if (activations == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0 ||
        options.maxT <= 0 ||
        options.maxU <= 0)
        return RNNT_STATUS_INVALID_VALUE;

    CpuRNNT<float> rnnt(minibatch, options.maxT, options.maxU, alphabet_size, workspace, 
                            options.blank_label, options.num_threads, options.batch_first);

    if (gradients != NULL)
        return rnnt.cost_and_grad(activations, gradients,
                                    costs,
                                    flat_labels, label_lengths,
                                    input_lengths);
    else
        return rnnt.score_forward(activations, costs, flat_labels,
                                    label_lengths, input_lengths);
}


rnntStatus_t get_workspace_size(int maxT, int maxU,
                               int minibatch,
                               bool gpu,
                               size_t* size_bytes)
{
    if (minibatch <= 0 ||
        maxT <= 0 ||
        maxU <= 0)
        return RNNT_STATUS_INVALID_VALUE;

    *size_bytes = 0;

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(float) * maxT * maxU * 2;

    // blank & label log probability cache
    per_minibatch_bytes += sizeof(float) * maxT * maxU * 2;

    *size_bytes = per_minibatch_bytes * minibatch;

    return RNNT_STATUS_SUCCESS;
}

}
