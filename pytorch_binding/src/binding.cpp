#include <iostream>
#include <vector>

#include <numeric>

#include "rnnt.h"
#include "TH.h"

extern "C" int cpu_rnnt(THFloatTensor *log_probs,
                        THIntTensor *labels,
                        THIntTensor *input_lengths,
                        THIntTensor *label_lengths,
                        THFloatTensor *costs,
                        THFloatTensor *grads,
                        int blank_label) {

    float *probs_ptr = log_probs->storage->data + log_probs->storageOffset;
    float *grads_ptr;
    if (grads->storage) {
            grads_ptr = grads->storage->data + grads->storageOffset;
    } else {
            grads_ptr = NULL; // this will trigger the score forward code path
    }

    int *input_lengths_ptr = input_lengths->storage->data + input_lengths->storageOffset;
    int *labels_ptr = labels->storage->data + labels->storageOffset;
    int *label_lengths_ptr = label_lengths->storage->data + label_lengths->storageOffset;
    float *costs_ptr = costs->storage->data + costs->storageOffset;

    int minibatch_size = log_probs->size[0];
    int maxT = log_probs->size[1];
    int maxU = log_probs->size[2];
    int alphabet_size = log_probs->size[3];

    size_t cpu_size_bytes;
    get_workspace_size(maxT, maxU, alphabet_size, minibatch_size,
                       false, &cpu_size_bytes);

    float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];

    compute_rnnt_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_lengths_ptr,
                     input_lengths_ptr, alphabet_size,
                     minibatch_size, maxT, maxU, costs_ptr,
                     cpu_workspace, blank_label);

    delete cpu_workspace;
    return 1;
}
