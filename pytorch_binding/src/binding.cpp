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
                        int blank_label,
                        int num_threads,
                        int batch_first) {

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

    int maxT = log_probs->size[0];
    int maxU = log_probs->size[1];
    int minibatch_size = log_probs->size[2];
    int alphabet_size = log_probs->size[3];

	if (batch_first) {
		minibatch_size = log_probs->size[0];
		maxT = log_probs->size[1];
		maxU = log_probs->size[2];
	}

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.batch_first = batch_first;
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
    compute_rnnt_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_lengths_ptr,
                     input_lengths_ptr, alphabet_size,
                     minibatch_size, costs_ptr,
                     cpu_workspace, options);

    delete cpu_workspace;
    return 1;
}
