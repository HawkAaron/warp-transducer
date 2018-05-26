int cpu_rnnt(THFloatTensor *acts,
                THIntTensor *labels,
                THIntTensor *input_lengths,
                THIntTensor *label_lengths,
                THFloatTensor *costs,
                THFloatTensor *grads,
                int blank_label,
                int num_threads);