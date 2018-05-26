int gpu_rnnt(THCudaTensor *acts,
                THCudaIntTensor *labels,
                THCudaIntTensor *input_lengths,
                THCudaIntTensor *label_lengths,
                THFloatTensor *costs,
                THCudaTensor *grads,
                int blank_label,
                int num_threads);