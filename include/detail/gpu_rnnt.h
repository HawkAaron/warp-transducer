#pragma once

#include <tuple>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>

#if !defined(RNNT_DISABLE_OMP) && !defined(APPLE)
#include <omp.h>
#endif

#include "reduce.h"
#include "cpu_rnnt_kernel.h"

template<typename ProbT>
class GpuRNNT {
public:
    // Noncopyable
    GpuRNNT(int minibatch, int maxT, int maxU, int alphabet_size, void* workspace, 
            int blank, int num_threads, CUstream stream) :
        minibatch_(minibatch), maxT_(maxT), maxU_(maxU), alphabet_size_(alphabet_size), 
        workspace_(workspace), blank_(blank), num_threads_(num_threads), stream_(stream) {
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
#else
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        } else {
            num_threads_ = omp_get_max_threads();
        }
#endif
    };

    GpuRNNT(const GpuRNNT&) = delete;
    GpuRNNT& operator=(const GpuRNNT&) = delete;

    void log_softmax(const ProbT* const trans_acts, const ProbT* const pred_acts, ProbT* denom);
    void log_softmax_cpu(const ProbT* const trans_acts, const ProbT* const pred_acts, ProbT* denom);

    rnntStatus_t cost_and_grad(ProbT* const trans_acts,
                              ProbT* const pred_acts,
                              ProbT* trans_grad,
                              ProbT* pred_grad,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);

    rnntStatus_t score_forward(ProbT* const trans_acts,
                              ProbT* const pred_acts,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);

private:
    int minibatch_;
    int maxT_;
    int maxU_;
    int alphabet_size_; // Number of characters plus blank
    void* workspace_;
    int blank_;
    int num_threads_;
    CUstream stream_;
    
};

template<typename ProbT>
void
GpuRNNT<ProbT>::log_softmax(const ProbT* const ft, const ProbT* const gu, ProbT* denom) {

    // trans_acts + pred_acts -> log_softmax denominator
    // reduce_max(ft, gu, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, maxT_, maxU_, 0, stream_);
    // reduce_exp(ft, gu, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, maxT_, maxU_, 1, stream_);
    for (int mb = 0; mb < minibatch_; ++mb) {
        for (int t = 0; t < maxT_; ++t) {
            for (int u = 0; u < maxU_; ++u) {
                int t_offset = (mb * maxT_ + t)* alphabet_size_;
                int u_offset = (mb * maxU_ + u) * alphabet_size_;
                ProbT max_activation = neg_inf<ProbT>();

                for (int v = 0; v < alphabet_size_; ++v)
                    max_activation = std::max(max_activation, ft[v + t_offset] + gu[v + u_offset]);
                
                ProbT de = ProbT(0.);
                for (int v = 0; v < alphabet_size_; ++v) {
                    de += std::exp(ft[v + t_offset] + gu[v + u_offset] - max_activation);
                }

                // here only store denominator
                denom[(mb * maxT_ + t) * maxU_ + u] = -max_activation - std::log(de);
            }
        }
    }
}

template<typename ProbT>
rnntStatus_t
GpuRNNT<ProbT>::cost_and_grad(ProbT* const trans_acts_cpu,
                       ProbT* const pred_acts_cpu,
                       ProbT* trans_grads_cpu,
                       ProbT* pred_grads_cpu,
                       ProbT* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    // alphas & betas & denom
    size_t bytes = sizeof(ProbT) * maxT_ * maxU_ * 3;
    // forward-backward loglikelihood
    bytes += sizeof(ProbT) * 2;
    // labels
    bytes += sizeof(int) * (maxU_ - 1);
    // length
    bytes += sizeof(int) * 2;
    // acts & grads
    bytes += sizeof(ProbT) * (maxT_ + maxU_) * alphabet_size_ * 2;
    bytes *= minibatch_;

    void * gpu_workspace;
    // cudaMalloc(&gpu_workspace, bytes);
    gpu_workspace = malloc(bytes);
    size_t bytes_used = 0;
    // acts
    ProbT* trans_acts = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace));
    bytes_used += sizeof(ProbT) * maxT_ * alphabet_size_ * minibatch_;
    ProbT* pred_acts = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxU_ * alphabet_size_ * minibatch_;
    // grads
    ProbT* trans_grad = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxT_ * alphabet_size_ * minibatch_;
    ProbT* pred_grad = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxU_ * alphabet_size_ * minibatch_;
    // denom
    ProbT* denom = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    // alphas & betas
    ProbT* alphas = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    ProbT* betas = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    // logllh
    ProbT* llForward = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * minibatch_;
    ProbT* llBackward = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * minibatch_;
    // labels
    int* labels = reinterpret_cast<int*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(int) * (maxU_ - 1) * minibatch_;
    // length
    int* xlen = reinterpret_cast<int*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(int) * minibatch_;
    int* ylen = reinterpret_cast<int*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(int) * minibatch_;

    // cudaMemcpyAsync(trans_acts, trans_acts_cpu, sizeof(ProbT) * minibatch_ * maxT_ * alphabet_size_, cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(pred_acts, pred_acts_cpu, sizeof(ProbT) * minibatch_ * maxU_ * alphabet_size_, cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(trans_grad, trans_grads_cpu, sizeof(ProbT) * minibatch_ * maxT_ * alphabet_size_, cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(pred_grad, pred_grads_cpu, sizeof(ProbT) * minibatch_ * maxU_ * alphabet_size_, cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(labels, flat_labels, sizeof(int) * minibatch_ * (maxU_ - 1), cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(xlen, input_lengths, sizeof(int) * minibatch_, cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(ylen, label_lengths, sizeof(int) * minibatch_, cudaMemcpyHostToDevice, stream_);

    // denom
    auto start = std::chrono::high_resolution_clock::now();
    // log_softmax(trans_acts, pred_acts, denom);
    log_softmax(trans_acts_cpu, pred_acts_cpu, denom);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "log_softmax " << elapsed.count() * 1000 << " ms\n";
    // alphas
    start = std::chrono::high_resolution_clock::now();
    // compute_alphas_kernel<ProbT><<<1, minibatch_, 0, stream_>>>(trans_acts, pred_acts, denom, alphas, llForward, 
    //     xlen, ylen, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    trans_acts = static_cast<ProbT*>(trans_acts_cpu); pred_acts = static_cast<ProbT*>(pred_acts_cpu); 
    xlen = const_cast<int*>(input_lengths); ylen = const_cast<int*>(label_lengths); labels = const_cast<int*>(flat_labels);
    // printf("denom\n");
    // for (int mb = 0; mb < minibatch_; ++mb) {
    //     for (int t = 0; t < maxT_; ++t) {
    //         for (int u = 0; u < maxU_; ++u) {
    //             printf("%f ", denom[(mb*maxT_+t)*maxU_+u]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // printf("alphas\n");
    // for (int mb = 0; mb < minibatch_; ++mb) {
    //     for (int t = 0; t < maxT_; ++t) {
    //         for (int u = 0; u < maxU_; ++u) {
    //             printf("%f ", alphas[(mb*maxT_+t)*maxU_+u]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // printf("trans_acts\n");
    // for (int mb = 0; mb < minibatch_; ++mb) {
    //     for (int t = 0; t < maxT_; ++t) {
    //         for (int v = 0; v < alphabet_size_; ++v) {
    //             printf("%f ", trans_acts[(mb * maxT_ + t) * alphabet_size_ + v]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // printf("pred_acts\n");
    // for (int mb = 0; mb < minibatch_; ++mb) {
    //     for (int u = 0; u < maxU_; ++u) {
    //         for (int v = 0; v < alphabet_size_; ++v) {
    //             printf("%f ", pred_acts[(mb * maxU_ + u) * alphabet_size_ + v]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    compute_alphas_cpu(trans_acts, pred_acts, denom, alphas, llForward, 
        xlen, ylen, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "compute_alphas_kernel " << elapsed.count() * 1000 << " ms\n";
    // ProbT* alphas_cpu = static_cast<ProbT*>(workspace_);
    // cudaMemcpyAsync(alphas_cpu, alphas, sizeof(ProbT) * minibatch_ * maxT_ * maxU_, cudaMemcpyDeviceToHost, stream_);
    // printf("alphas\n");
    // for (int mb = 0; mb < minibatch_; ++mb) {
    //     for (int t = 0; t < maxT_; ++t) {
    //         for (int u = 0; u < maxU_; ++u) {
    //             printf("%f ", alphas[(mb * maxT_ + t) * maxU_ + u]);
    //         }
    //         printf("\n");
    //     }
    // }
    // betas
    start = std::chrono::high_resolution_clock::now();
    // compute_betas_kernel<ProbT><<<1, minibatch_, 0, stream_>>>(trans_acts, pred_acts, denom, betas, llBackward,
    //     xlen, ylen, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    compute_betas_cpu(trans_acts, pred_acts, denom, betas, llBackward,
        xlen, ylen, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "compute_betas_kernel " << elapsed.count() * 1000 << " ms\n";
    // ProbT* betas_cpu = static_cast<ProbT*>(workspace_);
    // cudaMemcpyAsync(betas_cpu, betas, sizeof(ProbT) * minibatch_ * maxT_ * maxU_, cudaMemcpyDeviceToHost, stream_);
    // printf("betas\n");
    // for (int mb = 0; mb < minibatch_; ++mb) {
    //     for (int t = 0; t < maxT_; ++t) {
    //         for (int u = 0; u < maxU_; ++u) {
    //             printf("%f ", betas[(mb * maxT_ + t) * maxU_ + u]);
    //         }
    //         printf("\n");
    //     }
    // }
    // gradient
    start = std::chrono::high_resolution_clock::now();
    // compute_grad_kernel<128, ProbT><<<minibatch_ * maxT_ * maxU_, 128, 0, stream_>>>(trans_grad, pred_grad, 
    //     trans_acts, pred_acts, denom, alphas, betas, llForward, xlen, ylen, labels, 
    //     minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    compute_grad_cpu<128, ProbT>(trans_grads_cpu, pred_grads_cpu, 
        trans_acts, pred_acts, denom, alphas, betas, llForward, xlen, ylen, labels, 
        minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "compute_grad_kernel " << elapsed.count() * 1000 << " ms\n";

    // cost
    // cudaMemcpyAsync(costs, llForward, sizeof(ProbT) * minibatch_, cudaMemcpyDeviceToHost, stream_);
    // cudaMemcpyAsync(trans_grads_cpu, trans_grad, sizeof(ProbT) * minibatch_ * maxT_ * alphabet_size_, cudaMemcpyDeviceToHost, stream_);
    // cudaMemcpyAsync(pred_grads_cpu, pred_grad, sizeof(ProbT) * minibatch_ * maxU_ * alphabet_size_, cudaMemcpyDeviceToHost, stream_);
    for (int mb = 0; mb < minibatch_; ++mb) costs[mb] = -llForward[mb];
    free(gpu_workspace);
    return RNNT_STATUS_SUCCESS;
}

template<typename ProbT>
rnntStatus_t
GpuRNNT<ProbT>::score_forward(ProbT* const trans_acts_cpu, 
                       ProbT* const pred_acts_cpu,
                       ProbT* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    // alphas & denom
    size_t bytes = sizeof(ProbT) * maxT_ * maxU_ * 2;
    // forward loglikelihood
    bytes += sizeof(ProbT);
    // labels
    bytes += sizeof(int) * (maxU_ - 1);
    // length
    bytes += sizeof(int) * 2;
    // acts
    bytes += sizeof(ProbT) * (maxT_ + maxU_) * alphabet_size_;
    bytes *= minibatch_;

    void * gpu_workspace;
    // cudaMalloc(&gpu_workspace, bytes);
    gpu_workspace = malloc(bytes);
    size_t bytes_used = 0;
    // acts
    ProbT* trans_acts = static_cast<ProbT*>(gpu_workspace);
    bytes_used += maxT_ * alphabet_size_ * minibatch_;
    ProbT* pred_acts = static_cast<ProbT*>(gpu_workspace) + bytes_used;
    bytes_used += maxU_ * alphabet_size_ * minibatch_;
    // denom
    ProbT* denom = static_cast<ProbT*>(gpu_workspace) + bytes_used;
    bytes_used += maxT_ * maxU_ * minibatch_;
    // alphas
    ProbT* alphas = static_cast<ProbT*>(gpu_workspace) + bytes_used;
    bytes_used += maxT_ * maxU_ * minibatch_;
    // logllh
    ProbT* llForward = static_cast<ProbT*>(gpu_workspace) + bytes_used;
    bytes_used += minibatch_;
    // labels
    int* labels = static_cast<int*>(gpu_workspace) + bytes_used;
    bytes_used += (maxU_ - 1) * minibatch_;
    // length
    int* xlen = static_cast<int*>(gpu_workspace) + bytes_used;
    bytes_used += minibatch_;
    int* ylen = static_cast<int*>(gpu_workspace) + bytes_used;
    bytes_used += minibatch_;

    // cudaMemcpyAsync(trans_acts, trans_acts_cpu, sizeof(ProbT) * minibatch_ * maxT_ * alphabet_size_, cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(pred_acts, pred_acts_cpu, sizeof(ProbT) * minibatch_ * maxU_ * alphabet_size_, cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(labels, flat_labels, sizeof(int) * minibatch_ * (maxU_ - 1), cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(xlen, input_lengths, sizeof(int) * minibatch_, cudaMemcpyHostToDevice, stream_);
    // cudaMemcpyAsync(ylen, label_lengths, sizeof(int) * minibatch_, cudaMemcpyHostToDevice, stream_);

    // log_softmax(trans_acts, pred_acts, denom);
    // compute_alphas_kernel<ProbT><<<1, minibatch_, 0, stream_>>>(trans_acts, pred_acts, denom, alphas, llForward, 
    //     xlen, ylen, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    // cudaMemcpyAsync(costs, llForward, sizeof(ProbT) * minibatch_, cudaMemcpyDeviceToHost, stream_);

    log_softmax(trans_acts_cpu, pred_acts_cpu, denom);
    trans_acts = static_cast<ProbT*>(trans_acts_cpu); pred_acts = static_cast<ProbT*>(pred_acts_cpu); 
    xlen = const_cast<int*>(input_lengths); ylen = const_cast<int*>(label_lengths); labels = const_cast<int*>(flat_labels);
    compute_alphas_cpu(trans_acts, pred_acts, denom, alphas, llForward, 
        xlen, ylen, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    for (int mb = 0; mb < minibatch_; ++mb) costs[mb] = -llForward[mb];
    free(gpu_workspace);
    return RNNT_STATUS_SUCCESS;
}
