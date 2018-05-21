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

template<typename T>
inline __host__ __device__ T neg_inf() { return -T(INFINITY); }

template<typename T>
inline __host__ __device__ T log_sum_exp(T a, T b) {
    if (a == neg_inf<T>()) return b;
    if (b == neg_inf<T>()) return a;
    if (a > b)
        return log1p(exp(b-a)) + a;
    else
        return log1p(exp(a-b)) + b;
}

template<typename ProbT>
class CpuRNNT {
public:
    // Noncopyable
    CpuRNNT(int minibatch, int maxT, int maxU, int alphabet_size, void* workspace, int blank, int num_threads) :
        minibatch_(minibatch), maxT_(maxT), maxU_(maxU), alphabet_size_(alphabet_size), 
        workspace_(workspace), blank_(blank), num_threads_(num_threads) {
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
#else
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        } else {
            num_threads_ = omp_get_max_threads();
        }
#endif
    };

    CpuRNNT(const CpuRNNT&) = delete;
    CpuRNNT& operator=(const CpuRNNT&) = delete;

    void log_softmax(const ProbT* const trans_acts, const ProbT* const pred_acts, ProbT* denom);

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
    class CpuRNNT_metadata {
    public:
        CpuRNNT_metadata(int T, int U, void* workspace, size_t bytes_used);
        ProbT* alphas;
        ProbT* betas;
        ProbT* denom;
    };

    class CpuRNNT_index {
    public:
        CpuRNNT_index(int U, int maxU, int minibatch, int alphabet_size);
        int U;
        int maxU;
        int minibatch;
        int alphabet_size;

        int operator()(int t, int u);
        int operator()(int t, int u, int v);
    };

    class CpuRNNT_logProbs {
    public:
        CpuRNNT_logProbs(ProbT* const trans_acts, ProbT* const pred_acts, ProbT* denom, 
                        CpuRNNT_index& idx, CpuRNNT* rnnt);
        ProbT* trans_acts;
        ProbT* pred_acts;
        ProbT* denom;
        CpuRNNT_index idx;
        CpuRNNT* rnnt;

        ProbT operator()(int t, int u, int v);
    };

    int minibatch_;
    int maxT_;
    int maxU_;
    int alphabet_size_; // Number of characters plus blank
    void* workspace_;
    int blank_;
    int num_threads_;
    
    ProbT cost_and_grad_kernel(ProbT* const trans_acts, ProbT* const pred_acts,
                               ProbT* const denom,
                               ProbT* trans_grad, ProbT* pred_grad,
                               const int* const labels, int T, int U, size_t bytes_used);
    
    ProbT compute_alphas(CpuRNNT_logProbs& logp, int T, int U,
                         ProbT* alphas, const int* const labels);
    
    ProbT compute_betas_and_grad(ProbT* trans_grad, ProbT* pred_grad, CpuRNNT_logProbs& logp,
                                 int T, int U, ProbT* alphas, ProbT* betas,
                                 const int* const labels, ProbT logll);
};

template<typename ProbT>
CpuRNNT<ProbT>::CpuRNNT_metadata::CpuRNNT_metadata(int T, int U, void* workspace, size_t bytes_used) {
    
    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * T * U;
    std::fill(alphas, alphas + T * U, neg_inf<ProbT>());
    betas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * T * U;
    std::fill(betas, betas + T * U, neg_inf<ProbT>());
}

template<typename ProbT>
CpuRNNT<ProbT>::CpuRNNT_index::CpuRNNT_index(int U, int maxU, int minibatch, int alphabet_size) : 
                    U(U), maxU(maxU), minibatch(minibatch), alphabet_size(alphabet_size) {}

template<typename ProbT>
inline int CpuRNNT<ProbT>::CpuRNNT_index::operator()(int t, int u) {
    return t * U + u;
}

template<typename ProbT>
inline int CpuRNNT<ProbT>::CpuRNNT_index::operator()(int t, int u, int v) {
    return (t * maxU + u) * alphabet_size + v;
}

template<typename ProbT>
CpuRNNT<ProbT>::CpuRNNT_logProbs::CpuRNNT_logProbs(ProbT* const trans_acts, 
        ProbT* const pred_acts, ProbT* denom, CpuRNNT_index& idx, CpuRNNT* rnnt) :
            trans_acts(trans_acts), pred_acts(pred_acts), denom(denom), idx(idx), rnnt(rnnt) {
    
    // log_softmax(trans_acts, pred_acts, denom);
}

template<typename ProbT>
void
CpuRNNT<ProbT>::log_softmax(const ProbT* const trans_acts, const ProbT* const pred_acts, ProbT* denom) {

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        for (int t = 0; t < maxT_; ++t) {
            for (int u = 0; u < maxU_; ++u) {
                int t_offset = (mb * maxT_ + t) * alphabet_size_;
                int u_offset = (mb * maxU_ + u) * alphabet_size_;
                ProbT max_activation = neg_inf<ProbT>();

                for (int v = 0; v < alphabet_size_; ++v)
                    max_activation = std::max(max_activation, trans_acts[v + t_offset] + pred_acts[v + u_offset]);
                
                ProbT de = ProbT(0.);
                for (int v = 0; v < alphabet_size_; ++v) {
                    de += std::exp(trans_acts[v + t_offset] + pred_acts[v + u_offset] - max_activation);
                }

                // here only store denominator
                denom[(mb * maxT_ + t) * maxU_ + u] = -max_activation - std::log(de);
            }
        }
    }
}

template<typename ProbT>
inline ProbT CpuRNNT<ProbT>::CpuRNNT_logProbs::operator()(int t, int u, int v) {
    return denom[idx(t, u)] + trans_acts[t * idx.alphabet_size + v] + pred_acts[u * idx.alphabet_size + v];
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::cost_and_grad_kernel(ProbT* const trans_acts, ProbT* const pred_acts,
                                    ProbT* const denom,
                                    ProbT* trans_grad, ProbT* pred_grad,
                                    const int* const labels,
                                    int T, int U, size_t bytes_used) {
    
    CpuRNNT_metadata rnntm(T, U, workspace_, bytes_used);

    CpuRNNT_index idx(U, maxU_, minibatch_, alphabet_size_);
    CpuRNNT_logProbs logp(trans_acts, pred_acts, denom, idx, this);

    // auto start = std::chrono::high_resolution_clock::now();
    ProbT llForward = compute_alphas(logp, T, U, rnntm.alphas, labels);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "compute_alphas " << elapsed.count() * 1000 << " ms\n";

    // start = std::chrono::high_resolution_clock::now();
    ProbT llBackward = compute_betas_and_grad(trans_grad, pred_grad, 
                                              logp, T, U,
                                              rnntm.alphas, 
                                              rnntm.betas,
                                              labels,
                                              llForward);
    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // std::cout << "compute_betas_and_grad " << elapsed.count() * 1000 << " ms\n";

    ProbT diff = std::abs(llForward - llBackward);
    if (diff > 1e-1) {
        printf("WARNING: Forward backward likelihood mismatch %f\n", diff);
    }

    return -llForward;
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::compute_alphas(CpuRNNT_logProbs& logp, int T, int U, 
                        ProbT* alphas, const int* const labels) {

    CpuRNNT_index& idx = logp.idx;
    alphas[0] = 0;
    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            if (u == 0 && t > 0) 
                alphas[idx(t, 0)] = alphas[idx(t-1, 0)] + logp(t-1, 0, blank_);
            if (t == 0 && u > 0) 
                alphas[idx(0, u)] = alphas[idx(0, u-1)] + logp(0, u-1, labels[u-1]);
            if (t > 0 && u > 0) {
                ProbT no_emit = alphas[idx(t-1, u)] + logp(t-1, u, blank_);
                ProbT emit = alphas[idx(t, u-1)] + logp(t, u-1, labels[u-1]);
                alphas[idx(t, u)] = log_sum_exp<ProbT>(emit, no_emit);
            }
        }
    }

    ProbT loglike = alphas[idx(T-1, U-1)] + logp(T-1, U-1, blank_);

    return loglike;
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::compute_betas_and_grad(ProbT* trans_grad, ProbT* pred_grad, 
                                CpuRNNT_logProbs& logp,
                                int T, int U, ProbT* alphas, ProbT* betas,
                                const int* const labels, ProbT logll) {

    CpuRNNT_index& idx = logp.idx;

    // zero grad, first touch
    memset(trans_grad, 0, sizeof(ProbT) * maxT_ * alphabet_size_);
    memset(pred_grad, 0, sizeof(ProbT) * maxU_ * alphabet_size_);

    betas[idx(T-1, U-1)] = logp(T-1, U-1, blank_);

    for (int t = T-1; t >= 0; --t) {
        int t_offset = t * alphabet_size_;
        for (int u = U-1; u >= 0; --u) {
            int u_offset = u * alphabet_size_;
            if (u == U-1 && t < T-1)
                betas[idx(t, U-1)] = betas[idx(t+1, U-1)] + logp(t, U-1, blank_);
            if (t == T-1 && u < U-1)
                betas[idx(T-1, u)] = betas[idx(T-1, u+1)] + logp(T-1, u, labels[u]);
            if (t < T-1 && u < U-1) {
                ProbT no_emit = betas[idx(t+1, u)] + logp(t, u, blank_);
                ProbT emit = betas[idx(t, u+1)] + logp(t, u, labels[u]);
                betas[idx(t, u)] = log_sum_exp<ProbT>(emit, no_emit);
            }
            // grad
            for (int v = 0; v < alphabet_size_; ++v) {
                ProbT lgpk = logp(t, u, v);
                ProbT grad = std::exp(alphas[idx(t, u)] + betas[idx(t, u)] + lgpk - logll);
                // grad to last blank transition
                if (v == blank_ && t == T-1 && u == U-1) grad -= 1;
                if (v == blank_ && t < T-1) {
                    grad -= std::exp(alphas[idx(t, u)] + lgpk - logll + betas[idx(t+1, u)]);
                }
                if (v == labels[u] && u < U-1) {
                    grad -= std::exp(alphas[idx(t, u)] + lgpk - logll + betas[idx(t, u+1)]);
                }
                trans_grad[t_offset + v] += grad;
                pred_grad[u_offset + v] += grad;
            }
        }
    }

    return betas[0];
}

template<typename ProbT>
rnntStatus_t
CpuRNNT<ProbT>::cost_and_grad(ProbT* const trans_acts,
                       ProbT* const pred_acts,
                       ProbT* trans_grads,
                       ProbT* pred_grads,
                       ProbT* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    // denom
    ProbT* denom = static_cast<ProbT*>(workspace_);
    size_t bytes_used = sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas & log_softmax denominator
    per_minibatch_bytes += sizeof(ProbT) * maxT_ * maxU_ * 2;

    log_softmax(trans_acts, pred_acts, denom);

#pragma omp parallel for 
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription
        const int t_offset = mb * maxT_ * alphabet_size_;
        const int u_offset = mb * maxU_ * alphabet_size_;

        costs[mb] = cost_and_grad_kernel(trans_acts + t_offset , pred_acts + u_offset,
                             denom + mb * maxT_ * maxU_,
                             trans_grads + t_offset, pred_grads + u_offset,
                             flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0),
                             T, U, bytes_used + mb * per_minibatch_bytes);
    }

    return RNNT_STATUS_SUCCESS;
}

template<typename ProbT>
rnntStatus_t
CpuRNNT<ProbT>::score_forward(ProbT* const trans_acts, 
                       ProbT* const pred_acts,
                       ProbT* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    // denom
    ProbT* denom = static_cast<ProbT*>(workspace_);
    size_t bytes_used = sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas & log_softmax denominator
    per_minibatch_bytes += sizeof(ProbT) * maxT_ * maxU_ * 2;

    log_softmax(trans_acts, pred_acts, denom);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription
        const int t_offset = mb * maxT_ * alphabet_size_;
        const int u_offset = mb * maxU_ * alphabet_size_;

        CpuRNNT_metadata rnntm(T, U, workspace_, bytes_used + mb * per_minibatch_bytes);

        CpuRNNT_index idx(U, maxU_, minibatch_, alphabet_size_);
        CpuRNNT_logProbs logp(trans_acts + t_offset, pred_acts + u_offset, denom + mb * maxT_ * maxU_, idx, this);

        costs[mb] = -compute_alphas(logp, T, U, rnntm.alphas, 
                            flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0));
    }

    return RNNT_STATUS_SUCCESS;
}
