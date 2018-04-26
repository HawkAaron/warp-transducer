#pragma once

#include <tuple>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>

#if !defined(RNNT_DISABLE_OMP) && !defined(APPLE)
#include <omp.h>
#endif

template<typename T>
inline T neg_inf() { return -std::numeric_limits<T>::infinity(); }

template<typename T>
inline T log_sum_exp(T a, T b) {
    if (a == neg_inf<T>()) return b;
    if (b == neg_inf<T>()) return a;
    if (a > b)
        return std::log1p(std::exp(b-a)) + a;
    else
        return std::log1p(std::exp(a-b)) + b;
}

template<typename ProbT>
class CpuRNNT {
public:
    // Noncopyable
    CpuRNNT(int minibatch, int maxT, int maxU, int alphabet_size, void* workspace, int blank, int num_threads, bool batch_first) :
        minibatch_(minibatch), maxT_(maxT), maxU_(maxU), alphabet_size_(alphabet_size), 
        workspace_(workspace), blank_(blank), num_threads_(num_threads), batch_first(batch_first) {
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

    rnntStatus_t cost_and_grad(ProbT* const log_probs,
                              ProbT* grads,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);
    
    rnntStatus_t score_forward(ProbT* const log_probs,
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
    };

    class CpuRNNT_index {
    public:
        CpuRNNT_index(int U, int maxU, int minibatch, int alphabet_size, bool batch_first);
        int U;
        int maxU;
        int minibatch;
        int alphabet_size;
        bool batch_first;

        int operator()(int t, int u);
        int operator()(int t, int u, int v);
    };

    int minibatch_;
    int maxT_;
    int maxU_;
    int alphabet_size_; // Number of characters plus blank
    void* workspace_;
    int blank_;
    int num_threads_;
    bool batch_first;
    
    ProbT cost_and_grad_kernel(const ProbT* const log_probs, ProbT* grad,
                               const int* const labels, int mb,
                               int T, int U, size_t bytes_used);
    
    ProbT compute_alphas(const ProbT* const log_probs, int T, int U,
                         ProbT* alphas, const int* const labels);
    
    ProbT compute_betas_and_grad(ProbT* grad, const ProbT* const log_probs,
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
CpuRNNT<ProbT>::CpuRNNT_index::CpuRNNT_index(int U, int maxU, int minibatch, int alphabet_size, bool batch_first) : 
                    U(U), maxU(maxU), minibatch(minibatch), alphabet_size(alphabet_size), batch_first(batch_first) {}

template<typename ProbT>
inline int CpuRNNT<ProbT>::CpuRNNT_index::operator()(int t, int u) {
    return t * U + u;
}

template<typename ProbT>
inline int CpuRNNT<ProbT>::CpuRNNT_index::operator()(int t, int u, int v) {
    if (batch_first)
        return (t * maxU + u) * alphabet_size + v;
    return (t * maxU + u) * minibatch * alphabet_size + v;
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::cost_and_grad_kernel(const ProbT* const log_probs, ProbT* grad,
                              const int* const labels,
                              int mb, int T, int U, size_t bytes_used) {
    
    CpuRNNT_metadata rnntm(T, U, workspace_, bytes_used);

    ProbT llForward = compute_alphas(log_probs, T, U, rnntm.alphas, labels);
    ProbT llBackward = compute_betas_and_grad(grad, log_probs, T, U,
                                              rnntm.alphas, 
                                              rnntm.betas,
                                              labels,
                                              llForward);

    ProbT diff = std::abs(llForward - llBackward);
    if (diff > 1e-1) {
        printf("WARNING: Forward backward likelihood mismatch %f\n", diff);
    }

    return -llForward;
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::compute_alphas(const ProbT* const log_probs, int T, int U, 
                        ProbT* alphas, const int* const labels) {

    CpuRNNT_index idx(U, maxU_, minibatch_, alphabet_size_, batch_first);

    alphas[0] = 0;

    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            if (u == 0 && t > 0)
                alphas[idx(t, 0)] = alphas[idx(t-1, 0)] + log_probs[idx(t-1, 0, blank_)];
            if (t == 0 && u > 0)
                alphas[idx(0, u)] = alphas[idx(0, u-1)] + log_probs[idx(0, u-1, labels[u-1])];
            if (t > 0 && u > 0) {
                ProbT no_emit = alphas[idx(t-1, u)] + log_probs[idx(t-1, u, blank_)];
                ProbT emit = alphas[idx(t, u-1)] + log_probs[idx(t, u-1, labels[u-1])];
                alphas[idx(t, u)] = log_sum_exp<ProbT>(emit, no_emit);
            }
        }
    }

    ProbT loglike = alphas[idx(T-1, U-1)] + log_probs[idx(T-1, U-1, blank_)];

    return loglike;
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::compute_betas_and_grad(ProbT* grad, const ProbT* const log_probs,
                                int T, int U, ProbT* alphas, ProbT* betas,
                                const int* const labels, ProbT logll) {

    CpuRNNT_index idx(U, maxU_, minibatch_, alphabet_size_, batch_first);

    betas[idx(T-1, U-1)] = log_probs[idx(T-1, U-1, blank_)];

    for (int t = T-1; t >= 0; --t) {
        for (int u = U-1; u >= 0; --u) {
            if (u == U-1 && t < T-1)
                betas[idx(t, U-1)] = betas[idx(t+1, U-1)] + log_probs[idx(t, U-1, blank_)];
            if (t == T-1 && u < U-1)
                betas[idx(T-1, u)] = betas[idx(T-1, u+1)] + log_probs[idx(T-1, u, labels[u])];
            if (t < T-1 && u < U-1) {
                ProbT no_emit = betas[idx(t+1, u)] + log_probs[idx(t, u, blank_)];
                ProbT emit = betas[idx(t, u+1)] + log_probs[idx(t, u, labels[u])];
                betas[idx(t, u)] = log_sum_exp<ProbT>(emit, no_emit);
            }
        }
    }

    ProbT loglike = betas[0];

    // Gradients w.r.t. log probabilities
    grad[idx(T-1, U-1, blank_)] = -std::exp(log_probs[idx(T-1, U-1, blank_)] + alphas[idx(T-1, U-1)] - loglike);
    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            if (t < T-1) {
                ProbT g = alphas[idx(t, u)] + betas[idx(t+1, u)];
                grad[idx(t, u, blank_)] = -std::exp(log_probs[idx(t, u, blank_)] + g - loglike);
            }
            if (u < U-1) {
                ProbT g = alphas[idx(t, u)] + betas[idx(t, u+1)];
                grad[idx(t, u, labels[u])] = -std::exp(log_probs[idx(t, u, labels[u])] + g - loglike);
            }
        }
    }

    return loglike;
}

template<typename ProbT>
rnntStatus_t
CpuRNNT<ProbT>::cost_and_grad(ProbT* const log_probs,
                       ProbT* grads,
                       ProbT* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(ProbT) * maxT_ * maxU_ * 2;

    // zero grads
    memset(grads, 0, sizeof(ProbT) * maxT_ * maxU_ * alphabet_size_);

#pragma omp parallel for 
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription
        int batch_size = alphabet_size_;
        if (batch_first) batch_size = maxT_ * maxU_ * alphabet_size_;

        costs[mb] = cost_and_grad_kernel(log_probs + mb * batch_size,
                             grads + mb * batch_size,
                             flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0),
                             mb, T, U, mb * per_minibatch_bytes);
    }

    return RNNT_STATUS_SUCCESS;
}

template<typename ProbT>
rnntStatus_t
CpuRNNT<ProbT>::score_forward(ProbT* const log_probs, 
                       ProbT* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(ProbT) * maxT_ * maxU_ * 2;

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription
        int batch_size = alphabet_size_;
        if (batch_first) batch_size = maxT_ * maxU_ * alphabet_size_;

        CpuRNNT_metadata rnntm(T, U, workspace_, mb * per_minibatch_bytes);

        costs[mb] = -compute_alphas(log_probs + mb * batch_size, T, U, 
                            rnntm.alphas, 
                            flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0));
    }

    return RNNT_STATUS_SUCCESS;
}
