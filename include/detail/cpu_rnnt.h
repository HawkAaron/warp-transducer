#pragma once

#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

#if !defined(CTC_DISABLE_OMP) && !defined(APPLE)
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
    CpuRNNT(int minibatch, int maxT, int maxU, int alphabet_size, void* workspace, int blank) :
        minibatch_(minibatch), maxT_(maxT), maxU_(maxU), alphabet_size_(alphabet_size), 
        workspace_(workspace), blank_(blank) {

    };

    CpuRNNT(const CpuRNNT&) = delete;
    CpuRNNT& operator=(const CpuRNNT&) = delete;

    rnntStatus_t cost_and_grad(ProbT* const log_probs,
                              ProbT* grads,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);
    
    rnntStatus_t score_forward(const ProbT* const log_probs,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);

private:
    class CpuRNNT_metadata {
    public:
        CpuRNNT_metadata(int mb, int T, int U, int alphabet_size, 
                         void* workspace, size_t bytes_used);
        ProbT* alphas;
        ProbT* betas;
    };

    class CpuRNNT_index {
    public:
        CpuRNNT_index(int U, int maxU, int alphabet_size);
        int U;
        int maxU;
        int alphabet_size;

        int operator()(int t, int u);
        int operator()(int t, int u, int v);
    };

    int minibatch_;
    int maxT_;
    int maxU_;
    int alphabet_size_; // Number of characters plus blank
    void* workspace_;
    int blank_;

    // Only for seperate input
    void log_softmax(const ProbT* const activations, ProbT* log_probs,
                     const int* const input_lengths, const int* const label_lengths);
    
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
CpuRNNT<ProbT>::CpuRNNT_metadata::CpuRNNT_metadata(int mb, int T, int U,
                                            int alphabet_size,
                                            void* workspace, size_t bytes_used) {
    
    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * T * U;
    std::fill(alphas, alphas + T * U, neg_inf<ProbT>());
    betas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * T * U;
    std::fill(betas, betas + T * U, neg_inf<ProbT>());
}

template<typename ProbT>
CpuRNNT<ProbT>::CpuRNNT_index::CpuRNNT_index(int U, int maxU, int alphabet_size) : 
                        U(U), maxU(maxU), alphabet_size(alphabet_size) {}

template<typename ProbT>
inline int CpuRNNT<ProbT>::CpuRNNT_index::operator()(int t, int u) {
    return t * U + u;
}

template<typename ProbT>
inline int CpuRNNT<ProbT>::CpuRNNT_index::operator()(int t, int u, int v) {
    return (t * maxU + u) * alphabet_size + v;
}

template<typename ProbT>
void
CpuRNNT<ProbT>::log_softmax(const ProbT* const activations, ProbT* log_probs,
                     const int* const input_lengths, const int* const label_lengths) {

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        for (int t = 0; t < input_lengths[mb]; ++t) {
            for (int u = 0; u <= label_lengths[mb]; ++u) {
                int col_offset = (mb * maxT_ * maxU_ + t * maxU_ + u) * alphabet_size_;
                ProbT max_activation = neg_inf<ProbT>();
                for (int v = 0; v < alphabet_size_; ++v)
                    max_activation = std::max(max_activation, activations[v + col_offset]);
                
                ProbT denom = ProbT(0.);
                for (int v = 0; v < alphabet_size_; ++v) {
                    denom += std::exp(activations[v + col_offset] - max_activation);
                }

                for (int v = 0; v < alphabet_size_; ++v) {
                    log_probs[v + col_offset] = activations[v + col_offset]
                                                - max_activation - std::log(denom);
                }
            }
        }
    }
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::cost_and_grad_kernel(const ProbT* const log_probs, ProbT* grad,
                              const int* const labels,
                              int mb, int T, int U, size_t bytes_used) {
    
    CpuRNNT_metadata rnntm(mb, T, U, alphabet_size_, workspace_, bytes_used);

    ProbT llForward = compute_alphas(log_probs, T, U, rnntm.alphas, labels);
    ProbT llBackward = compute_betas_and_grad(grad, log_probs, T, U,
                                              rnntm.alphas, 
                                              rnntm.betas,
                                              labels,
                                              llForward);

    ProbT diff = std::abs(llForward - llBackward);
    if (diff > 1e-8) {
        printf("WARNING: Forward backward likelihood mismatch %f\n", diff);
    }

    return -llForward;
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::compute_alphas(const ProbT* const log_probs, int T, int U, 
                        ProbT* alphas, const int* const labels) {

    CpuRNNT_index idx(U, maxU_, alphabet_size_);

    alphas[0] = 0;
    for (int t = 1; t < T; ++t) {
        alphas[idx(t, 0)] = alphas[idx(t-1, 0)] + log_probs[idx(t-1, 0, blank_)];
    }

    for (int u = 1; u < U; ++u) {
        alphas[idx(0, u)] = alphas[idx(0, u-1)] + log_probs[idx(0, u-1, labels[u-1])];
    }

    for (int t = 1; t < T; ++t) {
        for (int u = 1; u < U; ++u) {
            ProbT no_emit = alphas[idx(t-1, u)] + log_probs[idx(t-1, u, blank_)];
            ProbT emit = alphas[idx(t, u-1)] + log_probs[idx(t, u-1, labels[u-1])];
            alphas[idx(t, u)] = log_sum_exp<ProbT>(emit, no_emit);
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

    CpuRNNT_index idx(U, maxU_, alphabet_size_);

    betas[idx(T-1, U-1)] = log_probs[idx(T-1, U-1, blank_)];
    for (int t = T-2; t >= 0; --t) {
        betas[idx(t, U-1)] = betas[idx(t+1, U-1)] + log_probs[idx(t, U-1, blank_)];
    }

    for (int u = U-2; u >= 0; --u) {
        betas[idx(T-1, u)] = betas[idx(T-1, u+1)] + log_probs[idx(T-1, u, labels[u])];
    }

    for (int t = T-2; t >= 0; --t) {
        for (int u = U-2; u >= 0; --u) {
            ProbT no_emit = betas[idx(t+1, u)] + log_probs[idx(t, u, blank_)];
            ProbT emit = betas[idx(t, u+1)] + log_probs[idx(t, u, labels[u])];
            betas[idx(t, u)] = log_sum_exp<ProbT>(emit, no_emit);
        }
    }

    ProbT loglike = betas[0];

    // Gradients w.r.t. log probabilities
    grad[idx(T-1, U-1, blank_)] = alphas[idx(T-1, U-1)];
    for (int t = 0; t < T-1; ++t) {
        for (int u = 0; u < U; ++u) {
            grad[idx(t, u, blank_)] = alphas[idx(t, u)] + betas[idx(t+1, u)];
        }
    }

    for (int t = 0; t < T; t++) {
        for (int u = 0; u < U-1; ++u) {
            grad[idx(t, u, labels[u])] = alphas[idx(t, u)] + betas[idx(t, u+1)];
        }
    }

    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            for (int v = 0; v < alphabet_size_; ++v) {
                grad[idx(t, u, v)] = -std::exp(grad[idx(t, u, v)] - loglike);
            }
        }
    }

    return loglike;
}

template<typename ProbT>
void
CpuRNNT<ProbT>::cost_and_grad(ProbT* const log_probs,
                       ProbT* grads,
                       ProbT* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    // ProbT* log_probs = static_cast<ProbT *>(workspace_);

    // maxT_ = *std::max_element(input_lengths, input_lengths + minibatch_);
    // maxU_ = *std::max_element(label_lengths, label_lengths + minibatch_) + 1;

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(ProbT) * maxT_ * maxU_ * 2;

    // do log_softmax in mxnet
    // log_softmax(activations, log_probs, input_lengths);

#pragma omp parallel for 
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription
        const int batch_size = maxT_ * maxU_ * alphabet_size_;

        costs[mb] = cost_and_grad_kernel(grads + mb * batch_size,
                             log_probs + mb * batch_size,
                             flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0),
                             mb, T, U, mb * per_minibatch_bytes);
    }

    return RNNT_STATUS_SUCCESS;
}

template<typename ProbT>
void
CpuRNNT<ProbT>::score_forward(const ProbT* const log_probs, 
                       ProbT* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    // ProbT* log_probs = static_cast<ProbT *>(workspace_);

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(ProbT) * maxT_ * maxU_ * alphabet_size_ * 2;

    //
    // log_softmax(activations, log_probs, input_lengths);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription
        const int batch_size = maxT_ * maxU_ * alphabet_size_;

        CpuRNNT_metadata rnntm(mb, T, U, alphabet_size_, workspace_, 0);

        costs[mb] = -compute_alphas(log_probs + mb * batch_size, T, U, 
                            rnntm.alphas, 
                            flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0));
    }

    return RNNT_STATUS_SUCCESS;
}
