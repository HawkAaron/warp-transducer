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

    rnntStatus_t cost_and_grad(const ProbT* const trans_acts,
                              const ProbT* const pred_acts,
                              ProbT* trans_grad,
                              ProbT* pred_grad,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);
    
    rnntStatus_t score_forward(const ProbT* const trans_acts,
                              const ProbT* const pred_acts,
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
        CpuRNNT_index(int U, int maxU, int minibatch, int alphabet_size);
        int U;
        int maxU;
        int minibatch;
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
    int num_threads_;

    // Only for seperate input
    void log_softmax(const ProbT* const trans_acts, const ProbT* const pred_acts, ProbT* log_probs);
    
    ProbT cost_and_grad_kernel(const ProbT* const log_probs, ProbT* trans_grad, ProbT* pred_grad,
                               const int* const labels, int T, int U, size_t bytes_used);
    
    ProbT compute_alphas(const ProbT* const log_probs, int T, int U,
                         ProbT* alphas, const int* const labels);
    
    ProbT compute_betas_and_grad(ProbT* trans_grad, ProbT* pred_grad, const ProbT* const log_probs,
                                 int T, int U, ProbT* alphas, ProbT* betas,
                                 const int* const labels, ProbT logll);
};

template<typename ProbT>
CpuRNNT<ProbT>::CpuRNNT_metadata::CpuRNNT_metadata(int T, int U, void* workspace, size_t bytes_used) {
    
    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * T * U;
    std::fill(alphas, alphas + T * U, neg_inf<ProbT>());
    betas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * U;
    std::fill(betas, betas + U, neg_inf<ProbT>());
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
    return (t * maxU + u) * minibatch * alphabet_size + v;
}

template<typename ProbT>
void
CpuRNNT<ProbT>::log_softmax(const ProbT* const trans_acts, const ProbT* const pred_acts, ProbT* log_probs) {
// TBV UBV TUBV
#pragma omp parallel for
    for (int c = 0; c < minibatch_ * maxT_ * maxU_; ++c) {
        int mb = c % minibatch_;
        int tu = (c - mb) / minibatch_;
        int u = tu % maxU_;
        int t = (tu - u) / maxU_;
        int t_offset = (t * minibatch_ + mb) * alphabet_size_;
        int u_offset = (u * minibatch_ + mb) * alphabet_size_;

        int col_offset = c * alphabet_size_;
        ProbT max_activation = neg_inf<ProbT>();
        for (int v = 0; v < alphabet_size_; ++v)
            max_activation = std::max(max_activation, trans_acts[v + t_offset] + pred_acts[v + u_offset]);
        
        ProbT denom = ProbT(0.);
        for (int v = 0; v < alphabet_size_; ++v) {
            denom += std::exp(trans_acts[v + t_offset] + pred_acts[v + u_offset] - max_activation);
        }

        // TODO using label store blank and label 's probability
        for (int v = 0; v < alphabet_size_; ++v) {
            log_probs[v + col_offset] = trans_acts[v + t_offset] + pred_acts[v + u_offset]
                                        - max_activation - std::log(denom);
        }
    }

    printf("log softmax\n");
    for (int mb = 0; mb < minibatch_; ++mb) {
        printf("(%d, ...)\n", mb);
        for (int t = 0; t < maxT_; ++t) {
            for (int u = 0; u < maxU_; u++) {
                for (int v = 0; v < alphabet_size_; ++v) {
                    printf("%f ", log_probs[idx(t, u, v)])
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::cost_and_grad_kernel(const ProbT* const log_probs, 
                              ProbT* trans_grad, ProbT* pred_grad,
                              const int* const labels,
                              int T, int U, size_t bytes_used) {
    
    CpuRNNT_metadata rnntm(T, U, workspace_, bytes_used);

    ProbT llForward = compute_alphas(log_probs, T, U, rnntm.alphas, labels);
    ProbT llBackward = compute_betas_and_grad(trans_grad, pred_grad, 
                                              log_probs, T, U,
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

    CpuRNNT_index idx(U, maxU_, minibatch_, alphabet_size_);

    alphas[0] = 0;
    // TODO using one loop to optimize memory continuous fetching
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
CpuRNNT<ProbT>::compute_betas_and_grad(ProbT* trans_grad, ProbT* pred_grad, 
                                const ProbT* const log_probs,
                                int T, int U, ProbT* alphas, ProbT* betas,
                                const int* const labels, ProbT logll) {

    CpuRNNT_index idx(U, maxU_, minibatch_, alphabet_size_);

    ProbT beta_t, beta_u, beta_ = neg_inf<ProbT>();

    std::fill(betas, betas + U, neg_inf<ProbT>()); // right edge gradient
    betas[U-1] = 0; // last point gradient

    for (int t = T-1; t >= 0; --t) {
        int t_offset = t * minibatch_ * alphabet_size_;
        for (int u = U-1; u >= 0; --u) {
            int p_offset = u * minibatch_ * alphabet_size_;
            beta_t = beta_; // assign current to top
            beta_u = betas[u]; // assign last beta_u to right
            if (u == U-1) {
                beta_t = neg_inf<ProbT>(); // top edge gradient
            }
            if (t == T-1 && u == U-1) {
                beta_ = log_probs[idx(T-1, U-1, blank_)];
            }
            if (t < T-1) {
                beta_ = beta_u + log_probs[idx(t, u, blank_)];
            }
            if (u < U-1) {
                beta_ = log_sum_exp<ProbT>(beta_, log_probs[idx(t, u, labels[u])]);
            }
            betas[u] = beta_;
            // gradient
            for (int v = 0; v < alphabet_size_; ++v) {
                ProbT logpk = log_probs[idx(t, u, v)];
                ProbT grad1 = std::exp(alphas[idx(t, u)] + beta_ + logpk - logll);
                ProbT grad2 = alphas[idx(t, u)] + logpk - logll;
                if (v == blank_) {
                    grad2 += beta_u;
                }
                if (v == labels[u]) {
                    grad2 += beta_t;
                }
                grad1 -= std::exp(grad2);
                trans_grad[t_offset + v] += grad1;
                pred_grad[p_offset + v] += grad1;
            }
        }
    }

    printf("compute_betas_and_grad beta %f\n", beta_);
    return beta_; // the last beta_ is loglike
}

template<typename ProbT>
rnntStatus_t
CpuRNNT<ProbT>::cost_and_grad(const ProbT* const trans_acts,
                       const ProbT* const pred_acts,
                       ProbT* trans_grads,
                       ProbT* pred_grads,
                       ProbT* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    ProbT* log_probs = static_cast<ProbT *>(workspace_);

    size_t bytes_used = sizeof(ProbT) * minibatch_ * maxT_ * maxU_ * alphabet_size_;
    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(ProbT) * (maxT_ + 1) * maxU_;

    log_softmax(trans_acts, pred_acts, log_probs);

    // zero all grads
    std::fill(trans_grads, trans_grads + minibatch_ * maxT_ * alphabet_size_, 0);
    std::fill(pred_grads, pred_grads + minibatch_ * maxU_ * alphabet_size_, 0);

#pragma omp parallel for 
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription

        costs[mb] = cost_and_grad_kernel(log_probs + mb * alphabet_size_,
                             trans_grads + mb * alphabet_size_,
                             pred_grads + mb * alphabet_size_,
                             flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0),
                             T, U, bytes_used + mb * per_minibatch_bytes);
    }

    return RNNT_STATUS_SUCCESS;
}

template<typename ProbT>
rnntStatus_t
CpuRNNT<ProbT>::score_forward(const ProbT* const trans_acts, 
                       const ProbT* pred_acts,
                       ProbT* costs,
                       const int* const flat_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    ProbT* log_probs = static_cast<ProbT *>(workspace_);

    size_t bytes_used = sizeof(ProbT) * minibatch_ * maxT_ * maxU_ * alphabet_size_;
    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(ProbT) * (maxT_ + 1) * maxU_;

    log_softmax(trans_acts, pred_acts, log_probs);

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription

        CpuRNNT_metadata rnntm(T, U, workspace_, bytes_used + mb * per_minibatch_bytes);

        costs[mb] = -compute_alphas(log_probs + mb * alphabet_size_, T, U, 
                            rnntm.alphas, 
                            flat_labels + std::accumulate(label_lengths, label_lengths + mb, 0));
    }

    return RNNT_STATUS_SUCCESS;
}
