#pragma once

#include "rnnt_helper.h"

template<typename T>
inline T logp(const T* const denom, const T* const trans_acts, const T* const pred_acts, const int maxT, const int maxU, const int alphabet_size, int mb, int t, int u, int v) {
    T ret = denom[(mb * maxT + t) * maxU + u] + trans_acts[(mb * maxT + t) * alphabet_size + v] + pred_acts[(mb * maxU + u) * alphabet_size + v];
    // printf("%f ", ret);
    return ret;
}

template<typename Tp>
void compute_alphas_cpu(const Tp* const trans_acts, const Tp* const pred_acts, const Tp* const denom, Tp* malphas, Tp* llForward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    for (int tid = 0; tid < minibatch; tid++) {
        const int T = xlen[tid];
        const int U = ylen[tid] + 1;
        const int* labels = mlabels + tid * (maxU - 1); // mb label start point
        Tp* alphas = malphas + tid * maxT * maxU; // gpu is different with cpu
        alphas[0] = 0;

        for (int t = 0; t < T; ++t) {
            for (int u = 0; u < U; ++u) {
                if (u == 0 && t > 0)
                    alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, tid, t-1, 0, blank_);
                if (t == 0 && u > 0)
                    alphas[u] = alphas[u-1] + logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, tid, 0, u-1, labels[u-1]);
                if (t > 0 && u > 0) {
                    Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, tid, t-1, u, blank_);
                    Tp emit = alphas[t * maxU + u-1] + logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, tid, t, u-1, labels[u-1]);
                    alphas[t * maxU + u] = log_sum_exp<Tp>(emit, no_emit);
                }
                // if (u > 0) printf("%d ", labels[u-1]);
                // printf("%f ", alphas[t * maxU + u]);
            }
            // printf("\n");
        }

        Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_);
        llForward[tid] = loglike;
        // printf("end batch %d\n", tid);
    }
}

template<typename Tp>
void compute_betas_cpu(const Tp* const trans_acts, const Tp* const pred_acts, const Tp* const denom, Tp* mbetas, Tp* llBackward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    for (int tid = 0; tid < minibatch; tid++) {
        const int T = xlen[tid];
        const int U = ylen[tid] + 1;
        const int* labels = mlabels + tid * (maxU - 1);
        Tp* betas = mbetas + tid * maxT * maxU;
        betas[(T-1) * maxU + U-1] = logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_);

        // printf("betas\n");
        for (int t = T-1; t >=0; --t) {
            for (int u = U-1; u >= 0; --u) {
                if (u == U-1 && t < T-1)
                    betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, tid, t, U-1, blank_);
                if (t == T-1 && u < U-1)
                    betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, tid, T-1, u, labels[u]);
                if (t < T-1 && u < U-1) {
                    Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, tid, t, u, blank_);
                    Tp emit = betas[t * maxU + u+1] + logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, tid, t, u, labels[u]);
                    betas[t * maxU + u] = log_sum_exp<Tp>(emit, no_emit);
                }
                // if (u < U-1) printf("%d ", labels[u]);
                // printf("%f ", betas[t * maxU + u]);
            }
            // printf("\n");
        }

        llBackward[tid] = betas[0];
    }
}

template<int NT, typename Tp>
void compute_grad_cpu(Tp* trans_grad, Tp* pred_grad, const Tp* const trans_acts, const Tp* const pred_acts, const Tp* const denom, Tp* malphas, Tp* mbetas, const Tp* const logll, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    for (int col = 0; col < minibatch * maxT * maxU; col++) {
        for (int tid = 0; tid < NT; tid++) {
            int idx = tid;

            int u = col % maxU;
            int bt = (col - u) / maxU;
            int t = bt % maxT;
            int mb = (bt - t) / maxT;

            const int T = xlen[mb];
            const int U = ylen[mb] + 1;
            const int* labels = mlabels + mb * (maxU - 1);
            const int offset = mb * maxT * maxU;
            Tp* alphas = malphas + offset;
            Tp* betas = mbetas + offset;

            while (idx < alphabet_size) {
                Tp logpk = logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, mb, t, u, idx);
                Tp grad = exp(alphas[t * maxU + u] + betas[t * maxU + u] + logpk - logll[mb]);
                // grad to last blank transition
                if (idx == blank_ && t == T-1 && u == U-1) grad -= 1;
                if (idx == blank_ && t < T-1) {
                    grad -= exp(alphas[t * maxU + u] + logpk - logll[mb] + betas[(t+1) * maxU + u]);
                }
                if (idx == labels[u] && u < U-1) {
                    grad -= exp(alphas[t * maxU + u] + logpk - logll[mb] + betas[t * maxU + u+1]);
                }
                trans_grad[(mb * maxT + t) * alphabet_size + idx] += grad;
                pred_grad[(mb * maxU + u) * alphabet_size + idx] += grad;

                idx += NT;
            }
        }
    }

    // for (int mb = 0; mb < minibatch; ++mb) {
    //     const int* labels = mlabels + mb * (maxU - 1);
    //     int offset = mb * maxT * maxU;
    //     Tp* alphas = malphas + offset;
    //     Tp* betas = mbetas + offset;
    //     int T = xlen[mb];
    //     int U = ylen[mb] + 1;
    //     for (int t = 0; t < maxT; ++t) {
    //         int t_offset = (mb * maxT + t) * alphabet_size;
    //         for (int u = 0; u < maxU; ++u) {
    //             int u_offset = (mb * maxU + u) * alphabet_size;
    //             for (int v = 0; v < alphabet_size; ++v) {
    //                 Tp lgpk = logp(denom, trans_acts, pred_acts, maxT, maxU, alphabet_size, mb, t, u, v);
    //                 Tp grad = exp(alphas[t * maxU + u] + betas[t * maxU + u] + lgpk - logll[mb]);
    //                 // grad to last blank transition
    //                 if (v == blank_ && t == T-1 && u == U-1) grad -= 1;
    //                 if (v == blank_ && t < T-1) {
    //                     grad -= exp(alphas[t * maxU + u] + lgpk - logll[mb] + betas[(t+1) * maxU + u]);
    //                 }
    //                 if (v == labels[u] && u < U-1) {
    //                     grad -= exp(alphas[t * maxU + u] + lgpk - logll[mb] + betas[t * maxU + u+1]);
    //                 }
    //                 trans_grad[t_offset + v] += grad;
    //                 pred_grad[u_offset + v] += grad;
    //             }
    //         }
    //     }
    // }
}