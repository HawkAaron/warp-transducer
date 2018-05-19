#include <cmath>
#include <cstdlib>
#include <random>
#include <tuple>
#include <vector>

#include <chrono>

#include <iostream>

#include <rnnt.h>

#include "test.h"

bool run_test(int B, int T, int L, int A, int num_threads) {
    std::mt19937 gen(2);

    auto start = std::chrono::high_resolution_clock::now();
    int trans_len = B * T * A;
    int pred_len = B * (L + 1) * A;
    float * trans_acts = genActs(trans_len);
    float * pred_acts = genActs(pred_len);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "genActs elapsed time: " << elapsed.count() * 1000 << " ms\n";

    std::vector<std::vector<int>> labels;
    std::vector<int> sizes;

    for (int mb = 0; mb < B; ++mb) {
        labels.push_back(genLabels(A, L));
        sizes.push_back(T);
    }

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }

    std::vector<float> costs(B);

    float * trans_grads = new float[trans_len];
    float * pred_grads = new float[pred_len];

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L + 1;
    options.blank_label = 0;
    options.loc = RNNT_GPU;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    options.stream = stream;

    float* trans_acts_gpu;
    float* pred_acts_gpu;
    cudaMalloc(&trans_acts_gpu, trans_len * sizeof(float));
    cudaMalloc(&pred_acts_gpu, pred_len * sizeof(float));
    cudaMemcpyAsync(trans_acts_gpu, trans_acts, trans_len * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(pred_acts_gpu, pred_acts, pred_len * sizeof(float), cudaMemcpyHostToDevice, stream);

    float* trans_grads_gpu;
    float* pred_grads_gpu;
    cudaMalloc(&trans_grads_gpu, trans_len * sizeof(float));
    cudaMalloc(&pred_grads_gpu, pred_len * sizeof(float));


    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, L+1, B, A,
                                     true,
                                     &gpu_alloc_bytes),
                    "Error: get_workspace_size in run_test");
    
    void* rnnt_gpu_workspace;

    // average time
    std::vector<float> time;
    for (int i = 0; i < 10; ++i) {
        start = std::chrono::high_resolution_clock::now();
        cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
        throw_on_error(compute_rnnt_loss(trans_acts_gpu, pred_acts_gpu, 
                                        trans_grads_gpu, pred_grads_gpu,
                                        flat_labels.data(), label_lengths.data(),
                                        sizes.data(),
                                        A, B,
                                        costs.data(),
                                        rnnt_gpu_workspace,
                                        options),
                        "Error: compute_rnnt_loss (0) in run_test");
        cudaFree(rnnt_gpu_workspace);
        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;
        time.push_back(elapsed.count() * 1000);
        std::cout << "compute_rnnt_loss elapsed time: " << elapsed.count() * 1000 << " ms\n";
    }

    float sum = 0;
    for (int i = 0; i < 10; ++i) {
        sum += time[i];
    }
    std::cout << "average 10 time cost: " << sum / time.size() << " ms\n";

    float cost = std::accumulate(costs.begin(), costs.end(), 0.);
    cudaFree(trans_acts_gpu);
    cudaFree(pred_acts_gpu);
    cudaFree(trans_grads_gpu);
    cudaFree(pred_grads_gpu);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Arguments: <Batch size> <Time step> <Label length> <Alphabet size>\n";
        return 1;
    }

    int B = atoi(argv[1]);
    int T = atoi(argv[2]);
    int L = atoi(argv[3]);
    int A = atoi(argv[4]);
    std::cout << "Arguments: " \
                << "\nBatch size: " << B \
                << "\nTime step: " << T \
                << "\nLabel length: " << L \
                << "\nAlphabet size: " << A \
                << std::endl;
    
    int num_threads = 1;
    if (argc >= 6) {
        num_threads = atoi(argv[5]);
        std::cout << "Num threads: " << num_threads << std::endl;
    }

    run_test(B, T, L, A, num_threads);
}