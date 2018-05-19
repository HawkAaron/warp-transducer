#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include <iostream>

#include <rnnt.h>

#include "test.h"

bool small_test() {
    const int B = 1;
    const int alphabet_size = 5;
    const int T = 2;
    const int U = 3;

    std::vector<float> trans_acts = {0.41949576139450073 ,0.31693804264068604 ,0.7670461535453796 ,0.8889783024787903 ,0.5395874977111816 ,
                                    0.24531936645507812 ,0.8488685488700867 ,0.5839998722076416 ,0.04401040077209473 ,0.3734890818595886 };

    std::vector<float> pred_acts = {0.6518162488937378 ,0.9520350098609924 ,0.40872979164123535 ,0.42321133613586426 ,0.4537124037742615 ,
                                    0.49220818281173706 ,0.5218477845191956 ,0.43106764554977417 ,0.6558153033256531 ,0.20882707834243774 ,
                                    0.3555586338043213 ,0.24547159671783447 ,0.0288771390914917 ,0.42015254497528076 ,0.9517340660095215 };

    float expected_score = 5.3452;

    std::vector<int> labels = {1, 2};
    std::vector<int> label_lengths = {2};

    std::vector<int> lengths;
    lengths.push_back(T);

    float score;

    rnntOptions options{};
    options.maxT = T;
    options.maxU = U;
    options.loc = RNNT_CPU;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, U, B, alphabet_size,
                                      false,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(trans_acts.data(), pred_acts.data(),
                                    NULL, NULL,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    &score,
                                    rnnt_cpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss in small_test");

    free(rnnt_cpu_workspace);
    const float eps = 1e-4;

    const float lb = expected_score - eps;
    const float ub = expected_score + eps;

    return (score > lb && score < ub);
}

bool options_test() {
    const int alphabet_size = 6;
    const int T = 4;
    const int L = 3;
    const int minibatch = 2;

    std::vector<float> trans_acts = {0.20836473 , 0.68488914 , 0.8508703 , 0.5761989 , 0.19992691 , 0.8066367 , 
                                    0.9547191 , 0.39503795 , 0.91791826 , 0.66354334 , 0.52238625 , 0.30651063 , 
                                    0.51816165 , 0.83243364 , 0.29219377 , 0.35955018 , 0.6904012 , 0.85133505 , 
                                    0.87253755 , 0.06845909 , 0.74267465 , 0.7473852 , 0.6735857 , 0.814946 , 
                                    0.72159135 , 0.27252442 , 0.20181221 , 0.7149978 , 0.7996553 , 0.594097 , 
                                    0.98958284 , 0.7198998 , 0.32019693 , 0.49187315 , 0.82729864 , 0.72087383 , 
                                    0.5865227 , 0.84655076 , 0.73006225 , 0.24205732 , 0.2466054 , 0.93103373 , 
                                    0.6253804 , 0.56404036 , 0.59297657 , 0.6260772 , 0.23223883 , 0.041093946};
    std::vector<float> pred_acts = {0.061166406 , 0.14563453 , 0.563884 , 0.66322905 , 0.19838423 , 0.18207806 , 
                                    0.9041121 , 0.31825322 , 0.10209769 , 0.44423354 , 0.7338143 , 0.22434187 , 
                                    0.7007454 , 0.65526026 , 0.5205059 , 0.30149776 , 0.6051819 , 0.1901899 , 
                                    0.6904842 , 0.30375922 , 0.61894506 , 0.032821894 , 0.75227857 , 0.8265933 , 
                                    0.79737663 , 0.86086124 , 0.4400267 , 0.8985075 , 0.3717013 , 0.9338418 , 
                                    0.91288275 , 0.6805384 , 0.019013822 , 0.84054446 , 0.52986646 , 0.27262968};

    std::vector<float> expected_trans_grads = {0.18298208713531494 , -0.1863221824169159 , 0.24029867351055145 , 0.31508156657218933 , 0.17985235154628754 , -0.7318925261497498 , 
                                                0.3619690537452698 , -0.06026348099112511 , 0.08730498701334 , 0.25879740715026855 , 0.22078825533390045 , -0.8685961961746216 , 
                                                0.27959010004997253 , 0.03639250993728638 , -0.061875589191913605 , 0.19594523310661316 , 0.30305129289627075 , -0.753103494644165 , 
                                                0.385039746761322 , 0.06576273590326309 , -0.18127907812595367 , 0.2354261875152588 , 0.28227531909942627 , -0.7872248888015747 , 
                                                0.26263949275016785 , -0.10713391751050949 , 0.13813626766204834 , 0.16261409223079681 , 0.2808478772640228 , -0.7371038794517517 , 
                                                0.3360159993171692 , -0.1639198213815689 , 0.1387038677930832 , 0.1545054018497467 , 0.2540736496448517 , -0.7193790674209595 , 
                                                0.29537156224250793 , -0.274471253156662 , 0.2312944084405899 , 0.18443721532821655 , 0.16433767974376678 , -0.6009695529937744 , 
                                                0.4264937937259674 , -0.4397970736026764 , 0.22451627254486084 , 0.4020277261734009 , 0.20442542433738708 , -0.8176661133766174};
    std::vector<float> expected_pred_grads = {0.2336210012435913 , -0.7242842316627502 , 0.49153390526771545 , 0.4382175803184509 , 0.2350836545228958 , -0.6741719245910645 , 
                                            0.5180250406265259 , 0.26190635561943054 , -0.7572638988494873 , 0.29955601692199707 , 0.3859791159629822 , -0.7082027196884155 , 
                                            0.4579349160194397 , 0.31794747710227966 , 0.35017895698547363 , 0.26747676730155945 , 0.3649044632911682 , -1.7584426403045654 , 
                                            0.5328136086463928 , -0.7057985663414001 , 0.3393358588218689 , 0.2234761267900467 , 0.5142948627471924 , -0.9041219353675842 , 
                                            0.4035489857196808 , -0.5864231586456299 , 0.24028921127319336 , 0.3576064705848694 , 0.20584842562675476 , -0.6208699345588684 , 
                                            0.38415825366973877 , 0.30689966678619385 , 0.15302574634552002 , 0.3225018382072449 , 0.18354131281375885 , -1.35012686252594};

    // Calculate the expected scores analytically
    std::vector<double> expected_scores(2);
    expected_scores[0] = 9.8610;
    expected_scores[1] = 8.7003;

    std::vector<int> labels = {1, 2, 1, 1};

    std::vector<int> label_lengths = {2, 2};

    std::vector<int> lengths = {4, 4};

    std::vector<float> trans_grads(alphabet_size * T * minibatch);
    std::vector<float> pred_grads(alphabet_size * L * minibatch);

    std::vector<float> scores(2);

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L;
    options.loc = RNNT_CPU;
    options.num_threads = 1;
    options.blank_label = 5;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, L, minibatch,
                                      alphabet_size, false,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in options_test");

    void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(trans_acts.data(), pred_acts.data(),
                                    trans_grads.data(), pred_grads.data(),
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    scores.data(),
                                    rnnt_cpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss in options_test");

    free(rnnt_cpu_workspace);

    const double eps = 1e-4;

    bool result = true;
    // transcription activations gradient check
    for (int i = 0; i < trans_grads.size(); i++) {
        const double lb = expected_trans_grads[i] - eps;
        const double ub = expected_trans_grads[i] + eps;
        if (!(trans_grads[i] > lb && trans_grads[i] < ub)) {
            std::cerr << "grad mismatch in options_test"
                      << " expected grad: " << expected_trans_grads[i]
                      << " calculated score: " << trans_grads[i]
                      << " !(" << lb << " < " << trans_grads[i]
                      << " < " << ub << ")" << std::endl;
            result = false;
        }
    }

    // prediction activations gradient check
    for (int i = 0; i < pred_grads.size(); i++) {
        const double lb = expected_pred_grads[i] - eps;
        const double ub = expected_pred_grads[i] + eps;
        if (!(pred_grads[i] > lb && pred_grads[i] < ub)) {
            std::cerr << "grad mismatch in options_test"
                      << " expected grad: " << expected_pred_grads[i]
                      << " calculated score: " << pred_grads[i]
                      << " !(" << lb << " < " << pred_grads[i]
                      << " < " << ub << ")" << std::endl;
            result = false;
        }
    }

    for (int i = 0; i < 2; i++) {
        const double lb = expected_scores[i] - eps;
        const double ub = expected_scores[i] + eps;
        if (!(scores[i] > lb && scores[i] < ub)) {
            std::cerr << "score mismatch in options_test"
                      << " expected score: " << expected_scores[i]
                      << " calculated score: " << scores[i]
                      << " !(" << lb << " < " << scores[i]
                      << " < " << ub << ")" << std::endl;
            result = false;
        }
    }
    return result;
}

bool inf_test() {
    const int alphabet_size = 15;
    const int T = 50;
    const int L = 10;
    const int minibatch = 1;

    std::vector<int> labels = genLabels(alphabet_size, L-1);
    labels[0] = 2;
    std::vector<int> label_lengths = {L-1};

    std::vector<float> trans_acts(alphabet_size * T * minibatch);
    std::vector<float> pred_acts(alphabet_size * L * minibatch);
    genActs(trans_acts);
    genActs(pred_acts);

    std::vector<int> sizes;
    sizes.push_back(T);

    std::vector<float> trans_grads(alphabet_size * T * minibatch);
    std::vector<float> pred_grads(alphabet_size * L * minibatch);

    float cost;

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L;
    options.loc = RNNT_CPU;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, L, minibatch,
                                      alphabet_size, false,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in inf_test");

    void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(trans_acts.data(), pred_acts.data(),
                                    trans_grads.data(), pred_grads.data(),
                                    labels.data(), label_lengths.data(),
                                    sizes.data(),
                                    alphabet_size,
                                    sizes.size(),
                                    &cost,
                                    rnnt_cpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss in inf_test");

    free(rnnt_cpu_workspace);

    bool status = true;
    status &= !std::isinf(cost);

    for (int i = 0; i < alphabet_size * T * minibatch; ++i)
        status &= !std::isnan(trans_grads[i]);
    for (int i = 0; i < alphabet_size * L; ++i)
        status &= !std::isnan(pred_grads[i]);

    return status;
}

float numeric_grad(std::vector<float>& acts, std::vector<float>& trans_acts, std::vector<float>& pred_acts,
                std::vector<int>& flat_labels, std::vector<int>& label_lengths,
                std::vector<int> sizes, int alphabet_size, int minibatch, 
                void* rnnt_cpu_workspace, rnntOptions& options, std::vector<float>& num_grad) {

    float epsilon = 1e-2;

    for (int i = 0; i < num_grad.size(); ++i) {

        std::vector<float> costsP1(minibatch);
        std::vector<float> costsP2(minibatch);

        acts[i] += epsilon;
        throw_on_error(compute_rnnt_loss(trans_acts.data(), pred_acts.data(),
                                        NULL, NULL,
                                        flat_labels.data(), label_lengths.data(),
                                        sizes.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP1.data(),
                                        rnnt_cpu_workspace,
                                        options),
                       "Error: compute_rnnt_loss (1) in grad_check");

        acts[i] -= 2 * epsilon;
        throw_on_error(compute_rnnt_loss(trans_acts.data(), pred_acts.data(),
                                        NULL, NULL,
                                        flat_labels.data(), label_lengths.data(),
                                        sizes.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP2.data(),
                                        rnnt_cpu_workspace,
                                        options),
                       "Error: compute_rnnt_loss (2) in grad_check");

        float costP1 = std::accumulate(costsP1.begin(), costsP1.end(), 0.);
        float costP2 = std::accumulate(costsP2.begin(), costsP2.end(), 0.);

        acts[i] += epsilon;
        num_grad[i] = (costP1 - costP2) / (2 * epsilon);
    }
}

bool grad_check(int T, int L, int alphabet_size,
                  std::vector<float>& trans_acts,
                  std::vector<float>& pred_acts,
                  const std::vector<std::vector<int>>& labels,
                  const std::vector<int>& sizes, float tol) {

    const int minibatch = labels.size();

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }

    std::vector<float> costs(minibatch);

    std::vector<float> trans_grads(trans_acts.size());
    std::vector<float> pred_grads(pred_acts.size());

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L;
    options.loc = RNNT_CPU;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, L, sizes.size(),
                                      alphabet_size, false,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in grad_check");

    void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(trans_acts.data(), pred_acts.data(),
                                    trans_grads.data(), pred_grads.data(),
                                    flat_labels.data(), label_lengths.data(),
                                    sizes.data(),
                                    alphabet_size,
                                    minibatch,
                                    costs.data(),
                                    rnnt_cpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss (0) in grad_check");

    float cost = std::accumulate(costs.begin(), costs.end(), 0.);

    std::vector<float> num_trans_grad(trans_grads.size());
    std::vector<float> num_pred_grad(pred_grads.size());

    //perform 2nd order central differencing
    numeric_grad(trans_acts, trans_acts, pred_acts, flat_labels, label_lengths, sizes,
            alphabet_size, minibatch, rnnt_cpu_workspace, options, num_trans_grad);
    numeric_grad(pred_acts, trans_acts, pred_acts, flat_labels, label_lengths, sizes,
            alphabet_size, minibatch, rnnt_cpu_workspace, options, num_pred_grad);

    free(rnnt_cpu_workspace);

    float diff_trans = rel_diff(trans_grads, num_trans_grad);
    float diff_pred = rel_diff(pred_grads, num_pred_grad);

    return (diff_trans < tol) && (diff_pred < tol);
}

bool run_tests() {
    std::vector<std::tuple<int, int, int, int, float>> problem_sizes =
       {std::make_tuple(20, 50, 15, 1, 1e-4),
        std::make_tuple(5, 10, 5, 65, 1e-4)
       };

    std::mt19937 gen(2);

    bool status = true;
    for (auto problem : problem_sizes) {
        int alphabet_size, T, L, minibatch;
        float tol;
        std::tie(alphabet_size, T, L, minibatch, tol) = problem;

        std::vector<float> trans_acts(alphabet_size * T * minibatch);
        std::vector<float> pred_acts(alphabet_size * L * minibatch);
        genActs(trans_acts);
        genActs(pred_acts);

        std::vector<std::vector<int>> labels;
        std::vector<int> sizes;
        for (int mb = 0; mb < minibatch; ++mb) {
            int actual_length = L - 1;
            labels.push_back(genLabels(alphabet_size, actual_length));
            sizes.push_back(T);
        }

        status &= grad_check(T, L, alphabet_size, trans_acts, pred_acts, labels, sizes, tol);
    }

    return status;
}

int main(void) {
    if (get_warprnnt_version() != 1) {
        std::cerr << "Invalid Warp-transducer version." << std::endl;
        return 1;
    }

    std::cout << "Running CPU tests" << std::endl;

    bool status = true;
    status &= small_test();
    printf("finish small_test %d\n", status);
    status &= options_test();
    printf("finish options_test %d\n", status);
    status &= inf_test();
    printf("finish inf_test %d\n", status);
    status &= run_tests();
    printf("finished %d\n", status);

    if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}
