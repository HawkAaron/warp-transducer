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

    std::vector<float> activations = {0.1, 0.6, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.6, 0.1, 0.1,
                                      0.1, 0.1, 0.2, 0.8, 0.1,
                                      0.1, 0.6, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.2, 0.1, 0.1,
                                      0.7, 0.1, 0.2, 0.1, 0.1};

    // Calculate the score analytically
    std::vector<float> log_probs(activations.size());
    softmax(activations.data(), alphabet_size, T, log_probs.data(), true);
    float expected_score = 4.495666;

    std::vector<int> labels = {1, 2};
    std::vector<int> label_lengths = {2};

    std::vector<int> lengths;
    lengths.push_back(T);

    float score;

    rnntOptions options{};
    options.loc = RNNT_CPU;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(lengths[0], label_lengths[0],
                                      lengths.size(), false,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(log_probs.data(), NULL,
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    T, U,
                                    &score,
                                    rnnt_cpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in small_test");

    free(rnnt_cpu_workspace);
    const float eps = 1e-6;

    const float lb = expected_score - eps;
    const float ub = expected_score + eps;

    return (score > lb && score < ub);
}

int offset(int t, int n, int a) {
    constexpr int minibatch = 2;
    constexpr int alphabet_size = 6;
    return (t * minibatch + n) * alphabet_size + a;
}

bool options_test() {
    const int alphabet_size = 6;
    const int T = 5;
    const int minibatch = 2;

    std::vector<float> activations =
            {0.06535690384862791, 0.7875301411923206, 0.08159176605666074,
              0.5297155426466327, 0.7506749639230854, 0.7541348379087998,
              0.6097641124736383, 0.8681404965673826, 0.6225318186056529,

             0.6685222872103057, 0.8580392805336061, 0.16453892311765583,
              0.989779515236694, 0.944298460961015, 0.6031678586829663,
              0.9467833543605416, 0.666202507295747, 0.28688179752461884,

             0.09418426230195986, 0.3666735970751962, 0.736168049462793,
              0.1666804425271342, 0.7141542198635192, 0.3993997272216727,
              0.5359823524146038, 0.29182076440286386, 0.6126422611507932,

             0.3242405528768486, 0.8007644367291621, 0.5241057606558068,
              0.779194617063042, 0.18331417220174862, 0.113745182072432,
              0.24022162381327106, 0.3394695622533106, 0.1341595066017014,

            0.5055615569388828, 0.051597282072282646, 0.6402903936686337,
              0.43073311517251, 0.8294731834714112, 0.1774668847323424,
              0.3207001991262245, 0.04288308912457006, 0.30280282975568984,

             0.6751777088333762, 0.569537369330242, 0.5584738347504452,
              0.08313242153985256, 0.06016544344162322, 0.10795752845152584,
              0.7486153608562472, 0.943918041459349, 0.4863558118797222,

             0.4181986264486809, 0.6524078485043804, 0.024242983423721887,
              0.13458171554507403, 0.3663418070512402, 0.2958297395361563,
              0.9236695822497084, 0.6899291482654177, 0.7418981733448822,

             0.25000547599982104, 0.6034295486281007, 0.9872887878887768,
              0.5926057265215715, 0.8846724004467684, 0.5434495396894328,
              0.6607698886038497, 0.3771277082495921, 0.3580209022231813};

    std::vector<float> expected_grads = // from tensorflow
          {-0.4322264564338117, -0.5677735435661883, 0.0,
              -0.36565009313836844, 0.0, -0.20212345042782007,
              -0.20212345042782007, 0.0, 0.0,

             -0.16521672442463506, -0.2670097320091765, 0.0,
              -0.3943653886107811, 0.0, -0.2382944365367636,
              -0.44041788696458367, 0.0, 0.0,

             -0.052129794015740985, -0.11308693040889405, 0.0,
              -0.18313786985332664, 0.0, -0.3243144491663483,
              -0.7647323361309323, 0.0, 0.0,

             0.0, -0.052129794015740985, 0.0,
              0.0, 0.0, -0.23526766386906767,
              -1.0, 0.0, 0.0,

            -0.7161424128232795, -0.2838575871767207, 0.0,
              -0.18382932237365335, -0.10002826480306751, 0.0,
              -0.10002826480306751, 0.0, 0.0,

             -0.41121794618117213, -0.3049244666421072, 0.0,
              -0.3295759402552584, -0.15917784876050195, 0.0,
              -0.2592061135635692, 0.0, 0.0,

             -0.11607642141651396, -0.29514152476465827, 0.0,
              -0.2865333615432337, -0.3381841034766833, 0.0,
              -0.5973902170402529, 0.0, 0.0,

             0.0, -0.11607642141651396, 0.0,
              0.0, -0.4026097829597475, 0.0,
              -1.0, 0.0, 0.0};


    // Calculate the expected scores analytically
    std::vector<double> expected_scores(2);
    auto& a = activations;
    expected_scores[0] = 4.2806528590890736;
    expected_scores[1] = 3.9384369822503591;

    // now take the log to account for the softmax
    for (auto& a : activations) {
        a = std::log(a);
    }

    std::vector<int> labels = {1, 2, 1, 1};

    std::vector<int> label_lengths = {2, 2};

    std::vector<int> lengths = {4, 4};

    std::vector<float> grads(alphabet_size * T * minibatch);

    std::vector<float> scores(2);

    ctcOptions options{};
    options.loc = CTC_CPU;
    options.num_threads = 1;
    options.blank_label = 5;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), lengths.data(),
                                      alphabet_size, lengths.size(), options,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in options_test");

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_ctc_loss(activations.data(), grads.data(),
                                    labels.data(), label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    lengths.size(),
                                    scores.data(),
                                    ctc_cpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in options_test");

    free(ctc_cpu_workspace);

    const double eps = 1e-4;

    bool result = true;
    for (int i = 0; i < grads.size(); i++) {
        const double lb = expected_grads[i] - eps;
        const double ub = expected_grads[i] + eps;
        if (!(grads[i] > lb && grads[i] < ub)) {
            std::cerr << "grad mismatch in options_test"
                      << " expected grad: " << expected_grads[i]
                      << " calculated score: " << grads[i]
                      << " !(" << lb << " < " << grads[i]
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

    std::vector<int> labels = genLabels(alphabet_size, L);
    labels[0] = 2;
    std::vector<int> label_lengths = {L};

    std::vector<float> acts = genActs(alphabet_size * T * minibatch);

    for (int i = 0; i < T; ++i)
        acts[alphabet_size * i + 2] = -1e30;

    std::vector<int> sizes;
    sizes.push_back(T);

    std::vector<float> grads(alphabet_size * T);

    float cost;

    ctcOptions options{};
    options.loc = CTC_CPU;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), sizes.data(),
                                      alphabet_size, sizes.size(), options,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in inf_test");

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_ctc_loss(acts.data(), grads.data(),
                                    labels.data(), label_lengths.data(),
                                    sizes.data(),
                                    alphabet_size,
                                    sizes.size(),
                                    &cost,
                                    ctc_cpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in inf_test");

    free(ctc_cpu_workspace);

    bool status = true;
    status &= std::isinf(cost);

    for (int i = 0; i < alphabet_size * T; ++i)
        status &= !std::isnan(grads[i]);

    return status;
}

float grad_check(int T, int alphabet_size,
                  std::vector<float>& acts,
                  const std::vector<std::vector<int>>& labels,
                  const std::vector<int>& sizes) {

    float epsilon = 1e-2;

    const int minibatch = labels.size();

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }

    std::vector<float> costs(minibatch);

    std::vector<float> grads(acts.size());

    ctcOptions options{};
    options.loc = CTC_CPU;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), sizes.data(),
                                      alphabet_size, sizes.size(), options,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in grad_check");

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_ctc_loss(acts.data(), grads.data(),
                                    flat_labels.data(), label_lengths.data(),
                                    sizes.data(),
                                    alphabet_size,
                                    minibatch,
                                    costs.data(),
                                    ctc_cpu_workspace,
                                    options),
                   "Error: compute_ctc_loss (0) in grad_check");

    float cost = std::accumulate(costs.begin(), costs.end(), 0.);

    std::vector<float> num_grad(grads.size());

    //perform 2nd order central differencing
    for (int i = 0; i < T * alphabet_size * minibatch; ++i) {

        std::vector<float> costsP1(minibatch);
        std::vector<float> costsP2(minibatch);

        acts[i] += epsilon;
        throw_on_error(compute_ctc_loss(acts.data(), NULL,
                                        flat_labels.data(), label_lengths.data(),
                                        sizes.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP1.data(),
                                        ctc_cpu_workspace,
                                        options),
                       "Error: compute_ctc_loss (1) in grad_check");

        acts[i] -= 2 * epsilon;
        throw_on_error(compute_ctc_loss(acts.data(), NULL,
                                        flat_labels.data(), label_lengths.data(),
                                        sizes.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP2.data(),
                                        ctc_cpu_workspace,
                                        options),
                       "Error: compute_ctc_loss (2) in grad_check");

        float costP1 = std::accumulate(costsP1.begin(), costsP1.end(), 0.);
        float costP2 = std::accumulate(costsP2.begin(), costsP2.end(), 0.);

        acts[i] += epsilon;
        num_grad[i] = (costP1 - costP2) / (2 * epsilon);
    }

    free(ctc_cpu_workspace);

    float diff = rel_diff(grads, num_grad);

    return diff;
}

bool run_tests() {
    std::vector<std::tuple<int, int, int, int, float>> problem_sizes =
       {std::make_tuple(20, 50, 15, 1, 1e-5),
        std::make_tuple(5, 10, 5, 65, 1e-4)
       };

    std::mt19937 gen(2);

    bool status = true;
    for (auto problem : problem_sizes) {
        int alphabet_size, T, L, minibatch;
        float tol;
        std::tie(alphabet_size, T, L, minibatch, tol) = problem;

        std::vector<float> acts = genActs(alphabet_size * T * minibatch);

        std::vector<std::vector<int>> labels;
        std::vector<int> sizes;
        for (int mb = 0; mb < minibatch; ++mb) {
            int actual_length = L;
            labels.push_back(genLabels(alphabet_size, actual_length));
            sizes.push_back(T);
        }

        float diff = grad_check(T, alphabet_size, acts, labels, sizes);

        status &= (diff < tol);
    }

    return status;
}

int main(void) {
    if (get_warpctc_version() != 2) {
        std::cerr << "Invalid WarpCTC version." << std::endl;
        return 1;
    }

    std::cout << "Running CPU tests" << std::endl;

    bool status = true;
    status &= small_test();
    status &= options_test();
    status &= inf_test();
    status &= run_tests();

    if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}
