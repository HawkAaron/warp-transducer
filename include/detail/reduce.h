#pragma once

rnntStatus_t reduce_exp(const float* const acts, float* denom, int rows, int cols, bool minus, cudaStream_t stream);
rnntStatus_t reduce_max(const float* const acts, float* denom, int rows, int cols, bool minus, cudaStream_t stream);
