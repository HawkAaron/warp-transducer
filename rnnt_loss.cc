/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file rnnt_loss.cc
 * \brief
 * \author Mingkun Huang
*/

#include "./rnnt_loss-inl.h"
#include "./rnnt_include/detail/cpu_transducer.h"

namespace mshadow {

template <typename DType>
void compute_rnnt_cost(const Tensor<cpu, 4, DType> activations, // BTUV
                             DType *costs, DType *grads, int *labels,
                             int *label_lengths, int *data_lengths,
                             void *workspace, int train, int blank_label) {
  int minibatch = static_cast<int>(activations.size(0));
  int maxT = static_cast<int>(activations.size(1));
  int maxU = static_cast<int>(activations.size(2));
  int alphabet_size = static_cast<int>(activations.size(3));

  mxnet_warprnnt::CpuRNNT<DType> rnnt(minibatch, maxT, maxU, alphabet_size, workspace, blank_label);
  if (train) {
    rnnt.cost_and_grad(activations.dptr_, grads, costs, labels,
                             label_lengths, data_lengths);
  } else {
    rnnt.score_forward(activations.dptr_, costs, labels, label_lengths,
                             data_lengths);
  }
}

}  // namespace mshadow

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<cpu>(RNNTLossParam param, int dtype) {
  return new RNNTLossOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *RNNTLossProp::CreateOperatorEx(Context ctx,
                                        std::vector<TShape> *in_shape,
                                        std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(RNNTLossParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_RNNTLoss, RNNTLossProp)
    .describe(R"code(RNN Transducer Loss.

The shapes of the inputs and outputs:

- **data**: `(batch_size, sequence_length, label_length + 1, alphabet_size)`
- **label**: `(batch_size, label_sequence_length)`
- **out**: `(batch_size)`

The `data` tensor consists of sequences of activation vectors (after applying softmax),
with i-th channel in the last dimension corresponding to i-th label
for i between 0 and alphabet_size-1 (i.e always 0-indexed).
Alphabet size should include one additional value reserved for blank label.
When `blank_label` is ``"first"``, the ``0``-th channel is be reserved for
activation of blank label, or otherwise if it is "last", ``(alphabet_size-1)``-th channel should be
reserved for blank label.

``label`` is an index matrix of integers. When `blank_label` is ``"first"``,
the value 0 is then reserved for blank label, and should not be passed in this matrix. Otherwise,
when `blank_label` is ``"last"``, the value `(alphabet_size-1)` is reserved for blank label.

``out`` is a list of RNNT loss values, one per example in the batch.

See *Sequence Transduction with Recurrent Neural Networks*, A. Graves. for more
information on the definition and the algorithm.

)code" ADD_FILELINE)
    .add_argument("data", "NDArray-or-Symbol", "Input data to the rnnt_loss op.")
    .add_argument("label", "NDArray-or-Symbol",
                  "Ground-truth labels for the loss.")
    .add_argument("data_lengths", "NDArray-or-Symbol",
                  "Lengths of data for each of the samples. Only required "
                  "when use_data_lengths is true.")
    .add_argument("label_lengths", "NDArray-or-Symbol",
                  "Lengths of labels for each of the samples. Only required "
                  "when use_label_lengths is true.")
    .add_arguments(RNNTLossParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_RNNTLoss).add_alias("_contrib_rnnt_loss");

}  // namespace op
}  // namespace mxnet
