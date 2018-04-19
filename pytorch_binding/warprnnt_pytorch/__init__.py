import torch
import warprnnt_pytorch as warp_rnnt
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn import Module
from torch.nn.modules.loss import _assert_no_grad

from ._warp_rnnt import *


class _RNNT(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens,
                        size_average, blank_label):
        acts = acts.contiguous()
        loss_func = warp_rnnt.cpu_rnnt
        grads = torch.zeros_like(acts) if acts.requires_grad else torch.zeros(0)
        minibatch_size = acts.size(0)
        costs = torch.zeros(minibatch_size)
        loss_func(acts,
                  labels,
                  act_lens,
                  label_lens,
                  costs,
                  grads,
                  blank_label)

        costs = Variable(torch.FloatTensor([costs.sum()]))

        if size_average:
            # Compute the avg. log-probability per batch sample.
            grads = grads / minibatch_size
            costs = costs / minibatch_size

        ctx.save_for_backward(Variable(grads))
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        grads, = ctx.saved_variables
        return grads, None, None, None, None, None


class RNNTLoss(Module):
    """
    Parameters:
        size_average (bool): normalize the loss by the batch size
            (default: `False`)
        blank_label (bool): default 0
    """
    def __init__(self, size_average=False, blank_label=0):
        super(RNNTLoss, self).__init__()
        self.rnnt = _RNNT.apply
        self.size_average = size_average
        self.blank_label = blank_label

    def forward(self, acts, labels, act_lens, label_lens):
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        assert 1 <= len(labels.size()) <= 2  # labels must be 1 dimensional
        if len(labels.shape) > 1:
            labels = torch.cat([labels[i, :j] for i, j in enumerate(label_lens)])
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)
        return self.ctc(acts, labels, act_lens, label_lens, self.size_average,
                        self.length_average)
