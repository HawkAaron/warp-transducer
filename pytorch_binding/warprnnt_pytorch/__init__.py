import torch
import warprnnt_pytorch as warp_rnnt
from torch.autograd import Function
from torch.nn import Module

from ._warp_rnnt import *

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "gradients only computed for acts - please " \
        "mark other tensors as not requiring gradients"


class _RNNT(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, size_average, blank_label):
        is_cuda = True if acts.is_cuda else False
        acts = acts.contiguous()
        labels = labels.contiguous()
        act_lens = act_lens.contiguous()
        label_lens = label_lens.contiguous()
        loss_func = warp_rnnt.gpu_rnnt if is_cuda else warp_rnnt.cpu_rnnt
        grads = torch.zeros_like(acts) if ctx.requires_grad else torch.zeros(0, device=acts.device)
        minibatch_size = acts.size(0)
        costs = torch.zeros(minibatch_size)
        loss_func(acts,
                  labels,
                  act_lens,
                  label_lens,
                  costs,
                  grads,
                  blank_label,
                  0)

        costs = torch.FloatTensor([costs.sum()])

        if size_average:
            # Compute the avg. log-probability per batch sample.
            grads = grads / minibatch_size
            costs = costs / minibatch_size

        ctx.grads = grads
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grads, None, None, None, None, None


class RNNTLoss(Module):
    """
    Parameters:
        size_average (bool): normalize the loss by the batch size
            (default: `False`)
        blank_label (int): default 0
    """
    def __init__(self, size_average=False, blank_label=0):
        super(RNNTLoss, self).__init__()
        self.rnnt = _RNNT.apply
        self.size_average = size_average
        self.blank_label = blank_label

    def forward(self, acts, labels, act_lens, label_lens):
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        assert len(labels.size()) == 2  # labels must be 2 dimensional
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)
        return self.rnnt(acts, labels, act_lens, label_lens, self.size_average, self.blank_label)
