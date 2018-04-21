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
                    size_average, blank_label, batch_first):
        is_cuda = True if acts.is_cuda else False
        acts = acts.cpu().contiguous()
        loss_func = warp_rnnt.cpu_rnnt
        grads = torch.zeros_like(acts) if ctx.requires_grad else torch.zeros(0)
        minibatch_size = acts.size(0) if batch_first else acts.size(2)
        costs = torch.zeros(minibatch_size).cpu()
        loss_func(acts,
                  labels,
                  act_lens,
                  label_lens,
                  costs,
                  grads,
                  blank_label,
                  batch_first)

        costs = torch.FloatTensor([costs.sum()])

        if is_cuda:
            costs = costs.cuda()
            grads = grads.cuda()

        if size_average:
            # Compute the avg. log-probability per batch sample.
            grads = grads / minibatch_size
            costs = costs / minibatch_size

        ctx.grads = Variable(grads)
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grads, None, None, None, None, None, None


class RNNTLoss(Module):
    """
    Parameters:
        size_average (bool): normalize the loss by the batch size
            (default: `False`)
        blank_label (bool): default 0
    """
    def __init__(self, size_average=False, blank_label=0, batch_first=True):
        super(RNNTLoss, self).__init__()
        self.rnnt = _RNNT.apply
        self.size_average = size_average
        self.blank_label = blank_label
        self.batch_first = batch_first

    def forward(self, acts, labels, act_lens, label_lens):
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        assert 1 <= len(labels.size()) <= 2  # labels must be 1 dimensional
        if len(labels.shape) > 1:
            labels = torch.cat([labels[i, :j] for i, j in enumerate(label_lens.data)])
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)
        return self.rnnt(acts, labels, act_lens, label_lens, 
                    self.size_average, self.blank_label, self.batch_first)
