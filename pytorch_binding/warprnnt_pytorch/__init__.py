import torch
import warprnnt_pytorch as warp_rnnt
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn import Module
from torch.nn.modules.loss import _assert_no_grad

from ._warp_rnnt import *


class _RNNT(Function):
    @staticmethod
    def forward(ctx, trans_acts, pred_acts, labels, act_lens, label_lens,
                    size_average, blank_label):
        is_cuda = True if trans_acts.is_cuda else False
        trans_acts = trans_acts.cpu().contiguous()
        pred_acts = pred_acts.cpu().contiguous()
        loss_func = warp_rnnt.cpu_rnnt
        trans_grads = torch.zeros_like(trans_acts) if ctx.requires_grad else torch.zeros(0)
        pred_grads = torch.zeros_like(pred_acts) if ctx.requires_grad else torch.zeros(0)
        minibatch_size = trans_acts.size(1)
        costs = torch.zeros(minibatch_size).cpu()
        loss_func(trans_acts, pred_acts,
                  labels.int().cpu(),
                  act_lens.int().cpu(),
                  label_lens.int().cpu(),
                  costs,
                  trans_grads, pred_grads,
                  blank_label,
                  0)

        costs = torch.FloatTensor([costs.sum()])

        if is_cuda:
            costs = costs.cuda()
            trans_grads = trans_grads.cuda()
            pred_grads = pred_grads.cuda()

        if size_average:
            # Compute the avg. log-probability per batch sample.
            trans_grads = trans_grads / minibatch_size
            pred_grads = pred_grads / minibatch_size
            costs = costs / minibatch_size

        ctx.grads = Variable(trans_grads), Variable(pred_grads)
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grads[0], ctx.grads[1], None, None, None, None, None


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

    def forward(self, trans_acts, pred_acts, labels, act_lens, label_lens):
        """
        trans_acts: Tensor of (batch x seqLength x outputDim) containing output from transcription network
        pred_acts: Tensor of (batch x labelLength x outputDim) containing output from prediction network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the transcription network
        label_lens: Tensor of (batch) containing label length of each example
        """
        assert 1 <= len(labels.size()) <= 2
        if len(labels.shape) > 1:
            labels = torch.cat([labels[i, :j] for i, j in enumerate(label_lens.data)])
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)
        # acts should be batch first
        if not self.batch_first:
            trans_acts = trans_acts.transpose(0, 1)
            pred_acts = pred_acts.transpose(0, 1)
        return self.rnnt(trans_acts, pred_acts, labels, 
                act_lens, label_lens, self.size_average, self.blank_label)
