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
    def forward(ctx, trans_acts, pred_acts, labels, act_lens, label_lens,
                    size_average, blank_label):
        is_cuda = True if trans_acts.is_cuda else False
        # TODO remove cpu rnnt flat_labels
        if not is_cuda and len(labels.shape) > 1:
            labels = torch.cat([labels[i, :j] for i, j in enumerate(label_lens)])
        trans_acts = trans_acts.contiguous()
        pred_acts = pred_acts.contiguous()
        labels = labels.contiguous()
        loss_func = warp_rnnt.gpu_rnnt if is_cuda else warp_rnnt.cpu_rnnt

        device = trans_acts.device
        trans_grads = torch.zeros_like(trans_acts) if ctx.requires_grad else torch.zeros(0, device=device)
        pred_grads = torch.zeros_like(pred_acts) if ctx.requires_grad else torch.zeros(0, device=device)

        minibatch_size = trans_acts.size(0)
        costs = torch.zeros(minibatch_size)
        loss_func(trans_acts, pred_acts,
                  labels,
                  act_lens,
                  label_lens,
                  costs,
                  trans_grads, pred_grads,
                  blank_label,
                  0)

        costs = torch.FloatTensor([costs.sum()])

        if size_average:
            # Compute the avg. log-probability per batch sample.
            trans_grads = trans_grads / minibatch_size
            pred_grads = pred_grads / minibatch_size
            costs = costs / minibatch_size

        ctx.grads = trans_grads, pred_grads
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
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)
        # acts should be batch first
        if not self.batch_first:
            trans_acts = trans_acts.transpose(0, 1)
            pred_acts = pred_acts.transpose(0, 1)
        return self.rnnt(trans_acts, pred_acts, labels, 
                act_lens, label_lens, self.size_average, self.blank_label)
