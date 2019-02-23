import torch
import warprnnt_pytorch as warp_rnnt
from torch.autograd import Function
from torch.nn import Module

from .warp_rnnt import *

__all__ = ['RNNTLoss']

class RNNTLoss(Function):
    """
    Parameters:
        blank (int, optional): blank label. Default: 0.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 
            'mean': the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: 'mean'
    """
    def __init__(self, blank=0, reduction='mean'):
        super(RNNTLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(self, trans_acts, pred_acts, labels, act_lens, label_lens):
        """
        trans_acts: Tensor of (batch x seqLength x outputDim) containing output from transcription network
        pred_acts: Tensor of (batch x labelLength x outputDim) containing output from prediction network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the transcription network
        label_lens: Tensor of (batch) containing label length of each example
        """
        is_cuda = trans_acts.is_cuda
        assert trans_acts.is_cuda == pred_acts.is_cuda

        certify_inputs(trans_acts, pred_acts, labels, act_lens, label_lens)

        # TODO remove cpu rnnt flat_labels
        if not is_cuda and len(labels.shape) > 1:
            labels = torch.cat([labels[i, :j] for i, j in enumerate(label_lens)])

        loss_func = warp_rnnt.gpu_rnnt if is_cuda else warp_rnnt.cpu_rnnt

        device = trans_acts.device
        assert trans_acts.requires_grad == pred_acts.requires_grad
        requires_grad = trans_acts.requires_grad
        trans_grads = torch.zeros_like(trans_acts) if requires_grad else torch.zeros(0, device=device)
        pred_grads = torch.zeros_like(pred_acts) if requires_grad else torch.zeros(0, device=device)

        minibatch_size = trans_acts.size(0)
        costs = torch.zeros(minibatch_size)
        loss_func(trans_acts, pred_acts,
                  labels,
                  act_lens,
                  label_lens,
                  costs,
                  trans_grads, pred_grads,
                  self.blank,
                  0)

        if self.reduction in ['sum', 'mean']:
            costs = torch.FloatTensor([costs.sum()])
            if self.reduction == 'mean':
                costs /= minibatch_size
                trans_grads /= minibatch_size
                pred_grads /= minibatch_size

        costs = costs.to(trans_acts.device)
        self.grads = trans_grads, pred_grads

        return costs

    def backward(self, grad_output):
        return self.grads[0].mul_(grad_output), self.grads[1].mul_(grad_output), None, None, None

def check_type(var, t, name):
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))

def check_contiguous(var, name):
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))

def check_dim(var, dim, name):
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))

def certify_inputs(trans_acts, pred_acts, labels, lengths, label_lengths):
    check_type(trans_acts, torch.float32, "trans_acts")
    check_type(pred_acts, torch.float32, "pred_acts")
    check_type(labels, torch.int32, "labels")
    check_type(label_lengths, torch.int32, "label_lengths")
    check_type(lengths, torch.int32, "lengths")
    check_contiguous(trans_acts, "trans_acts")
    check_contiguous(pred_acts, "pred_acts")
    check_contiguous(labels, "labels")
    check_contiguous(label_lengths, "label_lengths")
    check_contiguous(lengths, "lengths")

    if lengths.shape[0] != trans_acts.shape[0]:
        raise ValueError("must have a length per example.")
    if label_lengths.shape[0] != trans_acts.shape[0]:
        raise ValueError("must have a label length per example.")
    if trans_acts.shape[2] != pred_acts.shape[2]:
        raise ValueError("vocabulary size must equal.")

    check_dim(trans_acts, 3, "trans_acts")
    check_dim(pred_acts, 3, "pred_acts")
    check_dim(labels, 2, "labels")
    check_dim(lengths, 1, "lenghts")
    check_dim(label_lengths, 1, "label_lenghts")
    max_T = torch.max(lengths)
    max_U = torch.max(label_lengths)
    T = trans_acts.shape[1]
    U = pred_acts.shape[1]
    if T != max_T:
        raise ValueError("Input length mismatch")
    if U != max_U + 1:
        raise ValueError("Output length mismatch")

