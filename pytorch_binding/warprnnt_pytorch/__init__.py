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

    def forward(self, acts, labels, act_lens, label_lens):
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        is_cuda = acts.is_cuda

        certify_inputs(acts, labels, act_lens, label_lens)

        loss_func = warp_rnnt.gpu_rnnt if is_cuda else warp_rnnt.cpu_rnnt
        grads = torch.zeros_like(acts) if acts.requires_grad else torch.zeros(0, device=acts.device)
        minibatch_size = acts.size(0)
        costs = torch.zeros(minibatch_size)
        loss_func(acts,
                  labels,
                  act_lens,
                  label_lens,
                  costs,
                  grads,
                  self.blank,
                  0)

        if self.reduction in ['sum', 'mean']:
            costs = torch.FloatTensor([costs.sum()])
            if self.reduction == 'mean':
                costs /= minibatch_size
                grads /= minibatch_size

        costs = costs.to(acts.device)
        self.grads = grads

        return costs

    def backward(self, grad_output):
        grad_output = grad_output.view(-1, 1, 1, 1).to(self.grads)
        return self.grads.mul_(grad_output), None, None, None

def check_type(var, t, name):
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))

def check_contiguous(var, name):
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))

def check_dim(var, dim, name):
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))

def certify_inputs(log_probs, labels, lengths, label_lengths):
    check_type(log_probs, torch.float32, "log_probs")
    check_type(labels, torch.int32, "labels")
    check_type(label_lengths, torch.int32, "label_lengths")
    check_type(lengths, torch.int32, "lengths")
    check_contiguous(log_probs, "log_probs")
    check_contiguous(labels, "labels")
    check_contiguous(label_lengths, "label_lengths")
    check_contiguous(lengths, "lengths")

    if lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a length per example.")
    if label_lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a label length per example.")

    check_dim(log_probs, 4, "log_probs")
    check_dim(labels, 2, "labels")
    check_dim(lengths, 1, "lenghts")
    check_dim(label_lengths, 1, "label_lenghts")
    max_T = torch.max(lengths)
    max_U = torch.max(label_lengths)
    T, U = log_probs.shape[1:3]
    if T != max_T:
        raise ValueError("Input length mismatch")
    if U != max_U + 1:
        raise ValueError("Output length mismatch")

