"""
Tests for the C implementation of the sequence transducer.

From outside the package directory, run
`python -m transducer.test.`
"""
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
import torch
import torch.autograd as autograd
import torch.nn as nn

import os
import subprocess

from warprnnt_pytorch import RNNTLoss
from transducer_np import RNNTLoss as rnntloss

parser = argparse.ArgumentParser(description='MXNet RNN Transducer Test.')
parser.add_argument('B', type=int, default=1, help='batch size')
parser.add_argument('T', type=int, default=300, help='time step')
parser.add_argument('U', type=int, default=100, help='prediction step')
parser.add_argument('V', type=int, default=60, help='vocab size')
parser.add_argument('--np', default=False, action='store_true', help='use numpy loss')
parser.add_argument('--add', default=False, action='store_true', help='add_network')
args = parser.parse_args()

fn = rnntloss() if args.np else RNNTLoss()

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def wrap_and_call():
    print('initial gpu memory usage: {}'.format(get_gpu_memory_map()))
    batch_size = args.B
    vocab_size = args.V
    input_len = args.T
    output_len = args.U
    trans_acts = torch.zeros(batch_size, input_len, vocab_size).uniform_().cuda()
    pred_acts = torch.zeros(batch_size, output_len + 1, vocab_size).uniform_().cuda()
    labels = torch.zeros(batch_size, output_len).uniform_(1, vocab_size-1).int().cuda()
    
    trans_acts = autograd.Variable(trans_acts, requires_grad=True)
    pred_acts = autograd.Variable(pred_acts, requires_grad=True)
    labels = autograd.Variable(labels)
    lengths = autograd.Variable(torch.IntTensor([input_len] * batch_size).cuda())
    label_lengths = autograd.Variable(torch.IntTensor([output_len - 1] * batch_size).cuda())

    gpu_memory = batch_size * (input_len + (output_len + 1)) * vocab_size * 4

    print('before compute gradient, gpu memory assume: {:.3f}GB = {:.3f}MB, actual {}'.format(gpu_memory / (1<<30), gpu_memory / (1<<20), get_gpu_memory_map()))
    if args.add:
        start = time.time()
        costs = fn(trans_acts, pred_acts, labels, lengths, label_lengths)
        end = time.time()
        print('cpu loss time: {:.3f} s\n'.format(end-start))
        # grads to trans_acts, pred_acts
        gpu_memory += batch_size * (input_len + (output_len + 1)) * vocab_size * 4
    else:
        # joint
        acts = trans_acts.unsqueeze(dim=2) + pred_acts.unsqueeze(dim=1)
        log_probs = nn.functional.log_softmax(acts, dim=3)
        start = time.time()
        costs = fn(log_probs, labels, lengths, label_lengths)
        end = time.time()
        print('add network cpu loss time: {:.3f} s\n'.format(end-start))
        # acts & log_probs & grad to log_probs
        gpu_memory += batch_size * input_len * (output_len + 1) * vocab_size * 4 * 3
    print('after compute gradient, gpu memory assume: {:.3f}GB = {:.3f}MB, actual {}'.format(gpu_memory / (1<<30), gpu_memory / (1<<20), get_gpu_memory_map()))

    start = time.time()
    costs.backward()
    end = time.time()
    # grads to trans_acts, pred_acts
    gpu_memory += batch_size * (input_len + (output_len + 1)) * vocab_size * 4
    if not args.add:
        # grads to acts
        gpu_memory += batch_size * input_len * (output_len + 1) * vocab_size * 4
    # grad to log_probs is not retained
    # if not args.add:
    #     gpu_memory -= batch_size * input_len * (output_len + 1) * vocab_size * 4

    print('after backward, gpu memory assume: {:.3f}GB = {:.3f}MB, actual {}'.format(gpu_memory / (1<<30), gpu_memory / (1<<20), get_gpu_memory_map()))
    print('backward time: {:.3f} s'.format(end-start))
    print('GPU memory comsume: {:.3f}GB = {:.3f}MB'.format(gpu_memory / (1<<30), gpu_memory / (1<<20)))
    print()
    torch.cuda.empty_cache()

def time_test(blank=0):
    start = time.time()
    iters = 1
    for _ in range(iters):
        wrap_and_call()
    end = time.time()

    print("Time per iteration: {:.3f}(s)".format((end-start)/iters))

if __name__ == "__main__":
    time_test()
