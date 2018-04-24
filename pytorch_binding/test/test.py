import torch
import numpy as np
from warprnnt_pytorch import RNNTLoss

from transducer_np import RNNTLoss as rnntloss

B = 1
T = 4
U = 3
V = 5

def generate_sample():
    trans_acts = torch.autograd.Variable(torch.zeros(B, T, V).uniform_(), requires_grad=True)
    pred_acts = torch.autograd.Variable(torch.zeros(B, U, V).uniform_(), requires_grad=True)

    label = torch.autograd.Variable(torch.zeros(B, U-1).uniform_(1, V-1).int())

    act_length = torch.autograd.Variable(torch.IntTensor([T] * B))
    label_length = torch.autograd.Variable(torch.IntTensor([U-1] * B))

    joint = torch.nn.functional.log_softmax(trans_acts.unsqueeze(dim=2) + pred_acts.unsqueeze(dim=1), dim=3)

    loss = RNNTLoss()(joint, label, act_length, label_length)
    print(loss)
    loss.backward()

    print('trans_acts')
    print(trans_acts.data.numpy())
    print('pred_acts')
    print(pred_acts.data.numpy())
    print('label')
    print(label.data.numpy())
    print('grad to trans_acts')
    print(trans_acts.grad.data.numpy())
    print('grad to pred_acts')
    print(pred_acts.grad.data.numpy())

def test_add_network():
    trans_acts = torch.autograd.Variable(torch.Tensor([[[0.06860042, 0.02114975, 0.9319364,  0.32975072, 0.13327402],
                                                        [0.12564015, 0.43477476, 0.5019661,  0.6318551,  0.56487304],
                                                        [0.7360039,  0.39700246, 0.50331944, 0.6365183,  0.19341111],
                                                        [0.7335776,  0.9189112,  0.65545017, 0.89854324, 0.0246467 ]]]), requires_grad=True)


    pred_acts = torch.autograd.Variable(torch.Tensor([[[0.1928187,  0.63991195, 0.7839058,  0.8364832,  0.55822134],
                                                        [0.11235255, 0.44893646, 0.46855223, 0.20775604, 0.9215501 ],
                                                        [0.1369242,  0.6790915,  0.3493181,  0.72844136, 0.46717978]]]), requires_grad=True)

    # acts = trans_acts.unsqueeze(dim=2) + pred_acts.unsqueeze(dim=1)
    # acts = torch.nn.functional.log_softmax(acts, dim=3)

    label = torch.autograd.Variable(torch.IntTensor([[1, 2]]))

    act_length = torch.autograd.Variable(torch.IntTensor([T] * B))
    label_length = torch.autograd.Variable(torch.IntTensor([U-1] * B))

    loss = RNNTLoss()(trans_acts, pred_acts, label, act_length, label_length)
    loss.backward()

    expected_cost = 9.3370

    expected_trans_grads = np.array([[[-0.85036933, -0.17704993,  0.4379695,   0.32460052,  0.26484928],
                                    [-0.86005664, -0.01435287,  0.13164294,  0.36458912,  0.37817746],
                                    [-0.7415191,   0.10448333,  0.01432155,  0.36094004,  0.2617742 ],
                                    [-0.7804755,   0.3140569,  -0.12812297,  0.4177104,   0.17683122]]])
    expected_pred_grads = np.array([[[-0.80384177, -0.66368896,  0.6344613,   0.51796013,  0.31510928],
                                    [-0.65244085,  0.36658645, -0.5477955,   0.34172153,  0.49192834],
                                    [-1.7761381,   0.5242399,   0.36914518,  0.60815847,  0.27459458]]])

    assert np.allclose(loss.data.numpy(), expected_cost), \
        "costs mismath."
    assert np.allclose(trans_acts.grad.data.numpy(), expected_trans_grads), \
        "trans gradient mismatch."
    assert np.allclose(pred_acts.grad.data.numpy(), expected_pred_grads), \
        "pred gradient mismathc."

test_add_network()