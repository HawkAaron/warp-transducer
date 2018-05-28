import numpy as np
import tensorflow as tf
from warprnnt_tensorflow import rnnt_loss

trans_acts = tf.placeholder(tf.float32, [None, None, None])
pred_acts = tf.placeholder(tf.float32, [None, None, None])
labels = tf.placeholder(tf.int32, [None, None])
input_length = tf.placeholder(tf.int32, [None])
label_length = tf.placeholder(tf.int32, [None])

B = 2; T = 4; U = 3; V = 6; blank = 5

costs = rnnt_loss(trans_acts, pred_acts, labels, input_length, label_length, blank)
grad = tf.gradients(costs, [trans_acts, pred_acts])

a = np.array([0.20836473 , 0.68488914 , 0.8508703 , 0.5761989 , 0.19992691 , 0.8066367 ,
            0.9547191 , 0.39503795 , 0.91791826 , 0.66354334 , 0.52238625 , 0.30651063 ,
            0.51816165 , 0.83243364 , 0.29219377 , 0.35955018 , 0.6904012 , 0.85133505 ,
            0.87253755 , 0.06845909 , 0.74267465 , 0.7473852 , 0.6735857 , 0.814946 ,
            0.72159135 , 0.27252442 , 0.20181221 , 0.7149978 , 0.7996553 , 0.594097 ,
            0.98958284 , 0.7198998 , 0.32019693 , 0.49187315 , 0.82729864 , 0.72087383 ,
            0.5865227 , 0.84655076 , 0.73006225 , 0.24205732 , 0.2466054 , 0.93103373 ,
            0.6253804 , 0.56404036 , 0.59297657 , 0.6260772 , 0.23223883 , 0.041093946], dtype=np.float32).reshape(B, T, V)

b = np.array([0.061166406 , 0.14563453 , 0.563884 , 0.66322905 , 0.19838423 , 0.18207806 ,
            0.9041121 , 0.31825322 , 0.10209769 , 0.44423354 , 0.7338143 , 0.22434187 ,
            0.7007454 , 0.65526026 , 0.5205059 , 0.30149776 , 0.6051819 , 0.1901899 ,
            0.6904842 , 0.30375922 , 0.61894506 , 0.032821894 , 0.75227857 , 0.8265933 ,
            0.79737663 , 0.86086124 , 0.4400267 , 0.8985075 , 0.3717013 , 0.9338418 ,
            0.91288275 , 0.6805384 , 0.019013822 , 0.84054446 , 0.52986646 , 0.27262968], dtype=np.float32).reshape(B, U, V)

c = np.array([[1, 2], [1, 1]], dtype=np.int32)
d = np.array([4, 4], dtype=np.int32)
e = np.array([2, 2], dtype=np.int32)

feed = {trans_acts: a, pred_acts: b, labels: c, input_length: d, label_length: e}
with tf.Session() as sess:
    cost, trans_grad, pred_grad = sess.run([costs, grad[0], grad[1]], feed_dict=feed)
    print(cost)
    print(trans_grad)
    print(pred_grad)
