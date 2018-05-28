import tensorflow as tf
import numpy as np
from warprnnt_tensorflow import rnnt_loss
from tensorflow.python.client import device_lib

def is_gpu_available():
    """Returns whether Tensorflow can access a GPU."""
    return any(x.device_type == 'GPU' for x in device_lib.list_local_devices())

class WarpRNNTTest(tf.test.TestCase):

    def _run_rnnt(self, trans_acts, pred_acts, labels,
                    input_lengths, label_lengths,
                    expected_costs, expected_trans_grads,
                    expected_pred_grads, blank, use_gpu=False):
        self.assertEquals(trans_acts.shape, expected_trans_grads.shape)
        self.assertEquals(pred_acts.shape, expected_pred_grads.shape)
        trans_acts_t = tf.constant(trans_acts)
        pred_acts_t = tf.constant(pred_acts)
        labels_t = tf.constant(labels)
        input_lengths_t = tf.constant(input_lengths)
        label_lengths_t = tf.constant(label_lengths)
        costs = rnnt_loss(trans_acts_t, pred_acts_t, labels_t, input_lengths_t, label_lengths_t, blank)

        trans_grads, pred_grads = tf.gradients(costs, [trans_acts_t, pred_acts_t])

        self.assertShapeEqual(expected_costs, costs)

        self.assertShapeEqual(expected_trans_grads, trans_grads)
        self.assertShapeEqual(expected_pred_grads, pred_grads)

        log_dev_placement = False
        if not use_gpu:
            config = tf.ConfigProto(log_device_placement=log_dev_placement,
                                    device_count={'CPU': 0})
        else:
            config = tf.ConfigProto(log_device_placement=log_dev_placement,
                                    allow_soft_placement=False)
        
        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, config=config) as sess:
            (tf_costs, tf_trans_grad, tf_pred_grad) = sess.run([costs, trans_grads, pred_grads])
            self.assertAllClose(tf_costs, expected_costs, atol=1e-6)
            self.assertAllClose(tf_trans_grad, expected_trans_grads, atol=1e-6)
            self.assertAllClose(tf_pred_grad, expected_pred_grads, atol=1e-6)
    
    def test_forward(self):
        # Softmax activations for the following inputs:
        trans_acts = np.array([[[0.06860042, 0.02114975, 0.9319364,  0.32975072, 0.13327402],
                                [0.12564015, 0.43477476, 0.5019661,  0.6318551,  0.56487304],
                                [0.7360039,  0.39700246, 0.50331944, 0.6365183,  0.19341111],
                                [0.7335776,  0.9189112,  0.65545017, 0.89854324, 0.0246467 ]]], dtype=np.float32)

        pred_acts = np.array([[[0.1928187,  0.63991195, 0.7839058,  0.8364832,  0.55822134],
                                [0.11235255, 0.44893646, 0.46855223, 0.20775604, 0.9215501 ],
                                [0.1369242,  0.6790915,  0.3493181,  0.72844136, 0.46717978]]], dtype=np.float32)
        labels = np.array([[1, 2]], dtype=np.int32)
        input_lengths = np.array([4], dtype=np.int32)
        label_lengths = np.array([2], dtype=np.int32)

        trans_acts_t = tf.constant(trans_acts)
        pred_acts_t = tf.constant(pred_acts)
        labels_t = tf.constant(labels)
        input_lengths_t = tf.constant(input_lengths)
        label_lengths_t = tf.constant(label_lengths)
        costs = rnnt_loss(trans_acts_t, pred_acts_t, labels_t, input_lengths_t, label_lengths_t, blank)
        with self.test_session():
            print(costs.eval())

    def _test_basic(self, use_gpu):
        # Softmax activations for the following inputs:
        trans_acts = np.array([[[0.06860042, 0.02114975, 0.9319364,  0.32975072, 0.13327402],
                                [0.12564015, 0.43477476, 0.5019661,  0.6318551,  0.56487304],
                                [0.7360039,  0.39700246, 0.50331944, 0.6365183,  0.19341111],
                                [0.7335776,  0.9189112,  0.65545017, 0.89854324, 0.0246467 ]]], dtype=np.float32)

        pred_acts = np.array([[[0.1928187,  0.63991195, 0.7839058,  0.8364832,  0.55822134],
                                [0.11235255, 0.44893646, 0.46855223, 0.20775604, 0.9215501 ],
                                [0.1369242,  0.6790915,  0.3493181,  0.72844136, 0.46717978]]], dtype=np.float32)

        alphabet_size = 5

        labels = np.array([[1, 2]], dtype=np.int32)

        expected_costs = np.array([9.3370], dtype=np.float32)
        expected_trans_grads = np.array([[[-0.85036933, -0.17704993,  0.4379695,   0.32460052,  0.26484928],
                                        [-0.86005664, -0.01435287,  0.13164294,  0.36458912,  0.37817746],
                                        [-0.7415191,   0.10448333,  0.01432155,  0.36094004,  0.2617742 ],
                                        [-0.7804755,   0.3140569,  -0.12812297,  0.4177104,   0.17683122]]], dtype=np.float32)
        expected_pred_grads = np.array([[[-0.80384177, -0.66368896,  0.6344613,   0.51796013,  0.31510928],
                                        [-0.65244085,  0.36658645, -0.5477955,   0.34172153,  0.49192834],
                                        [-1.7761381,   0.5242399,   0.36914518,  0.60815847,  0.27459458]]], dtype=np.float32)

        input_lengths = np.array([4], dtype=np.int32)
        label_lengths = np.array([2], dtype=np.int32)
        
        self._run_rnnt(trans_acts, pred_acts, labels, input_lengths, label_lengths,
                expected_costs, expected_trans_grads, expected_pred_grads, 0, use_gpu)
    
    def test_basic_cpu(self):
        self._test_basic(use_gpu=False)
    
    def test_basic_gpu(self):
        if (is_gpu_available()):
            self._test_basic(use_gpu=True)
        else:
            print('Skipping GPU test, no gpus available')
    
    def _test_multiple_batches(self, use_gpu):
        B = 2; T = 4; U = 3; V = 6; blank = 5

        trans_acts = np.array([0.20836473 , 0.68488914 , 0.8508703 , 0.5761989 , 0.19992691 , 0.8066367 ,
                                0.9547191 , 0.39503795 , 0.91791826 , 0.66354334 , 0.52238625 , 0.30651063 ,
                                0.51816165 , 0.83243364 , 0.29219377 , 0.35955018 , 0.6904012 , 0.85133505 ,
                                0.87253755 , 0.06845909 , 0.74267465 , 0.7473852 , 0.6735857 , 0.814946 ,
                                0.72159135 , 0.27252442 , 0.20181221 , 0.7149978 , 0.7996553 , 0.594097 ,
                                0.98958284 , 0.7198998 , 0.32019693 , 0.49187315 , 0.82729864 , 0.72087383 ,
                                0.5865227 , 0.84655076 , 0.73006225 , 0.24205732 , 0.2466054 , 0.93103373 ,
                                0.6253804 , 0.56404036 , 0.59297657 , 0.6260772 , 0.23223883 , 0.041093946], dtype=np.float32).reshape(B, T, V)

        pred_acts = np.array([0.061166406 , 0.14563453 , 0.563884 , 0.66322905 , 0.19838423 , 0.18207806 ,
                            0.9041121 , 0.31825322 , 0.10209769 , 0.44423354 , 0.7338143 , 0.22434187 ,
                            0.7007454 , 0.65526026 , 0.5205059 , 0.30149776 , 0.6051819 , 0.1901899 ,
                            0.6904842 , 0.30375922 , 0.61894506 , 0.032821894 , 0.75227857 , 0.8265933 ,
                            0.79737663 , 0.86086124 , 0.4400267 , 0.8985075 , 0.3717013 , 0.9338418 ,
                            0.91288275 , 0.6805384 , 0.019013822 , 0.84054446 , 0.52986646 , 0.27262968], dtype=np.float32).reshape(B, U, V)

        expected_costs = np.array([9.8610, 8.7003], dtype=np.float32)
        expected_trans_grads = np.array([0.18298208713531494 , -0.1863221824169159 , 0.24029867351055145 , 0.31508156657218933 , 0.17985235154628754 , -0.7318925261497498 ,
                                        0.3619690537452698 , -0.06026348099112511 , 0.08730498701334 , 0.25879740715026855 , 0.22078825533390045 , -0.8685961961746216 ,
                                        0.27959010004997253 , 0.03639250993728638 , -0.061875589191913605 , 0.19594523310661316 , 0.30305129289627075 , -0.753103494644165 ,
                                        0.385039746761322 , 0.06576273590326309 , -0.18127907812595367 , 0.2354261875152588 , 0.28227531909942627 , -0.7872248888015747 ,
                                        0.26263949275016785 , -0.10713391751050949 , 0.13813626766204834 , 0.16261409223079681 , 0.2808478772640228 , -0.7371038794517517 ,
                                        0.3360159993171692 , -0.1639198213815689 , 0.1387038677930832 , 0.1545054018497467 , 0.2540736496448517 , -0.7193790674209595 ,
                                        0.29537156224250793 , -0.274471253156662 , 0.2312944084405899 , 0.18443721532821655 , 0.16433767974376678 , -0.6009695529937744 ,
                                        0.4264937937259674 , -0.4397970736026764 , 0.22451627254486084 , 0.4020277261734009 , 0.20442542433738708 , -0.8176661133766174], dtype=np.float32).reshape(B, T, V)

        expected_pred_grads = np.array([0.2336210012435913 , -0.7242842316627502 , 0.49153390526771545 , 0.4382175803184509 , 0.2350836545228958 , -0.6741719245910645 ,
                                        0.5180250406265259 , 0.26190635561943054 , -0.7572638988494873 , 0.29955601692199707 , 0.3859791159629822 , -0.7082027196884155 ,
                                        0.4579349160194397 , 0.31794747710227966 , 0.35017895698547363 , 0.26747676730155945 , 0.3649044632911682 , -1.7584426403045654 ,
                                        0.5328136086463928 , -0.7057985663414001 , 0.3393358588218689 , 0.2234761267900467 , 0.5142948627471924 , -0.9041219353675842 ,
                                        0.4035489857196808 , -0.5864231586456299 , 0.24028921127319336 , 0.3576064705848694 , 0.20584842562675476 , -0.6208699345588684 ,
                                        0.38415825366973877 , 0.30689966678619385 , 0.15302574634552002 , 0.3225018382072449 , 0.18354131281375885 , -1.35012686252594], dtype=np.float32).reshape(B, U, V)

        labels = np.array([[1, 2], [1, 1]], dtype=np.int32)
        input_lengths = np.array([4, 4], dtype=np.int32)
        label_lengths = np.array([2, 2], dtype=np.int32)

        self._run_rnnt(trans_acts, pred_acts, labels, input_lengths, label_lengths, 
                expected_costs, expected_trans_grads, expected_pred_grads, blank, use_gpu)

    def test_multiple_batches_cpu(self):
        self._test_multiple_batches(use_gpu=False)
    
    def test_multiple_batches_gpu(self):
        if (is_gpu_available()):
            self._test_multiple_batches(use_gpu=True)
        else:
            print('Skipping GPU test, no gpus available')

if __name__ == '__main__':
    tf.test.main()
