import tensorflow as tf
import argparse
import os
import argparse
import os

from models.model import Informer
from models.model import Informert

# from models.ExpInformer import ExpInformer
import warnings
warnings.filterwarnings('ignore')

#
# from IPython.core.display import display, HTML
#
#
# display(HTML("<style>.container { width:100% !important; }</style>"))

import math
import json
import numpy as np
import tensorflow as tf
import random
from sklearn.neighbors import KNeighborsClassifier
import timeit

####################################################
import tensorflow as tf
from embed import DataEmbedding
from attn import ProbAttention, FullAttention, AttentionLayer
from encoder import ConvLayer, Encoder, EncoderLayer


# parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
#
# parser.add_argument('--model', type=str, required=True, default='informer', help='model of the experiment')
#
# parser.add_argument('--data', type=str, required=True, default='encoder_inputs_xyz', help='data')
# parser.add_argument('--features', type=str, default='M', help='features [S, M]')
# parser.add_argument('--target', type=str, default='OT', help='target feature')
#
# parser.add_argument('--seq_len', type=int, default=50, help='input series length')
# parser.add_argument('--enc_in', type=int, default=60, help='encoder input size')
# parser.add_argument('--c_in', type=int, default=60, help='input size')
# parser.add_argument('--c_out', type=int, default=60, help='output size')
# parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
# parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
# parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
# parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
# parser.add_argument('--factor', type=int, default=5, help='prob sparse factor')
#
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
# parser.add_argument('--attn', type=str, default='prob', help='attention [prob, full]')
# parser.add_argument('--embed', type=str, default='fixed', help='embedding type [fixed, learned]')
# parser.add_argument('--activation', type=str, default='gelu', help='activation')
# parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
#
# parser.add_argument('--itr', type=int, default=2, help='each params run iteration')
# parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
# parser.add_argument('--des', type=str, default='test', help='exp description')
# parser.add_argument('--batch_size', type=int, default=64, help='input data batch size')
#
# parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
# parser.add_argument('--gpu', type=int, default=0, help='gpu')
#
# args = parser.parse_args()

# Exp = ExpInformer

#####################################################

# path = "D:/ZC/T1/Predict-Cluster-master/ucla_data/"

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

import math
import json
import numpy as np
import tensorflow as tf
import random
from sklearn.neighbors import KNeighborsClassifier
import timeit
import pickle


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# Load data

# change to your own path
path = "D:/ZC/prepressdate NTU120/"
# train data
# train_data = load_data(path+'cross_setup_data/raw_train_data.pkl')
# test_data = load_data(path+'cross_setup_data/raw_test_data.pkl')

train_data = load_data(path+'cross_subject_data/raw_train_data.pkl')
test_data = load_data(path+'cross_subject_data/raw_test_data.pkl')
# train_data = load_data(path + 'cross_view_data/raw_train_data.pkl')
# test_data = load_data(path + 'cross_view_data/raw_test_data.pkl')
print("Size of training data: ", len(train_data))
print("Size of test data: ", len(test_data))


# normalize data

def normalize_video(video):
    max_75 = np.amax(video, axis=0)
    min_75 = np.amin(video, axis=0)
    max_x = np.max([max_75[i] for i in range(0, 75, 3)])
    max_y = np.max([max_75[i] for i in range(1, 75, 3)])
    max_z = np.max([max_75[i] for i in range(2, 75, 3)])
    min_x = np.min([min_75[i] for i in range(0, 75, 3)])
    min_y = np.min([min_75[i] for i in range(1, 75, 3)])
    min_z = np.min([min_75[i] for i in range(2, 75, 3)])
    norm = np.zeros_like(video)
    for i in range(0, 75, 3):
        norm[:, i] = 2 * (video[:, i] - min_x) / (max_x - min_x) - 1
        norm[:, i + 1] = 2 * (video[:, i + 1] - min_y) / (max_y - min_y) - 1
        norm[:, i + 2] = 2 * (video[:, i + 2] - min_z) / (max_z - min_z) - 1
    return norm


for i in range(len(train_data)):
    train_data[i]['input'] = normalize_video(np.array(train_data[i]['input']))
for i in range(len(test_data)):
    test_data[i]['input'] = normalize_video(np.array(test_data[i]['input']))

# downsample

import math

dsamp_train = []
for i in range(len(train_data)):

    val = np.asarray(train_data[i]['input'])
    if val.shape[0] > 50:
        new_val = np.zeros((50, 75))
        diff = math.floor(val.shape[0] / 50)
        idx = 0
        for i in range(0, val.shape[0], diff):
            new_val[idx, :] = val[i, :]
            # new_val[random.randint(2, 48), :] = 0.0001
            # new_val[random.randint(2, 48), :] = 0.0001
            # new_val[random.randint(2, 48), :] = 0.0001


            idx += 1
            if idx >= 50:
                break
        dsamp_train.append(new_val)
    else:
        dsamp_train.append(val)

dsamp_test = []
for i in range(len(test_data)):
    val = np.asarray(test_data[i]['input'])
    if val.shape[0] > 50:
        new_val = np.zeros((50, 75))
        diff = math.floor(val.shape[0] / 50)
        idx = 0
        for i in range(0, val.shape[0], diff):
            new_val[idx, :] = val[i, :]
            idx += 1
            if idx >= 50:
                break
        dsamp_test.append(new_val)
    else:
        dsamp_test.append(val)

# dsamp_train = []
# for i in range(len(train_data)):
#
#     val = np.asarray(train_data[i]['input'])
#     if val.shape[0] > 50 and 8/10*val.shape[0]<50:
#         new_val = np.zeros((50, 75))
#         diff = math.floor(val.shape[0] / 50)
#         idx = 0
#         for i in range(0, val.shape[0], diff):
#             new_val[idx, :] = val[i, :]
#             # new_val[random.randint(2, 48), :] = 0.0001
#             # new_val[random.randint(2, 48), :] = 0.0001
#             # new_val[random.randint(2, 48), :] = 0.0001
#
#
#             idx += 1
#             if idx >= 50:
#                 break
#         dsamp_train.append(new_val)
#     elif 8/10 * val.shape[0] > 50:
#         new_val = np.zeros((50, 75))
#         diff = math.floor(8/10 * val.shape[0] / 50)
#         idx = 0
#         a = math.floor(1/10 * val.shape[0])
#         b = math.floor(9/10 * val.shape[0])
#         for i in range(a, b, diff):
#             new_val[idx, :] = val[i, :]
#             idx += 1
#             if idx >= 50:
#                 break
#         dsamp_train.append(new_val)
#     else:
#         dsamp_train.append(val)
#
# dsamp_test = []
# for i in range(len(test_data)):
#     val = np.asarray(test_data[i]['input'])
#     if val.shape[0] > 50 and 8 / 10 * val.shape[0] < 50:
#         new_val = np.zeros((50, 75))
#         diff = math.floor(val.shape[0] / 50)
#         idx = 0
#         for i in range(0, val.shape[0], diff):
#             new_val[idx, :] = val[i, :]
#             # new_val[random.randint(2, 48), :] = 0.0001
#             # new_val[random.randint(2, 48), :] = 0.0001
#             # new_val[random.randint(2, 48), :] = 0.0001
#
#             idx += 1
#             if idx >= 50:
#                 break
#         dsamp_test.append(new_val)
#     elif 8 / 10 * val.shape[0] > 50:
#         new_val = np.zeros((50, 75))
#         diff = math.floor(8 / 10 * val.shape[0] / 50)
#         idx = 0
#         a = math.floor(1 / 10 * val.shape[0])
#         b = math.floor(9 / 10 * val.shape[0])
#         for i in range(a, b, diff):
#             new_val[idx, :] = val[i, :]
#             idx += 1
#             if idx >= 50:
#                 break
#         dsamp_test.append(new_val)
#     else:
#         dsamp_test.append(val)




# Model
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell

class LinearSpaceDecoderWrapper(RNNCell):
    """Operator adding a linear encoder to an RNN cell"""

    def __init__(self, cell, output_size):
        """Create a cell with a linear encoder in space.

        Args:
          cell: an RNNCell. The input is passed through a linear layer.

        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

        print('output_size = {0}'.format(output_size))
        print(' state_size = {0}'.format(self._cell.state_size))

        # Tuple if multi-rnn
        if isinstance(self._cell.state_size, tuple):

            # Fine if GRU...
            insize = self._cell.state_size[-1]

            # LSTMStateTuple if LSTM
            if isinstance(insize, LSTMStateTuple):
                # insize = self._cell.state_size[-1]

                insize = insize.h

        else:
            # Fine if not multi-rnn
            insize = self._cell.state_size

        self.w_out = tf.get_variable("proj_w_out",
                                     [insize, output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
        self.b_out = tf.get_variable("proj_b_out", [output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.linear_output_size = output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_output_size

    def __call__(self, inputs, state, scope=None):
        """Use a linear layer and pass the output to the cell."""

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)

        # Apply the multiplication to everything
        output = tf.matmul(output, self.w_out) + self.b_out

        return output, new_state


class ResidualWrapper(RNNCell):
    """Operator adding residual connections to a given cell."""

    def __init__(self, cell):
        """Create a cell with added residual connection.

        Args:
          cell: an RNNCell. The input is added to the output.

        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell and add a residual connection."""

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)

        # Add the residual connection
        output = tf.add(output, inputs)

        return output, new_state


from tensorflow.python.ops.rnn import _transpose_batch_time


class Seq2SeqModelFS(object):
    def __init__(self, max_seq_len,max_seq_lent, input_size, rnn_size, batch_size, lr, train_keep_prob, decay_rate=0.95,
                 dtype=tf.float32):
        self.max_seq_len = max_seq_len
        self.max_seq_lent = max_seq_lent

        self.rnn_size = rnn_size
        self.batch_size = tf.placeholder_with_default(batch_size, shape=())
        self.input_size = input_size
        self.lr = tf.Variable(float(lr), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.lr.assign(self.lr * decay_rate)
        self.train_keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.global_step = tf.Variable(0, trainable=False)
        self.Informer = Informer(enc_in=75, c_out=75, seq_len=50, factor=5, d_model=512, n_heads=8, e_layers=3,
                                 d_ff=512, dropout=0.05, attn='prob', embed='fixed', activation='gelu')
        self.Informert = Informert(enc_in=50, c_out=50, seq_len=75, factor=5, d_model=512, n_heads=8, e_layers=3,
                                 d_ff=512, dropout=0.05, attn='prob', embed='fixed', activation='gelu')
        # self.InformerStack = InformerStack(enc_in=75, c_out=75, seq_len=50, factor=5, d_model=512, n_heads=8,
        #                                    e_layers=[3, 2, 1], d_ff=512, dropout=0.05, attn='prob', embed='fixed', activation='gelu')
        print('rnn_size = {0}'.format(rnn_size))

        with tf.variable_scope("inputs"):
            self.enc_xyz = tf.placeholder(dtype, shape=[None, self.max_seq_len, 512], name='enc_xyz')
            self.dec_xyz = tf.placeholder(dtype, shape=[None, self.max_seq_len, 512], name='dec_xyz')
            self.seq_len = tf.placeholder(tf.int32, [None])
            mask = tf.sign(tf.reduce_max(tf.abs(self.enc_xyz[:, 1:, :]), 2))
            self.enc_xyzt = tf.placeholder(dtype, shape=[None, self.max_seq_lent, 512], name='enc_xyzt')
            self.dec_xyzt = tf.placeholder(dtype, shape=[None, self.max_seq_lent, 512], name='dec_xyzt')
            self.seq_lent = tf.placeholder(tf.int32, [None])
            maskt = tf.sign(tf.reduce_max(tf.abs(self.enc_xyzt[:, 1:, :]), 2))
        with tf.variable_scope("prediction"):
            with tf.variable_scope("encoder"):
                with tf.variable_scope("encoder_xyz", reuse=tf.AUTO_REUSE):
                    cell_fw_xyz = [tf.nn.rnn_cell.BasicRNNCell(self.rnn_size // 2) for i in range(3)]
                    cell_bw_xyz = [tf.nn.rnn_cell.BasicRNNCell(self.rnn_size // 2) for i in range(3)]
                    tuple_xyz = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw_xyz, cell_bw_xyz, self.enc_xyz,
                                                                               dtype=tf.float32,
                                                                               sequence_length=self.seq_len)
                    cell_fw_xyzt = [tf.nn.rnn_cell.BasicRNNCell(self.rnn_size // 2) for i in range(3)]
                    cell_bw_xyzt = [tf.nn.rnn_cell.BasicRNNCell(self.rnn_size // 2) for i in range(3)]
                    tuple_xyzt = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw_xyzt, cell_bw_xyzt,
                                                                                self.enc_xyzt,
                                                                                dtype=tf.float32,
                                                                                sequence_length=self.seq_lent)
                    bi_xyz_h = tf.concat((tuple_xyz[1][-1], tuple_xyz[2][-1]), -1)
                    bi_xyz_ht = tf.concat((tuple_xyzt[1][-1], tuple_xyzt[2][-1]), -1)

                    self.enc_states = tuple_xyz[0]  # all encoder states [batch,time,2048]
                    self.enc_statest = tuple_xyzt[0]  # all encoder states [batch,time,2048]

                    self.bi_xyz_h = bi_xyz_h
                    self.bi_xyz_ht = bi_xyz_ht

                self.knn_state = self.bi_xyz_h
                self.knn_statet = self.bi_xyz_ht
            with tf.variable_scope("decoder"):
                with tf.variable_scope("decoder_xyz", reuse=tf.AUTO_REUSE):
                    cell_xyz__ = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)
                    cell_xyz_ = LinearSpaceDecoderWrapper(cell_xyz__, 512)
                    cell_xyz = ResidualWrapper(cell_xyz_)

                    def loop_fn(time, cell_output, cell_state, loop_state):
                        cell_state
                        """
                        Loop function that allows to control input to the rnn cell and manipulate cell outputs.
                        :param time: current time step
                        :param cell_output: output from previous time step or None if time == 0
                        :param cell_state: cell state from previous time step
                        :param loop_state: custom loop state to share information between different iterations of this loop fn
                        :return: tuple consisting of
                          elements_finished: tensor of size [bach_size] which is True for sequences that have reached their end,
                            needed because of variable sequence size
                          next_input: input to next time step
                          next_cell_state: cell state forwarded to next time step
                          emit_output: The first return argument of raw_rnn. This is not necessarily the output of the RNN cell,
                            but could e.g. be the output of a dense layer attached to the rnn layer.
                          next_loop_state: loop state forwarded to the next time step
                        """
                        if cell_output is None:
                            # time == 0, used for initialization before first call to cell
                            next_cell_state = self.bi_xyz_h
                            # the emit_output in this case tells TF how future emits look
                            emit_output = tf.zeros([512])
                        else:
                            # t > 0, called right after call to cell, i.e. cell_output is the output from time t-1.
                            # here you can do whatever ou want with cell_output before assigning it to emit_output.
                            # In this case, we don't do anything
                            next_cell_state = self.bi_xyz_h  # NOTE:IF NO-FS, use cell_state#
                            emit_output = cell_output

                            # check which elements are finished
                        elements_finished = (time >= max_seq_len - 1)
                        finished = tf.reduce_all(elements_finished)

                        # assemble cell input for upcoming time step
                        current_output = emit_output if cell_output is not None else None
                        # input_original = inputs_ta.read(time)  # tensor of shape (None, input_dim)
                        input_original = self.enc_xyz[:, 0, :]
                        if current_output is None:
                            # this is the initial step, i.e. there is no output from a previous time step, what we feed here
                            # can highly depend on the data. In this case we just assign the actual input in the first time step.
                            next_in = input_original
                        else:
                            # time > 0, so just use previous output as next input
                            # here you could do fancier things, whatever you want to do before passing the data into the rnn cell
                            # if here you were to pass input_original than you would get the normal behaviour of dynamic_rnn
                            next_in = current_output

                        next_input = tf.cond(finished,
                                             lambda: tf.zeros([self.batch_size, 512], dtype=tf.float32),
                                             # copy through zeros
                                             lambda: next_in)  # if not finished, feed the previous output as next input

                        # set shape manually, otherwise it is not defined for the last dimensions
                        next_input.set_shape([None, 512])

                        # loop state not used in this example
                        next_loop_state = None
                        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

                    outputs_ta, def_final_state_xyz, _ = tf.nn.raw_rnn(cell_xyz, loop_fn)
                    self.dec_outputs_xyz = _transpose_batch_time(outputs_ta.stack())

                with tf.variable_scope("decoder_xyz", reuse=tf.AUTO_REUSE):
                    cell_xyzt__ = tf.nn.rnn_cell.GRUCell(self.rnn_size)
                    cell_xyzt_ = LinearSpaceDecoderWrapper(cell_xyzt__, 512)
                    cell_xyzt = ResidualWrapper(cell_xyzt_)

                    def loop_fnt(time, cell_output, cell_state, loop_state):
                        cell_state
                        """
                        Loop function that allows to control input to the rnn cell and manipulate cell outputs.
                        :param time: current time step
                        :param cell_output: output from previous time step or None if time == 0
                        :param cell_state: cell state from previous time step
                        :param loop_state: custom loop state to share information between different iterations of this loop fn
                        :return: tuple consisting of
                          elements_finished: tensor of size [bach_size] which is True for sequences that have reached their end,
                            needed because of variable sequence size
                          next_input: input to next time step
                          next_cell_state: cell state forwarded to next time step
                          emit_output: The first return argument of raw_rnn. This is not necessarily the output of the RNN cell,
                            but could e.g. be the output of a dense layer attached to the rnn layer.
                          next_loop_state: loop state forwarded to the next time step
                        """
                        if cell_output is None:
                            # time == 0, used for initialization before first call to cell
                            next_cell_state = self.bi_xyz_ht
                            # the emit_output in this case tells TF how future emits look
                            emit_output = tf.zeros([512])
                        else:
                            # t > 0, called right after call to cell, i.e. cell_output is the output from time t-1.
                            # here you can do whatever ou want with cell_output before assigning it to emit_output.
                            # In this case, we don't do anything
                            next_cell_state = self.bi_xyz_ht  # NOTE:IF NO-FS, use cell_state#
                            emit_output = cell_output

                            # check which elements are finished
                        elements_finished = (time >= max_seq_lent - 1)
                        finished = tf.reduce_all(elements_finished)

                        # assemble cell input for upcoming time step
                        current_output = emit_output if cell_output is not None else None
                        # input_original = inputs_ta.read(time)  # tensor of shape (None, input_dim)
                        input_original = self.enc_xyzt[:, 0, :]
                        if current_output is None:
                            # this is the initial step, i.e. there is no output from a previous time step, what we feed here
                            # can highly depend on the data. In this case we just assign the actual input in the first time step.
                            next_in = input_original
                        else:
                            # time > 0, so just use previous output as next input
                            # here you could do fancier things, whatever you want to do before passing the data into the rnn cell
                            # if here you were to pass input_original than you would get the normal behaviour of dynamic_rnn
                            next_in = current_output

                        next_input = tf.cond(finished,
                                             lambda: tf.zeros([self.batch_size, 512], dtype=tf.float32),
                                             # copy through zeros
                                             lambda: next_in)  # if not finished, feed the previous output as next input

                        # set shape manually, otherwise it is not defined for the last dimensions
                        next_input.set_shape([None, 512])

                        # loop state not used in this example
                        next_loop_state = None
                        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

                    outputs_tat, def_final_state_xyzt, _ = tf.nn.raw_rnn(cell_xyzt, loop_fnt)
                    self.dec_outputs_xyzt = _transpose_batch_time(outputs_tat.stack())


            def loss_with_mask(pred, gt, mask):
                loss = tf.reduce_sum(tf.abs(pred - gt), 2) * mask
                loss = tf.reduce_sum(loss, 1)
                loss /= tf.reduce_sum(mask, 1)
                loss = tf.reduce_mean(loss)
                return loss
        with tf.variable_scope("pred_xyz", reuse=tf.AUTO_REUSE):
            pred_xyz2xyz = self.dec_outputs_xyz
            pred_xyz2xyzt = self.dec_outputs_xyzt

            self.loss = loss_with_mask(pred_xyz2xyz, self.enc_xyz[:, 1:, :], mask)
            self.losst = loss_with_mask(pred_xyz2xyzt, self.enc_xyzt[:, 1:, :], maskt)



        opt = tf.train.AdamOptimizer(self.lr)
        gradients, self.pred_vars = zip(*opt.compute_gradients(self.loss))
        gradientst, self.pred_varst = zip(*opt.compute_gradients(self.losst))

        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 25)
        clipped_gradientst, normt = tf.clip_by_global_norm(gradientst, 25)

        self.updates = opt.apply_gradients(zip(clipped_gradients, self.pred_vars), global_step=self.global_step)
        self.updatest = opt.apply_gradients(zip(clipped_gradientst, self.pred_varst), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.savert = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def step(self, session, encoder_inputs_xyz, decoder_inputs_xyz, seq_len, forward_only):
        encoder_inputs_xyz = self.Informer(encoder_inputs_xyz)
        decoder_inputs_xyz = self.Informer(decoder_inputs_xyz)

        encoder_inputs_xyz = list(encoder_inputs_xyz)
        decoder_inputs_xyz = list(decoder_inputs_xyz)

        encoder_inputs_xyz[0] = encoder_inputs_xyz[0].detach().numpy()
        decoder_inputs_xyz[0] = decoder_inputs_xyz[0].detach().numpy()

        # encoder_inputs_xyz = self.InformerStack(encoder_inputs_xyz)
        # decoder_inputs_xyz = self.InformerStack(decoder_inputs_xyz)
        #
        # encoder_inputs_xyz = list(encoder_inputs_xyz)
        # decoder_inputs_xyz = list(decoder_inputs_xyz)
        #
        # encoder_inputs_xyz[0] = encoder_inputs_xyz[0].detach().numpy()
        # decoder_inputs_xyz[0] = decoder_inputs_xyz[0].detach().numpy()

        if not forward_only:
            input_feed = {self.enc_xyz: encoder_inputs_xyz[0],
                          self.dec_xyz: decoder_inputs_xyz[0],
                          self.seq_len: seq_len}
            output_feed = [self.updates, self.loss]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1]

    def stept(self, session, encoder_inputs_xyzt, decoder_inputs_xyzt, seq_lent, forward_only):
        # encoder_inputs_xyzt = encoder_inputs_xyz.transpose(0, 2, 1)
        # decoder_inputs_xyzt = decoder_inputs_xyz.transpose(0, 2, 1)


        encoder_inputs_xyzt = self.Informert(encoder_inputs_xyzt)
        decoder_inputs_xyzt = self.Informert(decoder_inputs_xyzt)

        encoder_inputs_xyzt = list(encoder_inputs_xyzt)
        decoder_inputs_xyzt = list(decoder_inputs_xyzt)

        encoder_inputs_xyzt[0] = encoder_inputs_xyzt[0].detach().numpy()
        decoder_inputs_xyzt[0] = decoder_inputs_xyzt[0].detach().numpy()

        if not forward_only:
            input_feed = {self.enc_xyzt: encoder_inputs_xyzt[0],
                          self.dec_xyzt: decoder_inputs_xyzt[0],
                          self.seq_lent: seq_lent}
            output_feed = [self.updatest, self.losst]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1]
def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def mini_batch_classify(feature_xyz, labels, seq_len, batch_size):
    for start in range(0, len(feature_xyz), batch_size):
        end = min(start + batch_size, len(feature_xyz))
        yield feature_xyz[start:end], labels[start:end], seq_len[start:end]


# Hyperparameter
max_seq_lent = 19
max_seq_len = 13
rnn_size = 2048
input_size = 75
batch_size = 64
lr = 0.0001
train_keep_prob = 1.0
iterations = 50

# Model initialization

tf.reset_default_graph()
# FW
model = Seq2SeqModelFS(max_seq_len, max_seq_lent,input_size, rnn_size, batch_size, lr, train_keep_prob)
sess = get_session()
sess.run(tf.global_variables_initializer())

# Evaluate the acc before training

fea = []
lab = []
seq_len_new = []
for idx, data in enumerate(train_data):
    label = data["label"]
    val = np.asarray(data["input"])
    raw_len = val.shape[0]
    if raw_len > 50:
        seq_len_new.append(50)
        fea.append(dsamp_train[idx])
    else:
        seq_len_new.append(raw_len)
        pad_data = np.zeros((50, 75))
        pad_data[:raw_len, :] = dsamp_train[idx]
        fea.append(pad_data)
    one_hot_label = np.zeros((120,))
    one_hot_label[label] = 1.
    lab.append(one_hot_label)

test_fea = []
test_lab = []
test_seq_len_new = []
for idx, data in enumerate(test_data):
    label = data["label"]
    val = np.asarray(data["input"])
    raw_len = val.shape[0]
    if raw_len > 50:
        test_seq_len_new.append(50)
        test_fea.append(dsamp_test[idx])
    else:
        test_seq_len_new.append(raw_len)
        pad_data = np.zeros((50, 75))
        pad_data[:raw_len, :] = dsamp_test[idx]
        test_fea.append(pad_data)
    one_hot_label = np.zeros((120,))
    one_hot_label[label] = 1.
    test_lab.append(one_hot_label)


def get_feature(model, session, feature_xyz, batch_size, seq_len):
    feature_xyz = model.Informer(feature_xyz)
    # feature_xyz = model.InformerStack(feature_xyz)
    feature_xyz = list(feature_xyz)

    feature_xyz[0] = feature_xyz[0].detach().numpy()
    input_feed = {model.enc_xyz: feature_xyz[0],
                  model.dec_xyz: feature_xyz[0],
                  model.seq_len: seq_len, model.batch_size: batch_size}
    output_feed = [model.knn_state]
    outputs = session.run(output_feed, input_feed)
    return outputs[0]

def get_featuret(model, session,  feature_xyzt, batch_size, seq_lent):
##################################################################
    feature_xyzt = model.Informert(feature_xyzt)
    feature_xyzt = list(feature_xyzt)
    feature_xyzt[0] = feature_xyzt[0].detach().numpy()

    input_feed = {model.enc_xyzt: feature_xyzt[0],
                  model.dec_xyzt: feature_xyzt[0],
                  model.seq_lent: seq_lent, model.batch_size: batch_size}
    output_feed = [model.knn_statet]
    outputs = session.run(output_feed, input_feed)
    return outputs[0]

import copy


def FEATURET(a):
    b = copy.copy(a)
    for k in range(0, len(b)):


        a[k] = a[k].transpose(1, 0)
        feature_xyzt = a
    return feature_xyzt,b

def SEQL(a):
    b = copy.copy(a)
    for k in range(0, len(b)):
        a[k] = 75
        seq_lent = a

    return seq_lent,b

# knn_feature = []
# knn_featurem = []
#
# knn_featuret = []
# for encoder_inputs, labels, seq_len_enc in mini_batch_classify(fea, lab, seq_len_new, batch_size=64):
#     encoder_inputst, b = FEATURET(encoder_inputs)
#     encoder_inputs = b
#
#     seq_len_enct, c = SEQL(seq_len_enc)
#     seq_len_enc = c
#
#     result = get_feature(model, sess, encoder_inputs, len(encoder_inputs), seq_len_enc)
#     resultt = get_featuret(model, sess, encoder_inputst, len(encoder_inputst), seq_len_enct)
#
#     knn_feature.append(result)
#     knn_featuret.append(resultt)
# knn_feature = np.vstack(knn_feature)
# knn_featuret = np.vstack(knn_featuret)
# knn_featurem = np.concatenate((knn_feature, knn_featuret), axis=1, )
#
# test_knn_feature = []
# test_knn_featuret = []
# test_knn_featurem = []
# for encoder_inputs, labels, seq_len_enc in mini_batch_classify(test_fea, test_lab, test_seq_len_new,
#                                                                batch_size=64):
#     encoder_inputst, b = FEATURET(encoder_inputs)
#     encoder_inputs = b
#
#     seq_len_enct, c = SEQL(seq_len_enc)
#     seq_len_enc = c
#     result = get_feature(model, sess, encoder_inputs, len(encoder_inputs), seq_len_enc)
#     resultt = get_featuret(model, sess, encoder_inputst, len(encoder_inputst), seq_len_enct)
#     test_knn_feature.append(result)
#     test_knn_featuret.append(resultt)
# test_knn_feature = np.vstack(test_knn_feature)
# test_knn_featuret = np.vstack(test_knn_featuret)
# test_knn_featurem = np.concatenate((test_knn_feature, test_knn_featuret), axis=1, )

from sklearn.neighbors import KNeighborsClassifier
#
# neigh = KNeighborsClassifier(n_neighbors=1, metric='cosine')
# neigh.fit(knn_feature, np.argmax(lab, axis=1))
#
# neigh.score(test_knn_feature, np.argmax(test_lab, axis=1))
# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.reshape(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


def mini_batch(data, seq_len, input_size, batch_size):
    encoder_inputs = np.zeros((batch_size, seq_len, input_size), dtype=float)
    seq_len_enc = np.zeros((batch_size,), dtype=float)
    decoder_inputs = np.zeros((batch_size, seq_len, input_size), dtype=float)
    data_len = len(data)
    for i in range(batch_size):
        index = np.random.choice(data_len)
        data_sel = data[index]
        encoder_inputs[i, :data_sel.shape[0], :] = np.copy(data_sel)
        seq_len_enc[i] = data_sel.shape[0]
    return encoder_inputs, decoder_inputs, seq_len_enc

def mini_batcht(data, seq_len, input_size, batch_size):
    encoder_inputst = np.zeros((batch_size, seq_len,input_size), dtype=float)
    seq_len_enct = np.zeros((batch_size,), dtype=float)

    decoder_inputst = np.zeros((batch_size, seq_len,input_size), dtype=float)
    data_len = len(data)

    for i in range(batch_size):
        index = np.random.choice(data_len)
        data_sel = data[index]
        encoder_inputst[i, :data_sel.shape[0], :] = np.copy(data_sel)
        seq_len_enct[i] = data_sel.shape[1]
    encoder_inputst = encoder_inputst.transpose(0, 2, 1)
    decoder_inputst = decoder_inputst.transpose(0, 2, 1)

    return encoder_inputst, decoder_inputst, seq_len_enct
# training loop


start_time = timeit.default_timer()
knn_score = []
train_loss_li = []
train_losst_li = []

max_score = 0.0
for i in range(1, iterations + 1):
    encoder_inputs, decoder_inputs, seq_len_enc = mini_batch(dsamp_train, seq_len=50, input_size=75, batch_size=256)
    encoder_inputst, decoder_inputst, seq_len_enct = mini_batcht(dsamp_train, seq_len=50, input_size=75,
                                                                         batch_size=256)
    _, train_loss = model.step(sess, encoder_inputs, decoder_inputs, seq_len_enc, False)
    _, train_losst= model.stept(sess, encoder_inputst, decoder_inputst, seq_len_enct, False)

    if i % 1 == 0:
        print("step {0}:train loss:{1:.4f}".format(i, train_loss))
        train_loss_li.append(train_loss)
        print("step {0}:train loss:{1:.4f}".format(i, train_losst))
        train_losst_li.append(train_losst)
        end_time = timeit.default_timer()
        print("iteration {}:".format(i), end='')
        print(" using {:.2f} sec".format(end_time - start_time))
###########################################输出时间转TXT
        # train_time = []
        # time = end_time - start_time  # 损失样例
        # train_time.append(time)  # 损失加入到列表中
        # with open("D:/ZC/T5 NTU 60/train_timefull.txt", 'a+') as train_tim:
        #     train_tim.write(str(train_time) + '\n')
        start_time = end_time
    # check knn score every 200 iterations
    # if i % 200 == 0:
        knn_feature = []
        knn_featurem = []

        knn_featuret = []
        for encoder_inputs, labels, seq_len_enc in mini_batch_classify(fea, lab, seq_len_new, batch_size=64):
            encoder_inputst , b = FEATURET(encoder_inputs)
            encoder_inputs = b

            seq_len_enct,c=SEQL(seq_len_enc)
            seq_len_enc = c


            result = get_feature(model, sess, encoder_inputs, len(encoder_inputs), seq_len_enc)
            resultt = get_featuret(model, sess, encoder_inputst, len(encoder_inputst),seq_len_enct)

            knn_feature.append(result)
            knn_featuret.append(resultt)
        knn_feature = np.vstack(knn_feature)
        knn_featuret = np.vstack(knn_featuret)
        knn_featurem = np.concatenate((knn_feature, knn_featuret), axis=1, )



        test_knn_feature = []
        test_knn_featuret = []
        test_knn_featurem = []
        for encoder_inputs, labels, seq_len_enc in mini_batch_classify(test_fea, test_lab, test_seq_len_new,
                                                                       batch_size=64):
            encoder_inputst, b = FEATURET(encoder_inputs)
            encoder_inputs = b

            seq_len_enct, c = SEQL(seq_len_enc)
            seq_len_enc = c
            result = get_feature(model, sess, encoder_inputs, len(encoder_inputs), seq_len_enc)
            resultt = get_featuret(model, sess, encoder_inputst, len(encoder_inputst),seq_len_enct)
            test_knn_feature.append(result)
            test_knn_featuret.append(resultt)
        test_knn_feature = np.vstack(test_knn_feature)
        test_knn_featuret = np.vstack(test_knn_featuret)
        test_knn_featurem = np.concatenate((test_knn_feature, test_knn_featuret), axis=1, )

        # self.fc = nn.Linear(4096, 120)
        #
        # A = self.fc(knn_featurem)
        # B = self.fc(test_knn_featurem)
        #
        # acc1, acc5 = accuracy(B, target, topk=(1, 5))


        neigh = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        neigh.fit(knn_featurem, np.argmax(lab, axis=1))
        score = neigh.score(test_knn_featurem, np.argmax(test_lab, axis=1))
        knn_score.append(score)
        print(f"knn test score at {i}th iterations: ", score)
        # save the model: change to your own path
        if (score > max_score) :

        #     plot_knn_feature = []
        #
        #     j = 0
        # if j < 10:
        #     for feature_xyz, labels, seq_len in mini_batch_classify(fea, lab, seq_len_new, batch_size=64):
        #         j += 1
        #         result = get_feature(model, sess, feature_xyz, len(feature_xyz), seq_len)
        #         plot_knn_feature.append(result)
        #     plot_knn_feature = np.vstack(plot_knn_feature)
        #     import numpy as np
        #     import matplotlib.pyplot as plt
        #     from sklearn import manifold
        #
        #     tsne = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(knn_featurem)
        #     x_min, x_max = tsne.min(0), tsne.max(0)
        #     tsne_norm = (tsne - x_min) / (x_max - x_min)
        #
        #     t = [0 for i in range(120)]
        #     tsne_t = [0 for i in range(120)]
        #     for k in range(120):
        #         t[k] = np.argmax(lab, axis=1) == k
        #         tsne_t[k] = tsne_norm[t[k]]
        #         plt.scatter(tsne_t[k][:, 0], tsne_t[k][:, 1], 1,)
        #         # , label = 't_' + str(k)
        #     plt.legend(loc='upper left')
        #     plt.show()


            model.saver.save(sess,"D:/ZC/T5 NTU 60/T5/ntu_model",global_step=i)
            max_score = score
            print("Current KNN Max Score is {}".format(max_score))
    if i % 2 == 0:
        sess.run(model.learning_rate_decay_op)


class AEC(object):
    def __init__(self, input_size, batch_size, lr, dtype=tf.float32):

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X')

        # encoder
        self.fc1 = tf.layers.dense(self.X, 2048, activation='tanh')
        self.fc2 = tf.layers.dense(self.fc1, 1024, activation='tanh')
        self.fea = tf.layers.dense(self.fc2, 512, activation='tanh')

        # decoder
        self.fc3 = tf.layers.dense(self.fea, 1024, activation='tanh')
        self.fc4 = tf.layers.dense(self.fc3, 2048, activation='tanh')


        self.Y = tf.layers.dense(self.fc4, input_size, activation=None)

        self.loss = tf.reduce_mean(tf.square(self.X - self.Y))
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.opt = self.optimizer.minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def step(self, session, feature, forward_only=False):

        if forward_only:
            output = sess.run([self.fea], feed_dict={self.X: feature})
            return output
        else:
            _, train_loss = sess.run([self.opt, self.loss], feed_dict={self.X: feature})
            return train_loss


## AEC training loop

lr = 0.0001
input_size = 4096
batch_size = 16
epochs = 10000

sess.close()
tf.reset_default_graph()
aec_model = AEC(input_size, batch_size, lr)
sess = get_session()
sess.run(tf.global_variables_initializer())


def aec_mini_batch_classify(feature_xyz, batch_size):
    for start in range(0, len(feature_xyz), batch_size):
        end = min(start + batch_size, len(feature_xyz))
        yield feature_xyz[start:end]


aec_score = 0
for epoch in range(epochs):
    for feature in aec_mini_batch_classify(knn_featurem, batch_size):
        train_loss = aec_model.step(sess, feature)

    if epoch % 100 == 0:
        print("epoch:{0},loss:{1}".format(epoch, train_loss))
        denoise_all = []
        for feature in aec_mini_batch_classify(knn_featurem, batch_size):
            denoise = aec_model.step(sess, feature, forward_only=True)
            denoise_all.append(denoise[0])
        see = np.vstack(denoise_all)

        test_denoise_all = []
        for feature in aec_mini_batch_classify(test_knn_featurem, batch_size):
            denoise = aec_model.step(sess, feature, forward_only=True)
            test_denoise_all.append(denoise[0])
        test_see = np.vstack(test_denoise_all)

        neigh = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        neigh.fit(see, np.argmax(lab, axis=1))
        t_score = neigh.score(test_see, np.argmax(test_lab, axis=1))
        print("after {0} epoch knn score {1:.3f}".format(epoch, t_score))




