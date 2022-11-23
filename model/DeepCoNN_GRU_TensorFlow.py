https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
'''
DeepCoNN
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation.
In WSDM. ACM, 425-434.
'''


import tensorflow as tf
import numpy as np

class DeepCoNN_GRU_TensorFlow(object):
    def __init__(
            self, user_length,item_length, num_classes, user_vocab_size,item_vocab_size,fm_k,n_latent,user_num,item_num,
            embedding_size,hidden_size,filter_sizes, num_filters, l2_reg_lambda=0.0,l2_reg_V=0.0):
        self.input_u = tf.placeholder(tf.int32, [None, user_length], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, item_length], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None,1],name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            #self.embedded_users = tf.expand_dims(self.embedded_user, -1)

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            #self.embedded_items = tf.expand_dims(self.embedded_item, -1)

        def dropout():
            cell = tf.contrib.rnn.GRUCell(embedding_size)
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        with tf.name_scope("user_gru"):
            cells = [dropout() for _ in range(hidden_size)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,inputs=self.embedded_user, dtype=tf.float32)
            self.user_last = _outputs[:, -1, :]

        with tf.name_scope("item_gru"):
            cells = [dropout() for _ in range(hidden_size)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.embedded_item, dtype=tf.float32)
            self.item_last = _outputs[:, -1, :]


        with tf.name_scope("get_fea"):
            Wu = tf.get_variable(
                "Wu",
                shape=[num_filters_total, n_latent],
                initializer=tf.contrib.layers.xavier_initializer())
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_fea=tf.matmul(self.user_last, Wu) + bu
            #self.u_fea = tf.nn.dropout(self.u_fea,self.dropout_keep_prob)
            Wi = tf.get_variable(
                "Wi",
                shape=[num_filters_total, n_latent],
                initializer=tf.contrib.layers.xavier_initializer())
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_fea = tf.matmul(self.item_last, Wi) + bi
            #self.i_fea=tf.nn.dropout(self.i_fea,self.dropout_keep_prob)


        with tf.name_scope('fm'):
            self.z=tf.nn.relu(tf.concat(1,[self.u_fea,self.i_fea]))

            #self.z=tf.nn.dropout(self.z,self.dropout_keep_prob)

            WF1=tf.Variable(
                    tf.random_uniform([n_latent*2, 1], -0.1, 0.1), name='fm1')
            Wf2=tf.Variable(
                tf.random_uniform([n_latent*2, fm_k], -0.1, 0.1), name='fm2')
            one=tf.matmul(self.z,WF1)

            inte1=tf.matmul(self.z,Wf2)
            inte2=tf.matmul(tf.square(self.z),tf.square(Wf2))

            inter=(tf.square(inte1)-inte2)*0.5

            inter=tf.nn.dropout(inter,self.dropout_keep_prob)

            inter=tf.reduce_sum(inter,1,keep_dims=True)
            print inter
            b=tf.Variable(tf.constant(0.1), name='bias')

            self.predictions =one+inter+b
            print self.predictions

        with tf.name_scope("loss"):
            #losses = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))
            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy =tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))
