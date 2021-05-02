"""
Last update: 21.03.12. by KS.Kwon
"""


import tensorflow as tf
import numpy as np 

from tensorflow import keras
from tensorflow.keras import layers 

from inits import glorot, He, zeros


# MESSAGE PASSING function
class GGNNMsgPass(layers.Layer):
    def __init__(self, 
                 n_nodes,
                 n_edge_class,
                 input_dim,
                 batch_size,
                 eW_in=None,                 # matrices_in in original MPNN code
                 eW_out=None,                # matrices_out in original MPNN code
                 b=None,                     # message bias in origianl MPNN code
                 dropout=0.5,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_nodes = n_nodes
        self.n_edge_class = n_edge_class
        self.input_dim = input_dim
        self.batch_size = batch_size

        if eW_in is None:
            self.eW_in = tf.Variable(
                name='eW_in',
                initial_value=glorot(
                    [self.n_edge_class, self.input_dim, self.input_dim]
                )
            )
        else:
            self.eW_in = tf.Variable(name='eW_in', initial_value=eW_in)

        if eW_out is None:
            self.eW_out = tf.Variable(
                name='eW_out',
                initial_value=glorot(
                    [self.n_edge_class, self.input_dim, self.input_dim]
                )
            )
        else:
            self.eW_out = tf.Variable(name='eW_out', initial_value=eW_out)

        if b is None:
            self.b = tf.Variable(name='bias',
                initial_value=zeros([2*self.input_dim])
            )
        else:
            self.b = tf.Variable(name='bias', initial_value=b)

    def call(self, M_feat, M_adj):
        """
        Args:
            M_feat:     Feature matrix
                        [batch_size, n_nodes, input_dim]
            M_adj:      Adjacency matrix 
                        [batch_size, n_nodes, n_nodes]
        """
        
        # Generate Edge Class variable set (weight & bias)
        zeros = tf.constant(0.0, shape=[1, self.input_dim, self.input_dim])
        eW_in = tf.concat([zeros, self.eW_in], axis=0)
        eW_out = tf.concat([zeros, self.eW_out], axis=0)

        # Generate messages
        M_adj_T = tf.transpose(M_adj, [0, 2, 1])
        a_in = tf.gather(eW_in, M_adj)
        a_out = tf.gather(eW_out, M_adj_T)

        a_in = tf.transpose(a_in, [0, 1, 3, 2, 4])
        a_in = tf.reshape(
            a_in, 
            [-1, self.n_nodes*self.input_dim, self.n_nodes*self.input_dim]
        )

        a_out = tf.transpose(a_out, [0, 1, 3, 2, 4])
        a_out = tf.reshape(
            a_out,
            [-1, self.n_nodes*self.input_dim, self.n_nodes*self.input_dim]
        )

        messages = self.message_pass(M_feat, a_in, a_out)

        return messages

    def message_pass(self, M_feat, a_in, a_out):
        h_flat = tf.reshape(
            M_feat, [self.batch_size, self.n_nodes*self.input_dim, 1]
        )

        a_in_mul = tf.reshape(
            tf.matmul(a_in, h_flat), [self.batch_size*self.n_nodes, self.input_dim]
        )
        a_out_mul = tf.reshape(
            tf.matmul(a_out, h_flat), [self.batch_size*self.n_nodes, self.input_dim]
        )
        
        a_tmp = tf.concat([a_in_mul, a_out_mul], axis=1)
        a_t = a_tmp + self.b
        
        messages = tf.reshape(a_t, [self.batch_size, self.n_nodes, 2*self.input_dim])

        return messages


# VERTEX UPDATE function
class GRUUpdate(layers.Layer):
    def __init__(self,
                 n_nodes,
                 batch_size,
                 input_dim,
                 W_z=None,
                 W_r=None,
                 W_h=None,
                 U_z=None,
                 U_r=None,
                 U_h=None,
                 dropout=0.5,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.input_dim = input_dim

        if W_z is None:
            self.W_z = tf.Variable(
                name='W_z',
                initial_value=glorot(
                    [2*self.input_dim, self.input_dim]
                )
            )
        else:
            self.W_z = tf.Variable(name='W_z', initial_value=W_z)

        if W_r is None:
            self.W_r = tf.Variable(
                name='W_r',
                initial_value=glorot(
                    [2*self.input_dim, self.input_dim]
                )
            )
        else:
            self.W_r = tf.Variable(name='W_r', initial_value=W_r)

        if W_h is None:
            self.W_h = tf.Variable(
                name='W_h',
                initial_value=glorot(
                    [2*self.input_dim, self.input_dim]
                )
            )
        else:
            self.W_h = tf.Variable(name='W_h', initial_value=W_h)

        if U_z is None:
            self.U_z = tf.Variable(
                name='U_z',
                initial_value=glorot(
                    [self.input_dim, self.input_dim]
                )
            )
        else:
            self.U_z = tf.Variable(name='U_z', initial_value=U_z)

        if U_r is None:
            self.U_r = tf.Variable(
                name='U_r',
                initial_value=glorot(
                    [self.input_dim, self.input_dim]
                )
            )
        else:
            self.U_r = tf.Variable(name='U_r', initial_value=U_r)

        if U_h is None:
            self.U_h = tf.Variable(
                name='U_h',
                initial_value=glorot(
                    [self.input_dim, self.input_dim]
                )
            )
        else:
            self.U_h = tf.Variable(name='U_h', initial_value=U_h)

    def call(self, M_feat, messages, mask):
        """
        Args:
            M_feat:     input feature matrix
                        [batch_size, n_nodes, input_dim] 
            
            messages:   hidden feature matrix generated from GGNNMsgPass Layer
                        [batch_size, n_nodes, 2*input_dim] 
            
            mask:       masking list which indicate real nodes.
                        for real node 1, dummy node 0
                        [batch_size, n_nodes]
        """     
        
        # Reshape input data
        M_feat_rs = tf.reshape(
            M_feat, [self.batch_size*self.n_nodes, self.input_dim]
        )
        messages_rs = tf.reshape(
            messages, [self.batch_size*self.n_nodes, 2*self.input_dim]
        )

        # Intermediate products z_t, r_t, h_tilde
        z_t = tf.nn.sigmoid(
            tf.matmul(M_feat_rs, self.U_z) + tf.matmul(messages_rs, self.W_z)
        )
        r_t = tf.nn.sigmoid(
            tf.matmul(M_feat_rs, self.U_r) + tf.matmul(messages_rs, self.W_r)
        )
        h_tilde = tf.nn.tanh(
            tf.matmul(tf.multiply(M_feat_rs, r_t), self.U_h) + tf.matmul(messages_rs, self.W_h)
        )

        h_t = tf.multiply(1-z_t, M_feat_rs) + tf.multiply(z_t, h_tilde)
        
        mask_rs = tf.reshape(mask, [self.batch_size*self.n_nodes, 1])
        h_t_masked = tf.multiply(h_t, mask_rs)

        h_t_rs = tf.reshape(
            h_t_masked, [self.batch_size, self.n_nodes, self.input_dim]
        )

        return h_t_rs

# READOUT function
class GraphLevelOutput(layers.Layer):
    def __init__(self,
                 n_nodes,
                 batch_size,
                 input_dim,
                 hidden_dim,
                 FP_dim,
                 W_i=None,
                 b_i=None,
                 W_j=None,
                 b_j=None,
                 W_out_i=None,
                 b_out_i=None,
                 W_out_j=None,
                 b_out_j=None,
                 n_hidden_layers=1,
                 activation=tf.nn.relu,
                 dropout=0.5,
                 **kwargs):
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.FP_dim = FP_dim
        self.activation = activation
        self.dropout = dropout

        if W_i is None:
            self.W_i = tf.Variable(
                name='W_i',
                initial_value=He(
                    [self.input_dim, self.hidden_dim]
                )
            )
        else:
            self.W_i = tf.Variable(name='W_i', initial_value=W_i)

        if b_i is None:
            self.b_i = tf.Variable(
                name='b_i',
                initial_value=He(
                    [self.hidden_dim]
                )
            )
        else:
            self.b_i = tf.Variable(name='b_i', initial_value=b_i)

        if W_j is None:
            self.W_j = tf.Variable(
                name='W_j',
                initial_value=He(
                    [self.input_dim, self.hidden_dim]
                )
            )
        else:
            self.W_j = tf.Variable(name='W_j', initial_value=W_j)

        if b_j is None:
            self.b_j = tf.Variable(
                name='b_j',
                initial_value=He(
                    [self.hidden_dim]
                )
            )
        else:
            self.b_j = tf.Variable(name='b_j', initial_value=b_j)

        if W_out_i is None:
            self.W_out_i = tf.Variable(
                name='W_out_i',
                initial_value=He(
                    [self.hidden_dim, self.FP_dim]
                )
            )
        else:
            self.W_out_i = tf.Variable(name='W_out_i', initial_value=W_out_i)

        if b_out_i is None:
            self.b_out_i = tf.Variable(
                name='b_out_i',
                initial_value=He(
                    [self.FP_dim]
                )
            )
        else:
            self.b_out_i = tf.Variable(name='b_out_i', initial_value=b_out_i)

        if W_out_j is None:
            self.W_out_j = tf.Variable(
                name='W_out_j',
                initial_value=He(
                    [self.hidden_dim, self.FP_dim]
                )
            )
        else:
            self.W_out_j = tf.Variable(name='W_out_j', initial_value=W_out_j)

        if b_out_j is None:
            self.b_out_j = tf.Variable(
                name='b_out_j',
                initial_value=He(
                    [self.FP_dim]
                )
            )
        else:
            self.b_out_j = tf.Variable(name='b_out_j', initial_value=b_out_j)

    def call(self, inputs, mask, training=False):
        """
        Args:
            inputs: 
                    [batch_size, n_nodes, input_dim]
            mak:    
                    [batch_size, n_nodes]
        """

        inputs_rs = tf.reshape(
            inputs, [self.batch_size*self.n_nodes, self.input_dim]
        )
        mask_rs = tf.reshape(
            mask, [self.batch_size*self.n_nodes, 1]
        )

        h_nn_i = self.activation(tf.matmul(inputs_rs, self.W_i) + self.b_i)
        if training:
            h_nn_i = tf.nn.dropout(h_nn_i, self.dropout)

        h_nn_j = self.activation(tf.matmul(inputs_rs, self.W_j) + self.b_j)
        if training:
            h_nn_j = tf.nn.dropout(h_nn_j, self.dropout)
        
        out_nn_i = tf.nn.sigmoid(tf.matmul(h_nn_i, self.W_out_i) + self.b_out_i)
        out_nn_j = tf.matmul(h_nn_j, self.W_out_j) + self.b_out_j

        gated_activation = tf.multiply(out_nn_i, out_nn_j)
        gated_activation = tf.multiply(gated_activation, mask_rs)
        gated_activation = tf.reshape(
            gated_activation,
            [self.batch_size, self.n_nodes, self.FP_dim]
        )

        FP = tf.reduce_sum(gated_activation, axis=1)

        return FP


class Dropout_layer(layers.Layer):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 dropout=0.5,
                 W=None,
                 b=None,
                 activation=tf.nn.relu,
                 **kwargs):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout

        if W is None:
            self.W = tf.Variable(name='W_inter', initial_value=He([self.input_dim, self.output_dim]))
        else:
            self.W = tf.Variable(name='W_inter', initial_value=W)

        if b is None:
            self.b = tf.Variable(name='b_inter', initial_value=zeros([self.output_dim,]))
        else:
            self.b = tf.Variable(name='b_inter', initial_value=b)

    def call(self, inputs, training=False):
        linear_output = tf.matmul(inputs, self.W) + self.b

        if training:
            inputs = tf.nn.dropout(linear_output, rate=self.dropout, seed=1)
        
        out = self.activation(linear_output)

        return out


class Output_layer(layers.Layer):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 W=None,
                 b=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        if W is None:
            self.W = tf.Variable(name='W_log', initial_value=zeros([self.input_dim, self.output_dim]))
        else:
            self.W = tf.Variable(name='W_log', initial_value=W)

        if b is None:
            self.b = tf.Variable(name='b_log', initial_value=zeros([self.output_dim,]))
        else:
            self.b = tf.Variable(name='b_log', initial_value=b)

    def call(self, inputs):

        y_probs = tf.nn.softmax(tf.matmul(inputs, self.W) + self.b)
        score = tf.matmul(inputs, self.W) + self.b
        y_pred = y_probs[:, 1] >= 0.5
        y_pred = tf.cast(y_pred, tf.int64)

        return score, y_probs, y_pred