"""
Last update: 21.03.12. by KS.Kwon
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

from Layer import GGNNMsgPass, GRUUpdate, GraphLevelOutput, Dropout_layer, Output_layer


class Twoleg(models.Model):
    def __init__(self,
                 poc_n_nodes,
                 lig_n_nodes,
                 poc_n_edge_class,
                 lig_n_edge_class,
                 poc_input_dim,
                 lig_input_dim,
                 n_poc_prop_steps,  # Number of message passing propagation steps
                 n_lig_prop_steps,  # Number of message passing propagation steps                 
                 batch_size,
                 W_poc_msgpass,
                 b_poc_msgpass,
                 W_lig_msgpass,
                 b_lig_msgpass,
                 W_poc_vtx_update,
                 W_lig_vtx_update,
                 W_poc_readout,
                 b_poc_readout,
                 W_lig_readout,
                 b_lig_readout,
                 W_inter,
                 b_inter,
                 W_out,
                 b_out,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_poc_prop_steps = n_poc_prop_steps
        self.n_lig_prop_steps = n_lig_prop_steps

        poc_hidden_dim = [200]
        lig_hidden_dim = [200]
        poc_FP_dim = 200
        lig_FP_dim = 200

        poc_eW_in, poc_eW_out = W_poc_msgpass
        lig_eW_in, lig_eW_out = W_lig_msgpass

        poc_W_z, poc_W_r, poc_W_h, poc_U_z, poc_U_r, poc_U_h = W_poc_vtx_update
        lig_W_z, lig_W_r, lig_W_h, lig_U_z, lig_U_r, lig_U_h = W_lig_vtx_update

        poc_W_i, poc_W_j, poc_W_out_i, poc_W_out_j = W_poc_readout
        poc_b_i, poc_b_j, poc_b_out_i, poc_b_out_j = b_poc_readout
        lig_W_i, lig_W_j, lig_W_out_i, lig_W_out_j = W_lig_readout
        lig_b_i, lig_b_j, lig_b_out_i, lig_b_out_j = b_lig_readout

        self.POC_MsgPass = GGNNMsgPass(
            n_nodes=poc_n_nodes,
            n_edge_class=poc_n_edge_class,
            input_dim=poc_input_dim,
            batch_size=batch_size,
            eW_in=poc_eW_in,
            eW_out=poc_eW_out,
            b=b_poc_msgpass
        )
        self.POC_VtxUpdt = GRUUpdate(
            n_nodes=poc_n_nodes,
            batch_size=batch_size,
            input_dim=poc_input_dim,
            W_z=poc_W_z,
            W_r=poc_W_r,
            W_h=poc_W_h,
            U_z=poc_U_z,
            U_r=poc_U_r,
            U_h=poc_U_h
        )
        self.POC_ReadOut = GraphLevelOutput(
            n_nodes=poc_n_nodes,
            batch_size=batch_size,
            input_dim=2*poc_input_dim,
            hidden_dim=poc_hidden_dim[0],
            FP_dim=poc_FP_dim,
            W_i=poc_W_i,
            b_i=poc_b_i,
            W_j=poc_W_j,
            b_j=poc_b_j,
            W_out_i=poc_W_out_i,
            b_out_i=poc_b_out_i,
            W_out_j=poc_W_out_j,
            b_out_j=poc_b_out_j,
        )

        self.LIG_MsgPass = GGNNMsgPass(
            n_nodes=lig_n_nodes,
            n_edge_class=lig_n_edge_class,
            input_dim=lig_input_dim,
            batch_size=batch_size,
            eW_in=lig_eW_in,
            eW_out=lig_eW_out,
            b=b_lig_msgpass
        )
        self.LIG_VtxUpdt = GRUUpdate(
            n_nodes=lig_n_nodes,
            batch_size=batch_size,
            input_dim=lig_input_dim,
            W_z=lig_W_z,
            W_r=lig_W_r,
            W_h=lig_W_h,
            U_z=lig_U_z,
            U_r=lig_U_r,
            U_h=lig_U_h
        )
        self.LIG_ReadOut = GraphLevelOutput(
            n_nodes=lig_n_nodes,
            batch_size=batch_size,
            input_dim=2*lig_input_dim,
            hidden_dim=lig_hidden_dim[0],
            FP_dim=lig_FP_dim,
            W_i=lig_W_i,
            b_i=lig_b_i,
            W_j=lig_W_j,
            b_j=lig_b_j,
            W_out_i=lig_W_out_i,
            b_out_i=lig_b_out_i,
            W_out_j=lig_W_out_j,
            b_out_j=lig_b_out_j,
        )

        self.Interaction_layer = Dropout_layer(
            input_dim=poc_FP_dim+lig_FP_dim,
            output_dim=100,
            W=W_inter,
            b=b_inter
        )

        self.Output_layer = Output_layer(input_dim=100, output_dim=2, W=W_out, b=b_out)

    def call(self, inputs, training=False):
        M_poc_feat, M_poc_adj, poc_mask, M_lig_feat, M_lig_adj, lig_mask = inputs

        poc_h_list = [M_poc_feat]
        lig_h_list = [M_lig_feat]

        for i in range(self.n_poc_prop_steps):
            P_messages = self.POC_MsgPass(poc_h_list[-1], M_poc_adj)
            P_new_h = self.POC_VtxUpdt(poc_h_list[-1], P_messages, poc_mask)
            poc_h_list.append(P_new_h)
        
        poc_h_x = tf.concat([poc_h_list[-1], M_poc_feat], axis=2)
        poc_FP = self.POC_ReadOut(poc_h_x, poc_mask, training)

        for i in range(self.n_lig_prop_steps):
            L_messages = self.LIG_MsgPass(lig_h_list[-1], M_lig_adj)
            L_new_h = self.LIG_VtxUpdt(lig_h_list[-1], L_messages, lig_mask)
            lig_h_list.append(L_new_h)
        
        lig_h_x = tf.concat([lig_h_list[-1], M_lig_feat], axis=2)
        lig_FP = self.LIG_ReadOut(lig_h_x, lig_mask, training)

        inter_input = tf.concat([poc_FP, lig_FP], axis=1)
        inter_output = self.Interaction_layer(inter_input, training=training)
        class_score, class_prob, classification = self.Output_layer(inter_output)
        
        return class_score, class_prob, classification

    def loss(self, inputs, labels):
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=inputs)