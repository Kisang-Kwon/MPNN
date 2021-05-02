#!/usr/bin/env python
# coding: utf-8
'''
Last update: 21.03.13. by KS.Kwon

[21.03.13.]
- Input data
    - Protein:  PDB_extract_v11
    - Ligand:   AILION_v2

- Model structure
    - Graph layer:                  MPNN
        - Protein layer:            1
            - Message pass:         GGNN
            - Vertex update:        GRU
            - Readout:              GraphLevelOutput
                - hidden dim:       200
                - output dim:       200
                - activation:       ReLU
                - dropout:          0.5
            - Propagation step:     6

        - Ligand layer:             1
            - Message pass:         GGNN
            - Vertex update:        GRU
            - Readout:              GraphLevelOutput
                - hidden dim:       200
                - output dim:       200
                - activation:       ReLU
                - dropout:          0.5
            - Propagation step:     6

    - Interaction layer:            1
        - Dropout:                  0.5
    
    - Output layer:                 1
        - Prediction threshold:     0.5

    - Loss function:                Cross-entropy (softmax)
    - Optimizer:                    Adam
    - L2 regularization:            0.

'''

import os
import sys
import time
import numpy as np
import random
import csv
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import optimizers

from Utils import input_list_parsing, poc_load_data, lig_load_data, load_labels
from Model import Twoleg
from config import configuration
from Metrics import *

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


def train_Twoleg(f_config):
    print('======= Parameters ========')
    transfer, f_train_list, f_valid_list, pocket_dir, ligand_dir, poc_n_nodes, lig_n_nodes, epochs, learning_rate, batch_size, l2_param, checkpoint, prefix = configuration(f_config)
    print('===========================\n')

    print(time.strftime('Start: %x %X', time.localtime(time.time())))
    
    checkpoint_save = os.path.join(checkpoint, 'Params')

    # Data loading 
    train_list = input_list_parsing(f_train_list, pocket_dir, ligand_dir)
    valid_list = input_list_parsing(f_valid_list, pocket_dir, ligand_dir)
    
    random.shuffle(train_list)

    tr_total_batch = int(len(train_list) / batch_size)
    va_total_batch = int(len(valid_list) / batch_size)

    # Weights loading 
    if transfer == 'MPNN':
        f_params = os.path.join(checkpoint, 'Params/params.npy')
        params = np.load(f_params, allow_pickle=True)
        
        W_poc_msgpass, b_poc_msgpass, W_lig_msgpass, b_lig_msgpass = params[0:4]
        W_poc_vtx_update, W_lig_vtx_update = params[4:6]
        W_poc_readout, b_poc_readout = params[6:8]
        W_lig_readout, b_lig_readout = params[8:10]
        W_inter, b_inter, W_out, b_out = params[10:]        
        
        old_params = os.path.join(checkpoint, 'Params/old_params.npy')
        os.system(f'mv {params} {old_params}')
        
    elif transfer == 'None':
        os.makedirs(checkpoint_save)
        W_poc_msgpass = [None, None]
        b_poc_msgpass = None
        W_lig_msgpass = [None, None]
        b_lig_msgpass = None
        W_poc_vtx_update = [None, None, None, None, None, None]
        W_lig_vtx_update = [None, None, None, None, None, None]
        W_poc_readout = [None, None, None, None]
        b_poc_readout = [None, None, None, None]
        W_lig_readout = [None, None, None, None]
        b_lig_readout = [None, None, None, None]
        W_inter = None
        b_inter = None
        W_out = None
        b_out = None

    # Define model feature dimensions
    poc_check_data = np.load(train_list[0][0], allow_pickle=True)
    lig_check_data = np.load(train_list[0][1], allow_pickle=True)

    poc_input_dim = poc_check_data[0].shape[-1]
    # TODO: Deactivate bond feature matrix, Generate New type ligand adjacency matrix.
    #       The values in the ligad adjacency matrix from 0 to 4.
    #       (No bond for dummy nodes, single, double, triple, aromatic).
    lig_input_dim = lig_check_data[0].shape[-1]

    # Define a model
    model = Twoleg(poc_n_nodes=poc_n_nodes,
        lig_n_nodes=lig_n_nodes,
        poc_n_edge_class=1,
        lig_n_edge_class=5,
        poc_input_dim=poc_input_dim,
        lig_input_dim=lig_input_dim,
        n_poc_prop_steps=6,  # Number of message passing propagation steps
        n_lig_prop_steps=6,  # Number of message passing propagation steps                 
        batch_size=batch_size,
        W_poc_msgpass=W_poc_msgpass,
        b_poc_msgpass=b_poc_msgpass,
        W_lig_msgpass=W_lig_msgpass,
        b_lig_msgpass=b_lig_msgpass,
        W_poc_vtx_update=W_poc_vtx_update,
        W_lig_vtx_update=W_lig_vtx_update,
        W_poc_readout=W_poc_readout,
        b_poc_readout=b_poc_readout,
        W_lig_readout=W_lig_readout,
        b_lig_readout=b_lig_readout,
        W_inter=W_inter,
        b_inter=b_inter,
        W_out=W_out,
        b_out=b_out
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    O_loss = open(f'{checkpoint_save}/loss.csv', 'w')
    O_loss.write('Epoch,Train,Valid\n')
    O_acc = open(f'{checkpoint_save}/accuracy.csv', 'w')
    O_acc.write('Epoch,Train,Valid\n')

    loss_dict = defaultdict(list)
    acc_dict = defaultdict(list)
    for epoch in range(1, epochs+1):
        print('Epoch', epoch, time.strftime('Start: [%x %X]', time.localtime(time.time())))

        # Training
        for b_idx in range(1, tr_total_batch+1):
            # Mini batch data load
            batch_list = np.array(train_list[batch_size*(b_idx-1):batch_size*b_idx])
            
            tr_M_poc_feat, tr_M_poc_adj, tr_poc_mask, tr_PIDs, _ = poc_load_data(batch_list[:, 0])
            tr_M_lig_feat, tr_M_lig_adj, tr_lig_mask, tr_CIDs = lig_load_data(batch_list[:, 1])
            tr_labels = load_labels(batch_list[:, 2])
            
            # Convert to tensor variable
            tr_M_poc_feat = tf.convert_to_tensor(tr_M_poc_feat, dtype='float32')
            tr_M_poc_adj = tf.convert_to_tensor(tr_M_poc_adj, dtype='int32')
            tr_poc_mask = tf.convert_to_tensor(tr_poc_mask, dtype='float32')

            tr_M_lig_feat = tf.convert_to_tensor(tr_M_lig_feat, dtype='float32')
            tr_M_lig_adj = tf.convert_to_tensor(tr_M_lig_adj, dtype='int32')
            tr_lig_mask = tf.convert_to_tensor(tr_lig_mask, dtype='float32')
            
            tr_labels = tf.convert_to_tensor(tr_labels, dtype='float32')
            tr_PIDs = tf.convert_to_tensor(tr_PIDs)
            tr_CIDs = tf.convert_to_tensor(tr_CIDs)
            
            # Model call and calculate gradients
            with tf.GradientTape() as tape:
                tr_class_score, tr_class_prob, tr_classification = model(
                    (tr_M_poc_feat, tr_M_poc_adj, tr_poc_mask, tr_M_lig_feat, tr_M_lig_adj, tr_lig_mask),
                    training=True
                )
                tr_loss = model.loss(tr_class_score, tr_labels)

                if l2_param != 0.:
                    for param in model.trainable_variables:
                        tr_loss = tf.add(tr_loss, l2_param*tf.nn.l2_loss(param))
            
            # Weight update
            grads = tape.gradient(tr_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Metric 
            #tr_avg_loss += tr_loss
            #tr_scores_list.extend(tr_class_score)
            #tr_labels_list.extend(tr_labels)

        # Validation
        tr_avg_loss = 0.
        va_avg_loss = 0.
        tr_scores_list = []
        tr_labels_list = []
        va_scores_list = []
        va_labels_list = []
        for b_idx in range(1, tr_total_batch+1):
            # Mini batch data load
            batch_list = np.array(train_list[batch_size*(b_idx-1):batch_size*b_idx])
            
            tr_M_poc_feat, tr_M_poc_adj, tr_poc_mask, tr_PIDs, _ = poc_load_data(batch_list[:, 0])
            tr_M_lig_feat, tr_M_lig_adj, tr_lig_mask, tr_CIDs = lig_load_data(batch_list[:, 1])
            tr_labels = load_labels(batch_list[:, 2])
            
            # Convert to tensor variable
            tr_M_poc_feat = tf.convert_to_tensor(tr_M_poc_feat, dtype='float32')
            tr_M_poc_adj = tf.convert_to_tensor(tr_M_poc_adj, dtype='int32')
            tr_poc_mask = tf.convert_to_tensor(tr_poc_mask, dtype='float32')

            tr_M_lig_feat = tf.convert_to_tensor(tr_M_lig_feat, dtype='float32')
            tr_M_lig_adj = tf.convert_to_tensor(tr_M_lig_adj, dtype='int32')

            tr_lig_mask = tf.convert_to_tensor(tr_lig_mask, dtype='float32')
            
            tr_labels = tf.convert_to_tensor(tr_labels, dtype='float32')
            tr_PIDs = tf.convert_to_tensor(tr_PIDs)
            tr_CIDs = tf.convert_to_tensor(tr_CIDs)
            
            # Model call
            tr_class_score, tr_class_prob, tr_classification = model(
                (tr_M_poc_feat, tr_M_poc_adj, tr_poc_mask, tr_M_lig_feat, tr_M_lig_adj, tr_lig_mask),
                training=True
            )
            tr_loss = model.loss(tr_class_score, tr_labels)
            
            # Metric 
            #tr_avg_loss += tr_loss
            tr_avg_loss += np.sum(tr_loss)
            tr_scores_list.extend(tr_class_score)
            tr_labels_list.extend(tr_labels)

        for b_idx in range(1, va_total_batch+1):
            # Mini batch data load
            batch_list = np.array(valid_list[batch_size*(b_idx-1):batch_size*b_idx])

            va_M_poc_feat, va_M_poc_adj, va_poc_mask, va_PIDs, _ = poc_load_data(batch_list[:, 0])
            va_M_lig_feat, va_M_lig_adj, va_lig_mask, va_CIDs = lig_load_data(batch_list[:, 1])
            va_labels = load_labels(batch_list[:, 2])
            
            # Convert to tensor variable
            va_M_poc_feat = tf.convert_to_tensor(va_M_poc_feat, dtype='float32')
            va_M_poc_adj = tf.convert_to_tensor(va_M_poc_adj, dtype='int32')
            va_poc_mask = tf.convert_to_tensor(va_poc_mask, dtype='float32')

            va_M_lig_feat = tf.convert_to_tensor(va_M_lig_feat, dtype='float32')
            va_M_lig_adj = tf.convert_to_tensor(va_M_lig_adj, dtype='int32')
            va_lig_mask = tf.convert_to_tensor(va_lig_mask, dtype='float32')
            
            va_labels = tf.convert_to_tensor(va_labels, dtype='float32')
            va_PIDs = tf.convert_to_tensor(va_PIDs)
            va_CIDs = tf.convert_to_tensor(va_CIDs)
            
            # Model call
            va_class_score, va_class_prob, va_classification = model(
                (va_M_poc_feat, va_M_poc_adj, va_poc_mask, va_M_lig_feat, va_M_lig_adj, va_lig_mask),
                training=True
            )
            va_loss = model.loss(va_class_score, va_labels)
            
            # Metric
            #va_avg_loss += va_loss
            va_avg_loss += np.sum(va_loss)
            va_scores_list.extend(va_class_score)
            va_labels_list.extend(va_labels)

        # Save metrics
        tr_avg_loss = tr_avg_loss / (tr_total_batch * batch_size)
        va_avg_loss = va_avg_loss / (va_total_batch * batch_size)
        O_loss.write(f'[Epoch {epoch}],{float(tr_avg_loss)},{float(va_avg_loss)}\n')
        loss_dict['tr'].append(round(float(tr_avg_loss), 4))
        loss_dict['va'].append(round(float(va_avg_loss), 4))

        tr_acc = Accuracy(tr_scores_list, tr_labels_list)
        va_acc = Accuracy(va_scores_list, va_labels_list)

        O_acc.write(f'[Epoch {epoch}],{float(tr_acc)},{float(va_acc)}\n')
        acc_dict['tr'].append(round(float(tr_acc), 4))
        acc_dict['va'].append(round(float(va_acc), 4))
        
        print(f'Train loss: {round(float(tr_avg_loss), 4)}, Valid loss: {round(float(va_avg_loss), 4)} | Train acc: {round(float(tr_acc), 4)}, Valid acc: {round(float(va_acc), 4)}')

        # Save parameters every 5 epochs
        if epoch % 5 == 0:
            params = np.array([
                # POCKET_Message Passing layer
                [model.POC_MsgPass.eW_in, model.POC_MsgPass.eW_out],
                model.POC_MsgPass.b,

                # LIGAND Message Passing layer
                [model.LIG_MsgPass.eW_in, model.LIG_MsgPass.eW_out],
                model.LIG_MsgPass.b,

                # POCKET Vertex Update layer
                [model.POC_VtxUpdt.W_z, model.POC_VtxUpdt.W_r, model.POC_VtxUpdt.W_h,
                 model.POC_VtxUpdt.U_z, model.POC_VtxUpdt.U_r, model.POC_VtxUpdt.U_h],
                
                # LIGAND Vertex Update layer
                [model.LIG_VtxUpdt.W_z, model.LIG_VtxUpdt.W_r, model.LIG_VtxUpdt.W_h,
                 model.LIG_VtxUpdt.U_z, model.LIG_VtxUpdt.U_r, model.LIG_VtxUpdt.U_h],

                # POCKET Readout layer
                [model.POC_ReadOut.W_i, model.POC_ReadOut.W_j,
                 model.POC_ReadOut.W_out_i, model.POC_ReadOut.W_out_j],

                [model.POC_ReadOut.b_i, model.POC_ReadOut.b_j,
                 model.POC_ReadOut.b_out_i, model.POC_ReadOut.b_out_j],
                
                # LIGAND ReadOut layer
                [model.LIG_ReadOut.W_i, model.LIG_ReadOut.W_j,
                 model.LIG_ReadOut.W_out_i, model.LIG_ReadOut.W_out_j],

                [model.LIG_ReadOut.b_i, model.LIG_ReadOut.b_j,
                 model.LIG_ReadOut.b_out_i, model.LIG_ReadOut.b_out_j],

                # INTERACTION LAYER
                model.Interaction_layer.W, model.Interaction_layer.b,
                model.Output_layer.W, model.Output_layer.b
            ])
            np.save(f'{checkpoint_save}/params.npy', params)
            np.save(f'{checkpoint_save}/params_{epoch}.npy', params)
            Loss_plot(loss_dict, checkpoint, prefix, epochs)
            Accuracy_plot(acc_dict, checkpoint, prefix, epochs)

    O_loss.close()
    O_acc.close()
    
    Loss_plot(loss_dict, checkpoint, prefix, epochs)
    Accuracy_plot(acc_dict, checkpoint, prefix, epochs)

    print(time.strftime('End: %x %X', time.localtime(time.time())))

def eval_Twoleg(f_config):
    print('======= Parameters ========')
    f_test_list, pocket_dir, ligand_dir, poc_n_nodes, lig_n_nodes, batch_size, checkpoint, prefix = configuration(f_config)    
    print('===========================\n')

    checkpoint_load = os.path.join(checkpoint, 'Params')

    print(time.strftime('Start: %x %X', time.localtime(time.time())))

    # Data loading 
    test_list = input_list_parsing(f_test_list, pocket_dir, ligand_dir)
    te_total_batch = int(len(test_list) / batch_size)
    
    # Weights loading
    f_params = os.path.join(checkpoint_load, 'params.npy')
    params = np.load(f_params, allow_pickle=True)
    
    W_poc_msgpass, b_poc_msgpass, W_lig_msgpass, b_lig_msgpass = params[0:4]
    W_poc_vtx_update, W_lig_vtx_update = params[4:6]
    W_poc_readout, b_poc_readout = params[6:8]
    W_lig_readout, b_lig_readout = params[8:10]
    W_inter, b_inter, W_out, b_out = params[10:]

    # Model assessment
    poc_check_data = np.load(test_list[0][0], allow_pickle=True)
    lig_check_data = np.load(test_list[0][1], allow_pickle=True)

    poc_input_dim = poc_check_data[0].shape[-1]
    lig_input_dim = lig_check_data[0].shape[-1]
    
    model = Twoleg(poc_n_nodes=poc_n_nodes,
        lig_n_nodes=lig_n_nodes,
        poc_n_edge_class=1,
        lig_n_edge_class=5,
        poc_input_dim=poc_input_dim,
        lig_input_dim=lig_input_dim,
        n_poc_prop_steps=6,  # Number of message passing propagation steps
        n_lig_prop_steps=6,  # Number of message passing propagation steps                 
        batch_size=batch_size,
        W_poc_msgpass=W_poc_msgpass,
        b_poc_msgpass=b_poc_msgpass,
        W_lig_msgpass=W_lig_msgpass,
        b_lig_msgpass=b_lig_msgpass,
        W_poc_vtx_update=W_poc_vtx_update,
        W_lig_vtx_update=W_lig_vtx_update,
        W_poc_readout=W_poc_readout,
        b_poc_readout=b_poc_readout,
        W_lig_readout=W_lig_readout,
        b_lig_readout=b_lig_readout,
        W_inter=W_inter,
        b_inter=b_inter,
        W_out=W_out,
        b_out=b_out
    )
    
    class_scores = []
    class_probs = []
    classification = []
    labels = []
    PIDs = []
    CIDs = []

    for b_idx in range(1, te_total_batch+1):
        # Mini batch data load
        batch_list = np.array(test_list[batch_size*(b_idx-1):batch_size*b_idx])
        
        te_M_poc_feat, te_M_poc_adj, te_poc_mask, te_PIDs, _ = poc_load_data(batch_list[:, 0])
        te_M_lig_feat, te_M_lig_adj, te_lig_mask, te_CIDs = lig_load_data(batch_list[:, 1])
        te_labels = load_labels(batch_list[:, 2])
        
        # Convert to tensor variable
        te_M_poc_feat = tf.convert_to_tensor(te_M_poc_feat, dtype='float32')
        te_M_poc_adj = tf.convert_to_tensor(te_M_poc_adj, dtype='int32')
        te_poc_mask = tf.convert_to_tensor(te_poc_mask, dtype='float32')

        te_M_lig_feat = tf.convert_to_tensor(te_M_lig_feat, dtype='float32')
        te_M_lig_adj = tf.convert_to_tensor(te_M_lig_adj, dtype='int32')
        te_lig_mask = tf.convert_to_tensor(te_lig_mask, dtype='float32')
        
        te_labels = tf.convert_to_tensor(te_labels, dtype='float32')
        te_PIDs = tf.convert_to_tensor(te_PIDs)
        te_CIDs = tf.convert_to_tensor(te_CIDs)
        
        # Model call
        te_class_score, te_class_prob, te_classification = model(
            (te_M_poc_feat, te_M_poc_adj, te_poc_mask, te_M_lig_feat, te_M_lig_adj, te_lig_mask),
            training=False
        )

        class_scores.extend(te_class_score)
        class_probs.extend(te_class_prob)
        classification.extend(te_classification)
        labels.extend(te_labels)
        PIDs.extend(te_PIDs)
        CIDs.extend(te_CIDs)

    class_scores = np.array(class_scores)
    class_probs = np.array(class_probs)
    labels = np.array(labels)

    TP, TN, FP, FN, acc = Stats(class_scores, labels, checkpoint, prefix)
    auc = ROC_curve(class_probs, labels, checkpoint, prefix)

    filename = os.path.join(checkpoint, f'{prefix}_stats.txt')
    O_eval = open(filename, 'w')
    O_eval.write(f'True Positive: {TP}\nTrue Negative: {TN}\nFalse Positive: {FP}\nFalse Negative: {FN}\nAccuracy: {acc}\nAUC: {auc}\n')
    O_eval.close()

    predict_fpath = os.path.join(checkpoint, f'{prefix}_prediction.csv')
    Prediction(predict_fpath, PIDs, CIDs, classification, class_probs, labels)

    print(time.strftime('End: %x %X', time.localtime(time.time())))


def predict_Twoleg(f_config):
    print('======= Parameters ========')
    f_test_list, pocket_dir, ligand_dir, poc_n_nodes, lig_n_nodes, batch_size, checkpoint, prefix = configuration(f_config)    
    print('===========================\n')

    checkpoint_load = os.path.join(checkpoint, 'Params')

    print(time.strftime('Start: %x %X', time.localtime(time.time())))

    # Data loading 
    test_list = input_list_parsing(f_test_list, pocket_dir, ligand_dir)
    te_total_batch = int(len(test_list) / batch_size)
    
    # Weights loading
    f_params = os.path.join(checkpoint_load, 'params.npy')
    params = np.load(f_params, allow_pickle=True)
    
    W_poc_msgpass, b_poc_msgpass, W_lig_msgpass, b_lig_msgpass = params[0:4]
    W_poc_vtx_update, W_lig_vtx_update = params[4:6]
    W_poc_readout, b_poc_readout = params[6:8]
    W_lig_readout, b_lig_readout = params[8:10]
    W_inter, b_inter, W_out, b_out = params[10:]

    # Model assessment
    poc_check_data = np.load(test_list[0][0], allow_pickle=True)
    lig_check_data = np.load(test_list[0][1], allow_pickle=True)

    poc_input_dim = poc_check_data[0].shape[-1]
    lig_input_dim = lig_check_data[0].shape[-1]
    
    model = Twoleg(poc_n_nodes=poc_n_nodes,
        lig_n_nodes=lig_n_nodes,
        poc_n_edge_class=1,
        lig_n_edge_class=5,
        poc_input_dim=poc_input_dim,
        lig_input_dim=lig_input_dim,
        n_poc_prop_steps=6,  # Number of message passing propagation steps
        n_lig_prop_steps=6,  # Number of message passing propagation steps                 
        batch_size=batch_size,
        W_poc_msgpass=W_poc_msgpass,
        b_poc_msgpass=b_poc_msgpass,
        W_lig_msgpass=W_lig_msgpass,
        b_lig_msgpass=b_lig_msgpass,
        W_poc_vtx_update=W_poc_vtx_update,
        W_lig_vtx_update=W_lig_vtx_update,
        W_poc_readout=W_poc_readout,
        b_poc_readout=b_poc_readout,
        W_lig_readout=W_lig_readout,
        b_lig_readout=b_lig_readout,
        W_inter=W_inter,
        b_inter=b_inter,
        W_out=W_out,
        b_out=b_out
    )
    
    class_probs = []
    classification = []
    PIDs = []
    CIDs = []
    for b_idx in range(1, te_total_batch+1):
        # Mini batch data load
        batch_list = np.array(test_list[batch_size*(b_idx-1):batch_size*b_idx])
        
        te_M_poc_feat, te_M_poc_adj, te_poc_mask, te_PIDs, _ = poc_load_data(batch_list[:, 0])
        te_M_lig_feat, te_M_lig_adj, te_lig_mask, te_CIDs = lig_load_data(batch_list[:, 1])
        te_labels = load_labels(batch_list[:, 2])
        
        # Convert to tensor variable
        te_M_poc_feat = tf.convert_to_tensor(te_M_poc_feat, dtype='float32')
        te_M_poc_adj = tf.convert_to_tensor(te_M_poc_adj, dtype='int32')
        te_poc_mask = tf.convert_to_tensor(te_poc_mask, dtype='float32')

        te_M_lig_feat = tf.convert_to_tensor(te_M_lig_feat, dtype='float32')
        te_M_lig_adj = tf.convert_to_tensor(te_M_lig_adj, dtype='int32')
        te_lig_mask = tf.convert_to_tensor(te_lig_mask, dtype='float32')
        
        te_labels = tf.convert_to_tensor(te_labels, dtype='float32')
        te_PIDs = tf.convert_to_tensor(te_PIDs)
        te_CIDs = tf.convert_to_tensor(te_CIDs)
        
        # Model call
        te_class_score, te_class_prob, te_classification = model(
            (te_M_poc_feat, te_M_poc_adj, te_poc_mask, te_M_lig_feat, te_M_lig_adj, te_lig_mask),
            training=False
        )

        classification.extend(te_classification)
        class_probs.extend(te_class_prob)
        PIDs.extend(te_PIDs)
        CIDs.extend(te_CIDs)
    
    predict_fpath = os.path.join(checkpoint, f'{prefix}_prediction.csv')
    Prediction(predict_fpath, PIDs, CIDs, classification, class_probs)

    print(time.strftime('End: %x %X', time.localtime(time.time())))


if __name__ == '__main__':
    f_config = sys.argv[1]
    tasks = sys.argv[2]

    if tasks == 'train':
        train_Twoleg(f_config)
    elif tasks == 'eval':
        eval_Twoleg(f_config)
    elif tasks == 'predict':
        predict_Twoleg(f_config)
