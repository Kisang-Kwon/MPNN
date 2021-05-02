#!/usr/bin/env python
# coding: utf-8
'''
Last update: 21.01.22. by KS.Kwon

[21.01.22.]
- Added protein data 'residue_names' in protein.npy files
'''

import os
import sys
import numpy as np
import tensorflow as tf
import csv

from multiprocessing import Pool

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)



def input_list_parsing(f_input_list, pocket_dir, ligand_dir):
    Fopen = open(f_input_list)
    csvreader = csv.reader(Fopen)

    input_list = []
    for line in csvreader:
        f_poc = os.path.join(pocket_dir, f'{line[0]}.npy')
        f_lig = os.path.join(ligand_dir, f'{line[1]}.npy')
        label = line[2]
        smiles = line[3]
        input_list.append([f_poc, f_lig, label, smiles])
    
    return input_list


def AE_input_parsing(f_input_list, pocket_dir):
    Fopen = open(f_input_list)
    input_list = []
    for line in Fopen:
        line = line.rstrip()
        filepath = os.path.join(pocket_dir, f'{line}.npy')
        input_list.append(filepath)
    
    return input_list


def read_npy(f_npy):
    arr = np.load(f_npy, allow_pickle=True)
    return arr


def poc_load_data(input_list, n_threads=4):
    batch_data = []
    #for i in input_list:
    #    batch_data.append(np.load(i, allow_pickle=True))
    p = Pool(n_threads)
    batch_data = p.map(read_npy, input_list)
    batch_data = np.array(batch_data)
    p.close()

    poc_feat = []
    poc_adj = []
    poc_mask = []
    PIDs = []
    b_residue_names = []

    for e in batch_data:
        pf, pa, pm, pid, res_names = e
        
        poc_feat.append(pf)
        poc_adj.append(pa)
        poc_mask.append(pm)
        PIDs.append(pid)
        b_residue_names.append(res_names)

    return poc_feat, poc_adj, poc_mask, PIDs, b_residue_names
    

def lig_load_data(input_list, n_threads=4): 
    batch_data = []
    #for i in input_list:
    #    batch_data.append(np.load(i, allow_pickle=True))
    p = Pool(n_threads)
    batch_data = p.map(read_npy, input_list)
    batch_data = np.array(batch_data)
    p.close()
    
    lig_feat = []
    lig_adj = []
    lig_mask = []
    CIDs = []
    
    for e in batch_data:
        lf, la, lm, cid = e
        
        lig_feat.append(lf)
        lig_adj.append(la)
        lig_mask.append(lm)
        CIDs.append(cid)

    return lig_feat, lig_adj, lig_mask, CIDs


def load_labels(input_list):
    labels = []
    for l in input_list:
        label = np.zeros([2])
        l = int(l)
        label[l] = 1
        labels.append(label)

    return labels