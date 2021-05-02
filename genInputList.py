#!/usr/bin/env python
# coding: utf-8

'''
Last update: 20.10.23. by KS.Kwon
'''

import os
import sys
import pandas as pd
import numpy as np
import random
import argparse
import csv

from collections import defaultdict


def gen_input_list(ligdir, f_active_smi=None, f_decoy_smi=None):
    
    lig_list = [f.replace('.npy', '') for f in os.listdir(ligdir) if f.endswith('.npy')]
    ligand_dict = defaultdict(list)
    
    if f_active_smi is not None:
        f_smi = open(f_active_smi)
        csvreader = csv.reader(f_smi)
        next(csvreader)
        
        for lig in csvreader:
            CID = lig[0]
            canSmi = lig[1]
            PIDs = lig[3]
            
            if CID not in lig_list: 
                print(f'[Notice] {CID} has no feature data in the pre-processed ligand feature dataset.')
                continue

            for pid in PIDs.split(':'):
                ligand_dict[pid].append(f'{CID},1,{canSmi}')
        
        f_smi.close()

    if f_decoy_smi is not None:
        f_smi = open(f_decoy_smi)
        csvreader = csv.reader(f_smi)
        next(csvreader)

        for lig in csvreader:
            CID = lig[0]
            canSmi = lig[1]
            PIDs = lig[3]

            if CID not in lig_list: 
                print(f'[Notice] {CID} has no feature data in the pre-processed ligand feature dataset.')
                continue

            for pid in PIDs.split(':'):
                ligand_dict[pid].append(f'{CID},0,{canSmi}')

        f_smi.close()

    return ligand_dict


def seperate_data(outdir, prefix, ligand_dict, protein_dir, mode):

    tr = []
    va = []
    te = []

    n_tr = 0
    n_va = 0
    n_te = 0

    if os.path.isdir(outdir) is False:
        os.mkdir(outdir)

    f_tr = f'{prefix}_tr.csv'
    f_va = f'{prefix}_va.csv'
    f_te = f'{prefix}_te.csv'

    if mode == 'train':
        O_tr = open(os.path.join(outdir, f_tr), 'w')
        O_va = open(os.path.join(outdir, f_va), 'w')

    O_te = open(os.path.join(outdir, f_te), 'w')

    pocket_list = [f.replace('.npy', '') for f in os.listdir(protein_dir) if f.endswith('.npy')]

    for pdb in ligand_dict.keys():
        if pdb in pocket_list:
            if mode == 'train':
                assign = random.randint(1, 10)
            elif mode == 'eval':
                assign = 10

            if assign <= 7:
                tr.append(pdb)
                n_tr += len(ligand_dict[pdb])
                random.shuffle(ligand_dict[pdb])
                for data in ligand_dict[pdb]:
                    O_tr.write(f'{pdb},{data}\n')
            elif assign == 8:
                va.append(pdb)
                n_va += len(ligand_dict[pdb])
                random.shuffle(ligand_dict[pdb])
                for data in ligand_dict[pdb]:
                    O_va.write(f'{pdb},{data}\n')
            elif 9 <= assign:
                te.append(pdb)
                n_te += len(ligand_dict[pdb])
                random.shuffle(ligand_dict[pdb])
                for data in ligand_dict[pdb]:
                    O_te.write(f'{pdb},{data}\n')
        else:
            print(f'[Notice] {pdb}.npy file does not exist.')

    if mode == 'train':
        O_tr.close()
        O_va.close()
    
    O_te.close()
    
    f_data_rate = f'{prefix}_datarate.txt'
    O_data_rate = open(os.path.join(outdir, f_data_rate), 'w')
    O_data_rate.write(f'Train,Valid,Test\n')
    O_data_rate.write(f'{n_tr},{n_va},{n_te}')
    O_data_rate.close()
    
    if mode == 'train':
        f_tr_list = f'{prefix}_tr_list.txt'
        f_va_list = f'{prefix}_va_list.txt'
        
        O_tr_list = open(os.path.join(outdir, f_tr_list), 'w')
        O_va_list = open(os.path.join(outdir, f_va_list), 'w')
        
        O_tr_list.write('\n'.join(tr))
        O_va_list.write('\n'.join(va))

        O_tr_list.close()
        O_va_list.close()

    f_te_list = f'{prefix}_te_list.txt'
    O_te_list = open(os.path.join(outdir, f_te_list), 'w')
    O_te_list.write('\n'.join(te))
    O_te_list.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation')

    parser.add_argument('--active', dest='actives', type=str)
    parser.add_argument('--decoy', dest='decoys', type=str)
    parser.add_argument('--outdir', dest='outdir', type=str, required=True)
    parser.add_argument('--prefix', dest='prefix', required=True)
    parser.add_argument('--pocdir', dest='pocdir', required=True)
    parser.add_argument('--ligdir', dest='ligdir', required=True)
    parser.add_argument('--mode', dest='mode', default='train')

    parser.add_argument('-P', dest='protein_type')

    args = parser.parse_args()
    print(args)

    ''' Generate input list data (PID, CID, label, smiles) '''
    ligand_dict = gen_input_list(f_active_smi=args.actives, f_decoy_smi=args.decoys, ligdir=args.ligdir)
    
    ''' Seperate data with training, validation, testset '''
    seperate_data(args.outdir, args.prefix, ligand_dict, args.pocdir, args.mode)
