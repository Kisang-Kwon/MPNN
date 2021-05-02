import os

def configuration(f_config):

    Config = open(f_config)
    for line in Config:
        line = line.rstrip()
        p = line.split(' = ')[0]
        val = line.split(' = ')[1]

        if p == 'mode':
            mode = val
            print(f'mode [{mode}]')
        elif p == 'checkpoint':
            checkpoint = val
            print(f'checkpoint [{checkpoint}]')
        elif p == 'prefix':
            prefix = val
            print(f'prefix [{prefix}]')

        if p == 'transfer':
            transfer = val
            print(f'transfer [{transfer}]')

        elif p == 'train_list':
            train_list = val
            print(f'train_list [{train_list}]')
        elif p == 'valid_list':
            valid_list = val
            print(f'valid_list [{valid_list}]')
        elif p == 'test_list':
            test_list = val
            print(f'test_list [{test_list}]')

        elif p == 'pocket_dir':
            pocket_dir = val
            print(f'pocket_dir [{pocket_dir}]')
        elif p == 'ligand_dir':
            ligand_dir = val
            print(f'ligand_dir [{ligand_dir}]')

        elif p == 'poc_n_nodes':
            poc_n_nodes = int(val)
            print(f'poc_n_nodes [{poc_n_nodes}]')
        elif p == 'lig_n_nodes':
            lig_n_nodes = int(val)
            print(f'lig_n_nodes [{lig_n_nodes}]')

        elif p == 'epochs':
            epochs = int(val)
            print(f'epochs [{epochs}]')
        elif p == 'learning_rate':
            lr = float(val)
            print(f'learning_rate [{lr}]')
        elif p == 'batch_size':
            batch_size = int(val)
            print(f'batch_size [{batch_size}]')
        elif p == 'l2_regularization':
            l2_param = float(val)
            print(f'l2_param [{l2_param}]')


    if mode == 'train':
        return transfer, train_list, valid_list, pocket_dir, ligand_dir, poc_n_nodes, lig_n_nodes, epochs, lr, batch_size, l2_param, checkpoint, prefix
    elif mode == 'eval' or mode == 'predict':
        return test_list, pocket_dir, ligand_dir, poc_n_nodes, lig_n_nodes, batch_size, checkpoint, prefix
    elif mode == 'AE_train':
        return train_list, valid_list, pocket_dir, poc_n_nodes, epochs, lr, batch_size, l2_param, checkpoint, prefix
    elif mode == 'AE_eval':
        return test_list, pocket_dir, poc_n_nodes, batch_size, checkpoint, prefix
    elif mode == 'saliency':
        return test_list, pocket_dir, ligand_dir, poc_n_nodes, lig_n_nodes, batch_size, checkpoint, prefix