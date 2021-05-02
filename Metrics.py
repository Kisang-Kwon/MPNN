#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

from sklearn.metrics import roc_curve, auc, confusion_matrix


def Stats(class_scores, labels, checkpoint, prefix):
    class_prob = tf.nn.softmax(class_scores)

    cm = confusion_matrix(labels[:, 1], class_prob[:, 1] > 0.5)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    filename = os.path.join(checkpoint, f'{prefix}_Confusion_matrix.png')
    plt.savefig(filename, bbox_inches='tight')
    
    acc = (cm[1][1] + cm[0][0]) / np.sum(cm)

    return cm[1][1], cm[0][0], cm[0][1], cm[1][0], acc 


def ROC_curve(class_probs, labels, checkpoint, prefix):
    fpr, tpr, thresholds = roc_curve(labels[:, 1], class_probs[:, 1])
    AUC = auc(fpr, tpr)
    print(thresholds)
    
    fig1 = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (auc = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    
    filename = os.path.join(checkpoint, f'{prefix}_ROC_curve.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig1)

    return AUC


def Loss_plot(loss_dict, checkpoint, prefix, epochs):
    fig1 = plt.figure()
    plt.plot(range(1, len(loss_dict['tr'])+1), loss_dict['tr'])
    plt.plot(range(1, len(loss_dict['va'])+1), loss_dict['va'])

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss plot')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.xlim(0, epochs)
    plt.ylim(0.0, 5.0)

    filename = os.path.join(checkpoint, f'{prefix}_training_loss.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig1)


def Accuracy(class_scores, labels):
    class_scores = tf.convert_to_tensor(class_scores)
    labels = tf.convert_to_tensor(labels)

    class_prob = tf.nn.softmax(class_scores)
    pred = class_prob[:, 1] > 0.5

    T = 0
    F = 0
    for i, label in enumerate(labels[:, 1].numpy()):
        if label == pred[i]:
            T += 1
        else:
            F += 1
    
    accuracy = T / (T + F)
    print(T, F, round(accuracy, 4))
    return accuracy


def Accuracy_plot(acc_dict, checkpoint, prefix, epochs):
    fig1 = plt.figure()
    plt.plot(range(1, len(acc_dict['tr'])+1), acc_dict['tr'])
    plt.plot(range(1, len(acc_dict['va'])+1), acc_dict['va'])

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy plot')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.xlim(0, epochs)
    plt.ylim(0.0, 1.0)

    filename = os.path.join(checkpoint, f'{prefix}_training_accuracy.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig1)


def Prediction(filepath, PIDs, CIDs, classification, class_prob, labels=None):
    out_predict = open(filepath, 'w')
    
    if labels is None:
        T = []
        F = []
        for i, prediction in enumerate(classification):
            if prediction == 1:
                T.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Active')
            elif prediction == 0:
                F.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Inactive')
        
        out_predict.write('PID,CID,Class_prob,Prediction\n')
        out_predict.write('\n'.join(T) + '\n')
        out_predict.write('\n'.join(F))
    
    else:
        TP = []
        TN = []
        FP = []
        FN = []
        for i, prediction in enumerate(classification):
            if prediction == 1:
                if tf.argmax(labels[i]) == 1:
                    TP.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Active,TP')
                elif tf.argmax(labels[i]) == 0:
                    FN.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Active,FP')
            elif prediction == 0:
                if tf.argmax(labels[i]) == 1:
                    FN.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Inactive,FN')
                elif tf.argmax(labels[i]) == 0:
                    TN.append(f'{PIDs[i]},{CIDs[i]},{round(float(class_prob[i][1]), 4)},Inactive,TN')

        output = []
        output.extend(TP)
        output.extend(TN)
        output.extend(FP)
        output.extend(FN)

        out_predict.write('PID,CID,Class_prob,Prediction,Result\n')
        out_predict.write('\n'.join(output))

    out_predict.close()

if __name__ == '__main__':
    import csv

    testfile = open('/home/ailon26/00_working/11_tf_porting/twoleg/GCN/loss.csv')
    csvreader = csv.reader(testfile)
    next(csvreader)

    loss_dict = defaultdict(list)
    for line in csvreader:
        loss_dict['tr'].append(round(float(line[1]), 4))
        loss_dict['va'].append(round(float(line[2]), 4))

    Loss_plot(loss_dict, './', 'test')
