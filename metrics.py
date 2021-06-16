import torch
import torch.nn as nn


def tversky_loss(targets, inputs, beta=0.7, weights=None):
    batch_size = targets.size(0)
    loss = 0.0

    for i in range(batch_size):
        prob = inputs[i,0,:,:,:]
        ref = targets[i,0,:,:,:]
        alpha = 1.0-beta

        tp = (ref*prob).sum()
        fp = ((1-ref)*prob).sum()
        fn = (ref*(1-prob)).sum()
        tversky = (tp)/(tp + alpha*fp + beta*fn)
        loss = loss + (1-tversky)
    return loss/batch_size
 
def Diceloss(targets,inputs):
    dice = 0.0
    batch_size = targets.shape[0]
    for sample in range(batch_size):
        temp_dice = 0.0
        for i in range(targets.shape[1]):
            intersection = torch.sum(targets[sample,i,:,:,:]*inputs[sample,i,:,:,:])
            union = torch.sum(targets[sample,i,:,:,:])+torch.sum(inputs[sample,i,:,:,:])
            temp_dice+=2*intersection/(intersection+union)
        dice+=temp_dice
    print(dice)
    loss = 1-(dice/batch_size)
    return loss

def accuarcy(targets,inputs,threshold=0.5):
    acc = ((inputs>=threshold)==(targets>0)).float().mean()
    return acc

def precision(targets,inputs,threshold=0.2):
    tp = (((inputs*targets)>threshold).flatten(1).sum(1)>0).sum()
    prec = tp/(((inputs>threshold).flatten(1).sum(1)>0).sum()+1e-8)
    return prec

def recall(targets,inputs,threshold=0.2):
    tp = (((inputs*targets)>threshold).flatten(1).sum(1)>0).sum()
    recall = tp/(((targets>0).flatten(1).sum(1)>0).sum()+1e-8)
    return recall
   
def F1score(targets,inputs,threshold=0.2):
    prec = precision(targets,inputs,threshold)
    rec = recall(targets,inputs,threshold)
    F1 = (2*prec*rec)/(prec+rec)
    return F1
