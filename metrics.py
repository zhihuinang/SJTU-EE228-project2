import torch
import torch.nn as nn


def tversky_loss(targets, inputs, beta=0.7, weights=None):
    '''
    batch_size = targets.size(0)
    loss = 0.0

    for i in range(batch_size):
        prob = inputs[i]
        ref = targets[i]

        alpha = 1.0-beta

        tp = (ref*prob).sum()
        fp = ((1-ref)*prob).sum()
        fn = (ref*(1-prob)).sum()
        tversky = tp/(tp + alpha*fp + beta*fn)
        loss = loss + (1-tversky)
    return loss/batch_size
    '''
    gt = torch.flatten(targets)
    predict = torch.flatten(inputs)
    alpha = 1.0-beta

    tp = (gt*predict).sum()
    fp = ((1-gt)*predict).sum()
    fn = (gt*(1-predict)).sum()
    tversky = tp/(tp + alpha*fp + beta*fn)
    loss = 1-tversky
    return loss

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
