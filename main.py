import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import Unet3D
from dataset import ribdataset
from metrics import tversky_loss


def train(model,loader,optimizer,device):
    trainingloss = 0
    for i,datas in enumerate(loader):
        ct = datas['data'].to(device)
        label = datas['label'].to(device)

        output = model(ct)
        err = tversky_loss(label,output)
        err.backward()
        optimizer.step()
        trainingloss += err

        print('training loss: {}'.format(trainingloss/(i+1)))


def evaluation(model,loader,device):
    valloss = 0
    with torch.no_grad():
        for i,datas in enumerate(loader):
            ct = datas['data'].to(device)
            label = datas['label'].to(device)

            output = model(ct)
            err = tversky_loss(label,output)
            valloss += err

            print('training loss: {}'.format(valloss/(i+1)))


def main(args):

    output_dir = './output/'+args.task_id
    folder = os.path.exists(output_dir)
    if not folder:
        os.makedirs(output_dir)


    if torch.cuda.is_available():
        device = 'gpu'
    else:
        device = 'cpu'

    train_data = ribdataset(split='train')
    val_data = ribdataset(split='val')
    test_data = ribdataset(split='test')

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              num_workers=4,
                              shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=8,
                            num_workers=1)
    test_loader = DataLoader(test_data,
                             batch_size=8,
                             num_workers=1)


    model = Unet3D(5).to(device)
    optimizer = Adam(model.parameters(),lr=1e-3)

    for epoch in range(args.epoch):
        print("now training epoch {}".format(epoch+1))
        train(model,train_loader,optimizer,device)
    evaluation(model,val_loader,device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',required=True,type=str)
    parser.add_argument('--epoch',required=True,default=5,type=int)
    parser.add_argument('--batch_size',required=True,default=16,type=int)

    args= parser.parse_args()
    main(args)





