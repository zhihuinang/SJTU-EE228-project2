import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

from model import Unet3D
from dataset import lits_ribdataset,FracNetTrainDataset
from metrics import tversky_loss
from transfroms import Window,MinMaxNorm

def train(model,loader,optimizer,device,args):
    trainingloss = 0
    for i,datas in enumerate(loader):
        print('training {}/{}......'.format(i,len(loader)))
        ct,label = datas
        ct = ct.to(device)
        optimizer.zero_grad()

        output = model(ct).cpu()
        err = tversky_loss(label,output)

        trainingloss += err.item()
        err.backward()
        optimizer.step()
        print('training loss: {}'.format(trainingloss/((i+1)*args.batch_size)))
        torch.save(model.module.state_dict(),args.output_dir+'/model.pth')

def evaluation(model,loader,device,args):
    print('begin evaluation......')
    valloss = 0
    with torch.no_grad():
        for i,datas in enumerate(loader):
            ct,label = datas
            ct = ct.to(device)

            output = model(ct).cpu()
            err = tversky_loss(label,output)
            valloss += err

            print('evaluation loss: {}'.format(valloss/((i+1)*args.batch_size)))


def main(args):

    output_dir = '../../output/'+args.task_id
    args.output_dir = output_dir
    folder = os.path.exists(output_dir)
    if not folder:
        os.makedirs(output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    transform = [
        Window(-200, 1000),
        MinMaxNorm(-200, 1000)
    ]
    train_data = FracNetTrainDataset(split='train',transforms=transform)
    val_data = FracNetTrainDataset(split='val',transforms=transform)

    train_loader = FracNetTrainDataset.get_dataloader(train_data, 
                                                    batch_size = args.batch_size,
                                                    shuffle=True)
    val_loader = FracNetTrainDataset.get_dataloader(val_data,
                                                    batch_size=args.batch_size)


    model = Unet3D(1,1)
    model = nn.DataParallel(model)
    model.to(device)
    optimizer = Adam(model.parameters(),lr=1e-3)

    for epoch in range(args.epoch):
        print("now training epoch {}".format(epoch+1))
        train(model,train_loader,optimizer,device,args)
        evaluation(model,val_loader,device,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',required=True,type=str)
    parser.add_argument('--epoch',required=True,default=5,type=int)
    parser.add_argument('--batch_size',required=True,default=16,type=int)
    parser.add_argument('--gpu',default='0',type=str)
    args= parser.parse_args()
    main(args)





