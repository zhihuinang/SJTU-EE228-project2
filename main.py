import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import Unet3D
from dataset import ribdataset,lits_ribdataset
from metrics import tversky_loss

def train(model,loader,optimizer,device,args):
    trainingloss = 0
    for i,datas in enumerate(loader):
        print('training {}/{}......'.format(i,len(loader)))
        ct = datas['data'].to(device)
        label = datas['label']
        output = model(ct).cpu()
        err = tversky_loss(label,output)
        trainingloss = err.item()
        err.backward()
        optimizer.step()
        print('training loss: {}'.format(trainingloss))
        torch.save(model.state_dict(),args.output_dir+'/model.pth')

def evaluation(model,loader,device,args):
    print('begin evaluation......')
    valloss = 0
    with torch.no_grad():
        for i,datas in enumerate(loader):
            ct = datas['data'].to(device)
            label = datas['label']

            output = model(ct).cpu()
            err = tversky_loss(label,output)
            valloss += err

            print('evaluation loss: {}'.format(valloss/(i+1)))


def main(args):

    output_dir = '../../output/'+args.task_id
    args.output_dir = output_dir
    folder = os.path.exists(output_dir)
    if not folder:
        os.makedirs(output_dir)


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    train_data = lits_ribdataset(split='train',down_sample_rate=args.down_rate)
    val_data = lits_ribdataset(split='val',down_sample_rate=args.down_rate)

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=1)


    model = Unet3D(1,5).to(device)
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
    parser.add_argument('--down_rate',default=0.25,type=float)
    args= parser.parse_args()
    main(args)





