import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from model import Unet3D
from dataset import lits_ribdataset
from metrics import tversky_loss



def save_result(inputs,heights,outputs,name,args,output_file):
    for sample in range(2):
        output = outputs[sample,:,:,:,:]
        output = output.squeeze()
        output = output.numpy()
        max_value = output.max(axis=0,keepdims=1)
        output[output<max_value]=0
        

        in_put = inputs[sample,:,:,:,:]
        in_put = in_put.squeeze()
        uprate = 1/args.down_rate
        output = ndimage.zoom(output,(1,uprate,uprate,uprate),order=0)
        a,b,c,d = output.shape
        h = heights[sample]
        label_out = np.zeros((h,c,d))
        for i in range(5):
            if h<=b:
                confi = np.max(output[i,:h,:,:])
                output_chi = output[i,:h,:,:]
                output_chi = np.where(output_chi>=0.2,1,0)
                label_out = label_out+i*output_chi
            elif h>b:
                confi = np.max(output[i,:,:,:])
                output = np.pad(ouput,((0,h-output.shape[0]),(0,0),(0,0)),'constant',constant_values = 0)
                output_chi = output[i,:,:,:]
                output_chi = np.where(output_chi>=0.2,1,0)
                label_out = label_out+i*output_chi
            output_file.write(name[sample]+','+str(i)+','+str(confi)+','+str(i)+'\n')
        out = sitk.GetImageFromArray(label_out.astype(np.float64))
        sitk.WriteImage(out,args.output_dir+'/'+name[sample]+'-label.nii.gz')


def test(model,loader,device,args,output_file):
    with torch.no_grad():
        for i ,datas in enumerate(loader):
            print('doing test {}/{}'.format(i,len(loader)))
            name = datas['name']
            data = datas['data'].to(device)
            h = datas['height']
            output = model(data).cpu()
            save_result(data.cpu(),h,output,name,args,output_file)


def main(args):

    output_dir = '../output/'+args.task_id
    args.output_dir = output_dir
    folder = os.path.exists(output_dir)
    if not folder:
        os.makedirs(output_dir)


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    test_data = lits_ribdataset(split='test',down_sample_rate=args.down_rate)

    test_loader = DataLoader(test_data,
                             batch_size=2)

    
    model = Unet3D(1,5).to(device)
    model_dict = torch.load(args.output_dir+'/model.pth')
    model.load_state_dict(model_dict)

    output_file = open(output_dir+'/ribfrac-test-pred.csv','w')
    output_file.write('public_id,label_id,confidence,label_code\n')
    test(model,test_loader,device,args,output_file)
    output_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',required=True,type=str)
    parser.add_argument('--down_rate',default=0.25,type=float)
    args= parser.parse_args()
    main(args)





