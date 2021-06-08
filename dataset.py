from torch.utils.data import Dataset
import pandas as pd
import os
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
from tqdm import tqdm
import skimage.transform as skTrans
from nilearn.image import new_img_like,resample_to_img,resample_img
import warnings
warnings.filterwarnings('ignore')


class ribdataset(Dataset):
    def __init__(self,split='train'):
        self.split = split
        if split=='train':
            self.path = '../../data/ribfrac/train/'
            self.label_path = '../../data/ribfrac/train_label/'
            self.info_index = '../../data/ribfrac/ribfrac-train-info.csv'
        elif split == 'val':
            self.path = '../../data/ribfrac/val/'
            self.label_path = '../../data/ribfrac/val_label/'
            self.info_index = '../../data/ribfrac/ribfrac-val-info.csv'
        elif split == 'test':
            self.path = '../../data/ribfrac/test/'
            self.label_path = None
            self.info_index = None
        if self.info_index != None:
            self.df = pd.read_csv(self.info_index)
        for files in os.walk(self.path):
            self.data_list = files[2]
        
 



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        ct_name = self.data_list[index]
        prefix = ct_name[:-13]
        label_name = prefix+'-label.nii.gz'

        ct = nib.load(self.path+ct_name)
        scaled_ct = scale_image(ct, (0.25,0.25,0.25))
        
        ct_data = np.asarray(ct.dataobj)
        if self.split =='test':
            return {'data':np.expand_dims(ct_data,axis=0)}
        if ct_data.shape[2]<512:
           ct_data = np.pad(ct_data,((0,0),(0,0),(0,512-ct_data.shape[2])),'constant',constant_values = 0)

        ct_data = ct_data.astype('float32')


        label = nib.load(self.label_path+label_name)
        label = scale_image(label, (0.25,0.25,0.25))
        label = np.asarray(label.dataobj)
        if label.shape[2]<512:
           label = np.pad(label,((0,0),(0,0),(0,512-label.shape[2])),'constant',constant_values = 0)

        a,b,c = label.shape
        label_ch = np.zeros((1,5,a,b,c),dtype=np.uint8)
        tempdf = self.df.loc[self.df['public_id']==prefix]
        for i in tqdm(range(tempdf.shape[0])):
            label_code = tempdf.loc[tempdf['label_id']==i,'label_code']
            label_code = label_code.tolist()[0]
            if label_code==-1:
                continue
            else:
                temp = np.asarray(label)
                mask = np.asarray([temp==i])
                label_ch[:,label_code,:,:,:]+=np.squeeze(mask)
                label_ch = label_ch.astype('uint8')
       
        ct_data = np.expand_dims(ct_data,axis=0)
        return {'data':ct_data[:,192:320,192:320,192:320],'label':label_ch[:,:,192:320,192:320,192:320]}
        #return {'data':ct_data[:,:,:,:],'label':label_ch[:,:,:]}


def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    template = new_img_like(image, data=image.get_data(), affine=new_affine)
    return resample_to_img(template,image,interpolation="continuous")





class lits_ribdataset(Dataset):
    def __init__(self,split='train',down_sample_rate=0.25):
        self.split = split
        if split=='train':
            self.path = '../../data/ribfrac/train/'
            self.label_path = '../../data/ribfrac/train_label/'
            self.info_index = '../../data/ribfrac/ribfrac-train-info.csv'
        elif split == 'val':
            self.path = '../../data/ribfrac/val/'
            self.label_path = '../../data/ribfrac/val_label/'
            self.info_index = '../../data/ribfrac/ribfrac-val-info.csv'
        elif split == 'test':
            self.path = '../data/ribfrac/test/'
            self.label_path = None
            self.info_index = None
        if self.info_index != None:
            self.df = pd.read_csv(self.info_index)
        for files in os.walk(self.path):
            self.data_list = files[2]
        self.down_rate = down_sample_rate
 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        ct_name = self.data_list[index]
        prefix = ct_name[:-13]
        label_name = prefix+'-label.nii.gz'

        ct = sitk.ReadImage(self.path+ct_name,sitk.sitkInt16)
        ct = sitk.Cast(sitk.RescaleIntensity(ct),sitk.sitkUInt8)
        ct_data = sitk.GetArrayFromImage(ct)
        h = ct_data.shape[0]

        if ct_data.shape[0]<512:
           ct_data = np.pad(ct_data,((0,512-ct_data.shape[0]),(0,0),(0,0)),'constant',constant_values = 0)
        if ct_data.shape[0]>512:
            ct_data = ct_data[:512,:,:]
        ct_data = ndimage.zoom(ct_data,(self.down_rate,self.down_rate,self.down_rate),order=3)
        ct_data = ct_data.astype('float32')
        
        if self.split =='test':
            return {'data':np.expand_dims(ct_data,axis=0),'name':prefix,'height':h}


        

        label = sitk.ReadImage(self.label_path+label_name,sitk.sitkInt8)
        label_array = sitk.GetArrayFromImage(label)
        if label_array.shape[0]<512:
            label_array = np.pad(label_array,((0,512-label_array.shape[0]),(0,0),(0,0)),'constant',constant_values = 0)
        elif label_array.shape[0]>512:
            label_array = label_array[:512,:,:]
        label_array = ndimage.zoom(label_array,(self.down_rate,self.down_rate,self.down_rate),order=0)
        a,b,c = label_array.shape
        label_ch = np.zeros((1,5,a,b,c),dtype=np.uint8)
        tempdf = self.df.loc[self.df['public_id']==prefix]
        for i in range(tempdf.shape[0]):
            label_code = tempdf.loc[tempdf['label_id']==i,'label_code']
            label_code = label_code.tolist()[0]
            if label_code==-1:
                continue
            else:
                mask = np.where(label_array==i,1,0).astype('uint8')
                label_ch[:,label_code,:,:,:]+=np.squeeze(mask)
                label_ch = label_ch.astype('uint8')
        ct_data = np.expand_dims(ct_data,axis=0)
        return {'data':ct_data[:,:,:,:],'label':label_ch[:,:,:,:,:]}

