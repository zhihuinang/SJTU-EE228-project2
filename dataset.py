from torch.utils.data import Dataset
import pandas as pd
import os
import nibabel as nib
import numpy as np

class ribdataset(Dataset):
    def __init__(self,split='train'):
        if split=='train':
            self.path = '../data/train'
            self.label_path = '../data/train_label'
            self.info_index = '../data/ribfrac-train-info.csv'
        elif split == 'val':
            self.path = '../data/val'
            self.label_path = '../data/val_label'
            self.info_index = '../data/ribfrac-val-info.csv'
        elif split == 'test':
            self.path = '../data/test'
            self.label_path = None
            self.info_index = None

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
        ct_data = np.asarray(ct.dataobj)

        label = nib.load(self.label_path+label_name)
        a,b,c = label.shape
        label_ch = np.zeros((a,b,c,5))
        tempdf = self.df.loc[self.df['public_id']==prefix]
        for i in range(tempdf.shape[0]):
            label_code = tempdf.loc[i]['label_code']
            if label_code==-1:
                continue
            else:
                temp = label
                temp[temp==i]=1
                temp[temp!=i]=0
                label_ch[:,:,:,label_code]=temp

        return {'data':ct_data,'label':label_ch}

