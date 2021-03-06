from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
import nibabel as nib
import SimpleITK as sitk
import torch
from scipy import ndimage
import numpy as np
from tqdm import tqdm
import skimage.transform as skTrans
from nilearn.image import new_img_like,resample_to_img,resample_img
import warnings
warnings.filterwarnings('ignore')
from itertools import product
from skimage.measure import regionprops


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
        label_ch = np.zeros((5,a,b,c),dtype=np.uint8)
        tempdf = self.df.loc[self.df['public_id']==prefix]
        for i in range(tempdf.shape[0]):
            label_code = tempdf.loc[tempdf['label_id']==i,'label_code']
            label_code = label_code.tolist()[0]
            if label_code==-1:
                continue
            else:
                mask = np.where(label_array==i,1,0).astype('uint8')
                label_ch[label_code,:,:,:]+=np.squeeze(mask)
                label_ch = label_ch.astype('uint8')
            
        ct_data = np.expand_dims(ct_data,axis=0)
        return {'data':ct_data[:,:,:,:],'label':label_ch[:,:,:,:]}


class FracNetTrainDataset(Dataset):

    def __init__(self, split='train',crop_size=64,
            transforms=None, num_samples=4, train=True):
        self.split = split
        if split=='train':
            self.image_dir = '../../data/ribfrac/train/'
            self.label_dir = '../../data/ribfrac/train_label/'
        elif split == 'val':
            self.image_dir = '../../data/ribfrac/val/'
            self.label_dir = '../../data/ribfrac/val_label/'
        self.public_id_list = sorted([x.split("-")[0]
            for x in os.listdir(self.image_dir)])
        self.crop_size = crop_size
        self.transforms = transforms
        self.num_samples = num_samples
        self.train = train

    def __len__(self):
        return len(self.public_id_list)

    @staticmethod
    def _get_pos_centroids(label_arr):
        centroids = [tuple([round(x) for x in prop.centroid])
            for prop in regionprops(label_arr)]

        return centroids

    @staticmethod
    def _get_symmetric_neg_centroids(pos_centroids, x_size):
        sym_neg_centroids = [(x_size - x, y, z) for x, y, z in pos_centroids]

        return sym_neg_centroids

    @staticmethod
    def _get_spine_neg_centroids(shape, crop_size, num_samples):
        x_min, x_max = shape[0] // 2 - 40, shape[0] // 2 + 40
        y_min, y_max = 300, 400
        z_min, z_max = crop_size // 2, shape[2] - crop_size // 2
        spine_neg_centroids = [(
            np.random.randint(x_min, x_max),
            np.random.randint(y_min, y_max),
            np.random.randint(z_min, z_max)
        ) for _ in range(num_samples)]

        return spine_neg_centroids

    def _get_neg_centroids(self, pos_centroids, image_shape):
        num_pos = len(pos_centroids)
        sym_neg_centroids = self._get_symmetric_neg_centroids(
            pos_centroids, image_shape[0])

        if num_pos < self.num_samples // 2:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                self.crop_size, self.num_samples - 2 * num_pos)
        else:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                self.crop_size, num_pos)

        return sym_neg_centroids + spine_neg_centroids

    def _get_roi_centroids(self, label_arr):
        if self.train:
            # generate positive samples' centroids
            pos_centroids = self._get_pos_centroids(label_arr)

            # generate negative samples' centroids
            neg_centroids = self._get_neg_centroids(pos_centroids,
                label_arr.shape)

            # sample positives and negatives when necessary
            num_pos = len(pos_centroids)
            num_neg = len(neg_centroids)
            if num_pos >= self.num_samples:
                num_pos = self.num_samples // 2
                num_neg = self.num_samples // 2
            elif num_pos >= self.num_samples // 2:
                num_neg = self.num_samples - num_pos

            if num_pos < len(pos_centroids):
                pos_centroids = [pos_centroids[i] for i in np.random.choice(
                    range(0, len(pos_centroids)), size=num_pos, replace=False)]
            if num_neg < len(neg_centroids):
                neg_centroids = [neg_centroids[i] for i in np.random.choice(
                    range(0, len(neg_centroids)), size=num_neg, replace=False)]

            roi_centroids = pos_centroids + neg_centroids
        else:
            roi_centroids = [list(range(0, x, y // 2))[1:-1] + [x - y // 2]
                for x, y in zip(label_arr.shape, self.crop_size)]
            roi_centroids = list(product(*roi_centroids))

        roi_centroids = [tuple([int(x) for x in centroid])
            for centroid in roi_centroids]

        return roi_centroids

    def _crop_roi(self, arr, centroid):
        roi = np.ones(tuple([self.crop_size] * 3)) * (-1024)

        src_beg = [max(0, centroid[i] - self.crop_size // 2)
            for i in range(len(centroid))]
        src_end = [min(arr.shape[i], centroid[i] + self.crop_size // 2)
            for i in range(len(centroid))]
        dst_beg = [max(0, self.crop_size // 2 - centroid[i])
            for i in range(len(centroid))]
        dst_end = [min(arr.shape[i] - (centroid[i] - self.crop_size // 2),
            self.crop_size) for i in range(len(centroid))]
        roi[
            dst_beg[0]:dst_end[0],
            dst_beg[1]:dst_end[1],
            dst_beg[2]:dst_end[2],
        ] = arr[
            src_beg[0]:src_end[0],
            src_beg[1]:src_end[1],
            src_beg[2]:src_end[2],
        ]

        return roi

    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image)

        return image

    def __getitem__(self, idx):
        # read image and label
        public_id = self.public_id_list[idx]
        image_path = os.path.join(self.image_dir, f"{public_id}-image.nii.gz")
        label_path = os.path.join(self.label_dir, f"{public_id}-label.nii.gz")
        image = nib.load(image_path)
        label = nib.load(label_path)
        image_arr = image.get_fdata().astype(np.float)
        label_arr = label.get_fdata().astype(np.uint8)

        # calculate rois' centroids
        roi_centroids = self._get_roi_centroids(label_arr)

        # crop rois
        image_rois = [self._crop_roi(image_arr, centroid)
            for centroid in roi_centroids]
        label_rois = [self._crop_roi(label_arr, centroid)
            for centroid in roi_centroids]

        if self.transforms is not None:
            image_rois = [self._apply_transforms(image_roi)
                for image_roi in image_rois]

        image_rois = torch.tensor(np.stack(image_rois)[:, np.newaxis],
            dtype=torch.float)
        label_rois = (np.stack(label_rois) > 0).astype(np.float)
        label_rois = torch.tensor(label_rois[:, np.newaxis],
            dtype=torch.float)

        return image_rois, label_rois

    @staticmethod
    def collate_fn(samples):
        image_rois = torch.cat([x[0] for x in samples])
        label_rois = torch.cat([x[1] for x in samples])

        return image_rois, label_rois

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,
            num_workers=num_workers, collate_fn=FracNetTrainDataset.collate_fn)

class FracNetInferenceDataset(Dataset):

    def __init__(self, image_path, crop_size=64, transforms=None):
        image = nib.load(image_path)
        self.image_affine = image.affine
        self.image = image.get_fdata().astype(np.int16)
        self.crop_size = crop_size
        self.transforms = transforms
        self.centers = self._get_centers()

    def _get_centers(self):
        dim_coords = [list(range(0, dim, self.crop_size // 2))[1:-1]\
            + [dim - self.crop_size // 2] for dim in self.image.shape]
        centers = list(product(*dim_coords))

        return centers

    def __len__(self):
        return len(self.centers)

    def _crop_patch(self, idx):
        center_x, center_y, center_z = self.centers[idx]
        patch = self.image[
            center_x - self.crop_size // 2:center_x + self.crop_size // 2,
            center_y - self.crop_size // 2:center_y + self.crop_size // 2,
            center_z - self.crop_size // 2:center_z + self.crop_size // 2
        ]

        return patch

    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image)

        return image

    def __getitem__(self, idx):
        image = self._crop_patch(idx)
        center = self.centers[idx]

        if self.transforms is not None:
            image = self._apply_transforms(image)

        image = torch.tensor(image[np.newaxis], dtype=torch.float)

        return image, center

    @staticmethod
    def _collate_fn(samples):
        images = torch.stack([x[0] for x in samples])
        centers = [x[1] for x in samples]

        return images, centers

    @staticmethod
    def get_dataloader(dataset, batch_size, num_workers=0):
        return DataLoader(dataset, batch_size, num_workers=num_workers,
            collate_fn=FracNetInferenceDataset._collate_fn)

