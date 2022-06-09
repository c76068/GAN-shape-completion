"""
Define dataset and transforms
"""

from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import numpy as np
import torch
from skimage.transform import resize


def set_transform(resize=256):
    transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.ToTensor()
    ])

    return transform


class simpleDataset(Dataset):
    def __init__(self, folder, max_files=None, inputs=['vis'], include_stoppage=False, transform=None, resize_dim=256):
        self.folder = folder
        self.paths = glob.glob(os.path.join(folder,'*.npz'))
        self.transform = transform
        self.resize_dim = resize_dim
        self.inputs = inputs

        # ignore files that has 'stop' in filename if include_stoppage == False
        if not include_stoppage:
            self.paths = [f for f in self.paths if 'stop' not in f]


        if max_files and len(self.paths)>max_files:
            self.paths = self.paths[:max_files]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        filepath = self.paths[index]
        saved = np.load(filepath)
        saved = dict(zip((key for key in saved.keys()), (saved[key] for key in saved.keys())))
        saved['vis'] = (saved['psi']>0)*1.0

        X = np.stack([resize(saved[name],(self.resize_dim,self.resize_dim),order=0) for name in self.inputs],0)
        X = torch.Tensor(X)
        #X = torch.Tensor(resize((saved['psi']>0)*1.0,(self.resize_dim,self.resize_dim),order=0))
        Y = torch.Tensor(resize((saved['phi']>0)*1.0,(self.resize_dim,self.resize_dim),order=0)).unsqueeze(0)


        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)

        return X, Y


class CustomDataset(Dataset):
    """A custom dataset for loading 2D visibility data"""
    
    def __init__(self, data_dir, max_files, transform=None, gain_threshold=0, include_stoppage=False):
        """
        Initialize the dataset object
        
        :param data_dir: data directory 
        :param max_files: maximum number of data files to load 
        :param transform: optional transform to be applied on a sample 
        :param gain_threshold: threshold on the maximum gain of a sample 
        :param include_stoppage: whether or not to include stoppage samples 
        """
        
        # load files from data directory 
        self.data_dir = data_dir
        self.files = os.listdir(self.data_dir)[:max_files]
        self.files.sort()
        
        # ignore files that has 'stop' in filename if include_stoppage == False
        if not include_stoppage:
            self.files = [f for f in self.files if 'stop' not in f]
        
        # filter out samples whose maximum gain value is below gain_threshold 
        if gain_threshold > 0:
            self.files = [f for f in self.files 
                          if np.load(os.path.join(self.data_dir, f))['E'].max() >= gain_threshold]
        
        # assign IDs to samples 
        self.IDs = np.arange(len(self.files))
        
        # transform to be applied on a sample 
        self.transform = transform
        
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        
        return len(self.files)
    
    def __getitem__(self, idx):
        """Load and return a sample from the dataset at the given index idx"""
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # load sample 
        filename = self.files[idx]
        saved = np.load(os.path.join(self.data_dir, filename))
        
        ID = self.IDs[idx]  # sample ID
        psi = saved['psi']  # cumulative visibility
        phi = saved['phi']  # scene
        horizons = saved['horizons']  # frontier
        E = saved['E']  # gain function 
        
        # convert continuous to discrete 
        vis = 1.0*(psi>0)
        scene = 1.0*(phi>0)
        
        # form sample 
        sample = {'ID': ID, 'vis': vis, 'scene': scene, 'frontier': horizons, 'gain': E, 'max_gain': E.max()}
        
        # transform sample 
        if self.transform:
            sample = self.transform(sample)

        return sample

    
class ApplyTransform(Dataset):
    """Apply transformations to a Dataset"""
    
    def __init__(self, dataset, transform=None):
        """
        Initialize the dataset object 
        
        :param dataset: (Dataset) a Dataset that returns samples
        :param transform: (callable, optional) a function/transform to be applied on the sample
        """
        
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        """Load and return a sample from the dataset at the given index idx"""
        
        sample = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        """Return the number of samples in the dataset"""
        
        return len(self.dataset)
   
    
class ScaleGain(object):
    """Scale the gain function of a sample by either taking square root or dividing by maximum"""
    
    def __init__(self, mode):
        self.mode = mode
        
    def __call__(self, sample):
        
        if self.mode == 'none':
            pass
        elif self.mode == 'sqrt':
            sample['gain'] = np.sqrt(sample['gain'])
        elif self.mode == 'div_by_max':
            if sample['max_gain'] != 0:
                sample['gain'] /= sample['max_gain']
        else:
            raise ValueError
            
        return sample

    
class SelectInputsLabel(object):
    """Select inputs to the network"""
    
    def __init__(self, input_names, label_name):
        self.input_names = input_names
        self.label_name = label_name
        
    def __call__(self, sample):

        return {'ID': sample['ID'],
                'image': np.stack([sample[name] for name in self.input_names], 0),
                'label': sample[self.label_name],
                'max_gain': sample['max_gain']}

    
class ToTensor(object):
    """Convert ndarrays of a sample to tensors"""
    
    def __call__(self, sample):
        
        for key in sample.keys():
            sample[key] = torch.tensor(sample[key])
        
        return sample
    
