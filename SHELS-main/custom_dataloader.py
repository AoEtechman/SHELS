import glob
from subprocess import list2cmdline
import numpy as np
import torch 
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_path, transform):
        self.imgs_path = data_path
        file_list = glob.glob(self.imgs_path + "*")
        self.transform = transform 
        self.loader = default_loader
        self.data = []
        self.targets = []
        self.class_name_list = []
        i = 0
        self.class_map = {}
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            self.class_name_list.append(class_name)
            self.class_map[self.class_name_list[i]] = i
            for img_path in glob.glob(class_path+"/*.ppm"):
                self.data.append(img_path)
                self.targets.append(i)
            i+=1
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.classes = self.class_name_list 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        #path = os.path.join(sample)
        target = self.targets[idx]
        img = self.loader(path)
        if img is None:
            print(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target 



class GeneralDataset(Dataset):
    def __init__(self, data, targets, transform = None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        if torch.is_tensor(self.data):
            return self.data.size(dim = 0)
        elif isinstance(self.data, np.ndarray):
            return self.data.shape[0]
        return len(self.data)

    def __getitem__(self, idx):
        input_features = self.data[idx]
        if self.transform is not None:
            input_features = self.transform(input_features)
        target = self.targets[idx]
        return input_features, target
    

class GeneralImgDataset(Dataset):
    def __init__(self, data, targets, transform = None, rgb = False):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        if torch.is_tensor(self.data):
            return self.data.size(dim = 0)
        elif isinstance(self.data, np.ndarray):
            return self.data.shape[0]
        return len(self.data)

    def __getitem__(self, idx):
        input_features = self.data[idx]
        if not self.rgb:
            if isinstance(input_features, np.ndarray):
                input_features = Image.fromarray(input_features, mode="L")
            elif torch.is_tensor(input_features):
                input_features = Image.fromarray(input_features.numpy(), mode="L")
        else:
            if isinstance(input_features, np.ndarray):
                if input_features.shape[0] == 3: # should have color channels in the last dimension
                    input_features = input_features.transpose((1, 2, 0))
                input_features = Image.fromarray(input_features)
            elif torch.is_tensor(input_features):
                if input_features.shape[0] == 3:# should have color channels in the last dimension
                    input_features = torch.transpose(input_features, 0, 2)
                    input_features = torch.transpose(input_features, 0, 1)
               
                input_features = Image.fromarray(input_features.numpy())
        if self.transform is not None:
            input_features = self.transform(input_features)
        target = self.targets[idx]
        return input_features, target
    






