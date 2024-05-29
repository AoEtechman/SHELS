import torch
import os
import pickle

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data



# This file can be used to load and save datasets in the case of no internet access. 
# for example this is how you would load and save the fashion mnist dataset from pytorch


# transform = transforms.Compose([transforms.ToTensor()])
# trainset = datasets.FashionMNIST(root = '../data', train = True, download = True, transform = transform)
# testset  = datasets.FashionMNIST(root = '../data', train = False, download = True, transform = transform)
# torch.save(trainset, filename + '/fmnist/trainset.pt')
# torch.save(testset, filename + '/fmnist/testset.pt')
