import torch
import os
import pickle

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data


transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224])]) # was 224, 224
                                    #transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    # transform= transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = datasets.CIFAR10(root = '../data', train = True, download = True, transform = transform)
testset  = datasets.CIFAR10(root = '../data', train = False, download = True, transform = transform)
filename =  '/home/gridsan/aejilemele/LIS/SHELS-main/datasets'

torch.save(trainset,filename + '/cifar10/trainset.pt')
torch.save(testset, filename + '/cifar10/testset.pt')


transform = transforms.Compose([transforms.ToTensor()])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
trainset = datasets.MNIST(root = '../data', train = True, download = True, transform = transform)
testset  = datasets.MNIST(root = '../data', train = False, download = True, transform = transform)
torch.save(trainset, filename + '/mnist/trainset.pt')
torch.save(testset, filename + '/mnist/testset.pt')
