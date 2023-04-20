
from email.generator import Generator
from logging import PercentStyle
from random import random
import numpy as np
from copy import copy

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch
import pdb
from custom_dataloader import CustomDataset, GeneralDataset
from preprocess_audio_mnist import preprocess_data
from arguments import get_args, print_args

import scipy


def data_loader_CIFAR_10(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):
    
    args = get_args()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224])]) # was 224, 224
                                    #transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    # transform= transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #trainset = datasets.CIFAR10(root = '../data', train = True, download = True, transform = transform)
    #testset  = datasets.CIFAR10(root = '../data', train = False, download = True, transform = transform)
    trainset = torch.load('/home/gridsan/aejilemele/LIS/SHELS-main/datasets/cifar10/trainset.pt')
    testset = torch.load('/home/gridsan/aejilemele/LIS/SHELS-main/datasets/cifar10/testset.pt')

    classes = trainset.classes
    print(classes)
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        #print(ood_class, 'ood_class')
        trainset, testset = create_traintestset(ood_class, trainset,testset)
        if args.debugging:
            trainset = data.Subset(trainset, np.arange(0, 500))
            testset = data.Subset(testset, np.arange(0, 200))
        #print(torch.unique(torch.tensor(trainset.targets)), 'unique items in trainset')
        for i in range(0, len(ood_class)):
            #trainset_ood = datasets.CIFAR10(root = '../data', train = True, download = True, transform = transform)
            trainset_ood =  torch.load('/home/gridsan/aejilemele/LIS/SHELS-main/datasets/cifar10/trainset.pt')
            testset_ood =  torch.load('/home/gridsan/aejilemele/LIS/SHELS-main/datasets/cifar10/testset.pt')
            #testset_ood  = datasets.CIFAR10(root = '../data', train = False, download = True, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            if args.debugging:
                ood_trainset = data.Subset(ood_trainset, np.arange(0, 100))
                ood_testset  = data.Subset(ood_testset, np.arange(0, 100))
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)
    
    #print(torch.unique(torch.tensor(trainset.targets)), 'unique values in final thing passes into trainloader')
    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
     
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    if args.leave_one_out == True:
        np.random.seed(args.random_seed)
        np.random.shuffle(classes) # shuffles classes to choose which are iid and ood
        classes_idx_ID = classes[0:args.ID_tasks]
        trainset_one_out, testset_one_out = [], []
        # generate the id classes, then trainset and testset need to be a list of data loaders for each id class
        for id_class in classes_idx_ID:
            trainset_copy = train_set.clone()
            testset_copy = testset.clone()
            class_trainset, class_testset = create_OOD_dataset(id_class, trainset_copy, testset_copy)# although this says ood, the functionality that we desire is achieved by
            # this function. We simply want to get the training and testing points belonging to this specific id class which is achieved by this function
            trainset_one_out.append(class_trainset)
            testset_one_out.append(class_testset)
    # combine_set = []
    # combine_set.append(val_set)
    # combine_set.append(ood_trainset)
    # combine_set = data.ConcatDataset(combine_set)
    #train_set = trainset
    #print(torch.unique(torch.tensor(train_set.targets)), 'unique values in final thing passes into trainloader')
    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2, drop_last = True)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    if args.leave_one_out:
        return trainloader, valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset


def data_loader_GTSRB(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([112,112])]) # 112,112
                                    #transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    # transform= transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    root_dir_train = '../data/GTSRB/Final_Training/Images/'
    root_dir_test = '../data/GTSRB/Test/'
    trainset = CustomDataset(data_path = root_dir_train,transform = transform)
    testset  = CustomDataset(data_path = root_dir_test,transform = transform)
    classes = trainset.classes
    print(len(classes))
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        trainset, testset = create_traintestset(ood_class, trainset,testset)

        for i in range(0, len(ood_class)):
            trainset_ood = CustomDataset(data_path = root_dir_train, transform = transform)
            testset_ood  = CustomDataset(data_path = root_dir_test, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)

    train_data_len = 2*( int(0.88*len(trainset)/2.0))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])

    # bypassing val dataset


    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    return trainloader,trainloader, testloader, ood_trainset_list, ood_testset_list, classes, testset

def data_loader_SVHN(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):

    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([32,32])]) # was 32 32


    trainset = datasets.SVHN(root = '../data', split = 'train', download = True, transform = transform)
    testset  = datasets.SVHN(root = '../data', split = 'test', download = True, transform = transform)

    classes = [0,1,2,3,4,5,6,7,8,9]
    print(classes)
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        trainset, testset = create_traintestset_svhn(ood_class, trainset,testset)

        for i in range(0, len(ood_class)):
            trainset_ood = datasets.SVHN(root = '../data', split = 'train', download = True, transform = transform)
            testset_ood  = datasets.SVHN(root = '../data', split = 'test', download = True, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset_svhn(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)

    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    # combine_set = []
    # combine_set.append(val_set)
    # combine_set.append(ood_trainset)
    # combine_set = data.ConcatDataset(combine_set)

    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset


def data_loader_FashionMNIST(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):

    transform = transforms.Compose([transforms.ToTensor()])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    trainset = datasets.FashionMNIST(root = '../data', train = True, download = True, transform = transform)
    testset  = datasets.FashionMNIST(root = '../data', train = False, download = True, transform = transform)
    classes = trainset.classes
    print(classes)
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        trainset, testset = create_traintestset(ood_class, trainset,testset)

        for i in range(0, len(ood_class)):
            trainset_ood = datasets.FashionMNIST(root = '../data', train = True, download = True, transform = transform)
            testset_ood  = datasets.FashionMNIST(root = '../data', train = False, download = True, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)

    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    # combine_set = []
    # combine_set.append(val_set)
    # combine_set.append(ood_trainset)
    # combine_set = data.ConcatDataset(combine_set)

    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset

def data_loader_MNIST(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):

    args = get_args()
    transform = transforms.Compose([transforms.ToTensor()])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    #trainset = datasets.MNIST(root = '../data', train = True, download = True, transform = transform)
    #testset  = datasets.MNIST(root = '../data', train = False, download = True, transform = transform)
    trainset = torch.load('/home/gridsan/aejilemele/LIS/SHELS-main/datasets/mnist/trainset.pt')
    testset = torch.load('/home/gridsan/aejilemele/LIS/SHELS-main/datasets/mnist/testset.pt')
    classes = trainset.classes
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        trainset, testset = create_traintestset(ood_class, trainset,testset)
        if args.debugging:
            #trainset = data.Subset(trainset, np.arange(0, 500))
            #testset = data.Subset(testset, np.arange(0, 200)
            trainset.data = np.array(trainset.data)[np.arange(0, 500, 1, dtype = int)]
            trainset.targets = np.array(trainset.targets)[np.arange(0, 500, 1, dtype = int)]
            testset.data = np.array(testset.data)[np.arange(0, 200, 1, dtype = int)]
            testset.targets = np.array(testset.targets)[np.arange(0, 200, 1, dtype = int)]
        for i in range(0, len(ood_class)):
            #trainset_ood = datasets.MNIST(root = '../data', train = True, download = True, transform = transform)
            #testset_ood  = datasets.MNIST(root = '../data', train = False, download = True, transform = transform)
            trainset_ood = torch.load('/home/gridsan/aejilemele/LIS/SHELS-main/datasets/mnist/trainset.pt')
            testset_ood = torch.load('/home/gridsan/aejilemele/LIS/SHELS-main/datasets/mnist/testset.pt')
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            if args.debugging:
                #ood_trainset = data.Subset(ood_trainset, np.arange(0, 100))
                #ood_testset = data.Subset(ood_testset, np.arange(0, 100))
                 ood_trainset.data = np.array(ood_trainset.data)[np.arange(0, 100, 1, dtype = int)]
                 ood_trainset.targets = np.array(ood_trainset.targets)[np.arange(0, 100, 1, dtype = int)]
                 ood_testset.data = np.array(ood_testset.data)[np.arange(0, 200, 1, dtype = int)]
                 ood_testset.targets = np.array(ood_testset.targets)[np.arange(0, 200, 1, dtype = int)]
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)
    #print(trainset.targets, 'trainset targets')
    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    
    if args.leave_one_out:
        train_indices = train_set.indices
        val_indices = val_set.indices
        print(type(train_set.dataset.data))
        #print('testing data shape', train_set.dataset.data[train_indices].shape)
        #print('original data shape', train_set.dataset.data.shape)
        train_set_dataset = GeneralDataset(np.array(train_set.dataset.data)[train_indices].tolist(), np.array(train_set.dataset.targets)[train_indices])
        val_set_dataset = GeneralDataset(np.array(val_set.dataset.data)[val_indices].tolist(), np.array(val_set.dataset.targets)[val_indices])
        #print(train_set_dataset)
        #print(train_set_dataset.targets)
        #print(len(train_set), 'trainset length real')
        #print(len(train_set_dataset), 'trainset length')
        #print(len(val_set_dataset), 'valset length')
        np.random.seed(args.random_seed)
        classes_integer_list = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
        np.random.shuffle(classes_integer_list) # shuffles classes to choose which are iid and ood
        classes_idx_ID = classes_integer_list[0:args.ID_tasks]
        #print(classes_idx_ID)
        trainset_one_out, testset_one_out, valset_one_out = [], [], []

        
        # generate the id classes, then trainset and testset need to be a list of data loaders for each id class
        #valset_copy2 = copy(val_set_dataset) # this only needs to be copied once as we do not care how it is mutated as we will not use it
        for id_class in classes_idx_ID:


            #print(id_class, 'id class')
            # look into how to fix this
            trainset_copy = copy(train_set_dataset)
            #print(len(trainset_copy), 'dataset copy length')
            testset_copy = copy(testset)
            valset_copy = copy(val_set_dataset)




            class_trainset, class_testset, class_valset = create_OOD_dataset(id_class, trainset_copy, testset_copy, valset_copy)# although this says ood, the functionality that we desire is achieved by
            # this function. We simply want to get the training and testing points belonging to this specific id class which is achieved by this function
            #print(len(class_trainset), 'class trainset length')
            #print(class_trainset.data[0].shape, 'class trainset data shape')
            #class_valset, _ = create_OOD_dataset(id_class, valset_copy, valset_copy2)
            trainset_one_out.append(class_trainset)
            testset_one_out.append(class_testset)
            valset_one_out.append(class_valset)
    # combine_set = []
    # combine_set.append(val_set)
    # combine_set.append(ood_trainset)
    # combine_set = data.ConcatDataset(combine_set)
    #print('second batch size check', BATCH_SIZE)
    #print('type of trainset', type(train_set.dataset.data))
    #print(type(train_set.dataset.data[0]))
    if args.leave_one_out:
        BATCH_SIZE = 1
    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2, drop_last = True)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Te\st set length", len(testset))
    print("OOD set length", ood_testset_len)
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    if args.leave_one_out:
        return trainloader, valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset


def data_loader_Audio_MNIST(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):
    src_path, meta_data_path = r"/home/gridsan/aejilemele/LIS/AudioMNIST/data", r"/home/gridsan/aejilemele/LIS/AudioMNIST/AudioMNIST_meta.txt"
    Data, targets, classes = preprocess_data(src_path, meta_data_path)
    print('------------')
    d = Data.detach().clone()
    t = targets.detach().clone()
    trainset, testset = random_split_dataset(d, t)
    trainset_ood, testset_ood = random_split_dataset(d, t)
    
    print(classes)
    print('ood class')
    print(ood_class)

    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0


    if len(ood_class) > 0:
        trainset, testset = create_traintestset_audio(ood_class, trainset,testset)
        
        #trainset_ood = AudioDataset(trainset_ood.dataset.data[trainset_ood.indices],trainset_ood.dataset.targets[trainset_ood.indices] )
        # print(len(trainset_ood))
        #testset_ood = AudioDataset(testset_ood.dataset.data[testset_ood.indices],testset_ood.dataset.targets[testset_ood.indices] )
        # print(len(testset_ood))
 
        for i in range(len(ood_class)):
            # trainset_ood = datasets.MNIST(root = '../data', train = True, download = True, transform = transform)
            # testset_ood  = datasets.MNIST(root = '../data', train = False, download = True, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset_audio(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)
 
    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len


    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])

    # combine_set = []
    # combine_set.append(val_set)
    # combine_set.append(ood_trainset)
    # combine_set = data.ConcatDataset(combine_set)
    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)

    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2, drop_last = True)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset


def create_subdataset(classes, trainset, testset):

    #for i in range(len(ood_class)):

    idx = np.where(np.isin(np.array(trainset.targets), classes, invert = False))[0].tolist()
    accessed_targets = map(trainset.targets.__getitem__, idx)
    trainset.targets = list(accessed_targets)
    accessed_data = map(trainset.data.__getitem__, idx)
    trainset.data = list(accessed_data)

    idx = np.where(np.isin(np.array(testset.targets), classes, invert = False))[0].tolist()
    accessed_targets = map(testset.targets.__getitem__, idx)
    testset.targets = list(accessed_targets)
    accessed_data = map(testset.data.__getitem__, idx)
    testset.data = list(accessed_data)
    return trainset, testset



def create_traintestset_audio(ood_class, trainset, testset):

    #for i in range(len(ood_class)):
    tf_map = np.isin(np.array(trainset.dataset.targets[trainset.indices]), ood_class, invert = True)
    idx = np.where(tf_map)[0].tolist()# if in ood class set to false
    accessed_targets = map(trainset.dataset.targets[trainset.indices].__getitem__, idx)
    targets = list(accessed_targets)
    accessed_data = map(trainset.dataset.data[trainset.indices].__getitem__, idx)
    data = list(accessed_data)
    trainset = GeneralDataset(data, targets)

    idx = np.where(np.isin(np.array(testset.dataset.targets[testset.indices]), ood_class, invert = True))[0].tolist()
    accessed_targets = map(testset.dataset.targets[testset.indices].__getitem__, idx)
    test_targets = list(accessed_targets)
    accessed_data = map(testset.dataset.data[testset.indices].__getitem__, idx)
    test_data = list(accessed_data)
    testset = GeneralDataset(test_data, test_targets)
    return trainset, testset

def create_traintestset(ood_class, trainset, testset):

    #for i in range(len(ood_class)):

    idx = np.where(np.isin(np.array(trainset.targets), ood_class, invert = True))[0].tolist()
    accessed_targets = map(trainset.targets.__getitem__, idx)
    trainset.targets = list(accessed_targets)
    accessed_data = map(trainset.data.__getitem__, idx)
    trainset.data = list(accessed_data)

    idx = np.where(np.isin(np.array(testset.targets), ood_class, invert = True))[0].tolist()
    accessed_targets = map(testset.targets.__getitem__, idx)
    testset.targets = list(accessed_targets)
    accessed_data = map(testset.data.__getitem__, idx)
    testset.data = list(accessed_data)
    return trainset, testset

def create_OOD_dataset_audio(ood_class, ood_trainset, oodset):


    idx_ood = np.where(np.array(ood_trainset.dataset.targets) == ood_class)[0].tolist()
    accessed_targets = map(ood_trainset.dataset.targets.__getitem__, idx_ood)
    targets = list(accessed_targets)
    accessed_data = map(ood_trainset.dataset.data.__getitem__, idx_ood)
    data = list(accessed_data)
    ood_trainset = GeneralDataset(data, targets)




    idx_ood = np.where(np.array(oodset.dataset.targets) == ood_class)[0].tolist()
    accessed_targets = map(oodset.dataset.targets.__getitem__, idx_ood)
    targets = list(accessed_targets)
    accessed_data = map(oodset.dataset.data.__getitem__, idx_ood)
    data = list(accessed_data)
    oodset = GeneralDataset(data, targets)


    return ood_trainset, oodset

def create_OOD_dataset(ood_class, ood_trainset, oodset, valset = None):


    idx_ood = np.where(np.array(ood_trainset.targets) == ood_class)[0].tolist()
    accessed_targets = map(ood_trainset.targets.__getitem__, idx_ood)
    ood_trainset.targets = list(accessed_targets)
    accessed_data = map(ood_trainset.data.__getitem__, idx_ood)
    ood_trainset.data = list(accessed_data)




    idx_ood = np.where(np.array(oodset.targets) == ood_class)[0].tolist()
    accessed_targets = map(oodset.targets.__getitem__, idx_ood)
    oodset.targets = list(accessed_targets)
    accessed_data = map(oodset.data.__getitem__, idx_ood)
    oodset.data = list(accessed_data)

    if valset:
        idx_ood = np.where(np.array(valset.targets) == ood_class)[0].tolist()
        accessed_targets = map(valset.targets.__getitem__, idx_ood)
        valset.targets = list(accessed_targets)
        accessed_data = map(valset.data.__getitem__, idx_ood)
        valset.data = list(accessed_data)
        return ood_trainset, oodset, valset

    return ood_trainset, oodset

def create_traintestset_svhn(ood_class, trainset, testset):

    for i in range(len(ood_class)):


        idx = np.where(np.array(trainset.labels) != ood_class[i])[0].tolist()
        accessed_targets = map(trainset.labels.__getitem__, idx)
        trainset.labels = list(accessed_targets)
        accessed_data = map(trainset.data.__getitem__, idx)
        trainset.data = list(accessed_data)

        idx = np.where(np.array(testset.labels) != ood_class[i])[0].tolist()
        accessed_targets = map(testset.labels.__getitem__, idx)
        testset.labels = list(accessed_targets)
        accessed_data = map(testset.data.__getitem__, idx)
        testset.data = list(accessed_data)

    return trainset, testset

def create_OOD_dataset_svhn(ood_class, ood_trainset, oodset):


    idx_ood = np.where(np.array(ood_trainset.labels) == ood_class)[0].tolist()
    accessed_targets = map(ood_trainset.labels.__getitem__, idx_ood)
    ood_trainset.labels = list(accessed_targets)
    accessed_data = map(ood_trainset.data.__getitem__, idx_ood)
    ood_trainset.data = list(accessed_data)




    idx_ood = np.where(np.array(oodset.labels) == ood_class)[0].tolist()
    accessed_targets = map(oodset.labels.__getitem__, idx_ood)
    oodset.labels = list(accessed_targets)
    accessed_data = map(oodset.data.__getitem__, idx_ood)
    oodset.data = list(accessed_data)


    return ood_trainset, oodset

def random_split_dataset(d, t):
    dataset = GeneralDataset(d, t)
    length = len(dataset)

    train_len = int(length // (5/4))
   
    test_len = length - train_len

    return data.random_split(dataset, [train_len, test_len])


   
