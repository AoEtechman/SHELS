
from email.generator import Generator
from logging import PercentStyle
from random import random
import numpy as np
from copy import copy, deepcopy

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch
import pdb
from custom_dataloader import CustomDataset, GeneralImgDataset, GeneralDataset
from preprocess_audio_mnist import preprocess_data
from arguments import get_args, print_args

import scipy


###These are the functions for preparing the datasets and dataloaders for all of the datasets



def data_loader_tinyimagenet(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):
    
    args = get_args()
    norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    test_transform = transforms.Compose([transforms.ToTensor(), norm])
    if args.data_path == "":
        args.data_path = "datasets/tinyimagenet/"

    if args.data_path[-1] != "/":
        trainset_path = args.data_path + '/trainset.pt'
        testset_path = args.data_path + "/testset.pt"
        trainset = torch.load(trainset_path) # this code was run by submitting jobs that do not have access to the internet. If
                                                               # if Internet access is available, these lines in all of the loader functions
                                                               # should be replaced with the typical code used to load datasets in Pytorch. 
        testset = torch.load(testset_path)
    else:
        trainset_path = args.data_path + 'trainset.pt'
        testset_path = args.data_path + "testset.pt"

        trainset = torch.load(trainset_path)                                                                                               
        testset = torch.load(testset_path)


    classes = np.unique(trainset.targets)
    print(classes)
    ood_testset_list = [] 
    ood_trainset_list = [] 
    ood_testset_len = 0
    if len(ood_class) > 0:

        trainset, testset = create_traintestset(ood_class, trainset,testset)
       
        for i in range(0, len(ood_class)):
            trainset_ood =  torch.load(trainset_path) # we reload the datasets to avoid mutating the trainset and testset we created
            testset_ood =  torch.load(testset_path)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)
    
    
    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len

    
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])




    if args.leave_one_out:
        train_indices = train_set.indices
        val_indices = val_set.indices
      
        train_set_dataset = GeneralImgDataset(list(np.array(train_set.dataset.data)[train_indices]), list(np.array(train_set.dataset.targets)[train_indices]), train_transform, True)
        val_set_dataset = GeneralImgDataset(list(np.array(val_set.dataset.data)[val_indices]), list(np.array(val_set.dataset.targets)[val_indices]), train_transform, True)


        np.random.seed(args.random_seed)
        classes_integer_list = deepcopy(classes)
        np.random.shuffle(classes_integer_list) # shuffles classes to choose which are iid and ood
        classes_idx_ID = classes_integer_list[0:args.ID_tasks]  # the id class labels
        trainset_one_out, testset_one_out, valset_one_out = [], [], [] # lists of the leave one out trainsets, testsets, and valsets

        
        
        for id_class in classes_idx_ID:
            trainset_copy = copy(train_set_dataset)
            testset_copy = copy(testset)
            valset_copy = copy(val_set_dataset)

            class_trainset, class_testset, class_valset = create_OOD_dataset(id_class, trainset_copy, testset_copy, valset_copy)# although this says create ood datasets, the function at its core is to return a dataset with just the label that is passed in                                                                                                     
            trainset_one_out.append(class_trainset)
            testset_one_out.append(class_testset)
            valset_one_out.append(class_valset)


    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2, drop_last = True)
   

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)

    
    if args.leave_one_out:
        return trainloader, valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset




def data_loader_CIFAR_10(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):
    
    args = get_args()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224])]) # was 224, 224
                                    #transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    # transform= transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #trainset = datasets.CIFAR10(root = '../data', train = True, download = True, transform = transform)
    #testset  = datasets.CIFAR10(root = '../data', train = False, download = True, transform = transform)
    if args.data_path == "":
        args.data_path = "datasets/cifar10/"

    if args.data_path[-1] != "/":
        trainset_path = args.data_path + '/trainset.pt'
        testset_path = args.data_path + "/testset.pt"
        trainset = torch.load(trainset_path) 
        testset = torch.load(testset_path)
    else:
        trainset_path = args.data_path + 'trainset.pt'
        testset_path = args.data_path + "testset.pt"

        trainset = torch.load(trainset_path)                                                                                               
        testset = torch.load(testset_path)

   
    classes = trainset.classes
    print(classes)
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:

        trainset, testset = create_traintestset(ood_class, trainset,testset)
    
        for i in range(0, len(ood_class)):
            #trainset_ood = datasets.CIFAR10(root = '../data', train = True, download = True, transform = transform)
            trainset_ood =  torch.load(trainset_path)
            testset_ood =  torch.load(testset_path)
            #testset_ood  = datasets.CIFAR10(root = '../data', train = False, download = True, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)

    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len

     
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    if args.leave_one_out:
        train_indices = train_set.indices
        val_indices = val_set.indices
      
        train_set_dataset = GeneralImgDataset(list(np.array(train_set.dataset.data)[train_indices]), list(np.array(train_set.dataset.targets)[train_indices]), transform, True)
        val_set_dataset = GeneralImgDataset(list(np.array(val_set.dataset.data)[val_indices]), list(np.array(val_set.dataset.targets)[val_indices]), transform, True)


       
        np.random.seed(args.random_seed)
        classes_integer_list = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
        np.random.shuffle(classes_integer_list) # shuffles classes to choose which are iid and ood
        classes_idx_ID = classes_integer_list[0:args.ID_tasks]
        trainset_one_out, testset_one_out, valset_one_out = [], [], []

        
    
        for id_class in classes_idx_ID:
            trainset_copy = copy(train_set_dataset)
            testset_copy = copy(testset)
            valset_copy = copy(val_set_dataset)

            class_trainset, class_testset, class_valset = create_OOD_dataset(id_class, trainset_copy, testset_copy, valset_copy)# although this says ood, the functionality that we desire is achieved by
            trainset_one_out.append(class_trainset)
            testset_one_out.append(class_testset)
            valset_one_out.append(class_valset)



    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2, drop_last = True)
   
    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    
    if args.leave_one_out:
        return trainloader, valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out
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
  

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
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
   

    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset


def data_loader_FashionMNIST(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):
    
    args = get_args()
    transform = transforms.Compose([transforms.ToTensor()])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    #trainset = datasets.FashionMNIST(root = '../data', train = True, download = True, transform = transform)
    #testset  = datasets.FashionMNIST(root = '../data', train = False, download = True, transform = transform)
    if args.data_path == "":
        args.data_path = "datasets/fmnist/"

    if args.data_path[-1] != "/":
        trainset_path = args.data_path + '/trainset.pt'
        testset_path = args.data_path + "/testset.pt"
        trainset = torch.load(trainset_path) 
        testset = torch.load(testset_path)
    else:
        trainset_path = args.data_path + 'trainset.pt'
        testset_path = args.data_path + "testset.pt"

        trainset = torch.load(trainset_path)                                                                                               
        testset = torch.load(testset_path)

    classes = trainset.classes
    print(classes)
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        trainset, testset = create_traintestset(ood_class, trainset,testset)

        for i in range(0, len(ood_class)):
            #trainset_ood = datasets.FashionMNIST(root = '../data', train = True, download = True, transform = transform)
            #testset_ood  = datasets.FashionMNIST(root = '../data', train = False, download = True, transform = transform)
            trainset_ood = torch.load(trainset_path)
            testset_ood = torch.load(testset_path)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)

    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    if args.leave_one_out:
        train_indices = train_set.indices
        val_indices = val_set.indices
      
        train_set_dataset = GeneralImgDataset(np.array(train_set.dataset.data)[train_indices].tolist(), np.array(train_set.dataset.targets)[train_indices], transform)
        val_set_dataset = GeneralImgDataset(np.array(val_set.dataset.data)[val_indices].tolist(), np.array(val_set.dataset.targets)[val_indices], transform)

        np.random.seed(args.random_seed)
        classes_integer_list = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
        np.random.shuffle(classes_integer_list) # shuffles classes to choose which are iid and ood
        classes_idx_ID = classes_integer_list[0:args.ID_tasks]
        trainset_one_out, testset_one_out, valset_one_out = [], [], []

        
        for id_class in classes_idx_ID:
            trainset_copy = copy(train_set_dataset)
            testset_copy = copy(testset)
            valset_copy = copy(val_set_dataset)

            class_trainset, class_testset, class_valset = create_OOD_dataset(id_class, trainset_copy, testset_copy, valset_copy)# although this says ood, the functionality that we desire is achieved by
            trainset_one_out.append(class_trainset)
            testset_one_out.append(class_testset)
            valset_one_out.append(class_valset)


    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    if args.leave_one_out:
        return trainloader, valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset

def data_loader_MNIST(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):

    args = get_args()
    transform = transforms.Compose([transforms.ToTensor()])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    #trainset = datasets.MNIST(root = '../data', train = True, download = True, transform = transform)
    #testset  = datasets.MNIST(root = '../data', train = False, download = True, transform = transform)
    if args.data_path == "":
        args.data_path = "datasets/mnist/"
    
    if args.data_path[-1] != "/":
        trainset_path = args.data_path + '/trainset.pt'
        testset_path = args.data_path + "/testset.pt"
        trainset = torch.load(trainset_path) 
        testset = torch.load(testset_path)
    else:
        trainset_path = args.data_path + 'trainset.pt'
        testset_path = args.data_path + "testset.pt"

        trainset = torch.load(trainset_path)                                                                                               
        testset = torch.load(testset_path)

    
    classes = trainset.classes
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:

        trainset, testset = create_traintestset(ood_class, trainset,testset)

        for i in range(0, len(ood_class)):
            #trainset_ood = datasets.MNIST(root = '../data', train = True, download = True, transform = transform)
            #testset_ood  = datasets.MNIST(root = '../data', train = False, download = True, transform = transform)
            trainset_ood = torch.load(trainset_path)
            testset_ood = torch.load(testset_path)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)


    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    

    if args.leave_one_out:
        train_indices = train_set.indices
        val_indices = val_set.indices
    
        train_set_dataset = GeneralImgDataset(np.array(train_set.dataset.data)[train_indices].tolist(), np.array(train_set.dataset.targets)[train_indices], transform)
        val_set_dataset = GeneralImgDataset(np.array(val_set.dataset.data)[val_indices].tolist(), np.array(val_set.dataset.targets)[val_indices], transform)
        
        np.random.seed(args.random_seed)
        classes_integer_list = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
        np.random.shuffle(classes_integer_list) # shuffles classes to choose which are iid and ood
        classes_idx_ID = classes_integer_list[0:args.ID_tasks]
        trainset_one_out, testset_one_out, valset_one_out = [], [], []

        
        for id_class in classes_idx_ID:
           
            trainset_copy = copy(train_set_dataset)
            testset_copy = copy(testset)
            valset_copy = copy(val_set_dataset)

            class_trainset, class_testset, class_valset = create_OOD_dataset(id_class, trainset_copy, testset_copy, valset_copy)# although this says ood, the functionality that we desire is achieved by
            trainset_one_out.append(class_trainset)
            testset_one_out.append(class_testset)
            valset_one_out.append(class_valset)
    
   
    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2, drop_last = True)
  
    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)

    if args.leave_one_out:
        return trainloader, valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset


def data_loader_Audio_MNIST(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):
    ###Note there may be issues performing experiments with this dataset. We decided to focus on more traditional datasets and stopped iterating on this further
    src_path, meta_data_path = "../AudioMNIST/data", "../AudioMNIST/AudioMNIST_meta.txt"
    Data, targets, classes = preprocess_data(src_path, meta_data_path)
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


    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)

    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, drop_last = True)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2, drop_last = True)
    
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset


def create_subdataset(classes, trainset, testset):



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


   
