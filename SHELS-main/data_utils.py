import numpy as np
import data_loader

import pdb

from arguments import get_args, print_args

def single_dataset_loader(dataset, data_path, batch_size, ood_class_idx):
    args = get_args()
    ## load dataset1
    print("dataset", dataset)
    dataset = dataset.replace(" ", "")
    if dataset == "cifar10":
        if args.leave_one_out:
            trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out = data_loader.data_loader_CIFAR_10(data_path, batch_size, ood_class_idx)
        else:
            trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_loader.data_loader_CIFAR_10(data_path, batch_size, ood_class_idx)

    elif dataset == 'tinyimagenet':
        if args.leave_one_out:
            trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out = data_loader.data_loader_tinyimagenet(data_path, batch_size, ood_class_idx)
        else:
            trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_loader.data_loader_tinyimagenet(data_path, batch_size, ood_class_idx)

    elif dataset == "fmnist":
        if args.leave_one_out:
            trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out = data_loader.data_loader_FashionMNIST(data_path, batch_size, ood_class_idx)
        else:
            trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_loader.data_loader_FashionMNIST(data_path, batch_size, ood_class_idx)
        classes[0] = 'top'
    elif dataset == "mnist":
        if args.leave_one_out:
             trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out = data_loader.data_loader_MNIST(data_path, args.batch_size, ood_class_idx)
        else:
            trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_loader.data_loader_MNIST(data_path, batch_size, ood_class_idx)
    elif dataset == 'audiomnist':
        trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_loader.data_loader_Audio_MNIST(data_path, batch_size, ood_class_idx)
    elif dataset =="svhn":
        trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_loader.data_loader_SVHN(data_path, batch_size, ood_class_idx)
    elif dataset == "cubs":
        trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_loader.data_loader_CUBS(data_path, batch_size, ood_class_idx)
    elif dataset == "gtsrb":
        trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_loader.data_loader_GTSRB(data_path, batch_size, ood_class_idx)
 
 
    else:
        print("Invalid dataset ")
        exit()
    if args.leave_one_out:
        return  trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset

def mutliple_dataset_loader(data_path, dataset1, dataset2, batch_size):
    ood_class_idx = []

        ## load dataset1
    if dataset1 == "cifar10":
        trainloader,valloader, testloader, _, _, classes,testset = data_loader.data_loader_CIFAR_10(data_path, batch_size, ood_class_idx)

    elif dataset1 == "tinyimagenet":
        trainloader,valloader, testloader, _, _, classes,testset = data_loader.data_loader_tinyimagenet(data_path, batch_size, ood_class_idx)
    elif dataset1 == "fmnist":
        trainloader,valloader, testloader, _,_, classes,testset = data_loader.data_loader_FashionMNIST(data_path, batch_size, ood_class_idx)
        classes[0] = 'top'
    elif dataset1 == "mnist":
        trainloader,valloader, testloader, _, _, classes,testset  = data_loader.data_loader_MNIST(data_path, batch_size, ood_class_idx)
    elif dataset1 == 'audiomnist':
        trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_loader.data_loader_MNIST(data_path, batch_size, ood_class_idx)
    elif dataset1 == "cubs":
        trainloader,valloader, testloader, _, _, classes,testset = data_loader.data_loader_CUBS(data_path, batch_size, ood_class_idx)
    elif dataset1 == "gtsrb":
        trainloader,valloader, testloader, _, _, classes,testset = data_loader.data_loader_GTSRB(data_path, batch_size, ood_class_idx)
    elif dataset1 == "svhn":
        trainloader,valloader, testloader, _, _, classes,testset = data_loader.data_loader_SVHN(data_path, batch_size, ood_class_idx)
 

 
    else:
        print("Invalid dataset 1")
        exit()

    if dataset2 == "cifar10":
        trainloader_ood,valloader_ood, testloader_ood, _, _, classes_ood,testset_ood = data_loader.data_loader_CIFAR_10(data_path, batch_size, ood_class_idx)
    elif dataset2 == "tinyimagenet":
        trainloader_ood,valloader_ood, testloader_ood, _, _, classes_ood,testset_ood = data_loader.data_loader_tinyimagenet(data_path, batch_size, ood_class_idx)
    elif dataset2 == "fmnist":
        trainloader_ood,valloader_ood, testloader_ood, _, _, classes_ood,testset_ood = data_loader.data_loader_FashionMNIST(data_path, batch_size, ood_class_idx)
        classes_ood[0] = 'top'
    elif dataset2 == "mnist":
        trainloader_ood,valloader_ood, testloader_ood, _,_, classes_ood,testset_ood = data_loader.data_loader_MNIST(data_path, batch_size, ood_class_idx)
    elif dataset2 == 'audiomnist':
        trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_loader.data_loader_MNIST(data_path, batch_size, ood_class_idx)
    elif dataset2 == "svhn":
        trainloader_ood,valloader_ood, testloader_ood, _,_, classes_ood,testset_ood = data_loader.data_loader_SVHN(data_path, batch_size, ood_class_idx)
    elif dataset2 == "gtsrb":
        trainloader_ood,valloader_ood, testloader_ood, _,_, classes_ood,testset_ood = data_loader.data_loader_GTSRB(data_path, batch_size, ood_class_idx)
 

    else:
        print("Invalid dataset ")
        exit()

    return trainloader,valloader, testloader,classes,testset, testset_ood,classes_ood
