import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from train import Trainer
import random
from custom_dataloader import GeneralDataset
from copy import deepcopy

from arguments import get_args, print_args
import run_utils, data_utils
import json
import argparse
import os

if __name__ == "__main__":
    load_args = False


    if load_args:
        print("Loading arguments")
        parser =argparse.ArgumentParser()
        args = parser.parse_args()
        args = get_args()
        with open(args.save_path+'/args.txt', 'r') as f:
            args.__dict__ = json.load(f)
        args.batch_size = 1
        args.train = False
        args.load_checkpoint = True
        print_args(args)
        ## this is to make sure you set the arguement for OOD detection

    else:
        args = get_args()
        print_args(args)

        if args.train:
            with open(args.save_path+'/args.txt', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    if not os.path.exists(args.save_path+'/activations'):
        os.makedirs(args.save_path+'/activations')

   # data_path = r"C:\Users\aoeji\Downloads\LIS work\SHELS-main" ## this might vary for you
    data_path = '/home/gridsan/aejilemele/LIS/SHELS-main'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.empty_cache()

    #torch.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


    if args.experiment == "ood_detect":
        print(args.multiple_dataset)
        if not args.multiple_dataset:

            if args.single_ood_class:
                ## do round robin ood ood_detect
                output_dim = args.total_tasks - 1
                test_acc_list = []
                ood_acc_list = []
                for ood_class_idx in range(0, args.total_tasks):

                    trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_utils.single_dataset_loader(args.dataset1, data_path, args.batch_size, [ood_class_idx])
                    trainer = Trainer(output_dim, device, args)
                    print("OOD Class :", classes[ood_class_idx])
                    print('classes', classes)
                    run_utils.run(trainer,args,trainloader, valloader, testloader,[ood_class_idx], classes,[],0)
                    # TODO ood detection - from distribution_evaluation - DONE
                    if args.load_checkpoint:
                        test_acc, ood_acc = run_utils.do_ood_eval(trainloader, testloader,testset, ood_trainset_list,ood_testset_list, trainer, classes, [ood_class_idx], args,[])
                        test_acc_list.append(test_acc)
                        ood_acc_list.append(ood_acc)
                    print("Mean TEST acc :", np.mean(test_acc_list))
                    print("Mean OOD acc:", np.mean(ood_acc_list))
            else: #this is where we usually end up in

                print("More than one OOD class in the same dataset, this is set-up for continual learning")
                #ood_class_idx = np.arange(args.ID_tasks, args.total_tasks)
                output_dim = args.ID_tasks
                if args.load_list:
                    # class_list = []
                    # list_idx = np.arange(0,args.total_tasks)
                    # class_list.append(list_idx)
                    class_list = np.load('class_list1.npz', allow_pickle = True)['class_list']
                else:
                    list_idx = np.arange(0,args.total_tasks)
                    class_list = []

                ood_acc_list = []
                test_acc_list = []
                for exp_no in range(0,10):
                    print("EXP :", exp_no)
                    if args.load_list:
                        list_idx = class_list[exp_no]
                    else:
                        np.random.shuffle(list_idx) # shuffles classes to choose which are iid and ood
                        class_list.append(list_idx.copy())

                    classes_idx_OOD = list_idx[args.ID_tasks : args.total_tasks]

                    classes_idx_ID = list_idx[0:args.ID_tasks]


                    if args.leave_one_out:
                        #args.batch_size = 32 # set to 32 for training
                        trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset, trainset_one_out, testset_one_out, valset_one_out = data_utils.single_dataset_loader(args.dataset1, data_path, args.batch_size, classes_idx_OOD)
                    else:
                        trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_utils.single_dataset_loader(args.dataset1, data_path, args.batch_size, classes_idx_OOD)
                    
                    #trainer = Trainer(output_dim, device, args)

                    print(classes)
                    print("OOD Class :", np.array(classes)[classes_idx_OOD])
                    print("IND Class :", np.array(classes)[classes_idx_ID])
                    
                    if args.leave_one_out == True:
                        #original_batch_size = args.batch_size
                        #args.batch_size = 32 # set to 32 for training
                        best_etas = []

                        for i in range(args.ID_tasks):
                            
                            leave_one_out_trainer = Trainer(output_dim -  1, device, args)
                            leave_out_class = i

                            classes_idx_ID_one_out = list(classes_idx_ID[0:leave_out_class]) + list(classes_idx_ID[leave_out_class + 1:])
                            # trainer,args, trainloader,valloader, testloader, classes_idx_OOD, classes,classes_idx_ID, idx
                            
                            train_set = GeneralDataset([], [])
                            test_set = GeneralDataset([], [])
                            val_set = GeneralDataset([], [])
                            for j in range(args.ID_tasks):
                                if j != i:
                                    #print('type', type(train_set.data))
                                    #print(len(train_set.data))
                                    train_set.data += trainset_one_out[j].data
                                    train_set.targets += trainset_one_out[j].targets
                                    test_set.data += testset_one_out[j].data
                                    test_set.targets += testset_one_out[j].targets
                                    val_set.data += valset_one_out[j].data
                                    val_set.targets += valset_one_out[j].targets
                                    #print(len(train_set), 'length of trainset, should be growing')
                            #print(args.batch_size, 'batch size')
                            train_set.data = torch.stack((train_set.data)).unsqueeze(1).float()
                            
                            #print('train_set data shape', train_set.data.shape)
                            test_set.data  = torch.stack((test_set.data)).unsqueeze(1).float()
                            val_set.data = torch.stack((val_set.data)).unsqueeze(1).float()
                            trainloader_one_out = data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = 2, drop_last = True)
                            testloader_one_out  = data.DataLoader(test_set, batch_size = args.batch_size, shuffle = False, num_workers = 2, drop_last = True)
                            valloader_one_out   = data.DataLoader(val_set, batch_size = args.batch_size, shuffle = False, num_workers = 2, drop_last = True)
                            




                            trainloader_one_out_cheat = data.DataLoader(train_set, batch_size = 1, shuffle = True, num_workers = 2, drop_last = True) 
                            #testloader_one_out_cheat  = data.DataLoader(test_set, batch_size = 1, shuffle = False, num_worke$     
                            #valloader_one_out_cheat   = data.DataLoader(val_set, batch_size = 1, shuffle = False, num_worker$
                            print('about to enter cross validation training')
                            run_utils.run(leave_one_out_trainer, args, trainloader_one_out, valloader_one_out, testloader_one_out, [leave_out_class], classes, classes_idx_ID_one_out, exp_no, True )
                            print('finished training')
                            best_eta = run_utils.leave_one_out(trainloader_one_out_cheat, trainset_one_out[i], leave_one_out_trainer, classes, [i], args, classes_idx_ID_one_out, exp_no)
                            best_etas.append(best_eta)

                            


                        print('best etas for leave out classes: ', best_etas)
                        final_eta = np.mean(best_etas)
                        print('final eta', final_eta)
                        eta_save_path = args.save_path + '/leave_one_out_eta' + '/{}/eta.npz'.format(args.random_seed)
                        print(eta_save_path)
                        np.savez(eta_save_path,final_eta = final_eta, best_etas = best_etas)
                    # also save the classes that were used for this random seed
                        args.train = False

                    #if not args.leave_one_out:
                    trainer = Trainer(output_dim, device, args)
                    run_utils.run(trainer, args, trainloader, valloader, testloader,classes_idx_OOD, classes,classes_idx_ID,exp_no)
                    
                    if args.load_checkpoint:
                        if args.leave_one_out:
                            args.batch_size = 1
                            results = np.load(args.save_path + '/leave_one_out_eta' + '/{}/eta.npz'.format(args.random_seed))
                            final_eta = results['final_eta']
                            best_etas = results['best_etas']
                            eta = final_eta
                        else:
                            eta = 1

                        test_acc, ood_acc = run_utils.do_ood_eval(trainloader, valloader,testloader, testset, ood_trainset_list,ood_testset_list, trainer, classes,classes_idx_OOD,args,classes_idx_ID, eta , exp_no)
                        
                        print("TEST_Acc:", test_acc)
                        print("OOD_acc:", ood_acc)
                        ood_acc_list.append(ood_acc)
                        test_acc_list.append(test_acc)
                        if exp_no == 9:
                            combined_acc_list = np.add(ood_acc_list, test_acc_list)/2
                            print("MEAN OOD ACC :",np.mean(ood_acc_list))
                            print("STD OOD ACCC:", np.std(ood_acc_list))
                            print("MEAN TEST ACC :", np.mean(test_acc_list))
                            print("STD TEST ACC:", np.std(test_acc_list))
                            print("MEAN COMBINED ACC :", np.mean(combined_acc_list))
                            print("STD COMBINED ACC :", np.std(combined_acc_list))

                if not args.load_list:
                    np.savez('class_list_GTSRB.npz', class_list = class_list)

        else:
            print("OOD detection across datasets")
            output_dim = args.total_tasks ## this if for dataet1 -ID dataset
            trainloader,valloader, testloader,classes,testset, testset_ood,classes_ood = data_utils.mutliple_dataset_loader( data_path, args.dataset1, args.dataset2, args.batch_size)
            print("In dist classes", classes)
            print("OOD dist classes", classes_ood)
            classes_idx_OOD = np.arange(0,len(classes_ood))
            classes_idx_ID = np.arange(0,len(classes))
            trainer = Trainer(output_dim,device, args)
            run_utils.run(trainer, args, trainloader, valloader, testloader,classes_idx_OOD, classes,[],0)
            if args.load_checkpoint:
                test_acc, ood_acc = run_utils.do_ood_eval(trainloader,valloader, testloader,testset, [testset_ood],[testset_ood], trainer, classes, classes_idx_OOD, args, classes_idx_ID,0)
                print("combined acc:", (test_acc+ood_acc)/2)


