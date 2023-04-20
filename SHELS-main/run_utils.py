import torch
import torch.nn as nn
import numpy as np
import time
import torch.utils.data as data
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from custom_dataloader import GeneralDataset

from train import Trainer
import ood_utils
import layer_utils
import pdb
import pickle
import os


def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

def run(trainer,args, trainloader,valloader, testloader, classes_idx_OOD, classes,classes_idx_ID, idx, leave_one_out = False):

    if args.load_checkpoint and not leave_one_out:
        if len(classes_idx_OOD) == 1:
            checkpoint = torch.load(args.save_path + f'/{args.random_seed}' + '/model_{}.pt'.format(classes[classes_idx_OOD[0]]))
        else:
            checkpoint = torch.load(args.save_path + f'/{args.random_seed}' + '/model_{}.pt'.format(idx))
        trainer.model.load_state_dict(checkpoint)

    else:
        trainer.model.apply(weights_init_)


    if args.train:
        prev_loss = 1e30
        prev_loss_g = 1e30
        for z in range(0,1):
            print("TRAINING")
            if args.dataset1 == 'cifar10_old':
                alexnet = models.vgg16(pretrained=True)
                output_dim = args.ID_tasks
                alexnet.classifier[3] = nn.Linear(4096,1024)
                alexnet.classifier[6] = nn.Linear(1024, output_dim)
                alexnet_dict = alexnet.state_dict()
                model_dict = trainer.model.state_dict()
                pretrained_dict = {k : v for k,v in alexnet_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                trainer.model.load_state_dict(model_dict)
                #torch.nn.init.xavier_normal_(trainer.model.classifier[6].weight, gain =1)
                #torch.nn.init.xavier_normal_(trainer.model.classifier[3].weight, gain = 1)
                scheduler = optim.lr_scheduler.StepLR(trainer.optimizer, step_size = 4, gamma = 0.5)

            for epoch in range(args.epochs):
                start_time = time.time()
                print('working here')              
                train_loss, train_acc = trainer.optimize(trainloader,classes_idx_OOD, classes_idx_ID)
                #print('got past optimize')
                end_time = time.time()
                print(f'\t EPOCH: {epoch+1:.0f} | time elapsed: {end_time - start_time:.3f}')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
                loss, acc, act_avg_test,_,_ = trainer.evaluate(valloader,classes_idx_OOD,classes_idx_ID, extract_act = False)
                print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc*100:.2f}%')
                if args.dataset1 == 'cifar10_old':
                    scheduler.step()
                if loss < prev_loss:
                    prev_loss = loss
                    train_loss, train_acc = trainer.optimize(valloader,classes_idx_OOD, classes_idx_ID)
                    if len(classes_idx_OOD) == 1:
                        if not args.leave_one_out:
                            torch.save(trainer.model.state_dict(), args.save_path + f'/{args.random_seed}' +'/model_{}.pt'.format(classes[classes_idx_OOD[0]]))
                        else:
                            torch.save(trainer.model.state_dict(), args.save_path +  '/leave_one_out/{}/model_{}.pt'.format(args.random_seed, classes[classes_idx_OOD[0]]))
                    else:
                        torch.save(trainer.model.state_dict(), args.save_path + f'/{args.random_seed}' +'/model_{}.pt'.format(idx))

    else:
        
        print("TESTING")


def do_ood_eval(trainloader,valloader, testloader,testset, ood_trainset_list,ood_testset_list, trainer, classes, classes_idx_OOD,args,classes_idx_ID, eta, save_k = 0):

    ## the current implementation only works with a batch size of 1 which is probably not ideal

    ## 1. use traindata to compute thresholds for each class

    _, _, _,activations_list_train, _ = trainer.evaluate(trainloader, classes_idx_OOD, classes_idx_ID, extract_act = True)

    if len(classes_idx_OOD) == 1:
        np.savez(args.save_path + '/activations' + f'/{args.random_seed}' + '/act_full_train_{}.npz'.format(classes[classes_idx_OOD[0]]),**activations_list_train)
    else:
        np.savez(args.save_path + '/activations' f'/{args.random_seed}' + '/act_full_train_{}.npz'.format(save_k),**activations_list_train)

    if len(classes_idx_OOD) == 1:
        activations_list_train = dict(np.load(args.save_path + '/activations' f'/{args.random_seed}' + '/act_full_train_{}.npz'.format(classes[classes_idx_OOD[0]]), allow_pickle = True))
    else:
        activations_list_train = dict(np.load(args.save_path + '/activations' f'/{args.random_seed}' + '/act_full_train_{}.npz'.format(save_k),allow_pickle = True))
    
    new_activations_list_train = activations_list_train.copy()
   ## compute class wise thresholds
    if args.leave_one_out:
        thresholds_one_out, _, _ = ood_utils.compute_per_class_thresholds(activations_list_train, trainer, classes,classes_idx_OOD, args.ID_tasks, args.baseline_ood, eta)
        print(thresholds_one_out, 'leave one out thresholds')

    thresholds, _, _ = ood_utils.compute_per_class_thresholds(activations_list_train, trainer, classes,classes_idx_OOD, args.ID_tasks, args.baseline_ood, 1)
   # cheat_thresholds = ood_utils.cheat_thresholds(activations_list_train, trainer, classes, classes_idx_OOD, args.ID_tasks, args.baseline_ood)
    print(thresholds,'original thresholds')


    if not args.cont_learner:
            ## 2. with these thresholds evuate the test set accuracy
        test_acc = ood_utils.compute_test_Acc(testloader, thresholds, trainer, classes, classes_idx_OOD, args.save_path, args.ID_tasks,classes_idx_ID, args.baseline_ood, save_k)
        # test_loss, test_accuracy = evaluate_with_thresh(testloader, thresholds, trainer.model)
        # ## 3. evaluate ood detection data ## with varying data amounts
        ood_acc = ood_utils.compute_ood_Acc(ood_trainset_list, thresholds, trainer, classes, classes_idx_OOD, args.save_path, args.ID_tasks, args.multiple_dataset,classes_idx_ID,args.baseline_ood, args, save_k)


        return test_acc, ood_acc

    else:
        avg_act_all_layers, layer_indices = layer_utils.return_all_layer_activations(trainer, testloader)

        test_acc_full = []
        ood_acc_full = []
        
        id_classes = args.ID_tasks
        lr_mult = 2 # set to 2 for fmnist
        for k in range(0, len(ood_trainset_list),1):


            curr_ood_data = ood_trainset_list[k]


            # loss, acc, activations,activations_list_test, labels_list = trainer.evaluate(testloader, [classes_idx_OOD[k],0],classes_idx_ID, extract_act = False)
            #
            # print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc*100:.2f}%')

            ## incremental test data ood detetion
            if args.full_pipeline:
                testloader = data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 2)
                for p in range(0,3):
                    percent = 0.1*p
                    test_ood_acc = ood_utils.compute_incremental_test_acc(testloader, thresholds, trainer, classes, classes_idx_OOD, args.save_path,  len(classes_idx_ID),classes_idx_ID,percent,save_k)
                    print("total in OOD accuracy for sample test data", test_ood_acc)
                sample_data_len = int(0.1*len(curr_ood_data))
                rem_data_len = len(curr_ood_data) - sample_data_len
                sample_ood_data, remaining_ood_data = data.random_split(curr_ood_data,[sample_data_len, rem_data_len])
                ood_acc = ood_utils.compute_ood_Acc([sample_ood_data], thresholds, trainer, classes,[classes_idx_OOD[k]], args.save_path, len(classes_idx_ID), args.multiple_dataset, classes_idx_ID, args.baseline_ood, args, 0)
                print("ood_acc  :", ood_acc)

            else:
                ood_acc = 100

            if ood_acc > 30:
            # pdb.set_trace()
                batch_size = 32
                classes_idx_ID = np.array(np.insert(classes_idx_ID,len(classes_idx_ID),classes_idx_OOD[k]))
                ood_trainloader = data.DataLoader(curr_ood_data, batch_size = 1, shuffle = False, num_workers = 2)

                _, _, _, activations_list_new_class_cheat = trainer.ood_evaluate(ood_trainloader)

                # This has to be changed to just looking at the eta that best seperated the ood class from the id classes
                cheat_threshold, best_accuracy, best_id_accuracy, best_ood_accuracy, accuracies, etas, best_eta, id_accuracies, ood_accuracies, id_accuracy_original, ood_accuracy_original, total_accuracy_original = ood_utils.cheat_thresholds(new_activations_list_train, trainer, classes,classes_idx_OOD, id_classes + k, args.baseline_ood, activations_list_new_class_cheat, thresholds)

                length_curr_ood_data = len(curr_ood_data)
                print(length_curr_ood_data, 'length of ood data')
                length_testloader = len(testloader)
                print('test loader length. Batch size of 1 so should just be correct size', length_testloader) 
                testloader_weight = length_testloader/(length_testloader + length_curr_ood_data)
                ood_weight = length_curr_ood_data/(length_testloader + length_curr_ood_data)
                if args.leave_one_out:
                    ood_acc_one_out = ood_utils.compute_ood_Acc([curr_ood_data], thresholds_one_out, trainer, classes, [classes_idx_OOD[k]], args.save_path, id_classes + k, args.multiple_dataset, classes_idx_ID, args.baseline_ood , args, save_k)
                    test_acc_one_out = ood_utils.compute_test_Acc(testloader, thresholds_one_out, trainer, classes, classes_idx_OOD, args.save_path, id_classes + k, classes_idx_ID, args.baseline_ood, args, save_k)
                    total_acc_one_out = ood_acc_one_out * ood_weight + test_acc_one_out * testloader_weight
                    
                    print("leave one out ood acc on ood train data: ", ood_acc_one_out)
                    print("leave one out acc on id test data: ", test_acc_one_out)
                    print('total one out accuracy: ', total_acc_one_out)
                    print('\n')
                                    

                ood_acc_regular = ood_utils.compute_ood_Acc([curr_ood_data], thresholds, trainer, classes, [classes_idx_OOD[k]], args.save_path, id_classes + k, args.multiple_dataset, classes_idx_ID, args.baseline_ood , args, save_k)
                test_acc_regular = ood_utils.compute_test_Acc(testloader, thresholds, trainer, classes, classes_idx_OOD, args.save_path, id_classes + k , classes_idx_ID, args.baseline_ood, args, save_k)
                total_acc_regular = ood_acc_regular * ood_weight + test_acc_regular * testloader_weight
                print("regular ood acc on ood train data: ", ood_acc_regular)
                print("regular acc on id train data: ", test_acc_regular)
                print("regular total accuracy: ", total_acc_regular)
                print('\n')

                ood_acc_cheat = ood_utils.compute_ood_Acc([curr_ood_data], cheat_threshold, trainer, classes, [classes_idx_OOD[k]], args.save_path, id_classes + k, args.multiple_dataset, classes_idx_ID, args.baseline_ood , args, save_k)
                test_acc_cheat = ood_utils.compute_test_Acc(testloader, cheat_threshold, trainer, classes, classes_idx_OOD, args.save_path, id_classes + k, classes_idx_ID, args.baseline_ood, args, save_k)
                total_acc_cheat = ood_acc_cheat * ood_weight + test_acc_cheat * testloader_weight
                print("cheat ood acc on ood train data: ", ood_acc_cheat)
                print("cheat acc on id train data: ", test_acc_cheat)
                print('cheat total accuracy: ', total_acc_cheat)
                print('\n')
                

                

                #
                testset = [testset, ood_testset_list[k]]
                testset = data.ConcatDataset(testset)

                testloader = data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 2)
                testloader_cont = data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)

                
                # want to get accuracy here, and also with 1 std dev thresholds, compare with the cheat
                
                

                
                print(best_accuracy, 'optimal cheating accuracy on training data')
                print(best_id_accuracy, 'optimal cheating id accuracy on training data')
                print(best_ood_accuracy, 'optimal cheating ood accuracy on training data')
                state_dict = {'dataset': args.dataset1, 'best_eta': best_eta, 'best_thresholds': cheat_threshold, 'accuracies': accuracies, 'etas': etas, 'best_accuracy': best_accuracy, 'best_id_accuracy': best_id_accuracy, 'best_ood_accuracy': best_ood_accuracy, 'id_accuracies': id_accuracies, 'ood_accuracies': ood_accuracies, 'id_accuracy_original': id_accuracy_original, 'ood_accuracy_original': ood_accuracy_original, 'total_accuracy_original': total_accuracy_original}
                if args.leave_one_out:
                    state_dict_one_out = {'dataset': args.dataset1, 'one_out_eta': eta, 'cheat_eta': best_eta, 'one_out_thresholds': thresholds_one_out, 'test_acc_one_out': test_acc_one_out, 'ood_acc_one_out': ood_acc_one_out, 'test_acc_regular': test_acc_regular, 'ood_acc_regular': ood_acc_regular, 'ood_acc_cheat': ood_acc_cheat, 'test_acc_cheat': test_acc_cheat, 'total_acc_one_out': total_acc_one_out, 'total_acc_regular':total_acc_regular, 'total_acc_cheat':total_acc_cheat }

                current_act_avg_layers, test_acc_list,ood_acc_list,trainer = ood_utils.continual_learner(trainer, curr_ood_data, ood_testset_list[k], testloader_cont, avg_act_all_layers, layer_indices, batch_size,classes_idx_OOD[k],classes_idx_ID,lr_mult)  
                test_acc_full.append(test_acc_list)
                ood_acc_full.append(ood_acc_list)

                # get the new eta, try with weighted avgeraging based on number of classes that are id and ood.
                _, _, _, activations_list_new_class = trainer.ood_evaluate(ood_trainloader)
                if args.leave_one_out:
                    id_class_weight = (id_classes + k)/(id_classes + k + 1)
                    ood_class_weight = 1/ (id_classes + k + 1)
                    eta = eta * id_class_weight + best_eta * ood_class_weight
                    thresholds_one_out = ood_utils.update_thresholds(thresholds_one_out, activations_list_new_class,trainer,classes_idx_ID, eta)
                
                thresholds = ood_utils.update_thresholds(thresholds, activations_list_new_class,trainer,classes_idx_ID)

               
                
                
                cwd = os.getcwd()
                filename =  '/results/' +  args.dataset1 +  "/" + str(args.random_seed)
                print(filename) 
                dir = cwd + filename
                #dir = os.path.join(cwd, filename)
                os.makedirs(dir, exist_ok = True)
                with open(  os.path.join(dir,  f'number_of_id_classes:{id_classes + k}.pickle'), 'wb') as f:
                    pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                if args.leave_one_out:
                    one_out_filename =  '/results/' +  args.dataset1 +  "/leave_one_out/" + str(args.random_seed)
                    one_out_dir = cwd+ one_out_filename
                    os.makedirs(one_out_dir, exist_ok = True)
                    with open(  os.path.join(one_out_dir,  f'number_of_id_classes:{id_classes + k}.pickle'), 'wb') as f:
                        pickle.dump(state_dict_one_out, f, protocol=pickle.HIGHEST_PROTOCOL)


                   
                # fig, axs = plt.subplots(2, 2)
                # axs[0,0].plot(etas, accuracies)
                # axs[0,0].set_title('total accuracy curve')
                # #axs[0,0].xlabel('eta')
                # #axs[0,0].ylabel('accuracy')
                

                # axs[0,1].plot(etas, id_accuracies)
                # axs[0,1].set_title('id accuracy curve')
                # #axs[0,1].xlabel('eta')
                # #axs[0,1].ylabel('accuracy')
                

                # axs[1,1].plot(etas, ood_accuracies)
                # axs[1,1].set_title('ood accuracy w respect to eta')
                # #axs[1,1].xlabel('eta')
                # #axs[1,1].ylabel('accuracy')
                # for ax in axs.flat[:-1]:
                #     ax.set(xlabel='eta', ylabel='accuracy')
                # #plt.savefig(f'accuracy_curves{k}.png')
                # print(cheat_threshold)
                new_activations_list_train[str(id_classes +k)] = activations_list_new_class
                new_avg = current_act_avg_layers
                #
                for i in range(len(new_avg)-1):
                   # new_avg[i] =  (current_act_avg_layers[i] + avg_act_all_layers[i]*(1/(lr_mult*10)))
                   new_avg[i] =  (current_act_avg_layers[i] + avg_act_all_layers[i])
                avg_act_all_layers = new_avg
                lr_mult = 2 # 2 for fmnist

 
        # np.savez('cont_learner_with_another_nosp.npz', test_acc=test_acc_full, ood_acc= ood_acc_full)
        if args.full_pipeline:
            testloader = data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 2)
            for p in range(0,3):
                percent = 0.1*p
                test_ood_acc = ood_utils.compute_incremental_test_acc(testloader, thresholds, trainer, classes,classes_idx_OOD, args.save_path, len(classes_idx_ID),classes_idx_ID, percent,save_k)
                print("total in OOD accuracy for sample test data", test_ood_acc)
        print("Exiting Continual Learner experiment")
        exit()  ## the continula learner exits after running 1 experiment


def leave_one_out(trainloader, curr_ood_data, trainer, classes, classes_idx_OOD,args,classes_idx_ID, save_k = 0):
    _, _, _,activations_list_train, _ = trainer.evaluate(trainloader, classes_idx_OOD, classes_idx_ID, extract_act = True)

    assert(len(classes_idx_OOD) == 1)
    print(classes_idx_OOD)
    print(classes)

    
    #thresholds, _, _ = ood_utils.compute_per_class_thresholds(activations_list_train, trainer, classes,classes_idx_OOD, args.ID_tasks - 1, args.baseline_ood, 1)
    thresholds = [1,1,1,1] # dummy value

    np.savez(args.save_path + '/activations' +  '/leave_one_out/{}/act_full_train_{}.npz'.format(args.random_seed, classes[classes_idx_OOD[0]]),**activations_list_train)

    activations_list_train = dict(np.load(args.save_path + '/activations' + '/leave_one_out/{}/act_full_train_{}.npz'.format(args.random_seed, classes[classes_idx_OOD[0]]), allow_pickle = True))
  
    new_activations_list_train = activations_list_train.copy()
    
    curr_ood_data_copy = GeneralDataset(torch.stack((curr_ood_data.data)).unsqueeze(1).float(), curr_ood_data.targets)    
    id_classes = args.ID_tasks - 1

    #assert(len(ood_trainset_list) == 1)

    ood_trainloader = data.DataLoader(curr_ood_data_copy, batch_size = 1, shuffle = False, num_workers = 2)
    _, _, _, activations_list_new_class = trainer.ood_evaluate(ood_trainloader)

    _, _, _, _, _, _, best_eta, _, _, _, _, _ = ood_utils.cheat_thresholds(new_activations_list_train, trainer, classes,classes_idx_OOD, id_classes, args.baseline_ood, activations_list_new_class, thresholds)
   
    return best_eta
