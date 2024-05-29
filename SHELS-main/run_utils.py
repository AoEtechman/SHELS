import torch
import torch.nn as nn
import numpy as np
import time
import torch.utils.data as data
import torch.optim as optim
from torchvision import models
from custom_dataloader import GeneralImgDataset


import ood_utils
import layer_utils
import pickle
import os



# function for changing the batch size of a dataloader
def change_batch_size(dataloader, new_batch_size):
    
    dataset = dataloader.dataset
    sampler = dataloader.sampler
    collate_fn = dataloader.collate_fn
    num_workers = dataloader.num_workers
    pin_memory = dataloader.pin_memory
    pin_memory_device = dataloader.pin_memory_device
    drop_last = dataloader.drop_last
    timeout = dataloader.timeout
    worker_init_fn = dataloader.worker_init_fn
    multiprocessing_context = dataloader.multiprocessing_context
    generator = dataloader.generator
    prefetch_factor = dataloader.prefetch_factor
    
    # Create a new DataLoader object with the desired batch size and other attributes
    new_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=new_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device = pin_memory_device,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
        prefetch_factor = prefetch_factor
    )
    
    return new_dataloader

def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias != None:
            nn.init.constant_(m.bias, 0)

def run(trainer,args, trainloader,valloader, testloader, classes_idx_OOD, classes,classes_idx_ID, idx, leave_one_out = False):
    if leave_one_out:
        if args.load_checkpoint_one_out:
            if len(classes_idx_OOD) == 1:
                checkpoint = torch.load(args.save_path +  '/leave_one_out/{}/model_{}.pt'.format(args.random_seed, classes[classes_idx_OOD[0]]))
            else:
                checkpoint = torch.load(args.save_path +  '/leave_one_out/{}/model_{}.pt'.format(args.random_seed, idx))
            trainer.model.load_state_dict(checkpoint)
        else:
            trainer.model.apply(weights_init_)

    else:
        if args.load_checkpoint:
            if len(classes_idx_OOD) == 1:
                checkpoint = torch.load(args.save_path + f'/{args.random_seed}' + '/model_{}.pt'.format(classes[classes_idx_OOD[0]]))
            else:
                checkpoint = torch.load(args.save_path + f'/{args.random_seed}' + '/model_{}.pt'.format(idx))
            trainer.model.load_state_dict(checkpoint)
        else:
            trainer.model.apply(weights_init_)
            

    if leave_one_out:
        if args.leave_one_out_train:
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


    else:

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

    ## the current implementation only works with a batch size of 1 

   
    if trainloader.batch_size > 1:
        trainloader = change_batch_size(trainloader, 1)
    if testloader.batch_size > 1:
        testloader = change_batch_size(testloader, 1)

    args.batch_size = 1

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
   
    ## 1. use traindata to compute thresholds for each class

    # thresholds computed with leave one out optimization
    if args.leave_one_out:
        thresholds_one_out, _, _ = ood_utils.compute_per_class_thresholds(new_activations_list_train, trainer, classes,classes_idx_OOD, args.ID_tasks, args.baseline_ood, eta)
        print(thresholds_one_out, 'leave one out thresholds')

    # baseline thresholds
    thresholds, _, _ = ood_utils.compute_per_class_thresholds(new_activations_list_train, trainer, classes,classes_idx_OOD, args.ID_tasks, args.baseline_ood, 1)
    


    if not args.cont_learner:
        
        ## 2. with these thresholds evuate the test set accuracy
        test_acc = ood_utils.compute_test_Acc(testloader, thresholds, trainer, classes, classes_idx_OOD, args.save_path, args.ID_tasks,classes_idx_ID, args.baseline_ood, save_k)
        # test_loss, test_accuracy = evaluate_with_thresh(testloader, thresholds, trainer.model)
        # ## 3. evaluate ood detection data ## with varying data amounts
        ood_acc = ood_utils.compute_ood_Acc(ood_trainset_list, thresholds, trainer, classes, classes_idx_OOD, args.save_path, args.ID_tasks, args.multiple_dataset,classes_idx_ID,args.baseline_ood, args, save_k)


        return test_acc, ood_acc

    else:

        # return all model layer outputs and activations
        if args.dataset1 == "tinyimagenet":
            avg_act_all_layers, layer_indices, features_indices_mapping, prev_conv_net_mapping  = layer_utils.return_all_layer_activations_resnet(trainer, testloader)
        else:
            avg_act_all_layers, layer_indices  = layer_utils.return_all_layer_activations(trainer, testloader)


        test_acc_full = []
        ood_acc_full = []
        
        id_classes = args.ID_tasks
        lr_mult = 2 
        for k in range(0, len(ood_trainset_list),1):


            curr_ood_data = ood_trainset_list[k]

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

                # perform cheating to find the optimal eta 
                cheat_threshold, best_accuracy, best_id_accuracy, best_ood_accuracy, accuracies, etas, best_eta, id_accuracies, ood_accuracies, id_accuracy_original, ood_accuracy_original, total_accuracy_original = ood_utils.cheat_thresholds(new_activations_list_train, trainer, classes,classes_idx_OOD, id_classes + k, args.baseline_ood, activations_list_new_class_cheat, thresholds)
                print('cheat thresholds', cheat_threshold)
                print(thresholds,'original thresholds')

                length_curr_ood_data = len(curr_ood_data)
                
                
                length_testloader = len(testloader)
                

                testloader_weight = length_testloader/(length_testloader + length_curr_ood_data)
                ood_weight = length_curr_ood_data/(length_testloader + length_curr_ood_data)

                if args.leave_one_out:
                    print(thresholds_one_out,'leave one out thresholds')
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
                print("regular acc on id test data: ", test_acc_regular)
                print("regular total accuracy: ", total_acc_regular)
                print('\n')

                ood_acc_cheat = ood_utils.compute_ood_Acc([curr_ood_data], cheat_threshold, trainer, classes, [classes_idx_OOD[k]], args.save_path, id_classes + k, args.multiple_dataset, classes_idx_ID, args.baseline_ood , args, save_k)
                test_acc_cheat = ood_utils.compute_test_Acc(testloader, cheat_threshold, trainer, classes, classes_idx_OOD, args.save_path, id_classes + k, classes_idx_ID, args.baseline_ood, args, save_k)
                total_acc_cheat = ood_acc_cheat * ood_weight + test_acc_cheat * testloader_weight
                print("cheat ood acc on ood train data: ", ood_acc_cheat)
                print("cheat acc on id test data: ", test_acc_cheat)
                print('cheat total accuracy: ', total_acc_cheat)
                print('\n')
                

               
                testset = [testset, ood_testset_list[k]]

        
              
                num_id_batches = 5
                if args.leave_one_out:
                    # collect eval setting statistics
                    true_positives_one_out, false_positives_one_out, true_negatives_one_out, false_negatives_one_out = ood_utils.get_first_true_positive(testset, thresholds_one_out, num_id_batches, trainer, classes, classes_idx_OOD, id_classes + k, classes_idx_ID, args.baseline_ood, batch_size, args.random_seed + k, args, args.save_path, k)
                    print(f"true positives one out: {true_positives_one_out},  false positives one out: {false_positives_one_out}, true negatives one out: {true_negatives_one_out}, false negatives one out: {false_negatives_one_out}")
                
                # Collect eval setting statistics
                true_positives_regular, false_positives_regular, true_negatives_regular, false_negatives_regular = ood_utils.get_first_true_positive(testset, thresholds, num_id_batches, trainer, classes, classes_idx_OOD, id_classes + k, classes_idx_ID, args.baseline_ood, batch_size, args.random_seed + k, args, args.save_path, k)
                true_positives_cheat, false_positives_cheat, true_negatives_cheat, false_negatives_cheat = ood_utils.get_first_true_positive(testset, cheat_threshold, num_id_batches, trainer, classes, classes_idx_OOD, id_classes + k, classes_idx_ID, args.baseline_ood, batch_size, args.random_seed + k, args, args.save_path, k)

                print(f"true positives regular: {true_positives_regular},  false positives regular: {false_positives_regular}, true negatives regular: {true_negatives_regular}, false negatives regular: {false_negatives_regular}")
                print(f"true positives cheat: {true_positives_cheat},  false positives cheat: {false_positives_cheat}, true negatives cheat: {true_negatives_cheat}, false negatives cheat: {false_negatives_cheat} ")


                testset = GeneralImgDataset(data = testset[0].data + testset[1].data, targets = testset[0].targets + testset[1].targets, transform = testset[0].transform, rgb = args.rgb)
                print(testset.data[0].shape, "concated testset input shape")
                print(testset.targets[0], "concated testset output")


                testloader = data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 2)
                testloader_cont = data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)

                print(best_accuracy, 'optimal cheating accuracy on training data')
                print(best_id_accuracy, 'optimal cheating id accuracy on training data')
                print(best_ood_accuracy, 'optimal cheating ood accuracy on training data')


                # save results 
                state_dict = {'dataset': args.dataset1, 'best_eta': best_eta, 'best_thresholds': cheat_threshold, 'accuracies': accuracies, 'etas': etas, 'best_accuracy': best_accuracy, 'best_id_accuracy': best_id_accuracy, 'best_ood_accuracy': best_ood_accuracy, 'id_accuracies': id_accuracies, 'ood_accuracies': ood_accuracies, 'id_accuracy_original': id_accuracy_original, 'ood_accuracy_original': ood_accuracy_original, 'total_accuracy_original': total_accuracy_original}
                if args.leave_one_out:
                    state_dict_one_out = {'dataset': args.dataset1, 'one_out_eta': eta, 'cheat_eta': best_eta, 'one_out_thresholds': thresholds_one_out, 'test_acc_one_out': test_acc_one_out, 'ood_acc_one_out': ood_acc_one_out, 'test_acc_regular': test_acc_regular, 'ood_acc_regular': ood_acc_regular, 'ood_acc_cheat': ood_acc_cheat, 'test_acc_cheat': test_acc_cheat, 'total_acc_one_out': total_acc_one_out, 'total_acc_regular':total_acc_regular, 'total_acc_cheat':total_acc_cheat }
                    state_dict_queries_one_out = {"true_positives_one_out": true_positives_one_out,  "false_positives_one_out": false_positives_one_out, "true_negatives_one_out": true_negatives_one_out, "false_negatives_one_out": false_negatives_one_out}
                    state_dict_one_out = {**state_dict_one_out, **state_dict_queries_one_out}

                state_dict_queries = {"true_positives_regular": true_positives_regular,  "false_positives_regular": false_positives_regular, "true_negatives_regular": true_negatives_regular, "false_negatives_regular": false_negatives_regular, 
                                      "true positives cheat": true_positives_cheat,  "false_positives_cheat": false_positives_cheat, "true_negatives_cheat": true_negatives_cheat, "false_negatives_cheat": false_negatives_cheat }
                
                state_dict = {**state_dict, **state_dict_queries}

                # perform continual learning on new class
                if args.dataset1 == "tinyimagenet":
                    current_act_avg_layers, test_acc_list,ood_acc_list,trainer = ood_utils.continual_learner(trainer, curr_ood_data, ood_testset_list[k], testloader_cont, avg_act_all_layers, layer_indices, batch_size,classes_idx_OOD[k],classes_idx_ID,lr_mult, args, features_indices_mapping, prev_conv_net_mapping)  
                else:
                    current_act_avg_layers, test_acc_list,ood_acc_list,trainer = ood_utils.continual_learner(trainer, curr_ood_data, ood_testset_list[k], testloader_cont, avg_act_all_layers, layer_indices, batch_size,classes_idx_OOD[k],classes_idx_ID,lr_mult, args)  
                test_acc_full.append(test_acc_list)
                ood_acc_full.append(ood_acc_list)

                # get the new eta, try with weighted avgeraging based on number of classes that are id and ood.
                _, _, _, activations_list_new_class = trainer.ood_evaluate(ood_trainloader)
                new_activations_list_train[str(id_classes +k)] = np.array(activations_list_new_class)
                if args.leave_one_out:

                    # compute new eta estimate
                    eta = .5 * eta + .5 * best_eta
                    
                    # compute leave one out thresholds given eta estimate
                    thresholds_one_out, _, _ = ood_utils.compute_per_class_thresholds(new_activations_list_train, trainer, classes, classes_idx_OOD[k:], id_classes + k + 1, args.baseline_ood, eta)
                    print('thresholds one out', thresholds_one_out)
                thresholds = ood_utils.update_thresholds(thresholds, activations_list_new_class,trainer,classes_idx_ID)
               
                print('thresholds', thresholds)

               
                
                
                cwd = os.getcwd()
                filename =  '/results/' +  args.dataset1 +  "/" + str(args.random_seed)
                dir = cwd + filename
                os.makedirs(dir, exist_ok = True)
                with open(  os.path.join(dir,  f'number_of_id_classes:{id_classes + k}.pickle'), 'wb') as f:
                    pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                if args.leave_one_out:
                    one_out_filename =  '/results/'  +  "leave_one_out/" +  args.dataset1 + '/' + str(args.random_seed)
                    one_out_dir = cwd+ one_out_filename
                    os.makedirs(one_out_dir, exist_ok = True)
                    with open(  os.path.join(one_out_dir,  f'number_of_id_classes:{id_classes + k}.pickle'), 'wb') as f:
                        pickle.dump(state_dict_one_out, f, protocol=pickle.HIGHEST_PROTOCOL)

                new_avg = current_act_avg_layers

                for i in range(len(new_avg)-1):
                   new_avg[i] =  (current_act_avg_layers[i] + avg_act_all_layers[i])
                avg_act_all_layers = new_avg
                lr_mult = 2 
            

 
        if args.full_pipeline:
            testloader = data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 2)
            for p in range(0,3):
                percent = 0.1*p
                test_ood_acc = ood_utils.compute_incremental_test_acc(testloader, thresholds, trainer, classes,classes_idx_OOD, args.save_path, len(classes_idx_ID),classes_idx_ID, percent,save_k)
                print("total in OOD accuracy for sample test data", test_ood_acc)
        print("Exiting Continual Learner experiment")
        exit()  ## the continula learner exits after running 1 experiment


# computes etas to be stored during the first stage of leave one out threshold optimization
def leave_one_out(trainloader, curr_ood_data, trainer, classes, classes_idx_OOD,args,classes_idx_ID, save_k = 0):
    
    _, _, _,activations_list_train, _ = trainer.evaluate(trainloader, classes_idx_OOD, classes_idx_ID, extract_act = True)

    assert(len(classes_idx_OOD) == 1)
    print(classes_idx_OOD)
    print(classes)

    id_classes = args.ID_tasks - 1
    thresholds = [1 for i in range(id_classes)] 

    np.savez(args.save_path + '/activations' +  '/leave_one_out/{}/act_full_train_{}.npz'.format(args.random_seed, classes[classes_idx_OOD[0]]),**activations_list_train)

    activations_list_train = dict(np.load(args.save_path + '/activations' + '/leave_one_out/{}/act_full_train_{}.npz'.format(args.random_seed, classes[classes_idx_OOD[0]]), allow_pickle = True))
  
    new_activations_list_train = activations_list_train.copy()
    transform = curr_ood_data.transform
    rgb = curr_ood_data.rgb
    curr_ood_data_copy = GeneralImgDataset(curr_ood_data.data, curr_ood_data.targets, transform, rgb) 
    

    ood_trainloader = data.DataLoader(curr_ood_data_copy, batch_size = 1, shuffle = False, num_workers = 2)
    _, _, _, activations_list_new_class = trainer.ood_evaluate(ood_trainloader)

    # find the optimal eta
    _, _, _, _, _, _, best_eta, _, _, _, _, _ = ood_utils.cheat_thresholds(new_activations_list_train, trainer, classes,classes_idx_OOD, id_classes, args.baseline_ood, activations_list_new_class, thresholds)
   
    return best_eta
