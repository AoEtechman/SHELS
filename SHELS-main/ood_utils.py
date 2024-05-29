import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data

from weight_freezing import get_weights_mask, get_weights_mask_resnet
import layer_utils
from copy import deepcopy
from custom_dataloader import GeneralImgDataset





# Compute the detection thresholds for each class
def compute_per_class_thresholds(activations_train, trainer, classes, ood_class_idx, in_dist_classes, baseline_ood, eta):
    

    if len(ood_class_idx) == 1:
        ind_classes =  classes.copy()
        ind_classes.remove(classes[ood_class_idx[0]])
    else:
        ind_classes = classes[0: in_dist_classes]


    weights = trainer.model.classifier[trainer.weight_idx].weight.detach().cpu().numpy()
    thresholds = np.zeros(in_dist_classes)
    means = np.zeros(in_dist_classes)
    stds = np.zeros(in_dist_classes)

    total_train_data = 0
    total_true_y = 0

    for class_idx in range(0,in_dist_classes):
        y_list = []
        
        train_data = activations_train[str(class_idx)]

        total_train_data+=len(train_data)
        true_y = 0
        for idx in range(0,len(train_data)):
            train_feats = train_data[idx,:]
            y_label = []
            y_weighted = []
            for k in range(0, in_dist_classes):


                if baseline_ood:
                    norm = np.linalg.norm(train_feats)
                    norm = np.where(norm==0,1e-20,norm)
                    feat_norm = train_feats/norm

                    weight_norm = weights[k,:]/np.abs(np.linalg.norm(weights[k,:]))
                    weighted_feats = np.sum(feat_norm * weight_norm)
                else:
                    weighted_feats = np.sum(train_feats*weights[k,:])


                y_weighted.append(weighted_feats)

            if np.argmax(y_weighted) == class_idx:
                y_list.append(max(y_weighted))
                true_y +=1
	
        
        thresholds[class_idx] = np.mean(y_list) - np.std(y_list) * eta
        means[class_idx] = np.mean(y_list)
        stds[class_idx] = np.std(y_list)
        print(np.mean(y_list), 'mean')
        print(np.std(y_list), 'std dev')
        print("threshold", thresholds[class_idx])
        total_true_y+=true_y


    total_accuracy =  100*total_true_y/total_train_data
    print("total in distribution accuracy",total_accuracy)
    print("Computed thresholds using training data")

    for i in range(len(thresholds)): 
        if np.isnan(thresholds[i]): # if any thresholds are nan set them to zero
            thresholds[i] = 0
    return thresholds, means, stds

def cheat_thresholds(activations_train, trainer, classes, ood_class_idx, in_dist_classes, baseline_ood, activations_list_new_class, original_thresholds):
    '''
    This function is designed to 'cheat' and find the eta that produces the thresholds that best seperate a class from the rest of the in distribution classes. This is the core function used 
    to perform leave one out optimization. 
    '''

  

    weights = trainer.model.classifier[trainer.weight_idx].weight.detach().cpu().numpy()


    train_data_std_devs = [] # list of lists containing distance from mean of weighted feats of a class
    ID_class_properties = {}  # store mean and std dev
    total_y = 0
    total_y_id = 0
    print('in_dist_classes', in_dist_classes)

    for class_idx in range(in_dist_classes):

        train_data = activations_train[str(class_idx)]  #get all the activations of ground truth class
        correct_weighted_feats = [] # weighted feats that are correctly classified as the right class
        all_weighted_feats = []
        total_y += len(train_data)
        total_y_id +=  len(train_data)

        for idx in range(len(train_data)): # for every data point in class
            train_feats = train_data[idx]
           
            weighted_feats = [] # weighted feat projection of current data point and all of the classes

            for k in range(0, in_dist_classes): # loop through all potential classes

                current_class_weights = weights[k, :]
                
                if baseline_ood:
                    norm = np.linalg.norm(train_feats) # normalize all of the features
                    norm = np.where(norm==0,1e-20,norm) # if norm is zero offset to small epsilon
                    feat_norm = train_feats/norm # gives us a unit vector representation of our features

                    weight_norm = current_class_weights/np.abs(np.linalg.norm(current_class_weights)) # gives us unit vector representation of weights for class k
                    weighted_feat = np.sum(feat_norm * weight_norm) # see how our features project onto the weights of this class
                    weighted_feats.append(weighted_feat)
                else:
                    weighted_feats.append(np.sum(train_feats*current_class_weights))  #dot of features with current class weights
            
            index = np.argmax(weighted_feats) 
            if index == class_idx: # build list data points that the model classifies into the correct class
                correct_weighted_feats.append(weighted_feats[index])
            all_weighted_feats.append(weighted_feats[class_idx])    

        # find the best threshold for the current class
        mean = np.mean(correct_weighted_feats)
        std_dev = np.std(correct_weighted_feats)


        if len(correct_weighted_feats) == 0:
            print('we do not have any weighted feats that were classified correctly')
            mean = 0
            std_dev = 0
        else:
            print(f'number classified correctly for class {class_idx}', len(correct_weighted_feats)) 


        ID_class_properties[class_idx] = (mean, std_dev) # store properties
        std_devs_from_mean = [ (mean - weighted_feat)/std_dev for weighted_feat in correct_weighted_feats] # build list of how far away each correctly classified point is from the mean. analagous to z score but x and mu are switched
        train_data_std_devs.append(std_devs_from_mean) # build up list of lists



    #OOD data
    total_y_ood = 0
    all_weighted_feats_ood = [] # list of weighted feat projections against each class for all ood data points
    total_y += len(activations_list_new_class)
    total_y_ood += len(activations_list_new_class)

    for idx in range(0,len(activations_list_new_class)):
        feats = activations_list_new_class[idx]
        weighted_feats_ood = []

        for k in range(0, in_dist_classes):
            if baseline_ood:
                norm = np.linalg.norm(feats) # normalize all of the features
                norm = np.where(norm==0,1e-20,norm) # if norm is zero offset to small epsilon
                feat_norm = feats/norm # gives us a unit vector representation of our features

                weight_norm = weights[k,:]/np.abs(np.linalg.norm(weights[k,:])) # gives us unit vector representation of weights for class k
                weighted_feats_ood.append(np.sum(feat_norm * weight_norm)) # see how our features project onto the weights of this class
            else:
                weighted_feats_ood.append(np.sum(feats*weights[k,:])) 

        all_weighted_feats_ood.append(weighted_feats_ood)# append list of weighted activations for each data point with respect to each class



    # We now perform the optimization to find the best eta which is simply a linear search

    eta = set()

    # each correctly classified ID datapoint's distance from its class mean is a candidate value for eta
    for class_std_devs in train_data_std_devs:
        for std_dev in class_std_devs:
            eta.add(std_dev)

    eta = sorted(list(eta))
    print(len(eta), 'number of etas')

    best_accuracy = 0
    best_id_accuracy = 0
    best_ood_accuracy = 0
    best_thresholds = []
    best_total_accuracy = 0
    best_eta = 0

    accuracies = []
    accuracy_original = 0
    id_accuracies = []
    id_accuracies_original = 0
    ood_accuracies = []
    ood_accuracies_original = 0

    have_computed_original_accuracy_id = False
    have_computed_original_accuracy_ood = False
    have_computed_accuracy_original = False

    for n in eta:
        true_y = 0
        true_y_original = 0

        # id accuracy
        for class_std_devs in train_data_std_devs: # for sublist of class
            for std_dev in class_std_devs: 
                if std_dev < n: # if above current threshold. We switch the sign of the inequality because we reversed x and mu in the initial computation of z scores
                    true_y += 1
                if not have_computed_original_accuracy_id: # we also compute results for when eta is 1 which is the baseline formulation
                    if std_dev < 1:
                        true_y_original += 1

        id_accuracy = 100 * (true_y)/total_y_id
        id_accuracies.append(id_accuracy)
        if not have_computed_original_accuracy_id: 
            id_accuracies_original = (100 * true_y_original / total_y_id)
            have_computed_original_accuracy_id = True

        thresholds = [] # compute the actual thresholds corresponding to each class 
        for class_idx in ID_class_properties:
            threshold = (ID_class_properties[class_idx][0] - n * ID_class_properties[class_idx][1])
            thresholds.append(threshold)


        #ood accuracy
        true_y_ood = 0
        true_y_ood_original = 0
        for weighted_feats in all_weighted_feats_ood:
            index = np.argmax(weighted_feats)
            if max(weighted_feats) < thresholds[index]:
                true_y += 1
                true_y_ood += 1
            if not have_computed_original_accuracy_ood:
                if max(weighted_feats) < original_thresholds[index]:
                    true_y_ood_original  +=  1
                    true_y_original += 1
        if not have_computed_original_accuracy_ood:
            have_computed_original_accuracy_ood = True
            ood_accuracies_original = (100 *true_y_ood_original/total_y_ood)
        ood_accuracies.append(100 * (true_y_ood)/total_y_ood)
        
        
       
        totalaccuracy = 100*(true_y)/total_y 

        gmean = (id_accuracy * (100*true_y_ood/total_y_ood))**(1/2) # geometric mean

        if gmean == 0: # if either zero id or ood accuracy, then gmean is essentially just some transform of the accuracy 
                       # of whichever accuracy was non zero
            gmean = max((id_accuracy * .1)**(1/2), ((100*true_y_ood/total_y_ood) * .1)**(1/2))

        if not have_computed_accuracy_original:
            have_computed_accuracy_original = True
            accuracy_original = 100 * true_y_original / total_y

        accuracies.append(totalaccuracy)

        if gmean > best_accuracy: # update best values 
            best_accuracy = gmean 
            best_total_accuracy = totalaccuracy #new change that we made, note we also changed the return statement
            best_id_accuracy = id_accuracy
            best_ood_accuracy = true_y_ood / total_y_ood * 100
            best_thresholds = [thresholds]
            best_eta = n
    if len(best_thresholds) == 0:
        best_thresholds = [[ 0 for i in range(in_dist_classes)]]
        print('no eta values to search over so set thresholds to zero')


    print('best eta', best_eta)
    return best_thresholds[0], best_total_accuracy, best_id_accuracy, best_ood_accuracy, accuracies, eta, best_eta, id_accuracies, ood_accuracies, id_accuracies_original, ood_accuracies_original, accuracy_original



# compute the threshold for the new ID class
def update_thresholds(thresholds,activations_list_new_class,trainer,in_dist_classes, eta = 1):

    weights = trainer.model.classifier[trainer.weight_idx].weight.detach().cpu().numpy()

    # total_train_data = 0
    # total_true_y = 0
    y_list = []
    true_y = 0
    class_idx = len(in_dist_classes)

    thresholds = np.insert(thresholds,class_idx-1,0)
    for idx in range(0,len(activations_list_new_class)):
        feats = activations_list_new_class[idx]
        # weighted_feats = np.matmul(feats,weights[class_idx-1,:].T)
        # feats = activations_list_new_class[idx]

        weighted_feats = np.matmul(feats, weights.T)
        # print(weighted_feats)
        if np.argmax(weighted_feats) == class_idx-1:
            y_list.append(weighted_feats)
            true_y +=1
        # y_list.append(weighted_feats)
    thresholds[class_idx-1] = np.mean(y_list) - eta * np.std(y_list)
    if np.isnan(thresholds[class_idx-1]):
        thresholds[class_idx-1] = 0

    total_accuracy =  100*true_y/len(activations_list_new_class)
    # print("New class accuracy",total_accuracy)
    print("Updated thresholds using training data")
    # print(thresholds)
    return thresholds


# Compute id test accuracy
def compute_test_Acc(testloader, thresholds, trainer, classes, ood_class_idx, save_path, in_dist_classes, in_dist_class_list, baseline_ood,args, save_k = 0):

    _, _, activations,activations_test, _ = trainer.evaluate(testloader, ood_class_idx,in_dist_class_list, extract_act = True)

    if len(ood_class_idx) == 1:
        np.savez(save_path+'/activations/{}/act_full_test_{}.npz'.format(args.random_seed, classes[ood_class_idx[0]]),**activations_test)
    else:
        np.savez(save_path+'/activations/{}/act_full_test_{}.npz'.format(args.random_seed, save_k),**activations_test)


    if len(ood_class_idx) == 1:
        activations_test = dict(np.load(save_path+'/activations/{}/act_full_test_{}.npz'.format(args.random_seed, classes[ood_class_idx[0]]), allow_pickle = True))
    else:
        activations_test = dict(np.load(save_path+'/activations/{}/act_full_test_{}.npz'.format(args.random_seed, save_k),allow_pickle = True))

    weights = trainer.model.classifier[trainer.weight_idx].weight.detach().cpu().numpy()
    if len(ood_class_idx) == 1:
        ind_classes =  classes.copy()
        ind_classes.remove(classes[ood_class_idx[0]])
    else:
         ind_classes = classes[0: in_dist_classes]

    total_test_data = 0
    total_true_y = 0
    for class_idx in range(0,in_dist_classes):
        y_list = []
        test_data = activations_test[str(class_idx)]
        total_test_data+=len(test_data)
        true_y = 0
        for idx in range(0,len(test_data)):

            test_feats = test_data[idx,:]

            y_label = []
            y_weighted = []
            for k in range(0, in_dist_classes):

                if baseline_ood:
                    norm = np.linalg.norm(test_feats)
                    norm = np.where(norm==0,1e-20,norm)
                    feat_norm = test_feats/norm
                    weight_norm = weights[k,:]/np.abs(np.linalg.norm(weights[k,:]))
                    weighted_feats = np.sum(feat_norm * weight_norm)
                else:
                    weighted_feats = np.sum(test_feats*weights[k,:])

                y_weighted.append(weighted_feats)
            if max(y_weighted) > thresholds[class_idx] and np.argmax(y_weighted) == class_idx:
                y_list.append(max(y_weighted))
                true_y +=1
        print("accuracy {}".format(ind_classes[class_idx]), 100*true_y/len(test_data))
        total_true_y+=true_y
    total_test_accuracy =  100*total_true_y/total_test_data
    print("total in distribution accuracy", total_test_accuracy)
    return total_test_accuracy


# Collecting Eval setting statistics
def get_first_true_positive(testset, thresholds, num_batches, trainer, classes, ood_class_idx, in_dist_classes, in_dist_class_list, baseline_ood, batch_size, random_seed, args, save_path, k):

    print("thresholds", thresholds)
    torch.manual_seed(random_seed) # we want consistent shuffles to compare
    print(f"batch size: {batch_size}")
    print(f"length of id test set: {len(testset[0])}")
    print(f"len of ood testset: {len(testset[1])}")



    rgb = args.rgb


    
    id_test_set = GeneralImgDataset(data = testset[0].data, targets = testset[0].targets, transform = testset[0].transform, rgb = rgb)
    ood_test_set = GeneralImgDataset(data = testset[1].data, targets = testset[1].targets, transform = testset[1].transform, rgb = rgb) # check transform issue



    id_index = int(batch_size * num_batches) # the first num_batches batches should be all id
    

    indexes = torch.randperm(len(id_test_set)) # shuffle all of the data
    
    if torch.is_tensor(id_test_set.data[0]):
        id_test_set.data = torch.stack(id_test_set.data)[indexes]
        id_test_set.data = list(torch.unbind(id_test_set.data))
    else:
        id_test_set.data = np.stack(id_test_set.data)[indexes]
        id_test_set.data = list(id_test_set.data)

    if torch.is_tensor(id_test_set.targets[0]):
        id_test_set.targets = torch.stack(id_test_set.targets)[indexes]
        id_test_set.targets = list(torch.unbind(id_test_set.targets))
    else:
        id_test_set.targets = np.stack(id_test_set.targets)[indexes]
        id_test_set.targets = list(id_test_set.targets)

    first_n_batch_id_test_set = GeneralImgDataset(id_test_set.data[: id_index], id_test_set.targets[:id_index], transform = testset[0].transform, rgb = rgb)
    remaining_test_set = GeneralImgDataset(id_test_set.data[id_index:] + ood_test_set.data, id_test_set.targets[id_index:] + ood_test_set.targets, transform = testset[0].transform,rgb =  rgb) 
    
    indexes = torch.randperm(len(remaining_test_set))
    
    if torch.is_tensor(remaining_test_set.data[0]):
        remaining_test_set.data = torch.stack(remaining_test_set.data)[indexes]
        remaining_test_set.data = list(torch.unbind(remaining_test_set.data))
    else:
        remaining_test_set.data = np.stack(remaining_test_set.data)[indexes]
        remaining_test_set.data = list(remaining_test_set.data)

    if torch.is_tensor(remaining_test_set.targets[0]):
        remaining_test_set.targets = torch.stack(remaining_test_set.targets)[indexes]
        remaining_test_set.targets = list(torch.unbind(remaining_test_set.targets))
    else:
        remaining_test_set.targets = np.stack(remaining_test_set.targets)[indexes]
        remaining_test_set.targets = list(remaining_test_set.targets)


    shuffled_testset = data.ConcatDataset([first_n_batch_id_test_set, remaining_test_set])
    testloader_shuff = data.DataLoader(shuffled_testset, batch_size = 1, shuffle = False, num_workers = 2)

    _, _, _,activations_test, labels_list = trainer.evaluate(testloader_shuff, ood_class_idx,in_dist_class_list, extract_act = True, eval_for_queries = True)
    

    weights = trainer.model.classifier[trainer.weight_idx].weight.detach().cpu().numpy()


    test_data = np.array(activations_test)
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0


    for idx in range(len(test_data)):

        test_feats = test_data[idx,:]
        true_label = labels_list[idx]

        y_weighted = []
        for k in range(0, in_dist_classes):

            if baseline_ood:
                norm = np.linalg.norm(test_feats)
                norm = np.where(norm==0,1e-20,norm)
                feat_norm = test_feats/norm
                weight_norm = weights[k,:]/np.abs(np.linalg.norm(weights[k,:]))
                weighted_feats = np.sum(feat_norm * weight_norm)
            else:
                weighted_feats = np.sum(test_feats*weights[k,:])

            y_weighted.append(weighted_feats)

            

        if true_label < in_dist_classes:
            if max(y_weighted) > thresholds[true_label] and np.argmax(y_weighted) == true_label:
                
                true_negatives += 1 # correctly classified as ID
            else:
                false_positives += 1 # Incorrectly detected as OOD
        else:
            index = np.argmax(y_weighted)
            if max(y_weighted) > thresholds[index]:
                false_negatives += 1 # incorrectly classified as ID
            else:
                true_positives += 1 # corrected detected as OOD
                print("ending collection of statistics")
                break


    print("proxy id test acc", true_negatives/id_index)
    print("returning statistics")
    return true_positives, false_positives, true_negatives, false_negatives


                






# incremental ID test accuracy
def compute_incremental_test_acc(testloader, thresholds, trainer, classes, ood_class_idx, save_path, in_dist_classes, in_dist_class_list,begin_idx, save_k = 0):

    _, _, activations,activations_test, _ = trainer.evaluate(testloader, ood_class_idx,in_dist_class_list, extract_act = True)

    if len(ood_class_idx) == 1:
        np.savez(save_path+'/activations/act_full_test_{}.npz'.format(classes[ood_class_idx[0]]),**activations_test)
    else:
        np.savez(save_path+'/activations/act_full_test_{}.npz'.format(save_k),**activations_test)


    if len(ood_class_idx) == 1:
        activations_test = dict(np.load(save_path+'/activations/act_full_test_{}.npz'.format(classes[ood_class_idx[0]]), allow_pickle = True))
    else:
        activations_test = dict(np.load(save_path+'/activations/act_full_test_{}.npz'.format(save_k),allow_pickle = True))

    weights = trainer.model.classifier[trainer.weight_idx].weight.detach().cpu().numpy()

    if len(ood_class_idx) == 1:
        ind_classes =  classes.copy()
        ind_classes.remove(classes[ood_class_idx[0]])
    else:
         ind_classes = classes[0: len(in_dist_class_list)]

    total_test_data = 0
    total_true_y = 0
    for class_idx in range(0,len(in_dist_class_list)):

        test_data = activations_test[str(class_idx)]
        test_data_len = len(test_data)
        first_idx = int(begin_idx*test_data_len)
        last_idx = int((begin_idx+0.1)*test_data_len)
        # sample_len = int(0.1*len(test_data))
        total_test_data+=(last_idx-first_idx)
        true_y = 0
        for idx in range(first_idx,last_idx):

            test_feats = test_data[idx,:]

            weighted_feats = np.matmul(test_feats,weights.T)

            y_weighted = weighted_feats
            index = np.argmax(y_weighted)
            if max(y_weighted) < thresholds[index]: #and index == class_idx:
                true_y +=1
        per_ood_accuracy =  100*true_y/(last_idx-first_idx)
        # print("OOD accuracy {} :".format(ind_classes[class_idx]), per_ood_accuracy)
        total_true_y+=true_y
    total_ood_accuracy =  100*total_true_y/total_test_data
    # print("total in OOD accuracy", total_ood_accuracy)
    return total_ood_accuracy



def compute_ood_Acc(ood_data_list, thresholds, trainer, classes, ood_class_idx, save_path, in_dist_classes, multiple_dataset, in_dist_class_list, baseline_ood , args, save_k = 0):
    ood_acc_list = []
    #print(ood_data_list, 'ood_data_list')
    for i in range(0,len(ood_data_list)):
        if True:
            ood_trainloader = data.DataLoader(ood_data_list[i], batch_size = 1, shuffle = False, num_workers = 2)
        else:
            ood_trainloader = ood_data_list[i]
        loss, acc, act_avg_ood, activation_list_ood = trainer.ood_evaluate(ood_trainloader)

        if len(ood_class_idx) > 0:
            np.savez(save_path+'/activations/{}/act_full_ood_{}_{}.npz'.format(args.random_seed, classes[ood_class_idx[i]],save_k),act_ood = activation_list_ood)
        else:
            np.savez(save_path+'/activations/{}/act_full_ood_{}.npz'.format(args.random_seed, save_k),act_ood = activation_list_ood)


        if len(ood_class_idx) > 0:
            activations_ood = dict(np.load(save_path+'/activations/{}/act_full_ood_{}_{}.npz'.format(args.random_seed, classes[ood_class_idx[i]],save_k), allow_pickle = True))['act_ood']
        else:
            activations_ood = dict(np.load(save_path+'/activations/{}/act_full_ood_{}.npz'.format(args.random_seed, save_k), allow_pickle = True))['act_ood']

        weights = trainer.model.classifier[trainer.weight_idx].weight.detach().cpu().numpy()
        ood_data = activations_ood
        true_y = 0
        for idx in range(0,len(ood_data)):
            ood_feats = ood_data[idx,:]

            y_weighted = []
            for k in range(0, in_dist_classes):

                if baseline_ood:
                    norm = np.linalg.norm(ood_feats)
                    norm = np.where(norm==0,1e-20,norm)
                    feat_norm = ood_feats/norm

                    #feat_norm = ood_feats/(np.abs(np.linalg.norm(ood_feats)))
                    weight_norm = weights[k,:]/np.abs(np.linalg.norm(weights[k,:]))
                    weighted_feats = np.sum(feat_norm * weight_norm)
                else:
                    weighted_feats = np.sum(ood_feats*weights[k,:])
                y_weighted.append(weighted_feats)
            index = np.argmax(y_weighted)
            if max(y_weighted) <= thresholds[index]:
                true_y +=1
        total_ood_accuracy =  100*true_y/len(ood_data)
        ood_acc_list.append(total_ood_accuracy)
        if multiple_dataset:
            print("OOD accuracy :", total_ood_accuracy)
        else:
            print("OOD accuracy {} :".format(classes[ood_class_idx[i]]), total_ood_accuracy)
    mean_ood_acc = np.mean(ood_acc_list)
    return mean_ood_acc


# perform continual learning on new id class
def continual_learner(trainer, ood_traindata, ood_testdata, testloader, avg_act_all_layers, layer_indices, batch_size, ood_class,  in_dist_classes_list,lr_mult, args, features_indices_mapping = None, prev_conv_net_mapping = None):
    

    trainer.model.increment_classes(1) # increment the number of id classes by one
    trainer.optimizer = optim.Adam(trainer.model.parameters(), lr = trainer.learning_rate)

    trainer.output_dim+=1


    ood_trainloader = data.DataLoader(ood_traindata, batch_size = batch_size, shuffle = False, num_workers = 2)

    
    ood_testloader = data.DataLoader(ood_testdata, batch_size = batch_size, shuffle = False, num_workers = 2)


    # get the masks that govern model weight updates
    if args.dataset1 == "tinyimagenet":
        mask_c,weights_c,nodes_c, mask_f, weights_f, nodes_f, activations_norm = get_weights_mask_resnet(trainer.model,avg_act_all_layers, layer_indices, features_indices_mapping, prev_conv_net_mapping, trainer.output_dim)

    else:
        mask_c,weights_c,nodes_c, mask_f, weights_f, nodes_f, activations_norm = get_weights_mask(trainer.model,avg_act_all_layers, layer_indices, trainer.output_dim)


  
    trainer.model.update_model_weights(weights_f, weights_c, nodes_f, nodes_c)
    epochs = 8 # number of epochs to use for training new class data
    
    old_model = deepcopy(trainer.model) ## has the frozen weights
    prev_loss = 1e30
    test_acc_list = []
    ood_acc_list = []
    loss, acc_test, act_avg_test,_,_ = trainer.evaluate(testloader,[ood_class, ood_class],  in_dist_classes_list, extract_act = False)
    print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc_test*100:.2f}%')
    for epoch in range(epochs):

        train_loss, train_acc = trainer.optimize_cont(ood_trainloader,[ood_class],mask_c,mask_f, nodes_c, nodes_f, weights_c, weights_f,activations_norm, old_model,in_dist_classes_list)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        loss, acc_test, act_avg_test,_,_ = trainer.evaluate(testloader,[ood_class, ood_class],  in_dist_classes_list, extract_act = False)
        print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc_test*100:.2f}%')
        loss_ood, acc_ood, act_avg_test,_,_ = trainer.evaluate(ood_testloader,[ood_class,  ood_class], in_dist_classes_list, extract_act = False)
        print(f'\tOOD Loss: {loss_ood:.3f} | OOD Acc: {acc_ood*100:.2f}%')
        test_acc_list.append(acc_test)
        ood_acc_list.append(acc_ood)


    avg_act_all_layers, _ = layer_utils.return_all_layer_activations(trainer, ood_trainloader)

    return avg_act_all_layers, test_acc_list, ood_acc_list, trainer
