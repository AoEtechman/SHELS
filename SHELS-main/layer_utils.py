import numpy as np
import torch
import torch.nn as nn
from train import Trainer

import pdb

def return_all_layer_activations(trainer, data):
    """
    Returns the indices of relu activations in the model, as well as the avg layer activations over all batches 
    """
    layer_indices = []
    all_layer_activations = None

    # We store the indices of all of the relu activations in the model features if we are working with a typical cnn model with no skip connections
    # These will be the indices where we retrieve the activations of each Conv layer in the feature generation part of the model
    for i in range(0, len(trainer.model.features)):
        if isinstance(trainer.model.features[i], nn.ReLU):
            layer_indices.append(i)
    layer_indices.append(i) # we also store the output of the flatten layer of our CNN
    for j in range(0, len(trainer.model.classifier)):
        if isinstance(trainer.model.classifier[j], nn.ReLU):
            layer_indices.append(i+j+1)



    avg_act_all_layers = 0
    total_batches = 0
    trainer.model.eval()

    with torch.no_grad():
        for (x,y) in data:
            total_batches +=1
            x = x.to(trainer.device)
            y = y.to(trainer.device)

            layers ,y_pred = trainer.model(x) # layers is a list of all of the outputs of every single nn module in the model
            avg_layers = []
            for i in range(0, len(layers)):

                mean_layer = torch.mean(layers[i], dim = 0)

                avg_layers.append(mean_layer)
            avg_layers =  [ten.detach().cpu().numpy() for ten in avg_layers]
            #avg_act_all_layers+=np.array(avg_layers)
            avg_act_all_layers += np.array(avg_layers,dtype = object)

    all_layer_activations = avg_act_all_layers/total_batches

    return all_layer_activations, layer_indices


def return_all_layer_activations_resnet(trainer, data):


    """
    This function performs the same function as the last, however, because resnets have skip connections, the mapping between Relus and Conv layers is 
    more sophisticated. This function was built specifically for non pretrained resnet 18 models as specified in the pytorch documentation
    It returns the avg layer activations over all batches, the indices of the activations of the linear and flatten layers, the mapping of conv layer indices to their 
    corresponding relu layer activations, and the mapping of a conv layer to the previous conv layer in the architecture.  
    """

    
    features_indices_mapping = {}  # maps the index of the conv layer to the index of the relu that performs activation on the output of the afformentioned conv layer
    layer_indices = []
    all_layer_activations = []
    conv_net_indices = [] # indices of all of the conv layers in the model
    conv_net_flag = [] # list of boolean flag for each conv layer indicating whether the conv layer came from a downsample layer in the resnet model
    prev_conv_net_mapping = {} # maps the indice of a conv layer to the indice of the conv layer that came before it in the model
    relu_indices = []  

    for i, (feature, flag) in enumerate(zip(trainer.model.features, trainer.model.downsample_flag )):
        if isinstance(feature, nn.ReLU):
            relu_indices.append(i)
        else:
            if not flag:  # flag is a boolean indicating whether the nn module came from a downsample layer in the resnet. Downsample layers contain Conv layers, which occur in between 
                          # another Conv layer and its relu activation. See the model_resnet18 file for reference
                if isinstance(feature, nn.Conv2d):
                    conv_net_indices.append(i)
                    conv_net_flag.append(flag)
                    if len(conv_net_indices) >= 2:
                        prev_conv_net_mapping[i] = conv_net_indices[-2] # simply the previous conv layer
                    else:
                        prev_conv_net_mapping[i] = None # first conv layer has no previous conv layer

            else: # The input conv layer to the conv layer in the downsample layer is always three conv layers prior
                if isinstance(feature, nn.Conv2d):
                    conv_net_indices.append(i)
                    conv_net_flag.append(flag)
                    prev_conv_net_mapping[i] = conv_net_indices[-4]

    assert len(relu_indices) == len(conv_net_indices)

    for ix in range(len(conv_net_indices)): # we now match the index of the conv layer to the index of the relu that performs activation on the output of the afformentioned conv layer
        flag = conv_net_flag[ix]
        index = conv_net_indices[ix]
        relu_index = relu_indices[ix]
        if ix < len(conv_net_indices) - 1:
            next_flag = conv_net_flag[ix + 1]
            if next_flag != True and flag != True: # both not conv layers from a downsample layer, things proceed as normal
                features_indices_mapping[index] = relu_index
            elif next_flag != True and flag == True: # if the current conv layer came from a downsample layer, its relu activation is the relu at one index prior
                                                     # refer to the resnet model to see why. 
                features_indices_mapping[index] = relu_indices[ix - 1]
            elif next_flag == True: # if the next conv layer comes from a downsample layer, its relu activation is the relu at the next index,
                                    # refer to the resnet model to see why
                features_indices_mapping[index] = relu_indices[ix + 1]
        else:
            if flag == True:
                features_indices_mapping[index] = relu_indices[ix - 1]
            else:
                features_indices_mapping[index] = relu_index

    


    

    # the rest of the function follows the original return_all_layer_activations function
    layer_indices.append(i)
    for j in range(0, len(trainer.model.classifier)):
        if isinstance(trainer.model.classifier[j], nn.ReLU):
            layer_indices.append(i+j+1)



    avg_act_all_layers = 0
    total_batches = 0
    trainer.model.eval()

    with torch.no_grad():
        for (x,y) in data:
            total_batches +=1
            x = x.to(trainer.device)
            y = y.to(trainer.device)

            layers ,y_pred = trainer.model(x)
            avg_layers = []
            for i in range(0, len(layers)):

                mean_layer = torch.mean(layers[i], dim = 0)

                avg_layers.append(mean_layer)
            avg_layers =  [ten.detach().cpu().numpy() for ten in avg_layers]
            #avg_act_all_layers+=np.array(avg_layers)
            avg_act_all_layers += np.array(avg_layers,dtype = object)

    all_layer_activations = avg_act_all_layers/total_batches

    return all_layer_activations, layer_indices, features_indices_mapping, prev_conv_net_mapping


