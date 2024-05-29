import numpy as np
import torch
import torch.nn as nn

import pdb


def get_weights_mask(model, layer_act, layer_idx, output_dim):

    nodes_list_feats = []
    mask_list_feats = []
    weights_list_feats = []
    ## initialize nodes
    lower_nodes = np.zeros(1)
    curr_idx = 0
    activations_norm = []
    with torch.no_grad():

        for i in range(0,len(model.features)):


            if isinstance(model.features[i], nn.Conv2d):

                weights = model.features[i].weight.data.clone().detach()
                upper_nodes = np.zeros(model.features[i].out_channels)

                activations = layer_act[layer_idx[curr_idx]]  # get the activations of current layer
                curr_idx+=1
                activations = torch.from_numpy(activations)
                if len(activations.shape)>1:
                    activations = activations.norm(p = 2, dim=(1,2)).cpu().numpy()
                    activations_norm.append(activations)

                  
                ## lets freeze all weights
    
                mask = np.ones((weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]))
                
                ## find non important and important nodes of output channel from previous conv layer
                non_imp_idx_lower = np.argwhere(lower_nodes != 0)
                imp_idx_lower = np.argwhere(lower_nodes == 0)


                ## now lets look for unimportant nodes in the current conv layer output channels and unfreeze the weights coming into it
                non_imp_idx_upper = np.argwhere(activations == 0)[:,0]
                imp_idx_upper = np.argwhere(activations != 0)[:,0]
                upper_nodes[non_imp_idx_upper] = 1

                # dont update weights coming from important nodes to important nodes
                mask[imp_idx_upper,imp_idx_lower,:,:] = 0


                # zero the weights coming from non important nodes to important nodes
                if len(non_imp_idx_lower) > 0:
                    weights[imp_idx_upper,non_imp_idx_lower,:,:] = 0.0
                

                nodes_list_feats.append(upper_nodes)
                lower_nodes = upper_nodes
                mask_list_feats.append(mask)
                weights_list_feats.append(weights)


        ## nn.flatten activations
        activations = layer_act[layer_idx[curr_idx]]
        activations_norm.append(activations)
        lower_nodes = np.zeros(len(activations))
        non_imp_idx_lower = np.argwhere(activations == 0)[:,0]
        lower_nodes[non_imp_idx_lower] = 1
        # pdb.set_trace()
        curr_idx+=1
        nodes_list = []
        weights_list = []
        mask_list = []
        
        for i in range(0,len(model.classifier)):

            if isinstance(model.classifier[i], nn.Linear):

                weights = model.classifier[i].weight.data.clone().detach()
                upper_nodes = np.zeros(model.classifier[i].out_features)
                # pdb.set_trace()
                if i < len(model.classifier) - 1:
                    activations = layer_act[layer_idx[curr_idx]]
                    activations_norm.append(activations)
                    curr_idx+=1
                    # print(activations.shape)
                else:

                    activations = np.ones(output_dim)
                    activations[output_dim -1] = 0
                    activations_norm.append(activations)

                ## lets freeze all weights
                mask = np.ones((weights.shape[0], weights.shape[1]))
                ## fix the outgoing weights of lower unimporant nodes to 0
                non_imp_idx_lower = np.argwhere(lower_nodes == 1)
                imp_idx_lower = np.argwhere(lower_nodes == 0)
                # if len(non_imp_idx_lower) > 0:
                #     weights[:,non_imp_idx_lower] = 0.0

                ## now lets look for unimportant weights in the higher nodes and unfreeze the weights coming into it
                imp_idx_upper = np.argwhere(activations != 0)[:,0]
                non_imp_idx_upper = np.argwhere(activations == 0)[:,0]
                upper_nodes[non_imp_idx_upper] = 1
                mask[imp_idx_upper, imp_idx_lower] = 0
                if len(non_imp_idx_lower) > 0:
                    weights[imp_idx_upper,non_imp_idx_lower] = 0.0


                nodes_list.append(upper_nodes)
                lower_nodes = upper_nodes
                mask_list.append(mask)
                weights_list.append(weights)




    return mask_list, weights_list, nodes_list, mask_list_feats, weights_list_feats, nodes_list_feats, activations_norm




# same functionality as the previous function, we have just customized it for the resnet which has a more complex relationship between conv layers
def get_weights_mask_resnet(model, layer_act, layer_idx, features_indices_mapping, prev_conv_net_mapping, output_dim):

    nodes_list_feats = []
    mask_list_feats = []
    weights_list_feats = []
    ## initialize nodes
  
    curr_idx = 0
    activations_norm = []
    # layer_curr_idx = {}
    with torch.no_grad():

        for i in range(0,len(model.features)):


            if isinstance(model.features[i], nn.Conv2d):

                weights = model.features[i].weight.data.clone().detach()
                upper_nodes = np.zeros(model.features[i].out_channels)

                activations = layer_act[features_indices_mapping[i]]  # get the activations of this layer
 
                activations = torch.from_numpy(activations)
                if len(activations.shape)>1:
                    activations = activations.norm(p = 2, dim=(1,2)).cpu().numpy()
                    activations_norm.append(activations)

                prev_conv_layer_index = prev_conv_net_mapping[i]
                if prev_conv_layer_index == None:
                    lower_nodes = np.zeros(1)
                else:
                    lower_nodes = np.zeros(model.features[prev_conv_layer_index].out_channels)
    
                    prev_layer_activations = layer_act[features_indices_mapping[prev_conv_layer_index]]
                    prev_layer_activations = torch.from_numpy(prev_layer_activations)
                    if len(prev_layer_activations.shape)>1:
                        prev_layer_activations = prev_layer_activations.norm(p = 2, dim=(1,2)).cpu().numpy()


            
                    non_imp_idx_upper_prev = np.argwhere(prev_layer_activations == 0)[:,0]
                    lower_nodes[non_imp_idx_upper_prev] = 1



               
                mask = np.ones((weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]))
              
                non_imp_idx_lower = np.argwhere(lower_nodes != 0)
                imp_idx_lower = np.argwhere(lower_nodes == 0)
             
                non_imp_idx_upper = np.argwhere(activations == 0)[:,0]
                imp_idx_upper = np.argwhere(activations != 0)[:,0]
                upper_nodes[non_imp_idx_upper] = 1
                mask[imp_idx_upper,imp_idx_lower,:,:] = 0


                if len(non_imp_idx_lower) > 0:
                    weights[imp_idx_upper,non_imp_idx_lower,:,:] = 0.0
               

                nodes_list_feats.append(upper_nodes)

                mask_list_feats.append(mask)
                weights_list_feats.append(weights)


        ## nn.flatten activations
        activations = layer_act[layer_idx[curr_idx]]
        activations_norm.append(activations)
        lower_nodes = np.zeros(len(activations))
        non_imp_idx_lower = np.argwhere(activations == 0)[:,0]
        lower_nodes[non_imp_idx_lower] = 1

        curr_idx+=1
        nodes_list = []
        weights_list = []
        mask_list = []
        
        for i in range(0,len(model.classifier)):

            if isinstance(model.classifier[i], nn.Linear):

                weights = model.classifier[i].weight.data.clone().detach()
                upper_nodes = np.zeros(model.classifier[i].out_features)
                # pdb.set_trace()
                if i < len(model.classifier) - 1:
                    activations = layer_act[layer_idx[curr_idx]]
                    activations_norm.append(activations)
                    curr_idx+=1

                else:

                    activations = np.ones(output_dim)
                    activations[output_dim -1] = 0
                    activations_norm.append(activations)

                ## lets freeze all weights
                mask = np.ones((weights.shape[0], weights.shape[1]))
                ## fix the outgoing weights of lower unimporant nodes to 0
                non_imp_idx_lower = np.argwhere(lower_nodes == 1)
                imp_idx_lower = np.argwhere(lower_nodes == 0)
              

                ## now lets look for unimportant weights in the higher nodes and unfreeze the weights coming into it
                imp_idx_upper = np.argwhere(activations != 0)[:,0]
                non_imp_idx_upper = np.argwhere(activations == 0)[:,0]
                upper_nodes[non_imp_idx_upper] = 1
                mask[imp_idx_upper, imp_idx_lower] = 0
                if len(non_imp_idx_lower) > 0:
                    weights[imp_idx_upper,non_imp_idx_lower] = 0.0


                nodes_list.append(upper_nodes)
                lower_nodes = upper_nodes
                mask_list.append(mask)
                weights_list.append(weights)




    return mask_list, weights_list, nodes_list, mask_list_feats, weights_list_feats, nodes_list_feats, activations_norm
