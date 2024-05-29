import numpy
import torch

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    ## parser
    parser.add_argument('--dataset1', default="mnist", type=str, required = False)
    parser.add_argument('--dataset2', default="fmnist", type=str, required = False)

    parser.add_argument('--experiment', default="ood_detect", type = str, required = False) 
    parser.add_argument('--multiple_dataset', default = False, type=bool, required = False)

    parser.add_argument('--ID_tasks', default=5, type=int, required = False) # Number of ID classes
    parser.add_argument('--total_tasks', default=10, type=int, required = False) #total number of classes that will be used. I.e number of ID + number of OOD
    parser.add_argument('--single_ood_class', default = False, type = bool, required = False)  ## in this case does a round robin through all the classes


    parser.add_argument('--batch_size', default=32, type=int, required = False)
    parser.add_argument('--lr', default=0.0001, type=float, required = False)
    parser.add_argument('--epochs', default = 20, type = int, required = False)
    parser.add_argument('--epochs_g', default = 10, type = int, required = False)

    parser.add_argument('--cosine_sim', default = False, type = bool, required = False) # Whether or not cosine similarity is used
    parser.add_argument('--baseline', default = False, type = bool,required = False )
    parser.add_argument('--baseline_ood', default = False, type = bool,required = False )
    parser.add_argument('--sparsity_es', default = False, type = bool, required = False)
    parser.add_argument('--sparsity_gs', default = False, type = bool, required = False)

    parser.add_argument('--full_pipeline', default= False, type=bool, required = False)



    parser.add_argument('--train', default = False, type=bool, required = False)
    parser.add_argument('--load_checkpoint', default = False, type = bool, required = False) # Load model checkpoint
    parser.add_argument('--load_list', default = False, type = bool, required = False)
    parser.add_argument('--cont_learner', default = False, type = bool, required = False)
    parser.add_argument('--random_seed', default=5, type=int, required = False)

    parser.add_argument('--save_path', default="test", type = str, required = False) # model save path
    parser.add_argument("--data_path", default = "", type = str, required = False ) # data save path
    # parser.add_argument('--debugging', default = False, type = bool, required = False) 
    parser.add_argument('--leave_one_out', default = False, type = bool, required = False)  # perform leave one out threshold optimization
    parser.add_argument('--load_checkpoint_one_out', default = False, type = bool, required = False)  # load model check point if a pretrained models that were used for the first stage of leave one out optimization exist
    parser.add_argument('--leave_one_out_train', default = False, type = bool, required = False)  #perform training on leave one out models that are used for the first stage of leave one out optimization
    parser.add_argument('--rgb', default = False, type = bool, required = False) #whether the dataset has rgb images or black and white images
    args = parser.parse_args()

    return args


def print_args(args):

    print("Experiment :", args.experiment)
    print("Number of In-Dist tasks:", args.ID_tasks)
    print("Cosine Similarity :", args.cosine_sim)
    print("Exclusive Sparsity :", args.sparsity_es)
    print("Group Sparsity :", args.sparsity_gs)
    print("Training", args.train)
    print("dataset1 :", args.dataset1)

    print("learning_rate :", args.lr)
