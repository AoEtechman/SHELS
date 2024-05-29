# SHELS: Exclusive Feature Sets For Novelty Detection And Continual Learning Without Class Boundaires + leave one out optimization 

 ## Description

 This repo contains code for performing continual learning experiments with the method described in https://arxiv.org/abs/2206.13720. It also extends this method by providing a way to optimize for novelty detection threshold parameters as described in https://arxiv.org/abs/2309.02551. The optimization function that finds the optimal eta can be found in the function `cheat_thresholds` in the file 'ood_utils'. We also provide the code for a new eval setup in the continual learning setting. The purpose of this eval setting is to evaluate both the speed at which an agent detects novelty while constraining the number of task instances that it wrongly detects as novel. The code is defined in the `ood_utils` file under the function `get_first_true_positive`

 ## Code Structure
 The general structure of the code is as follows. `Main.py` is the outer shell that is used to run the code. `train.py` contains the code used for performing forward passes and computing losses. The `model` files contain the architectures of the various CNN models used for learning on the various datasets. The `data` files contain all the functionality for creating datasets and dataloaders. `ood_utils` contains most of the functions used in the continual learning setting including the functions for leave one out optimization, computing accuracies, and learning from new classes. `run_utils` provides higher level functions for running model training and well as running the continual learning experiments.

 ## 1. Requirements
  - pip install -r requirements.txt
   
    or 
  - conda create --name <env_name> --file requirements.txt


## 2. Datsets

  - Pytorch datasets for MNIST, FMNIST, SVHN and CIFAR10
  - GTSRB can be downloaded here https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
  - AudioMNIST can be retrieved from here https://github.com/soerenab/AudioMNIST/tree/master/data. Note current AudioMNIST model has shown some problems during continual learning that need to be fixed
  - TinyImagenet can be retrieved from https://www.kaggle.com/c/tiny-imagenet for example. Note current resnet model is not able to learn continually and undergoes forgetting when trying to train on a new class in the novelty accomodation stage.

    

## 3. Novelty detection 
   create directories to save the models and activations, example mkdir dir dir_bl, as well as the datasets if needed, example mkdir datadir
  ### MNIST (within-datatset)
  #### train 
    ### shels
     python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 5 --save_path ./dir --data_path ./datadir

    ### baseline
     python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --baseline True --train True --random_seed 5 --save_path ./dir_bl  --data_path ./datadir

   #### evaluation
     ### shels
      python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 5 --save_path ./dir  --data_path ./datadir

     ### baseline
      python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --baseline True --load_checkpoint True --random_seed 5 --save_path ./dir_bl --baseline_ood True  --data_path ./datadir

  To run experiments with different datasets, choose dataset1 argument from [mnist, fmnist, cifar10, svhn, gtsrb, audiomnist, tinyimagenet].
  
  Note : Be sure to specify the --total_tasks as well as --ID_tasks arguments, total number of classes and total number of ID classes respectively

  ### MNIST (ID) vs FMNIST (OOD) (across-datasets)
   #### train
    ### shels
     python main.py --dataset1 mnist --dataset2 fmnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --save_path ./dir  --data_path ./datadir

    ### baseline
     python main.py --dataset1 mnist --dataset2 fmnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --baseline True --train True --save_path ./dir_bl  --data_path ./datadir

   ### evaluation
    ### shels
     python main.py --dataset1 mnist --dataset2 fmnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --save_path ./dir  --data_path ./datadir

    ### baseline
     python main.py --dataset1 mnist --dataset2 fmnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --baseline True --load_checkpoint True --save_path ./dir_bl --baseline_ood True  --data_path ./datadir

 To run experiments with different datasets, choose dataset1 and dataset2 from [mnist, fmnist, cifar10, svhn, gtsrb, audiomnist, tinyimagenet]
    
  Note : Be sure to specify the --total_tasks as well as --ID_tasks arguments and ensure cosistent input dimension in data_loader.py for ID and OOD datasets

## 4. Novelty Accommodation and Novelty detection-Accommodation 
   
   ### MNIST
   #### train 
     python main.py --dataset1 mnist --ID_tasks 7 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 5 --save_path ./dir  --data_path ./datadir

   #### accommodation
    python main.py --dataset1 mnist --ID_tasks 7 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 5 --save_path ./dir --cont_learner True  --data_path ./datadir


   #### detection and accommodation
    python main.py --dataset1 mnist --ID_tasks 7 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 5 --save_path ./dir  --data_path ./datadir

  #### detection and accommodation and leave one out

    If training leave one out stage one model from scratch

    python main.py --dataset1 mnist --ID_tasks 6 --total_tasks 200 --batch_size 32 --lr 0.0001 --epochs 35 --cosine_sim True --sparsity_gs True --random_seed 1 --save_path ./dir  --cont_learner True  --load_checkpoint True --leave_one_out True --leave_one_out_train True --data_path ./datadir



    If already trained leave one out stage one model

    python main.py --dataset1 mnist --ID_tasks 6 --total_tasks 200 --batch_size 32 --lr 0.0001 --epochs 35 --cosine_sim True --sparsity_gs True --random_seed 1 --save_path ./dir  --cont_learner True  --load_checkpoint True  --leave_one_out True --load_checkpoint_one_out True --data_path ./datadir






  To run experiments with different datasets, choose dataset1 from [mnist, fmnist, cifar10, svhn, gtsrb, audiomnist, tinyimagenet]
  To load a preloaded set of experiments, use class_list_GTSRB.npz for GTSRB dataset and class_list1.npz for the other datasets by setting--load_list flag to True.
  
## Citing this work
  If you use this work please cite our paper.
      
  
 ``` 
  @misc{https://doi.org/10.48550/arxiv.2206.13720,
  doi = {10.48550/ARXIV.2206.13720}, 
  url = {https://arxiv.org/abs/2206.13720},
  author = {Gummadi, Meghna and Kent, David and Mendez, Jorge A. and Eaton, Eric},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {SHELS: Exclusive Feature Sets for Novelty Detection and Continual Learning Without Class Boundaries},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

