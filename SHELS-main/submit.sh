#!/bin/bash



source /etc/profile
module load anaconda/2022b
source activate shels


#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 5 --save_path /home/gridsan/aejilemele/LIS/SHELS-main
#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 5 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main
#python testing.py

#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 0 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist
#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 1 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist
#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 2 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist
#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 3 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist
#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 4 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist
#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 5 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist
#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 6 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist
#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 7 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist
#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 8 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist
#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 9 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist



#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 0 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 1 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 2 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 3 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 4 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 5 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 6 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 7 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 8 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 9 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist


#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 30 --cosine_sim True --sparsity_gs True --train True --random_seed 0 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10 
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 30 --cosine_sim True --sparsity_gs True --train True --random_seed 1 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 30 --cosine_sim True --sparsity_gs True --train True --random_seed 2 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 30 --cosine_sim True --sparsity_gs True --train True --random_seed 3 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 30 --cosine_sim True --sparsity_gs True --train True --random_seed 4 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 30 --cosine_sim True --sparsity_gs True --train True --random_seed 5 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 30 --cosine_sim True --sparsity_gs True --train True --random_seed 6 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 30 --cosine_sim True --sparsity_gs True --train True --random_seed 7 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 30 --cosine_sim True --sparsity_gs True --train True --random_seed 8 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 30 --cosine_sim True --sparsity_gs True --train True --random_seed 9 --save_path /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10

#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 5 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist  --cont_learner True
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 5 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist  --cont_learner True
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 5 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10  --cont_learner True

#


#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 0 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist --cont_learner True
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 0 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist  --cont_learner True
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 0 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10  --cont_learner True



#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 1 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist  --cont_learner True
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 1 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist  --cont_learner True 
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 1 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10  --cont_learner True


#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 2 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist  --cont_learner True 
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 2 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist  --cont_learner True 
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 2 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10  --cont_learner True


#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 3 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist  --cont_learner True 
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 3 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist  --cont_learner True
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 3 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10  --cont_learner True


#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 4 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist  --cont_learner True
#python main.py --dataset1 minst --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 4 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist  --cont_learner True
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 4 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10  --cont_learner True


#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 6 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist  --cont_learner True
#python main.py --dataset1 minst --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 6 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist  --cont_learner True
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 6 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10  --cont_learner True


#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 7 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist  --cont_learner True
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 7 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist  --cont_learner True
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 7 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10  --cont_learner True



#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 8 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist  --cont_learner True
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 8 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist  --cont_learner True
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 8 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10  --cont_learner True



#python main.py --dataset1 audiomnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 9 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/audiomnist  --cont_learner True
#python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 9 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/mnist  --cont_learner True
#python main.py --dataset1 cifar10 --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 9 --save_path  /home/gridsan/aejilemele/LIS/SHELS-main/models/cifar10  --cont_learner True


python create_tables.py


