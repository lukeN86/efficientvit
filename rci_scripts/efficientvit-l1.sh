#!/bin/bash
#SBATCH --job-name=efficientvit
#SBATCH --time=196:00:00
#SBATCH --partition=amdgpuextralong
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --output=/home/neumalu1/data/efficientvit/efficientvit_%j.out
# -------------------------------

ml fosscuda/2020b
ml Anaconda3

source activate efficientvit
export OMP_NUM_THREADS=14
torchrun --nnodes 1 --nproc_per_node=4 train_cls_model.py configs/cls/imagenet/l1.yaml --distributed --amp bf16 --data_provider.base_batch_size 256 --path ./exp/cls/imagenet/l1_b256/



