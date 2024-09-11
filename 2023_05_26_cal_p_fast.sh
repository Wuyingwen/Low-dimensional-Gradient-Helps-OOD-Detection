#!/bin/bash
# bash 2023_05_26_cal_p_fast.sh 264 25 500 50000
start_k=$1
interval_k=$2
k=$3
sample_num=$4

nproc_per_node=4
batch_size=10
# model='wideresnet' 
# model='wrt'
model='resnet18'
data='cifar10'
# model='resnet18'
# data='cifar100'
kernel='NFK' 
port='12433'

for ((i=start_k; i<k; i=i+interval_k))
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python 2023_05_26_cal_p_fast.py --nproc_per_node $nproc_per_node --batch_size $batch_size --model $model --data $data --kernel $kernel --k $k --start_k $i --interval_k $interval_k --sample_num $sample_num --port $port
done

# for ((i=start_k; i<k; i=i+interval_k))
# do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python 2023_05_26_cal_p_fast_2.py --nproc_per_node $nproc_per_node --batch_size $batch_size --model $model --data $data --kernel $kernel --k $k --start_k $i --interval_k $interval_k --sample_num $sample_num --port $port
# done