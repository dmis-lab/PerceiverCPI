#!/bin/bash 
echo "Please enter the GPU address"
read GPU
echo "Sellected GPU: $GPU"

#train
CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --data_path '' --separate_val_path '' --separate_test_path '' --metric mse --dataset_type regression --save_dir '' --target_columns label --epochs 150 --ensemble_size 2 --num_folds 1 --batch_size 50 --aggregation mean --dropout 0.1 --save_preds >& out_name.out &
#take inference
CUDA_VISIBLE_DEVICES=$GPU python predict.py --test_path '' --checkpoint_dir '' --preds_path ''