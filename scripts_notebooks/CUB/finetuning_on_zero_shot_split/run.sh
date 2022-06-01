#!/bin/bash

root="/path/to/repository/SpeciesRecognition"
gpu=0

script=zs-mp.py

# config paths
conf_path_rn='configs/config_rn50.yaml'
conf_path_vit='configs/config_vit32.yaml'

conf_path_rn_sweep='configs/config_sweep_rn50.yaml'
conf_path_vit_sweep='configs/config_sweep_vit32.yaml'


# ResNet

#base all no SS- and PS-Split
#python $script "$root" $conf_path_rn --wandb_log=yes --split=SS --simclr=no --robust_MSE=no --gpu=$gpu --add_to_run_name=base
#python $script "$root" $conf_path_rn --wandb_log=no --split=PS --simclr=no --robust_MSE=no --gpu=$gpu --add_to_run_name=base

# Proposed Split

#robust - mse
#python $script "$root" $conf_path_rn --wandb_log=yes --split=PS --simclr=no --robust_MSE=yes --gpu=$gpu --add_to_run_name=mse
#python $script "$root" $conf_path_rn --wandb_log=yes --split=PS --simclr=no --robust_MSE=yes --gpu=$gpu --add_to_run_name=mse_scale --clip_scale=0.2 --robust_scale=1.8

# simclr
#python $script "$root" $conf_path_rn --sweep=no --wandb_log=yes --split=PS --simclr=yes --simclr_preprocess=slip --gpu=$gpu --add_to_run_name=slip_preprocess
#python $script "$root" $conf_path_rn --sweep=no --wandb_log=yes --split=PS --simclr=yes --simclr_preprocess=own --gpu=$gpu --add_to_run_name=own_simclr_preprocess

# simple Data Augmentation in Training
python $script "$root" $conf_path_rn --sweep=no --wandb_log=no --split=PS --simclr=no --clip_data_augmentation=yes --gpu=$gpu --add_to_run_name=simple_data_aug_

#robust - wse
#python $script "$root" $conf_path_rn --wandb_log=yes --split=PS --simclr=no --robust_WSE_always=yes --alpha=0.9 --gpu=$gpu --add_to_run_name=wse_always_09
#python $script "$root" $conf_path_rn --wandb_log=yes --split=PS --simclr=no --robust_WSE_always=yes --alpha=0.5 --gpu=$gpu --add_to_run_name=wse_always_05
#python $script "$root" $conf_path_rn --wandb_log=yes --split=PS --simclr=no --robust_WSE_always=yes --alpha=0.1 --gpu=$gpu --add_to_run_name=wse_always_01
#python $script "$root" $conf_path_rn --wandb_log=yes --split=PS --simclr=no --robust_WSE_end=yes --alpha=0.9 --gpu=$gpu --add_to_run_name=wse_end_09
#python $script "$root" $conf_path_rn --wandb_log=yes --split=PS --simclr=no --robust_WSE_end=yes --alpha=0.5 --gpu=$gpu --add_to_run_name=wse_end_05
#python $script "$root" $conf_path_rn --wandb_log=yes --split=PS --simclr=no --robust_WSE_end=yes --alpha=0.1 --gpu=$gpu --add_to_run_name=wse_end_01

# #all
# python $script "$root" $conf_path_rn --wandb_log=yes --split=PS --simclr=yes --robust_MSE=yes --robust_WSE_always=yes --alpha=0.9 --gpu=$gpu

# Visual Transformer

#base all no SS and PS-Split
#python $script "$root" $conf_path_vit --wandb_log=yes --split=SS --simclr=no --robust_MSE=no --gpu=$gpu --add_to_run_name=base
#python $script "$root" $conf_path_vit --wandb_log=yes --split=PS --simclr=no --robust_MSE=no --gpu=$gpu --add_to_run_name=base

# Proposed Split

#robust - mse
#python zs-mp.py $conf_path_vit --wandb_log=yes --split=PS --simclr=no --robust_MSE=yes --gpu=$gpu --add_to_run_name=mse
#python zs-mp.py $conf_path_vit --wandb_log=yes --split=PS --simclr=no --robust_MSE=yes --gpu=$gpu --add_to_run_name=mse_scale --clip_scale=0.2 --robust_scale=1.8

# simclr
#python zs-mp.py $conf_path_vit --sweep=no --wandb_log=yes --split=PS --simclr=yes --simclr_preprocess=slip --gpu=$gpu --add_to_run_name=slip_preprocess
#python zs-mp.py $conf_path_vit --sweep=no --wandb_log=yes --split=PS --simclr=yes --simclr_preprocess=own --gpu=$gpu --add_to_run_name=own_simclr_preprocess

# simple Data Augmentation in Training
#python zs-mp.py $conf_path_vit --sweep=no --wandb_log=yes --split=PS --simclr=no --clip_data_augmentation=yes --gpu=$gpu --add_to_run_name=simple_data_aug_

#robust - wse
#python zs-mp.py $conf_path_vit --wandb_log=yes --split=PS --simclr=no --robust_WSE_always=yes --alpha=0.9 --gpu=$gpu --add_to_run_name=wse_always_09
#python zs-mp.py $conf_path_vit --wandb_log=yes --split=PS --simclr=no --robust_WSE_always=yes --alpha=0.5 --gpu=$gpu --add_to_run_name=wse_always_05
#python zs-mp.py $conf_path_vit --wandb_log=yes --split=PS --simclr=no --robust_WSE_always=yes --alpha=0.1 --gpu=$gpu --add_to_run_name=wse_always_01
#python zs-mp.py $conf_path_vit --wandb_log=yes --split=PS --simclr=no --robust_WSE_end=yes --alpha=0.9 --gpu=$gpu --add_to_run_name=wse_end_09
#python zs-mp.py $conf_path_vit --wandb_log=yes --split=PS --simclr=no --robust_WSE_end=yes --alpha=0.5 --gpu=$gpu --add_to_run_name=wse_end_05
#python zs-mp.py $conf_path_vit --wandb_log=yes --split=PS --simclr=no --robust_WSE_end=yes --alpha=0.1 --gpu=$gpu --add_to_run_name=wse_end_01