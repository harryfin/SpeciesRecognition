#!/bin/bash

root="/path/to/repository/SpeciesRecognition"

# Configs
conf_path_vit='configs/config_vit32.yaml'
conf_path_rn='configs/config_rn50.yaml'

script=cub-finetuning.py

python $script "$root" $conf_path_vit --wandb_log=no --gpu=0
#python $script "$root" $conf_path_rn --gpu=0

