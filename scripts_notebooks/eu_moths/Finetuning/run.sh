#!/bin/bash

root="/path/to/repository/SpeciesRecognition"
gpu=0

script=finetuning_or_sweep.py

sweep=yes

# config paths
conf_path_rn='configs/config_rn50.yaml'
conf_path_vit='configs/config_vit32.yaml'

conf_path_rn_sweep='configs/config_sweep_rn50.yaml'
conf_path_vit_sweep='configs/config_sweep_vit32.yaml'

# english labels
#python $script "$root" $conf_path_vit --sweep=no --wandb_log=yes --project_name=finetuning_on_eu_moths --label=english --gpu=$gpu
#python $script "$root" $conf_path_rn --sweep=no --wandb_log=yes --project_name=finetuning_on_eu_moths --label=english --gpu=$gpu

# biological labels
python $script "$root" $conf_path_vit --sweep=no --wandb_log=yes --project_name=finetuning_on_eu_moths --label=biological --gpu=$gpu
#python $script "$root" $conf_path_rn --sweep=no --wandb_log=yes --project_name=finetuning_on_eu_moths --label=biological --gpu=$gpu

#sweep
#python $script "$root" $conf_path_rn_sweep --sweep=yes --wandb_log=yes --project_name=finetuning_on_eu_moths --label=english --gpu=$gpu
