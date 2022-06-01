#!/bin/bash

root="/path/to/repository/SpeciesRecognition"
gpu=0

script=zero-shot-prediction.py


#config paths
conf_path_rn='configs/config_rn50.yaml'
conf_path_vit='configs/config_vit32.yaml'


# english labels
#python $script "$root" $conf_path_vit --wandb_log=yes --project_name=zsp_on_eu_moths --label=english --gpu=$gpu
#python $script "$root" $conf_path_rn --wandb_log=yes --project_name=zsp_on_eu_moths --label=english --gpu=$gpu

# biological labels
python $script "$root" $conf_path_vit --wandb_log=no --project_name=zsp_on_eu_moths --label=biological --gpu=$gpu
#python $script "$root" $conf_path_rn --wandb_log=yes --project_name=zsp_on_eu_moths --label=biological --gpu=$gpu
