#!/bin/bash

root="/path/to/repository/SpeciesRecognition"
gpu=0

script=zs_finetuning.py

# config paths
conf_path_rn='configs/config_rn50.yaml'
conf_path_vit='configs/config_vit32.yaml'

# biological labels
python $script "$root" $conf_path_rn --wandb_log=yes --label=biological --testsetting=zsl --add_to_run_name=bio-standard
#python $script "$root" $conf_path_vit --wandb_log=yes --label=biological --testsetting=zsl --add_to_run_name=bio-standard

# english labels
#python $script "$root" $conf_path_vit --wandb_log=yes --label=english --testsetting=zsl --add_to_run_name=eng-standard
#python $script "$root" $conf_path_rn --wandb_log=yes --label=english --testsetting=zsl --add_to_run_name=eng-standard