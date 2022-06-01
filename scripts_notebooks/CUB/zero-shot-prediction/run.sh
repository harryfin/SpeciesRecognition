#!/bin/bash

root='path/to/repository/SpeciesRecognition'
conf_path_vit='/config_vit32.yaml'
python zero-shot-prediction.py $root $conf_path_vit --sweep=no --wandb_log=yes

#conf_path_rn='/config_rn50.yaml'
#python zero-shot-prediction.py $root $conf_path_rn --sweep=no --wandb_log=yes