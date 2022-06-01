# Species Recognition
##  Zero-shot learning for animal species recognition by usage of multimodal models - fine-granular visual classification with CLIP

As part of a student project, I investigated whether a fine-grained classification is possible on a multimodal model. As a starting point, the [CLIP](https://github.com/openai/CLIP) model was used and fine-tuned. In addition, I studied if CLIP can preserve its zero-shot learning capability after fine-tuning on fine-granular datasets.

To improve the performance of the multimodal model, I performed experiments involving the [SLIP](https://github.com/facebookresearch/SLIP) model, [robust fine-tuning of zero-shot models](https://github.com/mlfoundations/wise-ft) and my implementation for representation-aware fine-tuning.

For the experiments, I used data sets such as [CUB_200_2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) or [Eu-Moths](https://inf-cv.uni-jena.de/home/research/datasets/eu_moths_dataset/#:~:text=This%20dataset%20consists%20of%20200,total%2C%20there%20are%202205%20images.), and for zero-shot learning, on CUB, I used data splits of [Yongqin Xian et al](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly). 

Included are:
1. PyTorch custom dataset APIs for CUB-200-2011, MMC, Eu-Moths,
2. a trainer to train, log and test multimodal models with PyTorch,
3. notebooks and scripts for fine-tuning and testing CLIP models (focus on the possibility of applying various class settings for zero-shot classification). 

# Install

## Environment

Move to the folder that contains the environment file , create the environment and activate it.
```
conda env create -f environment.yml
conda activate spec_rec_env
```

## Modules/Packages 

Move into the folder containing ```setup.py``` and do:

```bash
pip install --editable .
```

The ```--editable``` flag, allowing to make changes to the code. 


If no changes are necessary:

``` bash
pip install .
```

# Requirements/Adaptations
+ Access to [wandb](https://wandb.ai/site) is required.
+ Download datasets ([see: Import dataset](#Dataset))
+ Hyperparameters can be changed in the configs.
  

# Datasets

Put the datasets in the following locations or change the paths in the datasets_roots folder:
+ [CUB_200_2011:](https://www.vision.caltech.edu/datasets/cub_200_2011/)  ```~/Datasets/CUB_200_2011```
+ [Eu-Moths dataset:](https://inf-cv.uni-jena.de/home/research/datasets/eu_moths_dataset/#:~:text=This%20dataset%20consists%20of%20200,total%2C%20there%20are%202205%20images.) ```~/Datasets/eu-moths/ORIGINAL```
+ [MMC dataset:](https://github.com/kimbjerge/MCC-trap) ```~/Datasets/mmc```


Example to import the CUB_200_2011 dataset:
```python
from datasetCUB.Cub_class.class_cub import Cub2011
root = "~/path/to/Datasets/CUB_200_2011"
cub_training =  Cub2011(root, train = True)
cub_test = Cub2011(root, train = False)
```
# Usage

1. create virtual environment
1. install project 
3. download datasets and adjust the paths in ``root_directories\`` 
4. execute scripts

# Libraries

Folders in src contain files that can be imported as libraries.

File tree:
```
📦src
 ┣ 📂datasetCUB
 ┃ ┣ 📂Cub_class
 ┃ ┃ ┣ 📜class_cub.py
 ┃ ┃ ┣ 📜class_cub_for_simclr.py
 ┃ ┃ ┣ 📜class_cub_mp_splits.py
 ┃ ┃ ┣ 📜class_cub_split_sets.py
 ┃ ┃ ┗ 📜class_cub_with_data_augmentation.py
 ┃ ┣ 📂transformations
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜image_transformation.py
 ┃ ┃ ┗ 📜label_transformation.py
 ┃ ┗ 📜label_maps_CUB.py
 ┣ 📂datasetEuMoths
 ┃ ┣ 📜__init__.py
 ┃ ┗ 📜eu_moths.py
 ┣ 📜__init__.py
 ┣ 📜args_import.py
 ┣ 📜loss_functions.py
 ┣ 📜trainer.py
 ┣ 📜utils.py
 ┣ 📜utils_clip.py
 ┣ 📜utils_cub.py
 ┣ 📜utils_eu_moth.py
 ┗ 📜weight_space_ensembling.py
```

# Testing
```bash
pytest --verbose tests/
```

