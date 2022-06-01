Data and code for
[1] Zero-shot Learning - The Good, the Bad and the Ugly. Y. Xian, B. Schiele, Z. Akata. IEEE CVPR 2017.
[2] Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly. Y. Xian, C. Lampert, B. Schiele, Z. Akata. arXiv preprint arXiv:1707.00600
Cite the above paper if you are using this code.


evaluate.m
=================

Code for evaluating your method. Please check the code for more details


demo_eval.m
=================

Demo code for using evaluate.m


score_ale.m
=================

Score function for ale. 


Data
=================

We released both our proposed splits(PS) and resNet101 features of CUB, SUN, AWA1, AWA2 and APY datasets. 

The proposed splits consist of:
-allclasses.txt: list of names of all classes in the dataset
-trainvalclasses.txt: seen classes
-testclasses.txt: unseen classes
-trainclasses1/2/3.txt: 3 different subsets of trainvalclasses used for tuning the hyperparameters 
-valclasses1/2/3.txt: 3 different subsets of trainvalclasses used for tuning the hyperparameters


resNet101.mat includes the following fields:
-features: columns correspond to image instances
-labels: label number of a class is its row number in allclasses.txt
-image_files: image sources  


att_splits.mat includes the following fields:
-att: columns correpond to class attribute vectors normalized to have unit l2 norm, following the classes order in allclasses.txt 
-original_att: the original class attribute vectors without normalization
-trainval_loc: instances indexes of train+val set features (for only seen classes) in resNet101.mat
-test_seen_loc: instances indexes of test set features for seen classes
-test_unseen_loc: instances indexes of test set features for unseen classes

For ImageNet, we released our train/val/test class splits. Thanks to Wei-Lun Chao, Word2Vec can be downloaded at his github https://github.com/pujols/zero-shot-learning.
Regarding to the image features, we use the pretrained ResNet-101 model to extract ResNet features. No cropping is applied. 
Pretrained ResNet-101 model (caffe) can be downloaded here:
http://datasets.d2.mpi-inf.mpg.de/xian/Kaiming-ResNet-101.zip

CONTACT:
=================
Yongqin Xian
e-mail: yxian@mpi-inf.mpg.de
Computer Vision and Multimodal Computing, Max Planck Institute Informatics
Saarbruecken, Germany
http://d2.mpi-inf.mpg.de
