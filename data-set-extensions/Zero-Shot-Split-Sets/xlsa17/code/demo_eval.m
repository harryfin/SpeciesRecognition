%
% Demo of how to use evaluate.m  
% Y. Xian,  B. Schiele, Z. Akata. 
% Zero-shot Learning - The Good, the Bad and the Ugly. IEEE CVPR 2017.
% Cite the above paper if you are using this code.
%
%
% Yongqin Xian
% e-mail: yxian@mpi-inf.mpg.de
% Computer Vision and Multimodal Computing, Max Planck Institute Informatics
% Saarbruecken, Germany
% http://d2.mpi-inf.mpg.de

% load the pretrained model and normalization factor

clear all
evaluate(@score_ale, 'SUN', 'ale_SUN_1e-1_50.mat');
evaluate(@score_ale, 'CUB', 'ale_CUB_1e-1_50.mat');
evaluate(@score_ale, 'AWA', 'ale_AWA_1e-3_50.mat');
evaluate(@score_ale, 'APY', 'ale_APY_1e-3_50.mat');
