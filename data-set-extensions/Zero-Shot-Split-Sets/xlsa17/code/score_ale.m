function S = score_ale(X,  att, model_path)
%
% Score function of ALE in  
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
load(model_path, 'W', 'xtest_max', 'xtest_mean', 'xtest_variance');
% normalization
X = bsxfun(@rdivide, bsxfun(@minus, X, xtest_mean), xtest_variance);
X = X / xtest_max;
    
S = X * W * att;