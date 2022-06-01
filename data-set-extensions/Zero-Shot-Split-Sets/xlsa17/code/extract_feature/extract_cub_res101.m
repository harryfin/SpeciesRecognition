addpath('/BS/Deep_Fragments/work/MSc/caffe/matlab/');
cafferoot = '/BS/Deep_Fragments/work/MSc/caffe/';

datadir = '/BS/Deep_Fragments/work/MSc/CUB_200_2011/CUB_200_2011';
birdpath = [ datadir '/images' ];
birddirs = dir(birdpath);
fidx = 0;
cur_label = 1;
labels = zeros(100000,1);
image_files = cell(100000,1);
for d = 1:numel(birddirs),
    if strcmp(birddirs(d).name,'.') || strcmp(birddirs(d).name, '..'),
        continue;
    end
    fdir = [ birdpath '/' birddirs(d).name ];
    files = dir([fdir '/*jpg']);
    for f = 1:numel(files),
        fname = files(f).name;
        fpath = [ fdir '/' fname ];
        fidx = fidx + 1;
        image_files{fidx} = fpath;
        labels(fidx) = cur_label;
    end
    cur_label = cur_label + 1;
    fprintf('Reading dir %d of %d...\n', d, numel(birddirs));
end
image_files = image_files(1:fidx);
labels = labels(1:fidx);
mn = load('VGG_mean.mat');
model_file = [cafferoot '/models/residual_network/ResNet-101-model.caffemodel'];

% Init matcaffe.
net_file = [cafferoot '/models/residual_network/ResNet-101-deploy.prototxt'];
% use_gpu = 0;
% caffe('init', net_file, model_file, 'test');
% caffe('set_mode_gpu');

caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
phase = 'test';

net = caffe.Net(net_file, model_file, phase);
[ features, err_mask] = extract_feature(image_files, mn.image_mean, net);
labels = labels(find(~err_mask));
image_files = image_files(find(~err_mask));
% classnames = cell(size(features,1),1);
% imagenames = cell(size(features,1),1);
% for f = 1:size(features,2),
%     tmp = regexp(image_files{2+f}, '/','split');
%     classnames{f} = tmp{end-1};
%     imagenames{f} = tmp{end};
% end
% write_in_svmlight_CUB(features, classnames, imagenames);
