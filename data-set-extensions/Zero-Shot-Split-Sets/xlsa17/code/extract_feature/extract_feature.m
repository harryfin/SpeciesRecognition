function [F, err_mask] = extract_feature(image_files, mn, net)

    IMAGE_DIM = 224;

    num_images = length(image_files);
    %F = zeros(4096, num_images);
    %F = zeros(1024, num_images);
    F = zeros(2048, num_images);
    err_mask = zeros(numel(image_files), 1);

    % Feed forward batches of 10.
    batchsize = 10;
    fidx = 1;
    fidx_next = fidx;


    while true,
        imgs = zeros(IMAGE_DIM,IMAGE_DIM,3,batchsize, 'single');
        batch_idx = 1;
        for b = 1:batchsize+100000,
            fprintf(1,'im %d of %d [%s]\n', fidx_next, num_images, image_files{fidx_next});
            try
                tmp = imread(image_files{fidx_next});
            catch
                disp('Non-RGB image!!!!!!!');
                err_mask(fidx_next) = 1;
                fidx_next = fidx_next + 1;
                continue; 
            end
            imgs(:,:,:,batch_idx) = prepare(tmp, mn);
            batch_idx = batch_idx + 1;
            fidx_next = fidx_next + 1;
            if (fidx_next > num_images || batch_idx > batchsize ) 
                break;
            end
        end
        %fea = caffe('forward',{imgs});
        fea = net.forward({imgs});
        tmp = squeeze(fea{1});
        %tmp = reshape(tmp, [ size(tmp,1)*size(tmp,2)*size(tmp,3), batchsize ]);
        F(:,fidx:(fidx+batch_idx-2)) = tmp(:,1:batch_idx-1);
        fidx = fidx+batch_idx-1;
        if (fidx_next > num_images),
            break;
        end
    end
end

function im = prepare(im, data_mean)
  if size(im,3)==1,
      im = repmat(im,[1,1,3]);
  end
  IMAGE_DIM = 224;
  % convert from uint8 to single
  im = single(im);
  % reshape to a fixed size (e.g., 227x227)
  %im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
  im = imresize(im, [IMAGE_DIM IMAGE_DIM]);
  % permute from RGB to BGR and subtract the data mean (already in BGR)
  im = im(:,:,[3 2 1]) - data_mean; %imresize(data_mean, [224 224]);
  % flip width and height to make width the fastest dimension
  im = permute(im, [2 1 3]);
end
