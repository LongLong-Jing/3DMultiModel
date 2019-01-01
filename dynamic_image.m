% Read Image First
frame_list = dir('./test_frames/');
[frame_num,] = size(frame_list);
Clip_length = 16;
video_clip = zeros(112,112,3,Clip_length,'double'); 
for index = 3:1:frame_num  
    %skip the current folder and the parent folder
    frame_name = frame_list(index).name;
    img = imread(strcat('./test_frames/',frame_name));
    img = imresize(img,[112 112]);
    img = im2double(img);
    video_clip(:,:,:,index-2) = img(:,:,:); 
end
dynamic_img = RankPooling(video_clip);
figure(1)
imshow(dynamic_img);


function NewImg = RankPooling(X)
% approximate rank pooling
% ids indicates frame-video association (must be in range [1-N])
    ids = ones(1,size(X,4));
    nVideos = max(ids);   % The number of video
    for v=1:nVideos       % The number of videos
        % pool among frames
        indv = find(ids==v);   % The num of frame of every video
        if isempty(indv)
            error('Error: No frames in video %d',v);
        end
        N = numel(indv);
        % magic numbers
        fw = zeros(1,N);
        if N==1
            fw = 1;
        else
        for i=1:N
          fw(i) = sum((2*(i:N)-N-1) ./ (i:N));
        end
        end
        a = sum(bsxfun(@times,X(:,:,:,indv),...
        reshape(single(fw),[1 1 1 numel(indv)])),4);
    
        % !!!The dynamic image must be normalized to the rang of [0,1], Or the visualization will be not correct.!!!
        % !!! The following code is for normalization!!!
        %A1 A2 A3 is the three channels of the dynamic image
        A1 = a(:,:,1);
        A2 = a(:,:,2);
        A3 = a(:,:,3);
        
        % The first step is to subtract the mean of the channel for each pixels
        A1 = A1 - mean(A1(:));
        A2 = A2 - mean(A2(:));
        A3 = A3 - mean(A3(:));

        %The second step is to normalize the pixel value to the range of [0,1]
        [Max,I] = max(A1(:));
        [Min,I] = min(A1(:));
        A1 = (A1 - Min)/(Max - Min);

        [Max,I] = max(A2(:));
        [Min,I] = min(A2(:));
        A2 = (A2 - Min)/(Max - Min);

        [Max,I] = max(A3(:));
        [Min,I] = min(A3(:));
        A3 = (A3 - Min)/(Max - Min);
        NewImg = zeros(size(X,1),size(X,2),size(X,3),'double');
        NewImg(:,:,1) = A1;
        NewImg(:,:,2) = A2;
        NewImg(:,:,3) = A3;
    end
end