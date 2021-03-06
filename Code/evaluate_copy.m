%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace


% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.

predictions = zeros(200,1);
outliers = zeros(200,1);

distances = zeros(200, 1);

test(test<128) = 0;
test(test>=128)=255;

% going to try eroding test using a up down structuring element to reduce
% thickness. Some of the zeros are really thick.
testMod2 = test;

% for count =1: size(test, 1)
%     imageI = reshape(test(count, [1:784]), [28 28]);
%     element = strel('square', 2);
%     dilatedImage = imdilate(imageI, element);
%     test(count, 1:784) = reshape(dilatedImage, [1 784]);
% end
% 
for count = 1: size(test, 1)
    imageI = reshape(test(count,[1:784]), [28 28]);
    
%     theta = 0;
%     %rotate image
%     for i=1: size(imageI, 1)
%         for j =1: size(imageI, 2)
%             if imageI(i, j) >=128
%                 thisThet = atan( (j-size(imageI, 2) )/ (size(imageI, 1) - i));
%                 theta = [theta, thisThet];
%             end
%         end
%     end
%     %average theta
%     avThet = sum(theta)/(length(theta)-1);
%     imageI = imrotate(imageI, avThet);
%     imageI = imresize(imageI, [28 28]);
    
    % Dilate then erode
    element = strel('square', 2);
    dilated = imdilate(imageI, element);
    erroded = imerode(dilated, element);
    test(count, 1:784) = reshape(erroded, [1 784]);


    % Erode then dilate
%     element = strel('square', 2);
%     binaryImage = imbinarize(imageI);
%     errodedImage = bwskel(binaryImage);
%     skeleton = reshape(errodedImage, [1 784]);
%     a=zeros(size(skeleton));
%     a(skeleton==1) = 256;
%     a=imdilate(a, element);
%     test(count, 1:784) = a;
% 
%     element = strel('square', 2);
%     erroded = imerode(imageI, element);
%     dilated = imdilate(erroded, element);
%     test(count, 1:784) = reshape(dilated, [1 784]);

end

% loop through the test set, figure out the predicted number
for i = 1:200

testing_vector=test(i,:);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);
distances(i) = vec_distance;
predictions(i) = centroid_labels(prediction_index);

end


%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
% FILL IN
iqRange = iqr(distances);
sevenfive = prctile(distances, 75);
twentyfive = prctile(distances, 25);
outliers(distances>1.5*iqRange+sevenfive| distances<twentyfive-1.5*iqRange) =1;

outlierIndexes = find(outliers==1);
%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
% FILL IN
stem(1:200, outliers);

%% Plot the outliers
figure;

colormap('gray');

len = length(outlierIndexes);
plotsize = ceil(sqrt(len));

for ind=1:len
    
    centr=test(outlierIndexes(ind),[1:784]);
    subplot(plotsize,plotsize,ind);
    
    imagesc(reshape(centr,[28 28])');
    title(strcat('Outlier ',num2str(ind)))

end

%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o');
hold on;
plot(predictions,'x');
title('Predictions');

test=testMod2;
%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(correctlabels==predictions)

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
% FILL IN
distances = vecnorm(((ones(size(centroids, 1), 1)*data)- centroids(:, 1:size(data, 2)) )' );
[vec_distance, index]=min(distances);

end

