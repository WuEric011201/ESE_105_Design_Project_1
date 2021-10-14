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

test=testOG;
%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(correctlabels==predictions)

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
% FILL IN
distances = vecnorm(((ones(size(centroids, 1), 1)*data)- centroids(:, 1:size(data, 2)) )' );
[vec_distance, index]=min(distances);

end

