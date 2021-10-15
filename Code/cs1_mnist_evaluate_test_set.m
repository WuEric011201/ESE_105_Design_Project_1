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

% Do same preprocess
testOG= test;
test = preprocess(test);


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

%% Modification function from before
function modifiedData= preprocess(inputData)
    % First do a thresholding opperation. All pixels lighter than 150 are
    % removed, reducing noise
    threshold =150;
    modifiedData= inputData;
    modifiedData(modifiedData<threshold) = 0;
    modifiedData(modifiedData>=threshold)=255;    
    
    for count = 1: size(modifiedData, 1)
        % Reshape every row into their image
        imageI = reshape(modifiedData(count,[1:784]), [28 28]);
       
        % Perform a errosion using bwskel. Only the basic shape is left
        % after this opperation
        element = strel('square', 3);
        binaryImage = imbinarize(imageI);
        errodedImage = bwskel(binaryImage);
        a=zeros(size(errodedImage));
        a(errodedImage==1) = 255;
        % Rethicken the thin outline with a dilation using a 3 square
        imageI=imdilate(a, element);
        

        % Shave excess top pixels by removing all black 0 rows.
        while imageI(1, :) == zeros(1, size(imageI, 2))
            imageI(1,:) = [];
        end
        
        % Shave excess left pixels
        while imageI(:, 1) == zeros(size(imageI, 1), 1)
            imageI(:, 1) = [];
        end
        
        % Shave excess right pixels
        while imageI(:, size(imageI, 2)) == zeros(size(imageI, 1), 1)
            imageI(:, size(imageI, 2)) = [];
        end
       %Shave excess bottom pixels
        while imageI(size(imageI, 1), :) == zeros(1, size(imageI, 2))
            imageI(size(imageI, 1), :)=[];
        end
        
        % Doing this specifically for exactly vertical ones. Re add the
        % black excess space if it is a very vertical one (or else they get
        % stretched too much)
        if size(imageI, 2)/size(imageI, 1) >3
            if size(imageI, 2)<size(imageI, 1)
                imageI = [imageI, zeros(size(imageI, 1), size(imageI, 1)-size(imageI, 2))];
            end
            if size(imageI, 2)>size(imageI, 1)
                imageI = [imageI; zeros(size(imageI, 2)-size(imageI, 1), size(imageI, 2))];
            end
        end
        
        % Remake the images back into 28x28 after removing pixels
        imageI = imresize(imageI, [28 28]);

        % Reperform the threshold
        imageI(imageI<threshold)=0;
        imageI(imageI>=threshold)=255;
        
        % Save it back into modified data.
        modifiedData(count, 1:784) = reshape(imageI, [1 784]);

    end
end