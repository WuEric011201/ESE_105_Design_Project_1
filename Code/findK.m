
clear all;
close all;

%% In this script, you need to implement three functions as part of the k-means algorithm.
% These steps will be repeated until the algorithm converges:

  % 1. initialize_centroids
  % This function sets the initial values of the centroids
  
  % 2. assign_vector_to_centroid
  % This goes through the collection of all vectors and assigns them to
  % centroid based on norm/distance
  
  % 3. update_centroids
  % This function updates the location of the centroids based on the collection
  % of vectors (handwritten digits) that have been assigned to that centroid.


%% Initialize Data Set
% These next lines of code read in two sets of MNIST digits that will be used for training and testing respectively.

% training set (1500 images)
train=csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train=train(:,1:784);
train(:,785)=zeros(1500,1);

% testing set (200 images with 11 outliers)
test=csvread('mnist_test_200_woutliers.csv');
% store the correct test labels
correctlabels = test(:,785);
test=test(:,1:784);

% now, zero out the labels in "test" so that you can use this to assign
% your own predictions and evaluate against "correctlabels"
% in the 'cs1_mnist_evaluate_test_set.m' script
test(:,785)=zeros(200,1);

%% After initializing, you will have the following variables in your workspace:
% 1. train (a 1500 x 785 array, containins the 1500 training images)
% 2. test (a 200 x 785 array, containing the 200 testing images)
% 3. correctlabels (a 200 x 1 array containing the correct labels (numerical
% meaning) of the 200 test images

%% To visualize an image, you need to reshape it from a 784 dimensional array into a 28 x 28 array.
% to do this, you need to use the reshape command, along with the transpose
% operation.  For example, the following lines plot the first test image

figure;
colormap('gray'); % this tells MATLAB to depict the image in grayscale
testimage = reshape(test(89,[1:784]), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.

%% After importing, the array 'train' consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.

%% This next section of code calls the three functions you are asked to specify

max_iter= 10; % set the number of iterations of the algorithm
repeat = 20;

maxK = 40;


trainOG = train;
testOG= test;

train = preprocess(train);
test = preprocess(test);
%% The next line initializes the centroids.  Look at the initialize_centroids()
% function, which is specified further down this file.

%% Initialize an array that will store k-means cost at each iteration

cost_iteration = zeros(max_iter, 1);

%% This for-loop enacts the k-means algorithm
% First get rid of all the lightly colored squares to reduce noise

trainOG = train;
testOG= test;

train = preprocess(train);
test = preprocess(test);

distances = zeros(maxK-9, 1);

for k=10: maxK
centroids=initialize_centroids(train,k);
 
best_centroids = centroids;
 
lowestDistance = Inf;

centroid_labels = zeros(k, 1);
% Repeat  this repeat number of times
for rep = 1:repeat
    % disp(rep)
    stuck = 0;
    while ~isequal(ismember([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], centroid_labels), ones(1, 10)) && stuck< 20
        cost_iteration = zeros(max_iter, 1);

        
        for iter=1:max_iter
        %centroids(centroids<100) = 0;
        % FILL THIS IN!
            for i=1: size(train, 1)
                [train(i, 785), distance] = assign_vector_to_centroid(train(i, 1:784), centroids);
                cost_iteration(iter, 1)=cost_iteration(iter, 1)+distance;
            end
    
            centroids = update_Centroids(train, k);
        end
        centroid_labels=label_Data(train, centroids, trainsetlabels);
        
        stuck = stuck +1;
    end

    if cost_iteration(max_iter)< lowestDistance
        lowestDistance = cost_iteration(max_iter, 1);
        best_centroids = centroids;
    end
    
end

centroids=best_centroids;

distances(k-9, 1) = lowestDistance;


end

figure;
plot(10: maxK, distances, '-o');
title("Cost vs K value");
xlabel("K");
ylabel("Total Cost");

%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids
% There are other ways you might initialize centroids.
% ***Feel free to experiment.***
% Note that this function takes two inputs and emits one output (y).

function y=initialize_centroids(data,num_centroids)

centroids = zeros(num_centroids, size(data, 2));

random_index=randperm(size(data,1));

centroids(1, :)=data(random_index(1),:);

% data(random_index(1), :)=[];

for i = 2: num_centroids
    
    [index, ~] =find_farthest_distance(data(:, 1:784), centroids(1:i-1, :));
    %disp(index)
    centroids(i, :) = data(index, :);
end

y=centroids;

% centroids = zeros(num_centroids, size(data, 2));
% 
% counter = 0;
% 
% for i = 1:num_centroids
%     cluster = find(trainsetlabels==counter);
%     random_index=randperm(size(cluster, 1));
%     
%     centroids(i) = data(cluster(random_index(1)), :);
%     counter = counter+1;
%     if counter>9
%         counter=0;
%     end
% end
% 
% y=centroids;

% random_index=randperm(size(data,1));
% 
% centroids=data(random_index(1:num_centroids),:);
% 
% y=centroids;

end

%% Function to find farthest data point using norm/ distance for k meanas++

function [index, vec_distance] = find_farthest_distance(data, centroids)

% Distances is # cen x # data
distances = zeros(size(centroids, 1), size(data, 1));
for i = 1: size(centroids, 1)
    for j = 1: size(data, 1)
        distances(i, j) = norm(centroids(i, 1:size(data, 2)) - data(j, :));
    end
end
% distances
distance = sum(distances);

[vec_distance, index] = max(distance);
% distances = vecnorm(( ( ones(size(data, 1), 1)*centroid(1, 1:size(data, 2)) )- data )' );
% [vec_distance, index] = max(distances);

end

%% Function to pick the Closest Centroid using norm/distance
% This function takes two arguments, a vector and a set of centroids
% It returns the index of the assigned centroid and the distance between
% the vector and the assigned centroid.

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
% FILL THIS IN

% Assume the length of the centroids is at least 1
% Assign the closest centroid and distance to be that of the first
% closest=1;
distances = vecnorm(((ones(size(centroids, 1), 1)*data)- centroids(:, 1:size(data, 2)) )' );
[vec_distance, index]=min(distances);

end
%% Function to compute new centroids using the mean of the vectors currently assigned to the centroid.
% This function takes the set of training images and the value of k.
% It returns a new set of centroids based on the current assignment of the
% training images.

function new_centroids=update_Centroids(data,K)
% FILL THIS IN

% Find all vectors in cluster i
cluster = data( :,785);
tempCen = zeros(K, size(data, 2));
for i=1: K
    clusterI = data(cluster==i, 1:784);
    tempCen(i, 1:784)= sum(clusterI, 1)/(size(clusterI, 1));
end
new_centroids = tempCen;
end

%% Function to label the centroids by finding the most common label in that centroid

function centroid_labels = label_Data(data, centroids, trainingLabels)
cluster = data( :,785);

K = size(centroids, 1);

labels = zeros(K, 1);

for j=1: K
    labelsI = trainingLabels(cluster==j, 1);
    
    labels(j, 1)=mode(labelsI);
end
centroid_labels=labels;
end

%% Different preprocessing methods

function modifiedData= preprocess(inputData)
    threshold =150;

    modifiedData= inputData;
    
    modifiedData(modifiedData<threshold) = 0;
    modifiedData(modifiedData>=threshold)=255;    
    
    
    
    for count = 1: size(modifiedData, 1)
        imageI = reshape(modifiedData(count,[1:784]), [28 28]);
        
%         element = strel('square', 1);
%         imageI = imerode(imageI, element);
        

        
        element = strel('square', 3);
        binaryImage = imbinarize(imageI);
        errodedImage = bwskel(binaryImage);
        a=zeros(size(errodedImage));
        a(errodedImage==1) = 255;
        imageI=imdilate(a, element);
    

        
        if(count ==9)
            figure;
            colormap('gray');
            imagesc(imageI);
        end

        % Shave excess top pixels
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
                        
        %size(imageI)
        
%         while size(imageI, 2) < size(imageI, 1)
%             imageI = [imageI, zeros(size(imageI, 1), 1)];
%         end
%         while size(imageI, 1) < size(imageI, 2)
%             imageI = [zeros(1, size(imageI, 2)); imageI];
%         end
        
        if size(imageI, 2)/size(imageI, 1) >3
            if size(imageI, 2)<size(imageI, 1)
                imageI = [imageI, zeros(size(imageI, 1), size(imageI, 1)-size(imageI, 2))];
            end
            if size(imageI, 2)>size(imageI, 1)
                imageI = [imageI; zeros(size(imageI, 2)-size(imageI, 1), size(imageI, 2))];
            end
        end
        
        %size(imageI)
        
        imageI = imresize(imageI, [28 28]);

        
        imageI(imageI<threshold)=0;
        imageI(imageI>=threshold)=255;
        
        modifiedData(count, 1:784) = reshape(imageI, [1 784]);

    end
end