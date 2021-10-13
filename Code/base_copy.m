
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
testimage = reshape(test(121,[1:784]), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.

%% After importing, the array 'train' consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.

%% This next section of code calls the three functions you are asked to specify

%k= 15; % set k
max_iter= 10; % set the number of iterations of the algorithm
repeat = 15;

maxK=30;
%% The next line initializes the centroids.  Look at the initialize_centroids()
% function, which is specified further down this file.

% centroids=initialize_centroids(train,k);
% 
% best_centroids = centroids;
% 
% lowestDistance = Inf;
%% Initialize an array that will store k-means cost at each iteration

cost_iteration = zeros(max_iter, 1);

%% This for-loop enacts the k-means algorithm
train(train<128) = 0;
train(train>=128)=255;

for count = 1: size(train, 1)
    imageI = reshape(train(count,[1:784]), [28 28]);
    
%     theta = 0;
%     %rotate image
%     for i=1: size(imageI, 1)
%         for j =1: size(imageI, 2)
%             if imageI(i, j) >=128
%                 thisThet = atan( (j-size(imageI, 2) )/ (size(imageI, 1) - i));
%                 if thisThet>pi
%                     thisThet=thisThet-pi;
%                 end
%                 theta = [theta, thisThet];
%             end
%         end
%     end
%     %average theta
%     avThet = sum(theta)/(length(theta)-1);
%     imageI = imrotate(imageI, avThet);
%     imageI = imresize(imageI, [28 28]);
%     



%     element = strel('square', 2);
%     binaryImage = imbinarize(imageI);
%     errodedImage = bwskel(binaryImage);
%     skeleton = reshape(errodedImage, [1 784]);
%     a=zeros(size(skeleton));
%     a(skeleton==1) = 256;
%     a=imdilate(a, element);
%     train(count, 1:784) = a;

    element = strel('square', 2);
    dilated = imdilate(imageI, element);
    erroded = imerode(dilated, element);
    train(count, 1:784) = reshape(erroded, [1 784]);

%     element = strel('square', 2);
%     erroded = imerode(imageI, element);
%     dilated = imdilate(erroded, element);
%     train(count, 1:784) = reshape(dilated, [1 784]);
end

trainOG = train;

kdistances = zeros(maxK-9, 1);
for k = 10: maxK
centroids=initialize_centroids(train,k);

best_centroids = centroids;

lowestDistance = Inf;
    
    
for rep = 1:repeat
    cost_iteration = zeros(max_iter, 1);
    for iter=1:max_iter
        centroids(centroids<100) = 0;
        % FILL THIS IN!
        for i=1: size(train, 1)
            [train(i, 785), distance] = assign_vector_to_centroid(train(i, 1:784), centroids);
            cost_iteration(iter)=cost_iteration(iter)+distance;
        end
    
        centroids = update_Centroids(train, k);

    end
    if cost_iteration(max_iter)< lowestDistance
        lowestDistance = cost_iteration(max_iter);
        best_centroids = centroids;
    end

end
kdistances(k-9)=lowestDistance;

end

figure;
plot(10:maxK, kdistances, '-s');
title("K vs Lowest distance");
xlabel("K value");
ylabel("Lowest distance");


centroids=best_centroids;

%centroids(centroids<100) = 0;


centroid_labels=label_Data(train, centroids, trainsetlabels);
%% This section of code plots the k-means cost as a function of the number
% of iterations
% 
% figure;
% % FILL THIS IN!
% axis auto;
% plot(1:max_iter, cost_iteration);
% 
% %% This next section of code will make a plot of all of the centroids
% % Again, use help <functionname> to learn about the different functions
% % that are being used here.
% 
% figure;
% colormap('gray');
% 
% plotsize = ceil(sqrt(k));
% 
% for ind=1:k
%     
%     centr=centroids(ind,[1:784]);
%     subplot(plotsize,plotsize,ind);
%     
%     imagesc(reshape(centr,[28 28])');
%     title(strcat('Centroid ',num2str(ind)))
% 
% end

%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids
% There are other ways you might initialize centroids.
% ***Feel free to experiment.***
% Note that this function takes two inputs and emits one output (y).

function y=initialize_centroids(data,num_centroids)

random_index=randperm(size(data,1));

centroids=data(random_index(1:num_centroids),:);

y=centroids;

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
% Squared because Gaussian noise? IDK squared norm
distances = vecnorm(((ones(size(centroids, 1), 1)*data)- centroids(:, 1:size(data, 2)) )' );
%disp((centroids(:, size(data, 2))-(ones(size(centroids, 1), 1)*data))')
%size((centroids(:, 1:size(data, 2))))
% disp(distances)

[vec_distance, index]=min(distances);
% disp(index)
% disp(vec_distance)
% closest=1;
% distance = norm(centroids(1, :)-data)^2;
% disp(distance)
% % multiply 1 by data, get it same dimension as centroids then calculate
% % norm between everything
% 
% % 
% % Iterate through all the centroids, starting at 2
% for i=2: size(centroids, 1)
%     currentDist = norm(centroids(i, :)-data)^2;
%     disp(currentDist)
%     % If distance i is closer than norm closest, then make closest i
%     if distance > currentDist
%         closest=i;
%         distance = currentDist;
%     end
% end
% % Return everything
% index = closest;
% vec_distance = distance;
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
    %size(clusterI)
    %size(new_centroids)
    %sum(clusterI, 2)
    %sum(clusterI, 1)
    
    % tempCen(i, 1:784) = median(clusterI);
    
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