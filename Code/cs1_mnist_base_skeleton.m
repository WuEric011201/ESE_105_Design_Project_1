
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
testimage = reshape(test(9,[1:784]), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.

%% After importing, the array 'train' consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.

%% This next section of code calls the three functions you are asked to specify

k= 18; % set k
max_iter= 20; % set the number of iterations of the algorithm
repeat = 15;

% Save the original training and test sets, the start to preprocess the
% images so they work with k means better
trainOG = train;

train = preprocess(train);

%% The next line initializes the centroids.  Look at the initialize_centroids()
% function, which is specified further down this file.

centroids=initialize_centroids(train,k);

% Create variables to store the best centroids, set the default distance to
% infinity and label the centroids with all zero.
best_centroids = centroids;
 
lowestDistance = Inf;

centroid_labels = zeros(k, 1);
%% Initialize an array that will store k-means cost at each iteration

cost_iteration = zeros(max_iter, 1);

%% This for-loop enacts the k-means algorithm

% Repeat this a certain number of times just to try getting different
% minimals
for rep = 1:repeat
    % Make a stuck counter in case the k means takes too long to try
    % fining a way to get cluster labels that contains all 0-9 digits.
    stuck = 0;
    
    % Try your best to make sure at least 1 centroid has eacah of 0-9 as
    % its label
    while ~isequal(ismember([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], centroid_labels), ones(1, 10)) && stuck< 20
        cost_iteration = zeros(max_iter, 1);
        
        for iter=1:max_iter
        % FILL THIS IN!
            for i=1: size(train, 1)
                % Get the distance and index of centroid, save the centroid
                % to row 785 of train and add distance to the cost
                % iteration
                [train(i, 785), distance] = assign_vector_to_centroid(train(i, 1:784), centroids);
                cost_iteration(iter, 1)=cost_iteration(iter, 1)+distance;
            end
            % update centroid location
            centroids = update_Centroids(train, k);
        end
        % Label at the end
        centroid_labels=label_Data(train, centroids, trainsetlabels);
        
        stuck = stuck +1;
    end
    
    % If we found a new set of centroids with lowest distance we set it to
    % be the new lowest, update distance and centroids.
    if cost_iteration(max_iter)< lowestDistance
        lowestDistance = cost_iteration(max_iter, 1);
        best_centroids = centroids;
    end
    
end

centroids=best_centroids;
% Recalculate the labels for the best centroid
centroid_labels=label_Data(train, centroids, trainsetlabels);
%% This section of code plots the k-means cost as a function of the number
% of iterations

figure;
% FILL THIS IN!
axis auto;
plot(1:max_iter, cost_iteration, '-o');

%% This next section of code will make a plot of all of the centroids
% Again, use help <functionname> to learn about the different functions
% that are being used here.

figure;
colormap('gray');

plotsize = ceil(sqrt(k));

for ind=1:k
    
    centr=centroids(ind,[1:784]);
    subplot(plotsize,plotsize,ind);
    
    imagesc(reshape(centr,[28 28])');
    title(strcat('Centroid ',num2str(ind)))

end

%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids
% There are other ways you might initialize centroids.
% ***Feel free to experiment.***
% Note that this function takes two inputs and emits one output (y).

function y=initialize_centroids(data,num_centroids)
% The default code was not good enough for us. 
% Initialize the centroids using k means ++ instead

centroids = zeros(num_centroids, size(data, 2));

% Assign first centroid to a random data point
random_index=randperm(size(data,1));
centroids(1, :)=data(random_index(1),:);

for i = 2: num_centroids
    % Assign all the other centroids to the data point that is the farthest
    % away from all other centroids
    
    [index, ~] =find_farthest_distance(data(:, 1:784), centroids(1:i-1, :));
    centroids(i, :) = data(index, :);
end

y=centroids;



% random_index=randperm(size(data,1));
% 
% centroids=data(random_index(1:num_centroids),:);
% 
% y=centroids;

end

%% Function to find farthest data point using norm/ distance for k meanas++

function [index, vec_distance] = find_farthest_distance(data, centroids)

% Distances is # cen by # data
distances = zeros(size(centroids, 1), size(data, 1));
for i = 1: size(centroids, 1)
    for j = 1: size(data, 1)
        % Calculate the distance for each centroid i from data point j
        distances(i, j) = norm(centroids(i, 1:size(data, 2)) - data(j, :));
    end
end
% Sum all the columns to get sum of distances of all centroids from the
% point
distance = sum(distances);

% Find which data point is farthest from all the centroids
[vec_distance, index] = max(distance);

end

%% Function to pick the Closest Centroid using norm/distance
% This function takes two arguments, a vector and a set of centroids
% It returns the index of the assigned centroid and the distance between
% the vector and the assigned centroid.

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
% FILL THIS IN

% Find the distances by multiplying the data by a ones vector to duplicate
% it until it is repeated centroids # of times downwards. Subtract each of
% these with centroid to get the norm of each and return the smallest.
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
    % Grab the data in each cluster, and sum all of them and divide it by
    % the number of clusters for mean
    clusterI = data(cluster==i, 1:784);
    tempCen(i, 1:784)= sum(clusterI, 1)/(size(clusterI, 1));
end
new_centroids = tempCen;
end

%% Function to label the centroids by finding the most common label in that centroid

function centroid_labels = label_Data(data, centroids, trainingLabels)
% Grab all the cluster data
cluster = data( :,785);

K = size(centroids, 1);

labels = zeros(K, 1);

for j=1: K
    % Take the labels from the specific cluster
    labelsI = trainingLabels(cluster==j, 1);
    % Find the most frequently appearing one to set as the cluster label.
    labels(j, 1)=mode(labelsI);
end
centroid_labels=labels;
end

%% Different preprocessing methods

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