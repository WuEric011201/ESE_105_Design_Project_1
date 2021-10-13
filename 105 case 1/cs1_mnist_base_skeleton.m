
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
testimage = reshape(test(1,[1:784]), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.

%% After importing, the array 'train' consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.

%% This next section of code calls the three functions you are asked to specify

k= 20; % set k
max_iter= 10; % set the number of iterations of the algorithm

%% The next line initializes the centroids.  Look at the initialize_centroids()
% function, which is specified further down this file.

centroids=initialize_centroids(train,k);

%% Initialize an array that will store k-means cost at each iteration

cost_iteration = zeros(max_iter, 1);

%% This for-loop enacts the k-means algorithm
% Iterate for designated times
 iteration =0;
for iter=1:max_iter
    iteration = iteration +1
% Iterate through all of the points
train479 = [];
train479_store = [];
    for in = 1: size(train, 1)
        [train(in, 785), vec_distance] = assign_vector_to_centroid(train(in, (1:784)), centroids(:, (1:784)));
        if (train(in, 785)~=4 && train(in, 785)~=7 && train(in, 785)~=9)
            cost_iteration(iter, 1) = cost_iteration(iter, 1) + vec_distance;
            continue;
        end
        train479 = [train479; in];  % store the index of where in the row of the train the vector of 4 or 7 or 9 is stored
        train479_store = [train479_store; train(in, :)]; % store the the vector of 4 or 7 or 9 in the right order
    end
    
    centroids = update_Centroids(train, k);
    centroid_labels = auto_generate_labels(trainsetlabels,train);
    
    count = [sum(centroid_labels(:) == 4);  ...
        sum(centroid_labels(:) == 7); sum(centroid_labels(:) == 9)]; % count the centroids of each 4, 7, 9 
    total = sum(count, "All"); % count the total number of centroids
    index_centroid479 = zeros(total, 1); % Initialize the number of centroids that equal to the total number of
% centroids
    centroids479 = zeros(total, 785);
    if count(1)~=0
        index_centroid479(1:count(1)) = find(centroid_labels==4);
        centroids479(1:count(1), :) = initialize_centroids(train(train(:, 785)==index_centroid479(count(1)),:) ,count(1)); 
    end        
    
    if count(2)~=0
        index_centroid479( (count(1)+1) : (count(1)+count(2)) )=  find(centroid_labels==7) ;
        centroids479((count(1)+1) : (count(1)+count(2)) , :) = initialize_centroids(train(train(:, 785)==index_centroid479(count(1)+1), : ) ,count(2));
    end
    
    if count(3)~=0
        index_centroid479( (count(1)+count(2)+1) : total)=  find(centroid_labels==9) ;
        centroids479((count(1)+count(2)+1) : total, :) = initialize_centroids(train(train(:, 785)==index_centroid479(total), : ), count(3));
    end
  
    for in = 1: size(train479, 1)
        [train479_store(in, 785) , vec_distance] = assign_vector_to_centroid(train(train479(in), (1:784) ), ...
            centroids479(:, (1:784) ) );
        if 1 <= train479_store(in, 785) && train479_store(in, 785) <=count(1) % put the true index back to the train
            % this is a 4
                train(train479(in), 785) = index_centroid479(randi([1, count(1)]));
        elseif count(1)< train479_store(in, 785) && train479_store(in, 785) <=(count(1)+count(2))
                train(train479(in), 785) = index_centroid479(randi([1, count(2)]) + count(1));
        elseif (count(1)+count(2))<train479_store(in, 785) 
                train(train479(in), 785) = index_centroid479(randi([1, count(3)])+count(1)+count(2));
        end
        cost_iteration(iter, 1) = cost_iteration(iter, 1) + vec_distance;
    end
    % Assign to one of the new centroids
    
    centroids479 = update_Centroids(train479_store, total); 
    % put the renewed centroids of 479 back to the true "centroids"
    for in = 1 : sum(count, "All")
         centroids(index_centroid479(in), :) = centroids479(in);     
    end 

end


%% This section of code plots the k-means cost as a function of the number
% of iterations

figure;
stem(cost_iteration);


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

random_index=randperm(size(data,1)); % size is the number of rows; test case: size = 200; 200 random nums

centroids=data(random_index(1:num_centroids),:); % random k rows as centroids

y=centroids;

end

%% Function to pick the Closest Centroid using norm/distance
% This function takes two arguments, a vector and a set of centroids
% It returns the index of the assigned centroid and the distance between
% the vector and the assigned centroid.

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)

    % Calculate this point's the distance with all of the centroids
        vec_distance = norm(data - centroids(1, :));
        index = 1;
        for j = 2: size(centroids, 1)
            dis = norm(data - centroids(j, :));
            if(dis < vec_distance)
                vec_distance= dis;
                % Assign which centroid this point is attached to                 
                index = j;
            end
        end

end


%% Function to compute new centroids using the mean of the vectors currently assigned to the centroid.
% This function takes the set of training images and the value of k.
% It returns a new set of centroids based on the current assignment of the
% training images.

function new_centroids=update_Centroids(data,K)
    new_centroids = [];% zeros(K,size(data, 2));
 
    logic = findgroups(data(:, 785)) ;
    new_centroids(:, (1:784)) = splitapply(@mean,data(:, (1:784)),logic) ;
 
end

%% Function to auto generate centroid labels

function centroid_labels = auto_generate_labels(trainsetlabels,train)
    logic = findgroups(train(:, 785)) ;
    centroid_labels = splitapply(@mode,trainsetlabels,logic);
end