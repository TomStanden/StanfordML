function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
%number of features to reduce
num_feat = size(X,2);
%save the mean and standard deviation for each feature
%to scale future predictions suitably
mu = zeros(1, num_feat);
sigma = zeros(1, num_feat);

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

%create a loop to reduce features
for i = 1:num_feat
  mu(1,i) = mean(X(:,i));
  sigma(1,i) = std(X(:,i));
  X_norm(:,i) = (((X(:,i) - mu(1,i)))/(sigma(1,i)));
endfor







% ============================================================

end
