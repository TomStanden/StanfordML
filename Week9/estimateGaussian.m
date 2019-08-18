function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% a single octave sum will sum up all the rows 
% and leave an n dimensional vector for the n features
mu = (1/m)*sum(X);

%similarly for sigma^2 just need to minus the mean off each feature 
% and square it and then sum over all examples again 

squard_diff = (X - mu).^2;
sigma2 = (1/m)*sum(squard_diff);







% =============================================================


end
