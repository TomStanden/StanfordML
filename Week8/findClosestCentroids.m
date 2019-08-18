function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

m = size(X, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:m %loop over training examples
  for j = 1:K %loop over number of centroids
    %disp("Centroid number: "),disp(j)
    %work out the squared mod of the distance between a point 
    %and a centroid
    temp_dist = sum((X(i,:) - centroids(j, :)).^2);
   %if its the first distance then save it as the shortest
    if j == 1
      shortest_dist = temp_dist;
      closest_centroid = j;
    % if its the new dist is shorter than the previous shortest
    %save it
    elseif temp_dist < shortest_dist
      shortest_dist = temp_dist;
      closest_centroid = j;
    endif      
  endfor
  %store the centroid that this example is closest to
  idx(i) = closest_centroid;
endfor






% =============================================================

end

