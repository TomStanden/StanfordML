function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
pass = find(y == 1);
fail = find(y ~= 1);
j = (sum(-log(sigmoid(X(pass,:)*theta)))+ sum(- log(1 - sigmoid(X(fail,:)*theta))))./m;
reg = (lambda/(2*m))*sum(theta(2:length(theta)).^2);
J = j+reg;

%find the positions of each number in the y vector;
%one = find(y==1)
%notone = find(y~=1)
%two = find(y==2)
%nottwo = find(y~=2)
%three = find(y==3)
%notthree = find(y~=3)
%four = find(y==4)
%notfour = find(y~=4)
%five = find(y==5);
%notfive = find(y~=5);
%six = find(y==6);
%notsix = find(y~=6);
%seven = find(y==7);
%notseven = find(y~=7);
%eight = find(y==8);
%noteight = find(y~=8);
%nine = find(y==9);
%notnine = find(y~=9);
%nought = find(y==10);
%otnought = find(y~=10);

%j1 = (sum(-log(sigmoid(X(one,:)*theta)))+ sum(- log(1 - sigmoid(X(notone,:)*theta))))./m;
%j2 = (sum(-log(sigmoid(X(two,:)*theta)))+ sum(- log(1 - sigmoid(X(nottwo,:)*theta))))./m;
%j3 = (sum(-log(sigmoid(X(three,:)*theta)))+ sum(- log(1 - sigmoid(X(notthree,:)*theta))))./m;
%j4 = (sum(-log(sigmoid(X(four,:)*theta)))+ sum(- log(1 - sigmoid(X(notfour,:)*theta))))./m;
%j5 = (sum(-log(sigmoid(X(five,:)*theta)))+ sum(- log(1 - sigmoid(X(notfive,:)*theta))))./m;
%j6 = (sum(-log(sigmoid(X(six,:)*theta)))+ sum(- log(1 - sigmoid(X(notsix,:)*theta))))./m;
%j7 = (sum(-log(sigmoid(X(seven,:)*theta)))+ sum(- log(1 - sigmoid(X(notseven,:)*theta))))./m;
%j8 = (sum(-log(sigmoid(X(eight,:)*theta)))+ sum(- log(1 - sigmoid(X(noteight,:)*theta))))./m;
%j9 = (sum(-log(sigmoid(X(nine,:)*theta)))+ sum(- log(1 - sigmoid(X(notnine,:)*theta))))./m;
%j0 = (sum(-log(sigmoid(X(nought,:)*theta)))+ sum(- log(1 - sigmoid(X(notnought,:)*theta))))./m;

%j = j1 + j2 + j3 + j4 + j5 + j6 + j7 + j8 + j9 + j0;
%reg_term = (lambda/(2*m))*sum(theta(2:length(theta)).^2);
%J = j + reg_term;

gd = (X'*(sigmoid(X*theta) - y))./m;
% ^^ As before
% temporary theta vector with theta_0 set to zero as it is not involved in regularization
gdTheta = theta;
gdTheta(1) = 0;
%regularization term for gradient descent
REG = (lambda/m).*gdTheta;

grad = gd + REG;


% =============================================================

grad = grad(:);

end
