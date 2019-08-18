function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

pass = find(y == 1);
fail = find(y ~= 1);
j = (sum(-log(sigmoid(X(pass,:)*theta)))+ sum(- log(1 - sigmoid(X(fail,:)*theta))))./m;
reg = (lambda/(2*m))*sum(theta(2:length(theta)).^2);
J = j+reg;

gd = (X'*(sigmoid(X*theta) - y))./m;
% ^^ As before
% temporary theta vector with theta_0 set to zero as it is not involved in regularization
gdTheta = theta;
gdTheta(1) = 0;
%regularization term for gradient descent
REG = (lambda/m).*gdTheta;

grad = gd + REG;




% =============================================================

end
