function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% Add the row of ones to the X matrix
a1 = [ ones(m,1) X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% Unroll parameters 
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%{
STEP-BY-STEP spelling out of cost function equation 
Followed tutorial to get the y_matrix which turns y into a logistic matrix
Finding z2, a2, z3, a3 no issue
Calculating J proved very very fiddly, must have been getting the minus signs wrong
The double sum at the end just sums over all rows and columns of the j matrix
to obtain scalar cost
%}

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);


%first layer
z2 = a1*Theta1';
a2 = sigmoid(z2);

%add bias unit
a2 = [ones(length(a2),1) a2];

%second layer
z3 = a2*Theta2';
a3 = sigmoid(z3);

log1 = log(a3);

log2 = log(1 - a3);

p1 = (-y_matrix.*log1);

p2 = -(1 - y_matrix).*log2;

j = p1+p2;

unregJ = sum(sum(j))/m;


%%%%%%%%% REGULARIZATION %%%%%%%%
regTheta1 = Theta1(:,2:end).^2;
regTheta2 = Theta2(:,2:end).^2;
reg = (lambda/(2*m))*(sum(sum(regTheta1)) + sum(sum(regTheta2)));

J = unregJ + reg;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% use a loop over all m training examples
for t = 1:m
  %set a1 to the values of x(t) cos we are dealing with one example at a time
  A1 = [1 X(t,:)];
 % size(A1)
  %Run a feedforward pass
  Z2 = A1*Theta1';
 % size(Z2)
  A2 = sigmoid(Z2);
  A2 = [1 A2];
  Z3 = A2*Theta2';
  A3 = sigmoid(Z3);
  
  %calculate the errors
  %first/output layer is easy
  delta3 = A3 - y_matrix(t,:);
%  size(delta3)
%  size(Theta2)
  %error for hidden layer is more complicated
  %formula taken from pdf
  %split up into a and b to remove the bias unit when working backwards
  delta2a = (delta3*Theta2);
  delta2b = delta2a(2:end);
  delta2 = delta2b.*sigmoidGradient(Z2);
  
  Delta1 = A1'*delta2;
  Delta2 = A2'*delta3;
  %printf("GOT HERE\n")
  %size(Delta1)
  %size(Theta1_grad)
  %size(Delta2)
  %size(Theta2_grad)  
  %size(Theta1)
  %size(Theta2)
  
  Theta1_grad += Delta1';
  Theta2_grad += Delta2';
  
endfor

%don't regularize the bias unit so set to 0
RegTheta1 = Theta1;
RegTheta1(:,1) = 0;

RegTheta2 = Theta2;
RegTheta2(:,1) = 0;

Theta1_grad = Theta1_grad./m + (lambda/m)*RegTheta1;
Theta2_grad = Theta2_grad./m + (lambda/m)*RegTheta2;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
