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


% hypothesis
%size(X)
%size(theta)

%xTimesTheta = X * theta;

%h = sigmoid(xTimesTheta);

h = sigmoid(X * theta);

% cost
J = sum(-y .* log(h) - (1 - y) .* log(1 - h)) * (1/m);

% gradient
%h_minus_y = h - y;
%transX = X';
%transX_times_h_minus_y = transX * h_minus_y;
%gradAlt = transX_times_h_minus_y * (1/m);

grad = (X' * (h - y)) * (1/m);

%=======================
% TODO: need to understand the code below better
% i took it from 
% https://github.com/mridulnagpal/Andrew-Ng-ML-Course-Assignments/blob/master/machine-learning-ex2/ex2/costFunctionReg.m
h = X*theta;
h_new = sigmoid(h);

J = (1/m)*(sum(-y.*(log(h_new)) - (1.-y).*(log(1.-h_new))))+(lambda/(2*m))*(sum(theta .* theta)) - (lambda/(2*m))*(theta(1,1)*theta(1,1));
grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) ) + (lambda/m)*sum(theta) - (lambda/m)*(theta(1,1));


grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';


% =============================================================

end
