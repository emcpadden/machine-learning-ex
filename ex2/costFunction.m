function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

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

% =============================================================

end