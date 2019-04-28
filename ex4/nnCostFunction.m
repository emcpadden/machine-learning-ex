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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% ===== PART 1 ==============
% forward prop

% init the activations for layer 1 by adding the 1 bias column
a1 = [ones(m,1) X];

% calc the response and the activation of a2
z2 = (a1*Theta1');
a2 = sigmoid(z2);

% add the bias column to a2 so that it can be used to calc the next layer
a2 = [ones(size(a2, 1), 1) a2];

% calc the output of layer 3
z3 = (a2*Theta2');

% calc the activations for the output
h_theta = sigmoid(z3);
a3 = h_theta;

% need to convert the y values to a 10 place vector 
% first init the matrix to all zeros

y_matrix = zeros(m,num_labels);

% then, for each value of y with a 1 in the place that is
% the y answer, do this for every value of y
for i = 1:m
    y_matrix(i,y(i)) = 1;
end

%for i = 1:m     
%     term1 = -y_matrix(i,:) .* log(h_theta(i,:));
%     term2 = (ones(1,num_labels) - y_matrix(i,:)) .* log(ones(1,num_labels) - h_theta(i,:));
%     J = J + sum(term1 - term2);
%end
%J = J / m;

%term = (-1 * y_matrix .* log(h_theta)-(1-y_matrix) .* log(1-h_theta));
%sum_term = sum(term);
%sum_of_sum_term = sum(sum_term);
%Jalt = sum_of_sum_term/m;

J = 1/m * sum(sum(-1 * y_matrix .* log(h_theta)-(1-y_matrix) .* log(1-h_theta)));

reg = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

J = J + reg;


% ===== PART 2 ==============

% walk thru all of the training data
for t = 1:m

    % do the forward propagation
    
    % For the input layer, where l=1:
    a1 = [1; X(t,:)'];

    % For the hidden layers, where l=2:
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];

    % For the output layer
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % create a vector of all 0 except for a 1
    % in the index associated with that value of
    % y
    yy = ([1:num_labels]==y(t))';

    % For the delta values:
    delta_3 = a3 - yy;
    
    delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
    delta_2 = delta_2(2:end); % Taking of the bias row

    % delta_1 is not calculated because we do not associate error with the input    

    % Big delta update
    % accumulate the total values for theta 1 and 2
    Theta1_grad = Theta1_grad + delta_2 * a1';
    Theta2_grad = Theta2_grad + delta_3 * a2';
end

% calculate the theta 1 and 2 grad
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
