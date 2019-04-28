function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

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

mu = mean(X);
mu1 = mu(1,1);
mu2 = mu(1,2);
sigma = std(X);
sigma1 = sigma(1,1);
sigma2 = sigma(1,2);

X1 = X(:,1);
X2 = X(:,2);

MU1 = ones(length(X1), 1) .* mu1;
MU2 = ones(length(X2), 1) .* mu2;

X1TimesMu1 = (X1 - MU1);
X2TimesMu1 = (X2 - MU2);

SIGMA1 = ones(length(X1), 1) .* sigma1;
SIGMA2 = ones(length(X2), 1) .* sigma2;

X1_Norm = X1TimesMu1 ./ SIGMA1;
X2_Norm = X2TimesMu1 ./ SIGMA2;

norm = [X1_Norm, X2_Norm];
X_norm = norm;








% ============================================================

end
