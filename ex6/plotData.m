function plotData(x, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
%
% Note: This was slightly modified such that it expects y = 1 or y = 0

% Find Indices of Positive and Negative Examples

% plot(x, y, 'rx', 'MarkerSize', 5); % Plot the data 
% ylabel('Profit in $10,000s'); % Set the y−axis label 
% xlabel('Population of City in 10,000s'); % Set the x−axis label


% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);

% Plot Examples
plot(x(pos, 1), x(pos, 2), 'k+','LineWidth', 1, 'MarkerSize', 7)
hold on;
plot(x(neg, 1), x(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
hold off;

end
