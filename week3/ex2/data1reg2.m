clear ; close all; clc

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);
hold on;
plotData(X, y);

% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')


X = mapFeature(X(:,1), X(:,2));

initial_theta = zeros(size(X, 2), 1);

lambda = 1;

[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

plotDecisionBoundary(theta, X, y);

hold off;

p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);