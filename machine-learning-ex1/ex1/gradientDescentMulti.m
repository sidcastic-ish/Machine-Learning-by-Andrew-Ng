function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
data = load('ex1data2.txt'); 
y = data(:, 2);
m = length(y); % number of training examples
X = [ones(m, 1), data(:,1)]; 
J_history = zeros(num_iters, 1);
theta = zeros(2, 1); 
num_iters = 1500;
alpha = 0.01;
for iter = 1:num_iters,

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    num_thetas = length(theta);
	  new_theta = zeros(num_thetas,1);
	  for a = 1:num_thetas,
      inner_sum = 0;
		  for b = 1:m,
        inner_sum = inner_sum + ((X(b,:) * theta) - y(b)) * X(b,a);
	    end
	  new_theta(a) = theta(a) - (alpha / m * inner_sum);
	  end










    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
