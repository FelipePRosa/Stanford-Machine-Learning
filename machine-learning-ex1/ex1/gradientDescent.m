function [theta, J_history, T_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

      theta = theta - alpha * (1/m) * X'*((X*theta) - y);
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    T_history(iter,:) = theta;
end
  disp('Valor segunda iteração');
  disp (theta(1:2,:));
  disp(min(J_history));
  x=1:num_iters;
  figure;
  plot (x,J_history);
  xlabel ("Iterations");
  ylabel ("Squared Error Function");
  title ("Error Over Iterations");
end
