function [cost, grad] = likelihoodCost(v, h, theta, n_hidden, n_visible)

% unpack W,b,c
    W = reshape(theta(1: n_hidden * n_visible), n_hidden, n_visible);
    b = reshape(theta(n_hidden * n_visible+1:n_hidden * n_visible+n_visible),n_visible,1);
    c = reshape(theta(n_hidden * n_visible+n_visible + 1:end),n_hidden,1);
    cost = log(exp(h' * W * v + b' * v + c' * h));
    

    W_grad = zeros(n_hidden, n_visible);
    b_grad = zeros(n_visible,1);
    c_grad = zeros(n_hidden,1);
    
% caculate grad    
    
    
    
    
    grad = [W_grad(:); b_grad(:); c_grad(:)];

end