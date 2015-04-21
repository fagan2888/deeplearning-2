function [cost, grad] = likelihoodCost(v, theta, n_hidden, n_visible)

% unpack W,b,c
    imgNum = size(v,2);

    W = reshape(theta(1: n_hidden * n_visible), n_hidden, n_visible);
    b = reshape(theta(n_hidden * n_visible+1:n_hidden * n_visible+n_visible),n_visible,1);
    c = reshape(theta(n_hidden * n_visible+n_visible + 1:end),n_hidden,1);
    %cost = mean(v - );
    v_b = v;
    h = round(sigmoid(W * v_b + repmat(c,1,imgNum)));
    err = v - sigmoid(W' * h + repmat(b,1,imgNum));
    cost = mean(abs(err(:)));

    W_grad = zeros(n_hidden, n_visible);
    b_grad = zeros(n_visible,1);
    c_grad = zeros(n_hidden,1);
    
    v_k = zeros(size(v));
% caculate grad    
% gibbs sample hvh 
    h = rand(size(n_hidden, imgNum)) > sigmoid(W * v + repmat(c,1,imgNum));
    v_k(:) =  rand(size(v)) > sigmoid(W' * h + repmat(b,1,imgNum));
    
    s_v = sigmoid(W * v + repmat(c,1,imgNum));
    s_vk = sigmoid(W * v_k + repmat(c,1,imgNum));
    W_grad(:) = s_v * v' - s_vk * v_k'; 
    b_grad(:) = sum(v - v_k,2);
    c_grad(:) = sum(s_v - s_vk,2);
    
    grad = [W_grad(:); b_grad(:); c_grad(:)];

end