function rbmStruct = rbmtrain(v, n_hidden)
    imgNum = size(v,2);
    n_visible = size(v,1);
    
    rbmStruct = struct();
    
    
    batchImgNum = 100;
    maxepoch = 50;
    
    W = rand(n_hidden, n_visible);
    b = rand(n_visible,1);
    c = rand(n_hidden,1);
    
    eta = 1e-2;
    
    h = zeros(n_hidden, batchImgNum);
    v_k = zeros(n_visible, batchImgNum);
    for epoch = 1:maxepoch
        for i=1:imgNum/batchImgNum
            data = v(:,i:i+batchImgNum-1);
            
            % go up
            h(:) = rand(size(n_hidden, batchImgNum)) > sigmoid(W * data + repmat(c,1,batchImgNum));
            v_k(:) =  rand(size(data)) > sigmoid(W' * h + repmat(b,1,batchImgNum));
            
            s_v = sigmoid(W * data + repmat(c,1,batchImgNum));
            s_vk = sigmoid(W * v_k + repmat(c,1,batchImgNum));
            W_grad = s_v * data' - s_vk * v_k'; 
            b_grad = sum(data - v_k,2);
            c_grad = sum(s_v - s_vk,2);
            
            W = W - W_grad * eta;
            b = b - b_grad * eta;
            c = c - c_grad * eta;
        end
        fprintf('epoch:%d/50\n', epoch);
    end
    
    rbmStruct.W = W;
    rbmStruct.b = b;
    rbmStruct.c = c;
    rbmStruct.n_hidden = n_hidden;
    rbmStruct.n_visible = n_visible;
    
end