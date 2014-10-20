function p = prob(v,h,W,b,c)
    p = exp(h' * W * v + b'*v + c' * h);
end