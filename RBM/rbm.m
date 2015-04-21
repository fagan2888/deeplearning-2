%% RBM

n_visible  = 28 * 28;
n_hidden = 500;
lambda = 1e-3;
mu = 1e-3;

% load MNIST data
mnistData   = loadMNISTImages();
mnistLabels = loadMNISTLabels();

display_network(mnistData(:,1:64));

debug = 0;
if debug
    imgNum = 100;
    n_visible = 8*8;
    n_hidden = 4;
    mnistData = mnistData(1:n_visible, 1:imgNum);
    mnistLabels = mnistLabels(1:imgNum);
end
%% optimize theta

% theta = rand(n_hidden * n_visible + n_hidden + n_visible,1);
% 
% options.Display = 'iter';
% options.GradObj = 'on';
% options.MaxIter = 100;
% 
% theta = fminunc(@(p)likelihoodCost(mnistData,p,n_hidden,n_visible), theta, options);
% for i = 1:1000
%     [cost, grad] = likelihoodCost(mnistData, theta, n_hidden, n_visible);
%     theta = theta - 1e1 * grad;
%     fprintf('%d\t%6f\t%6f\n',i,cost,mean(abs(grad(:))));
%     if(mean(abs(grad(:)))<1e-4)
%         break;
%     end
% end

rbmStruct = rbmtrain(mnistData,n_hidden);

%%

display_network(rbmStruct.W');