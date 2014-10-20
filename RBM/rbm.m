%% RBM

n_visible  = 28 * 28;
n_hidden = 500;
lambda = 1e-3;
mu = 1e-3;

% load MNIST data
mnistData   = loadMNISTImages();
mnistLabels = loadMNISTLabels();

display_network(mnistData(:,1:64));

pause;

%% optimize theta
options.Display = 'iter';
options.GradObj = 'on';
options.MaxIter = 100;

theta = fminunc(@(p)likelihoodCost(v,h,p,n_hidden,n_visible), theta, options);

