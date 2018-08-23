% Demo of HFIT applied to a simple reinforcement learning model.
% Two models are compared:
%   Model 1: single learning rate (and inverse temperature)
%   Model 2: separate learning rates for positive and negative prediction errors
% Ground-truth data are generated from Model 1.

% ---------- generate simulated data ----------%

clear all;
rng default; % for reproducibility

% simulation parameters
N = 100;        % number of trials per subject
R = [0.2 0.8];  % reward probabilities

% parameter values for each agent
x = [8 0.1; 6 0.2; 2 0.1; 5 0.3];

% simulate data from RL agent on two-armed bandit
S = size(x,1);
for s = 1:S
    data(s) = rlsim(x(s,:),R,N);
    testdata(s) = rlsim(x(s,:),R,N*10);
end

% ------------ fit models --------------------%

% create hyperparameter structure
hyparam(1).name = 'inverse temperature hyperparameters';
hyparam(1).lb = [0 0]; % gamma shape and scale parameters
hyparam(1).ub = [100 10];
hyparam(1).logpdf = @(x) log(unifpdf(x(1), 0, 100)) + log(unifpdf(x(2), 0, 10)); 
%J = @(a,b) [psi(1,a) 1/b; 1/b a/b^2]; % Fisher information for Gamma (Yang and Berger, 1998)
%hyparam(1).logpdf = @(h) log(sqrt(det(J(h(1),h(2))))); % Jeffreys prior for Gamma
hyparam(1).rnd = @() [rand * 100 rand * 10];

hyparam(2).name = 'inverse temperature hyperparameters';
hyparam(2).lb = [1 1]; % beta parameters alpha and beta
hyparam(2).ub = [50 50];
hyparam(2).logpdf = @(x) log(unifpdf(x(1), 0, 50)) + log(unifpdf(x(2), 0, 50)); 
%J = @(a,b) [psi(1,a)-psi(1,a+b) -psi(1,a+b); -psi(1,a+b) psi(1,b)-psi(1,a+b)]; % Fisher information for Beta (Yang and Berger, 1998)
%hyparam(2).logpdf = @(h) log(sqrt(det(J(h(1),h(2))))); % Jeffreys prior for Beta
hyparam(2).rnd = @() [rand * 50 rand * 50];

% create parameter structure
param(1).name = 'inverse temperature';
param(1).hlogpdf = @(x,h) sum(log(gampdf(x,h(1),h(2))));
param(1).hrnd = @(h) gamrnd(h(1),h(2));
param(1).lb = 0;    % lower bound
param(1).ub = 50;   % upper bound

param(2).name = 'learning rate';
param(2).hlogpdf = @(x,h) sum(log(betapdf(x,h(1),h(2))));
param(2).hrnd = @(h) betarnd(h(1),h(2));
param(2).lb = 0;
param(2).ub = 1;

param(3) = param(2); % second learning rate for model 2
hyparam(3) = hyparam(2);


% run optimization
nstarts = 2;    % number of random parameter initializations

disp('... Fitting model 1');
results(1) = hfit_optimize(@rllik,hyparam(1:2),param(1:2),data,nstarts);

disp('... Fitting model 2');
results(2) = hfit_optimize(@rllik2,hyparam,param,data,nstarts);

% compute predictive probability for the two models on test data
logp(:,1) = mfit_predict(testdata,results(1));
logp(:,2) = mfit_predict(testdata,results(2));

%-------- plot results -----------%

r = corr(results(1).x(:),x(:));
disp(['Correlation between true and estimated parameters: ',num2str(r)]);
figure;
plot(results(1).x(:),x(:),'+k','MarkerSize',12,'LineWidth',4);
h = lsline; set(h,'LineWidth',4);
set(gca,'FontSize',25);
xlabel('Estimate','FontSize',25);
ylabel('Ground truth','FontSize',25);

bms_results = mfit_bms(results);
figure;
bar(bms_results.xp); colormap bone;
set(gca,'XTickLabel',{'Model 1' 'Model 2'},'FontSize',25,'YLim',[0 1]);
ylabel('Exceedance probability','FontSize',25);
title('Bayesian model comparison','FontSize',25);

figure;
d = logp(:,1)-logp(:,2);
m = mean(d);
se = std(d)/sqrt(S);
errorbar(m,se,'ok','MarkerFaceColor','k','MarkerSize',12,'LineWidth',4);
set(gca,'YLim',[-1 max(d)+1],'XLim',[0.5 1.5],'XTick',1,'XTickLabel',{'Model 1 vs. Model 2'},'FontSize',25);
ylabel('Relative log predictive prob.','FontSize',25);
hold on; plot([0.5 1.5],[0 0],'--r','LineWidth',3); % red line shows chance performance
title('Cross-validation','FontSize',25);

