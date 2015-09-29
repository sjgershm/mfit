% Demo of MFIT applied to a simple reinforcement learning model.
% Two models are compared:
%   Model 1: single learning rate (and inverse temperature)
%   Model 2: separate learning rates for positive and negative prediction errors
% Ground-truth data are generated from Model 1.

% ---------- generate simulated data ----------%

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

% create parameter structure
g = [1 5];  % parameters of the gamma prior
param(1).name = 'inverse temperature';
param(1).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(1).lb = 0;    % lower bound
param(1).ub = 50;   % upper bound

a = 1.2; b = 1.2;   % parameters of beta prior
param(2).name = 'learning rate';
param(2).logpdf = @(x) sum(log(betapdf(x,a,b)));
param(2).lb = 0;
param(2).ub = 1;

param(3) = param(2); % second learning rate for model 2

% run optimization
nstarts = 2;    % number of random parameter initializations
disp('... Fitting model 1');
results(1) = mfit_optimize(@rllik,param(1:2),data,nstarts);
disp('... Fitting model 2');
results(2) = mfit_optimize(@rllik2,param,data,nstarts);

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