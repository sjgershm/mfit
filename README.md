MFIT
====

Simple model-fitting tools.

Questions? Contact Sam Gershman (gershman@fas.harvard.edu).

—————

Quick start. Examples below use snippets from [mfit_demo_RL.m](demos/mfit_demo_RL.m):

1) Define your prior by constructing a parameter structure. Here is an example from a reinforcement learning model:

```matlab
g = [2 1];  % parameters of the gamma prior
param(1).name = 'inverse temperature';
param(1).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(1).lb = 0;    % lower bound
param(1).ub = 50;   % upper bound

a = 1.2; b = 1.2;   % parameters of beta prior
param(2).name = 'learning rate';
param(2).logpdf = @(x) sum(log(betapdf(x,a,b)));
param(2).lb = 0;
param(2).ub = 1;
```

Here `logpdf` takes as input a parameter (or multiple parameters, in the case of multiple subjects) and evaluates the log joint density. The fields `lb` and `ub` correspond to the lower and upper parameter bounds, respectively.

2) Call the optimizer, which finds the maximum a posteriori estimates of the parameters for each subject:
```matlab
results = mfit_optimize(@fun,param,data,nstarts)
```
Here `@fun` is a function handle for your log-likelihood function, which takes the following form:
```matlab
lik = fun(x,data,options)
```
where `x` is a vector of parameter values, `data` is a single subject's data structure. It must have a field called `N` (i.e., `data.N`) that specifies the number of observations for the subject. `options` is an additional input structure that may be omitted.

See [mfit_demo_RL.m](demos/mfit_demo_RL.m) for examples of Bayesian model comparison and cross-validation.
