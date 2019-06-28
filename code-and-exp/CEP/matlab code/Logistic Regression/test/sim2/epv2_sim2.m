addpath(genpath('../..'));
rng(0);

load('simulation2.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.05;
cfg.max_iter = 2000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= lrepv2(train,test, ts_mean, ts_var, cfg);
epv2.ll = logls;
epv2.kl = KLs;
epv2.auc = aucs;
epv2.time = time;
save('./epv2-sim2.mat', 'epv2');





