addpath(genpath('../..'));
rng(0);
load('simulation1.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.05;
cfg.max_iter = 1000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= lrep(train,test, ts_mean, ts_var, cfg);
ep.ll = logls;
ep.kl = KLs;
ep.auc = aucs;
ep.time = time;
save('./ep-sim1.mat', 'ep');





