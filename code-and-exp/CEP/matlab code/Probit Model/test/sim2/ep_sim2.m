addpath(genpath('../..'));
rng(0);

load('simulation2.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.1;
cfg.max_iter = 2000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= prep(train,test, ts_mean, ts_var, cfg);
ep.ll = logls;
ep.kl = KLs;
ep.auc = aucs;
ep.time = time;
save('./ep-sim2.mat', 'ep');





