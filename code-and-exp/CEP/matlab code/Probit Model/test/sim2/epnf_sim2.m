addpath(genpath('../..'));
rng(0);

load('simulation2.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.05;
cfg.max_iter = 500;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= prep_nf(train,test, ts_mean, ts_var, cfg);
epnf.ll = logls;
epnf.kl = KLs;
epnf.auc = aucs;
epnf.time = time;
save('./epnf-sim2.mat', 'epnf');





