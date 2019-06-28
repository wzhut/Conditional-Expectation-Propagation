addpath(genpath('../..'));
rng(0);

load('simulation1.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.05;
cfg.max_iter = 500;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= prvb(train,test, ts_mean, ts_var);
vb.ll = logls;
vb.kl = KLs;
vb.auc = aucs;
vb.time = time;
save('./vb-sim1.mat', 'vb');





