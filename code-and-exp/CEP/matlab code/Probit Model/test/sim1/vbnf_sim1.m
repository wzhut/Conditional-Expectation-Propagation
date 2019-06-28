addpath(genpath('../..'));
rng(0);

load('simulation1.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.05;
cfg.max_iter = 5000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= prvb_nf(train,test, ts_mean, ts_var);
vbnf.ll = logls;
vbnf.kl = KLs;
vbnf.auc = aucs;
vbnf.time = time;
save('./vbnf-sim1.mat', 'vbnf');





