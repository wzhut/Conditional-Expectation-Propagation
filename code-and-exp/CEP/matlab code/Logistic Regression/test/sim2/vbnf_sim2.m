addpath(genpath('../..'));
rng(0);

load('simulation2.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.05;
cfg.max_iter = 5000;
cfg.tol = 0;

[logl, KL, auc, logls, KLs, aucs, time]= lrvb_nf(train,test, ts_mean, ts_var);
vbnf.ll = logls;
vbnf.kl = KLs;
vbnf.auc = aucs;
vbnf.time = time;
save('./vbnf-sim2.mat', 'vbnf');





