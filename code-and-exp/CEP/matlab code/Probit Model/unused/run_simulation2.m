addpath(genpath('./lightspeed'));
addpath(genpath('./ghq'));
addpath(genpath('./minFunc_2012'));
rng(0);

load('./simulation/simulation2.mat', 'train', 'test', 'ts_mean', 'ts_var');

cfg.rho =0.05;
cfg.max_iter = 3000;
cfg.tol = 0;

% CEP
[logl, KL, auc, logls, KLs, aucs, time]= prcep(train,test, ts_mean, ts_var, cfg);
cep.ll = logls;
cep.kl = KLs;
cep.auc = aucs;
cep.time = time;
save('./simulation2.mat', 'cep');

% CEP2 
[logl, KL, auc, logls, KLs, aucs, time]= prcep2_diag(train,test, ts_mean, ts_var, cfg);
cep2.ll = logls;
cep2.kl = KLs;
cep2.auc = aucs;
cep2.time = time;
save('./simulation2.mat', 'cep2', '-append');

% EP fully factorized
[logl, KL, auc, logls, KLs, aucs, time]= prep(train,test, ts_mean, ts_var, cfg);
ep.ll = logls;
ep.kl = KLs;
ep.auc = aucs;
ep.time = time;
save('./simulation2.mat', 'ep', '-append');


cfg.max_iter = 500;
% EP non-fully factorized
[logl, KL, auc, logls, KLs, aucs, time]= prep_nf(train,test, ts_mean, ts_var, cfg);
epnf.ll = logls;
epnf.kl = KLs;
epnf.auc = aucs;
epnf.time = time;
save('./simulation2.mat', 'epnf', '-append');


% VB
[logl, KL, auc, logls, KLs, aucs, time]= prvb(train,test, ts_mean, ts_var);
vb.ll = logls;
vb.kl = KLs;
vb.auc = aucs;
vb.time = time;
save('./simulation2.mat', 'vb', '-append');

% VB nf
[logl, KL, auc, logls, KLs, aucs, time]= prvb_nf(train,test, ts_mean, ts_var);
vbnf.ll = logls;
vbnf.kl = KLs;
vbnf.auc = aucs;
vbnf.time = time;
save('./simulation2.mat', 'vbnf', '-append');

cfg.rho =0.005;
cfg.max_iter = 1000;
% % LP
% [logl, KL, auc, logls, KLs, aucs, time]= prlp(train,test, ts_mean, ts_var, cfg);
% lp.ll = logls;
% lp.kl = KLs;
% lp.auc = aucs;
% lp.time = time;
% save('./simulation2.mat', 'lp', '-append');
% 
% % LP nf
% [logl, KL, auc, logls, KLs, aucs, time]= prlp_nf(train,test, ts_mean, ts_var, cfg);
% lpnf.ll = logls;
% lpnf.kl = KLs;
% lpnf.auc = aucs;
% lpnf.time = time;
% save('./simulation2.mat', 'lpnf', '-append');




