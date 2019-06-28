clear all;
close all;

load('simulation2.mat');
load('lp-sim2.mat');
figure;
hold on;

% CEP
plot(cep.time, cep.kl, 'k', 'LineWidth',3);

% CEP2
plot(cep2.time, cep2.kl, 'r', 'LineWidth',3);

% EP
% plot(ep.time, ep.kl, 'b', 'LineWidth',3);

% EPv2
plot(epv2.time, epv2.kl, 'm', 'LineWidth',3);

% EP non-factorized

% plot(epnf.time, epnf.kl, 'y', 'LineWidth',3);

% VB
% plot(vb.time, vb.kl,'g', 'LineWidth',3);

% VBnf
% plot(vbnf.time, vbnf.kl,'c', 'LineWidth',3);

% LP

plot(lp.time, lp.kl, 'b', 'LineWidth',3);

set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
set(gca, 'FontSize', 20);
ylim([1, 10^5]);
xlim([9*10^-3,10^4]);
legend('CEP-1', 'CEP-2', 'EPv2', 'LP'); 
xlabel('Running time (seconds)', 'FontSize', 20);
ylabel('Approximiate KL', 'FontSize', 20);
hold off;