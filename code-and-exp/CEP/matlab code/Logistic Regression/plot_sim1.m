clear all;
close all;

load('simulation1.mat');
load('lp-sim1.mat');
load('lp-sim1-first.mat');
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
plot(vb.time, vb.kl,'g', 'LineWidth',3);

% VBnf
% plot(vbnf.time, vbnf.kl,'c', 'LineWidth',3);

% LP

plot([lpft.time, lp.time], [lpft.kl, lp.kl], 'b', 'LineWidth',3);

set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
set(gca, 'FontSize', 20);
ylim([0.001, 10^5]);
xlim([9*10^-3,10000]);
legend('CEP-1', 'CEP-2', 'EPv2', 'VB', 'LP'); 
xlabel('Running time (seconds)', 'FontSize', 20);
ylabel('Approximiate KL', 'FontSize', 20);
hold off;