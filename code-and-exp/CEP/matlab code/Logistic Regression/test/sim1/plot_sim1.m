clear all;
close all;


load('cep-sim1.mat');
load('cep2-sim1.mat');
load('epv2-sim1.mat');
load('vb-sim1.mat');
load('lp-sim1.mat');
figure;
hold on;

% CEP
plot(cep.time, cep.kl, 'k', 'LineWidth',3);

% CEP2
plot(cep2.time, cep2.kl, 'r', 'LineWidth',3);

% EP
plot(epv2.time, epv2.kl, 'b', 'LineWidth',3);

% EP non-factorized

% plot(epnf.time, epnf.kl, 'y', 'LineWidth',3);

% VB
% plot(vb.time, vb.kl,'g', 'LineWidth',3);

% VB nf
% plot(vbnf.time, vbnf.kl,'c', 'LineWidth',3);

% LP

plot(lp.time, lp.kl, 'c', 'LineWidth',3);

set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
%set(gca, 'FontSize', 20);
ylim([0.4, 2*10^3]);
yticks([10^0, 10^1, 10^2, 10^3]);
xlim([9*10^-3,10^4]);
xticks([10^-2, 10^0, 10^2, 10^4]);
%legend('CEP-1', 'CEP-2', 'EP', 'LP'); 
set(gca, 'FontSize', 30);
xlabel('Running time (seconds)', 'FontSize', 35);
ylabel('Approx. KL', 'FontSize', 35);
