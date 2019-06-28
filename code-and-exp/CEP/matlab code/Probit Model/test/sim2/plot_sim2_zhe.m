clear all;
close all;


load('cep-sim2.mat');
load('cep2-sim2.mat');
load('ep-sim2.mat');
load('vb-sim2.mat');
load('lp-sim2.mat');
figure;
hold on;

% CEP
plot(cep.time, cep.kl, 'k', 'LineWidth',5);

% CEP2
plot(cep2.time, cep2.kl, 'r', 'LineWidth',5);

% EP
plot(ep.time, ep.kl, 'b', 'LineWidth',5);

% EP non-factorized

% plot(epnf.time, epnf.kl, 'y', 'LineWidth',3);

% VB
% plot(vb.time, vb.kl,'g', 'LineWidth',3);

% VB nf
% plot(vbnf.time, vbnf.kl,'c', 'LineWidth',3);

% LP

plot(lp.time, lp.kl, 'c', 'LineWidth',5);

set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
set(gca, 'FontSize', 20);
ylim([10^1, 10^4]);
yticks([10^1, 10^2, 10^3, 10^4]);
xlim([10^-2,0.2*10^5]);
xticks([10^-2, 10^0, 10^2, 10^4]);
%legend('CEP-1', 'CEP-2', 'EP', 'LP'); 
set(gca, 'FontSize', 30);
xlabel('Running time (seconds)', 'FontSize', 35);
ylabel('Approx. KL', 'FontSize', 35);
%hold off;