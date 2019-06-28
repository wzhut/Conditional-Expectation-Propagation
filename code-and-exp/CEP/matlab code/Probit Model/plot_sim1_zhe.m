clear all;
close all;

load('simulation1.mat');
figure;
hold on;

% CEP
plot(ceptime, cepkl, 'k', 'LineWidth',3);

% CEP2
plot(cep2dtime, cep2dkl, 'r', 'LineWidth',3);

% EP
plot(eptime, epkl, 'b', 'LineWidth',3);

% VB
plot(vbtime, vbkl,'g', 'LineWidth',3);

set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
set(gca, 'FontSize', 20);
ylim([1.01, 10^5]);
xlim([9*10^-3,100]);
legend('CEP-1', 'CEP-2', 'EP', 'VB'); 
xlabel('Running time (seconds)', 'FontSize', 20);
ylabel('Approximiate KL', 'FontSize', 20);
hold off;