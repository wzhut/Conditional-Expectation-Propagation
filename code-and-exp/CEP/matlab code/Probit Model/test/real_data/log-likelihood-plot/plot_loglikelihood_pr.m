clear;
close all;

% load data

load('cep-realdata.mat');
load('cep2-realdata.mat');
load('ep-realdata.mat');
load('lp-realdata.mat');
load('vb-realdata.mat');

num_dataset = 6;
num_run = 5;
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};


for i = 1 : num_dataset
    cep_size = zeros(num_run, 1);
    cep2_size = zeros(num_run, 1);
    ep_size = zeros(num_run, 1);
    lp_size = zeros(num_run, 1);
    vb_size = zeros(num_run, 1);
    % average
    for j = 1 : num_run
        cep_size(j) = size(cep.lls{i,j}, 2);
        cep2_size(j) = size(cep2.lls{i,j}, 2);
        ep_size(j) = size(ep.lls{i,j}, 2);
        lp_size(j) = size(lp.lls{i,j}, 2);
        vb_size(j) = size(vb.lls{i,j}, 2);
    end
 
    cep_length = max(cep_size);
    cep2_length = max(cep2_size);
    ep_length = max(ep_size);
    lp_length = max(lp_size);
    vb_length = max(vb_size);
    
    cep_data = zeros(num_run, cep_length);
    cep2_data = zeros(num_run, cep2_length);
    ep_data = zeros(num_run, ep_length);
    lp_data = zeros(num_run, lp_length);
    vb_data = zeros(num_run, vb_length);
    
    for j = 1: num_run
        cep_data(j,1:cep_size(j)) = cep.lls{i,j};
        cep_data(j,setdiff(1:max(cep_size), 1:cep_size(j))) = cep.lls{i,j}(end);
        
        cep2_data(j,1:cep2_size(j)) = cep2.lls{i,j};
        cep2_data(j,setdiff(1:cep2_length, 1:cep2_size(j))) = cep2.lls{i,j}(end);
        
        ep_data(j,1:ep_size(j)) = ep.lls{i,j};
        ep_data(j,setdiff(1:ep_length, 1:ep_size(j))) = ep.lls{i,j}(end);
        
        lp_data(j,1:lp_size(j)) = lp.lls{i,j};
        lp_data(j,setdiff(1:lp_length, 1:lp_size(j))) = lp.lls{i,j}(end);
        
        vb_data(j,1:vb_size(j)) = vb.lls{i,j};
        vb_data(j,setdiff(1:vb_length, 1:vb_size(j))) = vb.lls{i,j}(end);
    end
    
    cep_data = mean(cep_data);
    cep2_data = mean(cep2_data);
    ep_data = mean(ep_data);
    lp_data = mean(lp_data);
    vb_data = mean(vb_data);
    
    subplot(2,3,i);
    hold on;
    plot(1:cep_length, cep_data, 'k', 'LineWidth',3);
    
    plot(1:cep2_length, cep2_data, 'r', 'LineWidth',3);
    
    plot(1:ep_length, ep_data, 'b', 'LineWidth',3);
    
    plot(1:lp_length, lp_data, 'c', 'LineWidth',3);
    
    plot(1:vb_length, vb_data, 'm', 'LineWidth',3);
    
    title(sprintf('%s', data_set{i}));
    
    set(gca, 'FontSize', 20);
%     ylim([0.1, 10^5]);
    xlim([0,200]);
    legend('CEP-1', 'CEP-2', 'EP', 'LP', 'VB'); 
    xlabel('Iteration', 'FontSize', 20);
    ylabel('Log-Likelihood', 'FontSize', 20);
    
    hold off;
    
end