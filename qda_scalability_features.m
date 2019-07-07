% QDA scalability based on the count of features.

% It works beautifully till the matrices have 1GB. 
%   runtime = 10e-08*ncol^2 + 0.0001*ncol

% Setting (the maximal matrix I can process is ~20k x 10k)
% It should hold: nrow >= 2*ncol_max.  
nrow = 10000;
ncol_max = 5000;

% Initialization
logger_x = [];
logger_scratch = [];
logger_incremental = [];
x = randn(nrow, ncol_max);
y = 1 + (rand(nrow, 1)>0.5);

for ncol=50:50:ncol_max
    % Data preparation for incremental versions
    x1 = x(:,1:ncol-1);
    x2 = x(:,ncol);

    % Measure
    tic;
    [~, new_struct] = qda_chol_incremental(x1, y);
    logger_scratch = [logger_scratch, toc]
    
    tic;
    obtained = qda_chol_incremental(x2, y, new_struct);
    logger_incremental = [logger_incremental, toc];
    
    logger_x = [logger_x, ncol];
end

%% Plot
clf
plot(logger_x, logger_scratch, 'b.:', 'linewidth', 2, 'markerSize', 20)
hold on
plot(logger_x, logger_incremental, 'r.-', 'linewidth', 2, 'markerSize', 20)
title('Training and scoring time for 10000 samples')
xlabel('Features')
ylabel('Runtime [s]')
legend('From scratch', 'Insert', 'location', 'northwest')

set(gcf, 'PaperPosition', [0 0.05 5 3]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 3.05]); %Keep the same paper size
saveas(gcf, 'scalability_features.pdf', 'pdf')
