% QDA scalability. We increment both, count of samples and count of
% features.

% We measure runtime of model training and scoring of the training data. 
% Once for the model from scratch. Once with incremental learning.

ncol_max = 3400; % The maximal speed-up is obtained around 9000 rows
nrow_max = 3*ncol_max;

% Initialization
logger_x = [];
logger_scratch = [];
logger_incremental = [];
x = rand(nrow_max, ncol_max) + 100000*eye(nrow_max, ncol_max);
y = 1 + (rand(nrow_max, 1)>0.5);

for ncol=2:50:ncol_max
    % Data preparation for incremental versions
    nrow = 3*ncol;
    x1 = x(1:nrow, 1:ncol-1);
    x2 = x(1:nrow, ncol);
    y_subset = y(1:nrow);

    % Measure runtime of QDA from scratch
    % It has a tiny advantage by working on the matrix without the new feature 
    tic;
    [~, new_struct] = qda_chol_incremental(x1, y_subset);
    logger_scratch = [logger_scratch, toc]
    
    tic;
    obtained = qda_chol_incremental(x2, y_subset, new_struct);
    logger_incremental = [logger_incremental, toc];
    
    logger_x = [logger_x, nrow];
end

%% Fit & plot
ft = fittype( 'poly1' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Robust = 'Bisquare';
x = logger_x(1:70)';
y = (logger_scratch(1:70)./logger_incremental(1:70))';
[fitresult, gof] = fit(x, y, ft, opts);

clf
h = plot(fitresult, x, y, '.');
title('Speed up of QDA insert vs. QDA from scratch')
ylabel('Speed up')
xlabel('Samples (features is 1/3 of samples)')
legend('Measured', 'Linear fit', 'location', 'northwest')
axis([0,10000, 0,12])

set(gcf, 'PaperPosition', [0 0.05 5 3]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 3.05]); %Keep the same paper size
saveas(gcf, 'scalability_both.pdf', 'pdf')
