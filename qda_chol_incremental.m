% Incremental Quadratic Discriminant Analysis
%   [D, NEW_STRUCT] = QDA_CHOL_INCREMENTAL(X, Y) trains the model on matrix X and label Y.
%   Both, X and Y must be numerical. Label Y can be multiclass.
%   D is a matrix of scored training data. NEW_STRUCT is a structure with the model parameters.
%
%   [D, NEW_STRUCT] = QDA_CHOL_INCREMENTAL(X, Y, OLD_STRUCT) updates model OLD_STRUCT with
%   a new feature vector X.
%
% Caveats:
%   1) It is only for classification -> does not work for dimensionality reduction.
%   2) Features must not be collinear -> does not work when #features > #samples.
%
% Example:
%    nrow = 12;
%    ncol =4;
%    rand ("seed", 2001); % Octave dialect
%    x = randn(nrow, ncol);
%    y = 1.0 + (rand(nrow, 1)>0.5)
%    x(y==1,:) = x(y==1,:) + 2;
%    [D, new_struct] = qda_chol_incremental(x(:,1:ncol-1),y)
%    qda_chol_incremental(x(:,ncol),y,new_struct)

function [D, new_struct] = qda_chol_incremental(x, y, old_struct)

new_struct = struct();

% If we do not pass anything precomputed, we have to calculate some basic
% statistics. X can be a matrix or a vector.
if nargin == 2    
    new_struct.d = size(x, 2);  % Count of features
    new_struct.ngroups = length(unique(y)); % Count of classes
    new_struct.all_data = x;
    
    for k = 1:new_struct.ngroups
        new_struct(k).means = mean(x(y==k,:), 1);
        new_struct(k).prior = sum(y==k)/length(y);
        new_struct(k).gsize = sum(y==k);
        new_struct(k).centered = bsxfun(@minus, x(y==k,:), new_struct(k).means);
        scatter = new_struct(k).centered' * new_struct(k).centered;
        covariance = scatter .* (1.0/(new_struct(k).gsize - 1));
        new_struct(k).R = chol(covariance);
        new_struct(k).log_det_sigma = 2*sum(log(diag( new_struct(k).R )));
        new_struct(k).A = bsxfun(@minus, new_struct(1).all_data, new_struct(k).means) / new_struct(k).R;
    end
else    
    % Update the statistics for each label class
    new_struct = old_struct;
    
    % Count of features
    new_struct(1).d = old_struct(1).d+1;
    
    % Input data
    new_struct(1).all_data = [old_struct(1).all_data, x];

    for k = 1:length(old_struct)
        % Conditional mean
        x_mean = mean(x(y==k));
        new_struct(k).means = [old_struct(k).means, x_mean];
        
        % Centered data
        x_centered = x(y==k)-x_mean;
        new_struct(k).centered = [old_struct(k).centered, x_centered];
        
        % The inserted part of the scatter matrix
        scatter_vector = x_centered' * new_struct(k).centered;
        
        % Scatter to covariance
        % Note: Matlab is not clever: picewise-division by a constant is 
        % ~2x slower than picewise-multiplication by 1/constant.
        covariance_vector = scatter_vector .* (1.0/(new_struct(k).gsize - 1));

        % Factorized covariance matrix
        try
            new_struct(k).R = cholinsert(old_struct(k).R, new_struct(1).d, covariance_vector');
        catch   % Needed because Matlab does not implement cholinsert
            new_struct(k).R = chol(new_struct(k).centered' * new_struct(k).centered .* (1.0/(new_struct(k).gsize - 1)));
        end
                 
        % Get logarithm of the determinant
        % The determinant of A is equal to the square of the product of the
        % diagonal elements of chol(A). 
        % For better numerical stability, we sum the logarithms instead
        % of logarithming the product. 
        new_struct(k).log_det_sigma = old_struct(k).log_det_sigma + 2*log(new_struct(k).R(end, end));
                
        % Calculate X * inv(chol(cov(X)))
        % We take the result of Cholesky decomposition and 
        % perform one iteration of forward substituion.
        % Can be solved with: 
        %   A = bsxfun(@minus, new_struct(1).all_data, new_struct(k).means) / new_struct(k).R;
        %   A = bsxfun(@minus, new_struct(1).all_data, new_struct(k).means) / chol(cov(new_struct(1).all_data(y==k,:)))
        new_struct(k).A = incremental_solve(new_struct(k).R, bsxfun(@minus, new_struct(1).all_data, new_struct(k).means), old_struct(k).A);
    end
end
    
% Scoring 
if nargin == 2 
    D = NaN(size(x,1), 2);

    for k = 1:new_struct(1).ngroups    
        % MVN relative log posterior density, by group, for each sample
        D(:,k) = log(new_struct(k).prior) - .5*(sum(new_struct(k).A .* new_struct(k).A, 2) + new_struct(k).log_det_sigma);
    end

    new_struct(1).D = D;
else
    D = old_struct(1).D;
    for k = 1:new_struct(1).ngroups    
        % MVN relative log posterior density, by group, for each sample
        D(:,k) = D(:,k) - .5 * new_struct(k).A(:,end).^2  +  0.5*old_struct(k).log_det_sigma - 0.5*new_struct(k).log_det_sigma;
    end 
    new_struct(1).D = D;
end

D = new_struct(1).D;
