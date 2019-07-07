% Incremental solution for X = B/R (X*R = B for X),
% where R is an upper triangular matrix via triangle matrix forward substitution.
% The solver assumes that we append matrices B and R with a new column.
%
% Solves X = B/R
%
%          |1 3 4|         |2 2 1|
% with B = |2 4 1| and R = |0 1 4|
%                          |0 0 3|
%
% assuming we already have a solution for:
%
%          |1 3|         |2 2|
%      B = |2 4| and R = |0 1|

function X_new = incremental_solve(R, B, X_old)
    ncol = size(B,2);
    X_new = [X_old, (B(:, ncol) - X_old(:, 1:ncol-1) * R(1:ncol-1,ncol)) / R(ncol,ncol)];
