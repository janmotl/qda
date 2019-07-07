# Quadratic Discriminant Analysis on a stream of features 

Traditional implementations of Quadratic Discriminant Analysis (QDA) work in an offline mode - when we want to extend the current model by a new feature, we have to retrain the model from scratch. But as we do so, we throw out a lot of information that can be reused. 

But this implementation of QDA reuses as much computation as possible when a new feature is added into the model. 

#### Requirements
Octave 4.4 or newer. While the code also works in MATLAB, `qrinsert` function is simulated with `qr` function, which is slower.

#### Example
```matlab
% Generate dummy data
nrow = 12;
ncol = 4;
rand("seed",2001); % Octave dialect
x = randn(nrow,ncol);
y = 1.0 + (rand(nrow,1)>0.5);
x(y==1,:) = x(y==1,:) + 2;

% Train the model
[scored,model] = qda_chol_incremental(x(:,1:ncol-1),y)

% Update the model with a new feature
[scored,model] = qda_chol_incremental(x(:,ncol),y,model)
```
