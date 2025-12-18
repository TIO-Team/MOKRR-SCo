function Y_pred = predict_binary_surrogate(model, O)

X = model.X;
theta = model.theta;
Alpha = model.Alpha;
lambda = model.lambda;

n = size(X, 1); 
m = size(O, 1); 

K_test = zeros(m, n);
for t = 1:m
    for i = 1:n
        h = sum(theta.* (O(t,:) ~= X(i,:)));
        K_test(t, i) = exp(-lambda * h);
    end
end

Y_pred = K_test * Alpha;  
end


