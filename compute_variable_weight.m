function var_weight = compute_variable_weight(X, subY)

[n, d] = size(X);
if size(subY,1) ~= n
    error('X 和 subY 行数必须一致');
end

theta = zeros(1, d);  
for i = 1:n-1
    for j = i+1:n
        delta = abs(subY(i) - subY(j));         % 子问题函数值差异
        diff = X(i,:) ~= X(j,:);                % 不同变量的位置
        theta = theta + delta * diff;           % 累加变量变化导致的函数差异
    end
end
% 归一化处理
var_weight = theta / (sum(theta) + eps);        % 避免除0
end
