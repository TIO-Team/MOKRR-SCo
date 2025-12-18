function offspring = generate_offspring(parents, mutation_rate, use_weighted_mutation, var_weight)

if nargin < 3
    use_weighted_mutation = false;
end
if nargin < 4
    var_weight = ones(1, size(parents, 2));
end

[k, d] = size(parents);

% 为每个父代选择交叉配对
idx1 = 1:k;  % 每个父代生成一个子代，不需要随机选择
p1 = parents(idx1, :);

%随机选择父代进行交叉
rand_mask = rand(k, d) < 0.5;  % 随机选择交叉的位
p2 = parents(randperm(k), :); % 随机选择一个父代进行交叉

child = p1;  % 初始化子代
child(rand_mask) = p2(rand_mask);  % 进行交叉操作

%加权或均匀变异
if use_weighted_mutation
    norm_weight = var_weight / (max(var_weight) + eps);
    prob_mut = mutation_rate * (1 + norm_weight);  
else
    prob_mut = mutation_rate * ones(1, d);
end

%变异
mut_mask = rand(k, d) < repmat(prob_mut, k, 1);
child(mut_mask) = 1 - child(mut_mask); 

offspring = child;
end


