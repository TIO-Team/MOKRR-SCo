function [parents, parent_idx] = select_parents(X, subY, num_parents, method)
% X: n x d 的历史存档解（01变量）
% subY: n x 1 子问题函数值（聚合函数值）
% num_parents: 需要选择的父代数量
% method: 父代选择策略，支持 'best', 'tournament', 'random'

if nargin < 4
    method = 'tournament';  % 默认使用锦标赛
end

n = size(X, 1);
parent_idx = zeros(1, num_parents);  % 保存索引
parents = zeros(num_parents, size(X,2));  % 保存父代

switch lower(method)
    case 'best'
        [~, sorted_idx] = sort(subY);  % 越小越好
        parent_idx = sorted_idx(1:num_parents);
        parents = X(parent_idx, :);

    case 'random'
        parent_idx = randperm(n, num_parents);
        parents = X(parent_idx, :);

    case 'tournament'
        k = 2;  % 锦标赛大小
        for i = 1:num_parents
            candidates = randperm(n, k);
            [~, best_idx] = min(subY(candidates));
            parent_idx(i) = candidates(best_idx);
            parents(i,:) = X(parent_idx(i), :);
        end
    otherwise
        error('未知的选择方法 "%s"', method);
end
end
