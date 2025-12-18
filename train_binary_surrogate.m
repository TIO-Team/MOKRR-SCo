function model = train_binary_surrogate(X, Y)
    % 输入：X: n x d 的训练样本（二进制矩阵）
    %       Y: n x k 的目标函数值矩阵
    % 输出：model: 包含 theta, Alpha, X, lambda 等信息的结构体



    lambda = 1.0;         % 加权 Hamming 核参数 
    lambda_reg = 0.01;    % 岭回归正则项

    [n, d] = size(X);
    k = size(Y, 2);

    %% 变量权重估计
    theta = zeros(1, d); 
    for i = 1:n-1
        Xi = X(i,:);      
        Yi = Y(i,:);
        for j = i+1:n
            delta = norm(Yi - Y(j,:), 2);  
            diff = Xi ~= X(j,:);           
            theta = theta + delta * diff; 
        end
    end
    theta = theta / (sum(theta) + eps);  % 避免除零错误

    %% 构造核矩阵
    K = zeros(n, n);
    for i = 1:n
        Xi = X(i,:);  
        for j = i:n
            % 计算加权Hamming距离
            hamming_dist = sum(theta .* (Xi ~= X(j,:))); 
            kij = exp(-lambda * hamming_dist);  
            K(i,j) = kij;
            K(j,i) = kij; 
        end
    end

    %% 训练岭回归模型
    Alpha = (K + lambda_reg * eye(n)) \ Y;  % 计算回归系数 

    %% 输出模型结构体
    model = struct();
    model.X = X;                % 训练样本
    model.theta = theta;        % 权重向量
    model.Alpha = Alpha;        % 回归系数
    model.lambda = lambda;      % 核参数
end




