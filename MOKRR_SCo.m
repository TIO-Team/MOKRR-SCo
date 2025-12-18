classdef MOKRR_SCo < ALGORITHM
% <multi> <binary> <expensive>
% Multi-objective Kernel Ridge Regression-Assisted Subproblem Co-optimization

    methods
        function main(Algorithm,Problem)
            %% 初始化
            %% Parameter setting
            warning off; 
            max_num_parents= Problem.N; %最大种群数量
            min_num_parents= 4;%最小
            num_parents=max_num_parents;%初始种群数量=最大种群数量
            use_weighted_mutation =true;            
            mutation_rate=1/Problem.D;%变异率
            inner_tmax=20;%进化求解子问题的迭代次数,内循环最大迭代次数   
            Population = Problem.Initialization();%% 参数初始种群，种群大小为Problem.N      
            Arc = Population; %将初始化得到的种群复制给存档 Arc 
            strategy='best'; %从存档A中选择父代来产生子代的方式
            
            %% 设置初始子问题个数
            Sub_problem_No=21;
            [W,~] = UniformPoint(Sub_problem_No,Problem.M);
            Sub_problem_No=size(W,1);


            %% 设置初始理想点
            Z=min(Population.objs,[],1); %理想点                    
            
            %% 每个解的子问题目标函数聚合值
            Arc_g_objs=zeros(size(Arc.objs,1),Sub_problem_No);
            for i = 1:Sub_problem_No               
                Arc_g_objs(:,i) = max(abs(Arc.objs-repmat(Z,size(Arc.objs,1),1)).*repmat(W(i,:),size(Arc.objs,1),1),[],2);  
            end      
            
            %% 迭代优化 Optimization
            while Algorithm.NotTerminated(Arc)                
                for k=1:Sub_problem_No  
                    
                    [parents, parent_idx] = select_parents(Arc.decs, Arc_g_objs(:,k), num_parents, strategy);                    
                    objs=Arc.objs;
                    parents_objs=objs(parent_idx,:); 
                    parents_g_objs=Arc_g_objs(parent_idx,k);
                    tempZ=Z; 
                    model = train_binary_surrogate(Arc.decs, Arc.objs);
                   
                    inner_t=1;
                    while inner_t<=inner_tmax                                    
                        var_weight = compute_variable_weight(parents, parents_g_objs);
                        offsprings = generate_offspring(parents, mutation_rate, use_weighted_mutation, var_weight);
                        off_objs_pred = predict_binary_surrogate(model, offsprings);
                        tempZ = min([tempZ;off_objs_pred]);
                        offspring_g_objs = max(abs(off_objs_pred-repmat(tempZ,size(offsprings,1),1)).*repmat(W(k,:),size(offsprings,1),1),[],2); 
                        parents_g_objs = max(abs(parents_objs-repmat(tempZ,size(offsprings,1),1)).*repmat(W(k,:),size(offsprings,1),1),[],2);                                                      
                        %基于第k个子问题的目标函数值进行优胜劣汰的环境选择
                        index=find(offspring_g_objs<=parents_g_objs);
                        parents(index,:)=offsprings(index,:); 
                        parents_objs(index,:)=off_objs_pred(index,:);
                        parents_g_objs(index)=offspring_g_objs(index);
                        inner_t=inner_t+1;                    
                    end 

                    %% 在最终种群中选择一个没有被评估过的最稀疏的解
                    % 计算种群中的点到Arc中解的汉明距离
                    dis=pdist2(parents,Arc.decs, 'hamming'); 
                    % 求种群中每个解到Arc中解的最小距离
                    mindis=min(dis,[],2);
                    % 对种群中的解基于最小距离从大到小排序
                    [~,sortid]=sort(mindis,'descend');
                    % 在种群中选择一个最稀疏的未评估过的解进行真实评估
                    for i=1:num_parents
                        test_id=sortid(i);
                        Final_Offspring=offsprings(test_id,:);
                        if ~ismember(Final_Offspring,Arc.decs,'rows')
                           off=Problem.Evaluation(Final_Offspring);
                           Arc=[Arc,off];  
                           Z=min(Z,off.obj);  
                           Arc_g_objs=[];
                               for i = 1:Sub_problem_No               
                                   Arc_g_objs(:,i) = max(abs(Arc.objs-repmat(Z,size(Arc.objs,1),1)).*repmat(W(i,:),size(Arc.objs,1),1),[],2);  
                               end 
                           break;
                        end
                    end
                    Algorithm.NotTerminated(Arc);
                    if Problem.FE==Problem.maxFE
                       break;
                    end
                end  
                %% 线性减少内循环优化代理子问题的种群大小，保证效率的同时，加快运行速度
                num_parents=round(min_num_parents+(max_num_parents-min_num_parents)*(1-Problem.FE/Problem.maxFE));                              
            end            
        end  
    end
end
