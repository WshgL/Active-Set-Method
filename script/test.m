% MATLAB 验证脚本

% 清除环境
clear; clc;

% 定义测试用例
test_cases = {};

% Test Case 1: 基本测试
test_cases{end+1} = struct(...
    'description', '基本测试',...
    'N', 3, 'M', 2,...
    'u_max', [5;5;5], 'u_min', [0;0;0], 'u_init', [1;1;1],...
    'b', [7;8],...
    'A', [1,2,3;4,5,6]);

% Test Case 2: 收紧的约束
test_cases{end+1} = struct(...
    'description', '收紧的约束',...
    'N', 3, 'M', 2,...
    'u_max', [2;2;2], 'u_min', [0.5;0.5;0.5], 'u_init', [1;1;1],...
    'b', [7;8],...
    'A', [1,2,3;4,5,6]);

% Test Case 3: 超定系统
test_cases{end+1} = struct(...
    'description', '超定系统',...
    'N', 2, 'M', 4,...
    'u_max', [10;10], 'u_min', [-10;-10], 'u_init', [0;0],...
    'b', [2;3;5;7],...
    'A', [1,2;2,1;3,4;4,3]);

% Test Case 4: 欠定系统
test_cases{end+1} = struct(...
    'description', '欠定系统',...
    'N', 4, 'M', 2,...
    'u_max', [5;5;5;5], 'u_min', [1;1;1;1], 'u_init', [2;2;2;2],...
    'b', [10;14],...
    'A', [1,2,1,2;2,1,2,1]);

% Test Case 5: 仅有上界
test_cases{end+1} = struct(...
    'description', '仅有上界',...
    'N', 3, 'M', 3,...
    'u_max', [2;2;2], 'u_min', [-Inf;-Inf;-Inf], 'u_init', [0;0;0],...
    'b', [1;2;3],...
    'A', eye(3));

% Test Case 6: 仅有下界
test_cases{end+1} = struct(...
    'description', '仅有下界',...
    'N', 3, 'M', 3,...
    'u_max', [Inf;Inf;Inf], 'u_min', [1;1;1], 'u_init', [0;0;0],...
    'b', [1;2;3],...
    'A', eye(3));

% Test Case 7: 无约束
test_cases{end+1} = struct(...
    'description', '无约束',...
    'N', 3, 'M', 2,...
    'u_max', [Inf;Inf;Inf], 'u_min', [-Inf;-Inf;-Inf], 'u_init', [0;0;0],...
    'b', [7;8],...
    'A', [1,2,3;4,5,6]);

% Test Case 8: 变量为负数
test_cases{end+1} = struct(...
    'description', '变量为负数',...
    'N', 2, 'M', 2,...
    'u_max', [0;0], 'u_min', [-5;-5], 'u_init', [-1;-1],...
    'b', [-3;-4],...
    'A', [1,2;3,4]);

% 运行所有测试用例
options = optimoptions('quadprog', 'Algorithm', 'active-set', 'Display', 'none');

for idx = 1:length(test_cases)
    tc = test_cases{idx};
    disp(['Test Case ', num2str(idx), ': ', tc.description]);
    N = tc.N;
    M = tc.M;
    u_max = tc.u_max;
    u_min = tc.u_min;
    u_init = tc.u_init;
    b = tc.b;
    A = tc.A;
    
    % 构建二次规划问题
    H = A' * A;
    f = -A' * b;
    
    % 求解二次规划问题
    [u_quadprog, fval, exitflag, output, lambda] = quadprog(H, f, [], [], [], [], u_min, u_max, u_init, options);
    
    % 输出结果
    disp('Optimal solution u:');
    disp(u_quadprog');
    disp(['Residual norm ||A*u - b||: ', num2str(norm(A*u_quadprog - b))]);
    disp(' ');
end