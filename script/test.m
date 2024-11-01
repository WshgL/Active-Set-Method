% MATLAB Verification Script

% Clear environment
clear; clc;

% Define test cases
test_cases = {};

% Test Case 1: Basic Test
test_cases{end+1} = struct(...
    'description', 'Basic Test',...
    'N', 3, 'M', 2,...
    'u_max', [5;5;5], 'u_min', [0;0;0], 'u_init', [1;1;1],...
    'b', [7;8],...
    'A', [1,2,3;4,5,6]);

% Test Case 2: Tightened Constraints
test_cases{end+1} = struct(...
    'description', 'Tightened Constraints',...
    'N', 3, 'M', 2,...
    'u_max', [2;2;2], 'u_min', [0.5;0.5;0.5], 'u_init', [1;1;1],...
    'b', [7;8],...
    'A', [1,2,3;4,5,6]);

% Test Case 3: Overdetermined System
test_cases{end+1} = struct(...
    'description', 'Overdetermined System',...
    'N', 2, 'M', 4,...
    'u_max', [10;10], 'u_min', [-10;-10], 'u_init', [0;0],...
    'b', [2;3;5;7],...
    'A', [1,2;2,1;3,4;4,3]);

% Test Case 4: Underdetermined System
test_cases{end+1} = struct(...
    'description', 'Underdetermined System',...
    'N', 4, 'M', 2,...
    'u_max', [5;5;5;5], 'u_min', [1;1;1;1], 'u_init', [2;2;2;2],...
    'b', [10;14],...
    'A', [1,2,1,2;2,1,2,1]);

% Test Case 5: Upper Bound Only
test_cases{end+1} = struct(...
    'description', 'Upper Bound Only',...
    'N', 3, 'M', 3,...
    'u_max', [2;2;2], 'u_min', [-Inf;-Inf;-Inf], 'u_init', [0;0;0],...
    'b', [1;2;3],...
    'A', eye(3));

% Test Case 6: Lower Bound Only
test_cases{end+1} = struct(...
    'description', 'Lower Bound Only',...
    'N', 3, 'M', 3,...
    'u_max', [Inf;Inf;Inf], 'u_min', [1;1;1], 'u_init', [0;0;0],...
    'b', [1;2;3],...
    'A', eye(3));

% Test Case 7: No Constraints
test_cases{end+1} = struct(...
    'description', 'No Constraints',...
    'N', 3, 'M', 2,...
    'u_max', [Inf;Inf;Inf], 'u_min', [-Inf;-Inf;-Inf], 'u_init', [0;0;0],...
    'b', [7;8],...
    'A', [1,2,3;4,5,6]);

% Test Case 8: Negative Variables
test_cases{end+1} = struct(...
    'description', 'Negative Variables',...
    'N', 2, 'M', 2,...
    'u_max', [0;0], 'u_min', [-5;-5], 'u_init', [-1;-1],...
    'b', [-3;-4],...
    'A', [1,2;3,4]);

% Run all test cases
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
    
    % Construct quadratic programming problem
    H = A' * A;
    f = -A' * b;
    
    % Solve the quadratic programming problem
    [u_quadprog, fval, exitflag, output, lambda] = quadprog(H, f, [], [], [], [], u_min, u_max, u_init, options);
    
    % Output results
    disp('Optimal solution u:');
    disp(u_quadprog');
    disp(['Residual norm ||A*u - b||: ', num2str(norm(A*u_quadprog - b))]);
    disp(' ');
end
