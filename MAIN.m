% Provided inertias (Iz), taking only the first 12 elements
Iz = [5.12e6, 8.42,0, 8.42,0, 8.42,0, 8.42, 8.85, 1.62, 2.06, 0.0037,0.377];

% Define the elements of Mz matrix
Mz = zeros(7,7); % Initialize a 6x6 matrix with zeros

% Fill the matrix according to the given structure
Mz(1,1) = Iz(1) + Iz(2)/3;
Mz(1,2) = Iz(2)/6;
Mz(2,1) = Mz(1,2);

Mz(2,2) = Iz(3) + Iz(2)/3 + Iz(4)/3;
Mz(2,3) = Iz(4)/6;
Mz(3,2) = Mz(2,3);

Mz(3,3) = Iz(5) + Iz(4)/3 + Iz(6)/3;
Mz(3,4) = Iz(6)/6;
Mz(4,3) = Mz(3,4);

Mz(4,4) = Iz(7) + Iz(6)/3 + Iz(8)/3;
Mz(4,5) = Iz(8)/6;
Mz(5,4) = Mz(4,5);

Mz(5,5) = Iz(9) + Iz(8)/3 + Iz(10)/3;
Mz(5,6) = Iz(10)/6;
Mz(6,5) = Mz(5,6);

Mz(6,6) = Iz(11) + Iz(10)/3+Iz(12)/3;
Mz(6,7) = Iz(12)/6;
Mz(7,6) = Mz(6,7);

Mz(7,7) = Iz(13) + Iz(12)/3;


Ixz = -467;
Inz = 1538;
%计算kz和mz
% 定义系统参数
m = [3231, 40.30,0, 40.30, 0,40.30,0, 40.30, 54.93, 40.52, 16.5, 5.92, 2310.5+50];
L = [120, 14.87,0, 14.87, 0,14.87, 0,14.87, 1, 40, 0.45, 0.95, 5.3];
g = 9.70; % 重力加速度 (m/s^2)
r = [0, 0.457,0, 0.457, 0,0.457,0, 0.457, 0, 0.20, 0, 0.025, 0, 0]; % r 数据

% 计算 mu_k 的值
mu = zeros(1,14);
mu(end) = 0; % mu_n+2 = 0
for k = length(m):-1:1
    mu(k) = m(k) + mu(k+1);
end


% 计算 K_k 的值
K = zeros(1, 12);
for k = 1:12
    K(k) = (m(k)/2 + mu(k+1)) * g * r(k)^2 / L(k);
end

% 定义 Kz 矩阵
Kz = [K(2)      -K(2)    0         0         0         0    0;
      -K(2)     K(2)+K(4) -K(4)    0         0         0   0;
      0         -K(4)    K(4)+K(6) -K(6)    0         0    0;
      0         0        -K(6)    K(6)+K(8) -K(8)    0    0;
      0         0        0        -K(8)    K(8)+K(10) -K(10)   0;
      0         0        0        0         -K(10)  K(10)+K(12)   -K(12)
      0         0        0         0          0        -K(12)      K(12)    ];
% Kz = [K(2)      -K(2)    0         0         0         0;
%       -K(2)     K(2)+K(4) -K(4)    0         0         0;
%       0         -K(4)    K(4)+K(6) -K(6)    0         0;
%       0         0        -K(6)    K(6)+K(8) -K(8)    0;
%       0         0        0        -K(8)    K(8)+K(10) -K(10);
%       0         0        0        0         -K(10)  K(10)];

% 显示 Kz 矩阵
disp('Matrix Kz:');
disp(Kz);


% 求解广义特征值问题 (M * ω^2 = K)
[eigenvectors, eigenvalues] = eig(Kz, Mz);  % 求解广义特征值问题

% 提取特征值 (ω^2)，特征值是对角线上的元素
omega_squared = diag(eigenvalues);

% 取平方根得到 ω
omega2 = sqrt(omega_squared);


% 求解广义特征值问题以获得特征频率和模态矩阵
Phi = eigenvectors;  % 模态矩阵（特征向量矩阵）
% 假设阻尼比 zeta
zeta = 1e-3;  % 假设阻尼比为 10^-3

% 构造模态阻尼矩阵 C_m
C_m_diag = 2 * zeta * omega2;
C_m = diag(C_m_diag);

% 通过模态矩阵 Phi 计算物理阻尼矩阵 C
C2 = Phi * C_m * Phi';

% 输出结果
disp('特征频率 omega2:');
disp(omega2);
disp('模态矩阵 Phi:');
disp(Phi);
disp('模态阻尼矩阵 C_m:');
disp(C_m);
disp('物理阻尼矩阵 C2:');
disp(C2);








% 定义参数
rho = [0.32, 0.5, 0.5, 0.5, 0.5, 0.463, 0.5, 0.5, 0.5, 0.5, 0.605]; % 质心位置比例
m = [3231, 40.30, 40.30, 40.30, 40.30, 54.93, 40.52, 16.5, 5.92, 33.5, 2277+50]; % 质量数组
L = [120, 14.87, 14.87, 14.87, 14.87, 1, 40, 0.45, 0.95, 0.7, 4.6]; % 长度数组
mu = [5820.57+50, 2589.57+50, 2549.27+50, 2508.97+50, 2468.67+50, 2428.37+50, 2373.44+50, 2332.92+50, 2316.42+50, 2310.5+50, 2277+50, 0]; % 累积质量数组
g = 9.70; % 重力加速度
lambda = 0.1; % 参数λ
Ix = [5.12e6, 742, 742, 742, 742, 6.51, 1489, 0.278, 0.445, 1.37, 8510];
% 初始化刚度矩阵Kx为一个对角矩阵
n = length(m); % 元素的数量
Kx = zeros(n+1, n+1); % 创建一个n+1 x n+1的零矩阵

% 计算对角线上的元素，根据公式 (A.2)
for i = 2:n+1
    if i == 2
        Kx(i, i) = (1 - rho(1)) * g * L(1) * mu(i)  - lambda * g * L(1) * mu(1);
    else
        Kx(i, i) = (m(i-1) * rho(i-1) + mu(i)) * g * L(i-1);
    end
end

% 由于公式中第一个元素为0，我们设置Kx(1,1)为0
Kx(1, 1) = 0;

% 显示结果
disp('Stiffness matrix Kx:');
disp(Kx);
% 初始化矩阵
Mx = zeros(n+1, n+1);

% 计算上三角部分（包括主对角线）
for i = 1:n+1
    for j = i:n+1
        if i == j
            if i == 1
                Mx(i, j) = mu(i); % 第一行第一列
            elseif i == 2
                Mx(i, j) = Ix(1) + (1 - rho(1))^2 * L(1)^2 * mu(2); % 修改的第二行第二个元素
            else
                Mx(i, j) = Ix(i-1) + (rho(i-1)^2 * m(i-1) + mu(i)) * L(i-1)^2; % 其他主对角线元素
            end
        else
            if i == 1
                if j == 2
                    Mx(i, j) = (1 - rho(i)) * L(i) * mu(j); % 第一行第二个元素
                else
                    Mx(i, j) = (rho(j-1) * m(j-1) + mu(j)) * L(j-1); % 第一行其他元素
                end
            elseif i == 2
                Mx(i, j) = (1 - rho(i-1)) * (rho(j-1) * m(j-1) + mu(j)) * L(i-1) * L(j-1); % 第二行的特定元素
            else
                Mx(i, j) = (rho(j-1) * m(j-1) + mu(j)) * L(i-1) * L(j-1); % 上三角其他元素
            end
        end
    end
end

% 填充下三角部分
Mx = Mx + triu(Mx, 1)';


% 输出结果
disp(Mx);



% 求解广义特征值问题 (M * ω^2 = K)
[V, D] = eig(Kx, Mx);  % 求解广义特征值问题

% 提取特征值 (ω^2)，特征值是对角线上的元素
omega_squared = diag(D);

% 取平方根得到 ω
omega1 = sqrt(omega_squared);


% 求解广义特征值问题以获得特征频率和模态矩阵
Phi = V;  % 模态矩阵（特征向量矩阵）
% 假设阻尼比 zeta
zeta = 1e-3;  % 假设阻尼比为 10^-3

% 构造模态阻尼矩阵 C_m
C_m_diag1 = 2 * zeta * omega1;
C_m = diag(C_m_diag1);

% 通过模态矩阵 Phi 计算物理阻尼矩阵 C
C1 = Phi * C_m * Phi';



% 输出结果
disp('特征频率 omega1:');
disp(omega1);
disp('模态矩阵 Phix:');
disp(Phi);
disp('模态阻尼矩阵 C_m1:');
disp(C_m);
disp('物理阻尼矩阵 C1:');
disp(C1);
% 显示特征频率 ω
disp('特征频率 ω 为：');
disp(omega1);


% 提取 MX 的最后一列元素并转换为行向量
MX1 = Mx(1:end, end)'; % 转置使其成为行向量

% 定义 KX1
KX1 = 61467.8427000000;

% 显示结果
disp('MX1 as a row vector:');
disp(MX1);
disp('Value of KX1:');
disp(KX1);

% 提取 MX 的最后一列元素并转换为行向量
MZ1 = Mz(1:end, end)'; % 转置使其成为行向量

% 定义 KX1
KZ1 = Kz(7,7);

Ixz = -467;
Inz = 1538;

% % 构建 M 和 K
% M = blkdiag(Mx,Mz,Inz);
% M(12,20)=Ixz;
% M(20,12)=Ixz;
% K = blkdiag(Kx, Kz,0);
% 
% 
% 
% C= blkdiag(C1,C2,0);
% disp(C);
M = blkdiag(Mx,Inz);
M(12,13)=Ixz;
M(13,12)=Ixz;
C= blkdiag(C1,0);
K = blkdiag(Kx,0);
