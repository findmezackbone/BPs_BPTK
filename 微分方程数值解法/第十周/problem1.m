%第十周作业 p164 1
clear;clc
n = 2:2; %考虑变步长

for i = n
    N = 2^i; 
    h = 1/N; %步长
    
    U = diag(ones(1,N-2),1); V = diag(ones(1,N-2),-1); I = diag(ones(1,N-1));
    D = (4/(h^2))*I-(1/(h^2))*U-(1/(h^2))*V;
    
    A = kron(I,D) + kron(U+V,-(1/(h^2))*I); %此即题目中矩阵A

    bigI = diag(ones(1,(N-1)^2)); %对应于0阶项前系数

    
end