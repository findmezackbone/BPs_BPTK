clear;clc
%第二问
n = 2:8; %变步长
Bias=[];

for i = n
    h = 2^(-i); %步长
    N = 2^i; 
    
    f = ones(N-1,1);

    %在构造三点差分矩阵
    A=(1/h^2)*(diag(2*ones(1,N-1))+diag(-1*ones(1,N-2),-1)+diag(-1*ones(1,N-2),1));
        
    %构造时间序列，改造三点差分矩阵
    tlist = [0];
    for j = 1:N-1
        v=j*h;
        tlist = [tlist,v];
        if v >= (1/2)
            A(j,:)=2*A(j,:);
        end
    end
    tlist = [tlist,1];
    
    %精确值
    Utrue = [];
    u1 = @(x) (x^2)/2-(5/12)*x;
    u2 = @(x) (x^2)/4-(5/24)*x-1/24;
    for j = tlist
        if j < 1/2
            Utrue = [Utrue;u1(j)];
        end
        if j >=1/2
            Utrue = [Utrue;u2(j)];
        end
    end

    %数值解
    Unum=inv(-A)*f;
    Unum = [0;Unum;0];

    l = abs(Unum-Utrue);

    bias = max(l);

    Bias = [Bias;bias];
end
plot(n,Bias,'bo-');title('绝对误差最大值与n关系');xlabel('n: h=2^{-n}');
plot(tlist,l,'r');title('当h=2^{-8}时,绝对误差与x_i关系');xlabel('x_i');
