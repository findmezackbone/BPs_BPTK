%p186.1  此问题的代码应该会非常复杂
%具体思路如下：
%1. 任取一个定解Poisson方程问题, 不妨u(x1,x2)=x1x2(x1-1)(x2-1),
%   这样在\Omega=[0,1]x[0,1]的边界上u(x1,x2)=0
%2. 考虑步长h=2^{-n}, 且不妨\Gamma={1/2}x[0,1], 并考虑左半部分为\Omega_1, 右半部分为\Omega_2的DN
%3. 在设立子程序的时, DN方法中的\theta


clear;clc
u = @(x1,x2) x1*x2*(x1-1)*(x2-1); %这是真解
Du = @(x1,x2) 2*x1*(x1-1) + 2*x2*(x2-1); %这是真解的LAPLACE, 因此后续考虑用DN方法求解Lu=-Du

k=40;
theta = 0.3;

n = 2 : 5; %变步长

Biaslist = [];

for p = n
    
    N = 2^(p); %此时网格内点个数为(N-1)x(N-1)
    hy = 2^(-p); hx = hy/2; %步长
    hylist=[];hxlist=[];
    for j = 1:N-1
        hylist=[hylist,hy*j];
    end
    for j = 1: 2*N-1
        hxlist = [hxlist,hx*j];
    end
    g0=zeros(N-1,1);
    Unum = DN2D(Du,g0,N,k,theta);
    Utrue = [];
    for j = 1:N-1
        for i = 1:2*N-1
            Utrue(j,i) = u(i*hx,j*hy);
        end
    end
    Biaslist= [Biaslist,max(max(abs(Utrue-Unum)))];
    

    subplot(3,2,p);mesh(hxlist,hylist,Utrue-Unum);title('迭代5次, theta=0.7时的误差');xlabel('h_{x2}=2^{-n}');

end
