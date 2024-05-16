clear;clc
%p144.3
N=10; %由于三点差分格式中需要u_N恰好对应于u(1)，因此先令N，在计算h
h=1/N;

f=ones(N,1);
tlist=[0];
for i=1:N-1
    tlist=[tlist,i*h];
end
tlist=[tlist,1];

%%
a=1;b=0;c=0;
U1num=solve3(N,a,b,c,f); %数值解
u1 = @(x) -x^2/2+3*x/4; U1true=[];
for j = tlist
    U1true=[U1true;u1(j)];
end

plot(tlist,U1true,'bo-');hold on;plot(tlist,U1num,'rx-');title('当a=1, b=c=0, N=10时, 虚拟点法精确解与数值解的图像');legend('精确解','数值解');

%%
a=1;b=0;c=0;

n=2:10;
Bias=[];
u1 = @(x) -(x^2)/2+3*x/4;

for i = n
    h=2^(-i);N=2^(i);
    f = ones(N,1);
    tlist = [0]; U1true=[];

    for j = 1:N
        tlist = [tlist,j*h];
    end

    for k = tlist
        U1true=[U1true;u1(k)];
    end
    U1num = solve3(N,a,b,c,f); 
    bias = abs(U1num-U1true)./abs(U1true);
    Bias=[Bias;bias(end)];
end
y = -2*n*(log(2)/log(exp(1)));
semilogy(n,Bias,'ro-'); hold on; plot(n,exp(y),'blue');title('数值解与真解的相对误差与n的关系');xlabel('n: h=2^{-n}');legend('数值解与近似解的误差阶','斜率-2ln2')

%%
a=1;b=-5;c=3;

n=2:10;
Bias=[];
l1 = (-b+sqrt(b^2+4*c))/(-2); l2 = (-b-sqrt(b^2+4*c))/(-2);
u1 = @(x) (exp(l2)*(1+l2)-1)/(c*(exp(l1)*(1+l1)-exp(l2)*(1+l2)))*exp(l1*x)-(exp(l1)*(1+l1)-1)/(c*(exp(l1)*(1+l1)-exp(l2)*(1+l2)))*exp(l2*x)+1/c;

for i = n
    h=2^(-i);N=2^(i);
    f = ones(N,1);
    tlist = [0]; U1true=[];

    for j = 1:N
        tlist = [tlist,j*h];
    end

    for k = tlist
        U1true=[U1true;u1(k)];
    end
    U1num = solve3(N,a,b,c,f); 
    bias = abs(U1num-U1true)./abs(U1true);
    Bias=[Bias;bias(end)];
end
y = -2*n*(log(2)/log(exp(1)));
semilogy(n,Bias,'ro-'); hold on; plot(n,exp(y),'blue');title('a=1, b=-5, c=-3时, 数值解与真解的相对误差与n的关系');xlabel('n: h=2^{-n}');legend('数值解与近似解的误差阶','斜率-2ln2')