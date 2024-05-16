clear;clc
%问题一
%首先构造三点差分格式所对应的矩阵
N=50; %由于三点差分格式中需要u_N恰好对应于u(1)，因此先令N，在计算h
h=1/N;

f=ones(N-1,1);
tlist=[0];
for i=1:N-1
    tlist=[tlist,i*h];
end
tlist=[tlist,1];



%%
a=1;b=0;c=0;
U1num=solve313(N,a,b,c,f); %数值解
u1 = @(x) x*(1-x)/(2*a); U1true=[];
for j = tlist
    U1true=[U1true;u1(j)];
end

plot(tlist,U1true,'bo-');hold on;plot(tlist,U1num,'rx-');title('当a=1, b=c=0时, 精确解与数值解的图像');legend('精确解','数值解');

%%
a=1;b=10;c=0;
U2num=solve313(N,a,b,c,f);
u2=@(x) (exp(b*x/a)-1)/(b*(1-exp(b/a)))+x/b; U2true=[];
for j = tlist
    U2true=[U2true;u2(j)];
end

plot(tlist,U2true,'bo-');hold on;plot(tlist,U2num,'rx-');title('当a=1, b=10, c=0时, 精确解与数值解的图像');legend('精确解','数值解');

%%
a=1;b=0;c=10;
U3num=solve313(N,a,b,c,f);
l1 = (b+sqrt(b^2+4*a*c))/(2*a); l2= (b-sqrt(b^2+4*a*c))/(2*a);
u3=@(x) ((exp(l2)-1)/(c*(exp(l1)-exp(l2))))*exp(l1*x)-((exp(l1)-1)/(c*(exp(l1)-exp(l2))))*exp(l2*x)+1/c; U3true=[];
for j = tlist
    U3true=[U3true;u3(j)];
end

plot(tlist,U3true,'bo-');hold on;plot(tlist,U3num,'rx-');title('当a=1, b=0, c=10时, 精确解与数值解的图像');legend('精确解','数值解');

%%
a=1;b=-10;c=0;
U4num=solve313(N,a,b,c,f);
u4=@(x) (exp(b*x/a)-1)/(b*(1-exp(b/a)))+x/b; U4true=[];
for j = tlist
    U4true=[U4true;u4(j)];
end

plot(tlist,U4true,'bo-');hold on;plot(tlist,U4num,'rx-');title('当a=1, b=-10, c=0时, 精确解与数值解的图像');legend('精确解','数值解');

%%
%汇总解在不同的条件下的图像
subplot(1,2,1); plot(tlist,U1true,'blue'); hold on; plot(tlist,U2true,'black'); hold on; plot(tlist,U3true,'bo-'); hold on; plot(tlist,U4true,'rx-'); title('精确解在不同条件下的图像');legend('b=0,c=0','b=10,c=0','b=0,c=10','b=-10,c=0')
subplot(1,2,2); plot(tlist,U1num,'blue'); hold on; plot(tlist,U2num,'black'); hold on; plot(tlist,U3num,'bo-'); hold on; plot(tlist,U4num,'rx-'); title('数值解在不同条件下的图像');legend('b=0,c=0','b=10,c=0','b=0,c=10','b=-10,c=0')

%%
%误差与网格关系:变步长h=2^(-n)，以a=1,b=c=0为例
n = 2:10;
%a=1;b=0;c=0; Bias = []; u1 = @(x) x*(1-x)/(2*a); U1=[];
%a=1;b=10;c=0; Bias = []; u1=@(x) (exp(b*x/a)-1)/(b*(1-exp(b/a)))+x/b; U1=[];
%a=1;b=0;c=10; Bias = []; l1 = (b+sqrt(b^2+4*a*c))/(2*a); l2= (b-sqrt(b^2+4*a*c))/(2*a); u1=@(x) ((exp(l2)-1)/(c*(exp(l1)-exp(l2))))*exp(l1*x)-((exp(l1)-1)/(c*(exp(l1)-exp(l2))))*exp(l2*x)+1/c; U1=[];
a=1;b=-10;c=0; Bias = []; u1=@(x) (exp(b*x/a)-1)/(b*(1-exp(b/a)))+x/b; U1=[];
for i = n
    h=2^(-i); N=2^(i);
    f=ones(N-1,1);
    tlist=[0];
    for j=1:N-1
        tlist=[tlist,j*h];
    end
    tlist=[tlist,1];
    U=[];
    U1 = solve313(N,a,b,c,f);
    for j = tlist
        
        U=[U;u1(j)];
    end
    bias = abs(U-U1);
    Bias=[Bias;max(bias)];
end

plot(n,Bias,'ro-'); title('步长一定时, 全局绝对误差的最大值与n的关系'); xlabel('i: \Delta{t}=2^{-i}');
% y3 = -2*n*(log(2)/log(exp(1))); 
% semilogy(n,exp(y3),'b'); hold on;semilogy(n,Bias,'ro-'); title('全局绝对误差最大值的收敛阶'); xlabel('i: \Delta{t}=2^{-i}'); legend('斜率-2ln2','数值解')