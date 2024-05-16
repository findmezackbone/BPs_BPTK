%%
%用AM AB三阶方法考虑收敛阶（初值为0or1）
clear;clc

k = 1000; %lambda
ui = 0; %初值
ti = 0; %最初时间
T = 1; %考虑在ti到ti+T上的数值解
f = @(t,u) k*(-u+cos(t)); 
u = @(t) (-k^2/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %初值为0的精确表达式
%u = @(t) (1/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %初值为1的精确表达式

RBAB3 = [];RBAM3=[];RBG3=[]; %后续用于存放各格式的末项相对误差
n = 2:10;

for i = n
    dt = 2^(-i); %变步长
    N = floor((T-ti)/dt);

    %时间节点
    Tlist = [ti];
    for j = 1:N
        Tlist = [Tlist;ti+j*dt];
    end
    
    %精确解
    Utrue = [ui];
    for j = 1:N
        Utrue = [Utrue;u(ti+j*dt)];
    end

    %AM4
    UAM3 = AM3(f,ti,Utrue,dt,T);

    UAB3 = AB3(f,ti,Utrue,dt,T);

    %Gear4
    UG3 = GEAR2(f,ti,Utrue,dt,T);
    
    RBAM3 = [RBAM3;(abs(Utrue(end)-UAM3(end))/abs(Utrue(end)))];
    RBAB3 = [RBAB3;(abs(Utrue(end)-UAB3(end))/abs(Utrue(end)))];
    RBG3 = [RBG3;(abs(Utrue(end)-UG3(end))/abs(Utrue(end)))];
    y1 = -4*n*(log(2)/log(exp(1)));
    y2 = -3*n*(log(2)/log(exp(1)));
    y3 = -2*n*(log(2)/log(exp(1)));
    y4 = -1*n*(log(2)/log(exp(1)));

end

%画图观察收敛阶
%semilogy(n,exp(y4),'black') ;hold on;semilogy(n,exp(y3),'yellow');hold on;semilogy(n,RBG4,'bo-');title('当初值为1, \lambda为1000, Gear4阶方法末项收敛阶');xlabel('i:\Delta{t}=2^{-i}');legend('斜率-ln2','斜率-2ln2','Gear4')
%semilogy(n,exp(y4),'black') ;hold on;semilogy(n,exp(y3),'yellow');hold on;semilogy(n,RBG3,'bo-');hold on;title('当初值为0, \lambda为1000, Gear二阶方法末项收敛阶');xlabel('i:\Delta{t}=2^{-i}');legend('斜率-ln2','斜率-2ln2','Gear2')
%semilogy(n,RBAB3,'rx-');hold on;
semilogy(n,RBAM3,'bo-');title('当初值为0, \lambda为1000, Adams三阶方法末项收敛阶');xlabel('i:\Delta{t}=2^{-i}');legend('AM3')
