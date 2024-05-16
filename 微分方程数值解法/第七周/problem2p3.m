%%
%直接考虑收敛阶（初值为0or1）
clear;clc

k = 1000; %lambda
ui = 0; %初值
ti = 0; %最初时间
T = 1; %考虑在ti到ti+T上的数值解
f = @(t,u) k*(-u+cos(t)); 
u = @(t) (-k^2/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %初值为0的精确表达式
%u = @(t) (1/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %初值为1的精确表达式

RBAB4 = [];RBAM4=[];RBG4=[]; %后续用于存放各格式的末项相对误差
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
    UAM4 = AM4(f,ti,Utrue,dt,T);

    UAB4 = AB4(f,ti,Utrue,dt,T);

    %Gear4
    UG4 = GEAR4(f,ti,Utrue,dt,T);
    
    RBAM4 = [RBAM4;(abs(Utrue(end)-UAM4(end))/abs(Utrue(end)))];
    RBAB4 = [RBAB4;(abs(Utrue(end)-UAB4(end))/abs(Utrue(end)))];
    RBG4 = [RBG4;(abs(Utrue(end)-UG4(end))/abs(Utrue(end)))];
    y1 = -4*n*(log(2)/log(exp(1)));
    y2 = -3*n*(log(2)/log(exp(1)));
    y3 = -2*n*(log(2)/log(exp(1)));
    y4 = -1*n*(log(2)/log(exp(1)));

end

semilogy(n,RBG4,'bo-');title('当初值为0, \lambda为1000, Gear4阶方法末项收敛阶');xlabel('i:\Delta{t}=2^{-i}');
%semilogy(n,RBAM4,'bo-');hold on;semilogy(n,RBAB4,'rx-');title('当初值为1, \lambda为1000, Adams四阶方法末项收敛阶');xlabel('i:\Delta{t}=2^{-i}');legend('AM4','AB4')
