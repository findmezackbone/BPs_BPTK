clear;clc
%考虑第二问中的收敛阶

k = 1000; %lambda
ui = 1; %初值
ti = 0; %最初时间
T = 1; %考虑在ti到ti+T上的数值解
f = @(t,u) k*(-u+cos(t)); 
%u = @(t) (-k^2/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %初值为0的精确表达式
u = @(t) (1/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %初值为1的精确表达式
RBEE = [];RBEI=[]; %后续用于存放各格式的末项相对误差

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

    %Euler显
    UEE = EulerExplicit(f,ti,ui,dt,T);

    %Gear4
    UEI = EulerImplicit(f,ti,ui,dt,T);
    
    RBEE = [RBEE;(abs(Utrue(end)-UEE(end))/abs(Utrue(end)))];
    RBEI = [RBEI;(abs(Utrue(end)-UEI(end))/abs(Utrue(end)))];
end

%画图观察收敛阶
semilogy(n,exp(-n*log(2)/log(exp(1))),'bo-');hold on;semilogy(n,RBEE,'gd-');hold on;semilogy(n,RBEI,'rx-');title('当初值为1, \lambda为100, 显式Euler与隐式Euler方法末项收敛阶');xlabel('i:\Delta{t}=2^{-i}');legend('斜率-ln2','EulerExplicit','EulerImplicit')
