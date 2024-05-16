clear;clc
a=-2;ui=1;T=3;ti=0;
dt = 0.08;

f = @(t,u) a*u; 
utrue = @(t) ui*exp(a*(t-ti));
N = floor((T-ti)/dt);
Utrue = zeros(N+1,1); Utrue(1)=ui; Tlist = zeros(N+1,1); Tlist(1) = ti;
for i = 1:N
    t0 = ti+i*dt;
    Tlist(i+1) = ti+i*dt;
    Utrue(i+1) = utrue(t0);
end


%第一问  精确的初值
UAM4 = AM4(f,ti,Utrue,dt,T); %AM三步四阶格式
UAB4 = AB4(f,ti,Utrue,dt,T); %AB四步四阶格式
UGEAR4 = GEAR4(f,ti,Utrue,dt,T); %GEAR4格式
subplot(3,2,1);plot(Tlist,abs(UAM4-Utrue),'bo-');hold on; plot(Tlist,abs(UAB4-Utrue),'gx-');hold on;plot(Tlist,abs(UGEAR4-Utrue),'r+-'); title('第一问：不同格式与精确解的绝对误差（精确初值）');legend('AM4','AB4','GEAR4')

%第二问 显示Euler格式计算初值
UE = EulerExplicit(f,ti,ui,dt,T);
P2UAM4 = AM4(f,ti,UE,dt,T); 
P2UAB4 = AB4(f,ti,UE,dt,T);
P2UGEAR4 = GEAR4(f,ti,UE,dt,T);
subplot(3,2,2);plot(Tlist,abs(P2UAM4-Utrue),'bo-');hold on; plot(Tlist,abs(P2UAB4-Utrue),'gx-');hold on;plot(Tlist,abs(P2UGEAR4-Utrue),'r+-'); title('第二问：不同格式与精确解的绝对误差（显式Euler求初值）');legend('AM4','AB4','GEAR4')

%第三问 用二至四阶KUTTA格式求初值
UK2 = Kutta2(f,ti,ui,dt,T);UK3 = Kutta3(f,ti,ui,dt,T);UK4 = Kutta4(f,ti,ui,dt,T);
K2UAM4 = AM4(f,ti,UK2,dt,T);K3UAM4 = AM4(f,ti,UK3,dt,T);K4UAM4 = AM4(f,ti,UK4,dt,T);
K2UAB4 = AB4(f,ti,UK2,dt,T);K3UAB4 = AB4(f,ti,UK3,dt,T);K4UAB4 = AB4(f,ti,UK4,dt,T);
K2UGEAR4 = GEAR4(f,ti,UK2,dt,T);K3UGEAR4 = GEAR4(f,ti,UK3,dt,T);K4UGEAR4 = GEAR4(f,ti,UK4,dt,T);
subplot(3,2,3);plot(Tlist,abs(K2UAM4-Utrue),'bo-');hold on; plot(Tlist,abs(K2UAB4-Utrue),'gx-');hold on;plot(Tlist,abs(K2UGEAR4-Utrue),'r+-'); title('第三问：不同格式与精确解的绝对误差（Kutta二阶求初值）');legend('AM4','AB4','GEAR4')
subplot(3,2,4);plot(Tlist,abs(K3UAM4-Utrue),'bo-');hold on; plot(Tlist,abs(K3UAB4-Utrue),'gx-');hold on;plot(Tlist,abs(K3UGEAR4-Utrue),'r+-'); title('第三问：不同格式与精确解的绝对误差（Kutta三阶求初值）');legend('AM4','AB4','GEAR4')
subplot(3,2,5);plot(Tlist,abs(K4UAM4-Utrue),'bo-');hold on; plot(Tlist,abs(K4UAB4-Utrue),'gx-');hold on;plot(Tlist,abs(K4UGEAR4-Utrue),'r+-'); title('第三问：不同格式与精确解的绝对误差（Kutta四阶求初值）');legend('AM4','AB4','GEAR4')


%%
%AM4格式不同初值
subplot(1,2,1);plot(Tlist,abs(UAM4-Utrue),'ro-');hold on;plot(Tlist,abs(K2UAM4-Utrue),'blackx-');hold on;plot(Tlist,abs(K3UAM4-Utrue),'b+-');hold on;plot(Tlist,abs(K4UAM4-Utrue),'gx-');title('A-M四阶格式在不同初值条件与精确解的绝对误差');legend('精确初值','Kutta2初值','Kutta3初值','Kutta4初值')
subplot(1,2,2);plot(Tlist,abs(UAM4-Utrue),'ro-');hold on;plot(Tlist,abs(K3UAM4-Utrue),'b+-');hold on;plot(Tlist,abs(K4UAM4-Utrue),'gx-');title('A-M四阶格式在不同初值条件与精确解的绝对误差');legend('精确初值','Kutta3初值','Kutta4初值')


%% 
%AB4格式不同初值
subplot(1,2,1);plot(Tlist,abs(UAB4-Utrue),'ro-');hold on;plot(Tlist,abs(K2UAB4-Utrue),'blackx-');hold on;plot(Tlist,abs(K3UAB4-Utrue),'b+-');hold on;plot(Tlist,abs(K4UAB4-Utrue),'gx-');title('A-B四阶格式在不同初值条件与精确解的绝对误差');legend('精确初值','Kutta2初值','Kutta3初值','Kutta4初值')
subplot(1,2,2);plot(Tlist,abs(UAB4-Utrue),'ro-');hold on;plot(Tlist,abs(K3UAB4-Utrue),'b+-');hold on;plot(Tlist,abs(K4UAB4-Utrue),'gx-');title('A-B四阶格式在不同初值条件与精确解的绝对误差');legend('精确初值','Kutta3初值','Kutta4初值')

%%
%GEAR4格式不同初值
subplot(1,2,1);plot(Tlist,abs(UGEAR4-Utrue),'ro-');hold on;plot(Tlist,abs(K2UGEAR4-Utrue),'blackx-');hold on;plot(Tlist,abs(K3UGEAR4-Utrue),'b+-');hold on;plot(Tlist,abs(K4UGEAR4-Utrue),'gx-');title('GEAR四阶格式在不同初值条件与精确解的绝对误差');legend('精确初值','Kutta2初值','Kutta3初值','Kutta4初值')
subplot(1,2,2);plot(Tlist,abs(UGEAR4-Utrue),'ro-');hold on;plot(Tlist,abs(K3UGEAR4-Utrue),'b+-');hold on;plot(Tlist,abs(K4UGEAR4-Utrue),'gx-');title('GEAR四阶格式在不同初值条件与精确解的绝对误差');legend('精确初值','Kutta3初值','Kutta4初值')
