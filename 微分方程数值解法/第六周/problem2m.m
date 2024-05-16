clear;clc;

a=-2;ui=1;T=1;ti=0; %初始参数 dt在后续研究精度会变动
f = @(t,u) a*u; 
utrue = @(t) ui*exp(a*(t-ti));

n = 2:8; 
%1)相同格式 不同初值的收敛精度
blistTAM4=[];blistEAM4=[];blistK2AM4=[];blistK3AM4=[];blistK4AM4=[];
blistTAB4=[];blistEAB4=[];blistK2AB4=[];blistK3AB4=[];blistK4AB4=[];
blistTG4=[];blistEG4=[];blistK2G4=[];blistK3G4=[];blistK4G4=[];

for i = n %观察在不同的步长下 相同的迭代格式在不同的初值下的收敛阶
    dt = (T-ti)/(2^i); %变步长
    N = floor((T-ti)/dt); %后续有N+1个点包括初值
    
    %先求离散精确解
    U = [ui];
    for j = 1:N
        U = [U;utrue(ti+j*dt)];
    end

    %再用显式Euler求离散近似解 用作后续的初值
    UE = EulerExplicit(f,ti,ui,dt,T);

    %再用二至四阶Kutta方法求离散近似解 用作后续的初值
    UK2 = Kutta2(f,ti,ui,dt,T);
    UK3 = Kutta3(f,ti,ui,dt,T);
    UK4 = Kutta4(f,ti,ui,dt,T);

    %首先考虑不同初值对AM4的收敛阶影响
    TAM4 = AM4(f,ti,U,dt,T); TB = abs(TAM4(end)-U(end))/abs(U(end)); %用精确初值AM4方法的末项相对误差
    EAM4 = AM4(f,ti,UE,dt,T); EB = abs(EAM4(end)-U(end))/abs(U(end));
    K2AM4 = AM4(f,ti,UK2,dt,T); K2B = abs(K2AM4(end)-U(end))/abs(U(end));
    K3AM4 = AM4(f,ti,UK3,dt,T); K3B = abs(K3AM4(end)-U(end))/abs(U(end));
    K4AM4 = AM4(f,ti,UK4,dt,T); K4B = abs(K4AM4(end)-U(end))/abs(U(end));
    blistTAM4 = [blistTAM4;TB];
    blistEAM4 = [blistEAM4;EB];
    blistK2AM4 = [blistK2AM4;K2B];
    blistK3AM4 = [blistK3AM4;K3B];
    blistK4AM4 = [blistK4AM4;K4B];

    %不同初值对AB4的收敛阶影响
    TAB4 = AB4(f,ti,U,dt,T); TB = abs(TAB4(end)-U(end))/abs(U(end)); %用精确初值AB4方法的末项相对误差
    EAB4 = AB4(f,ti,UE,dt,T); EB = abs(EAB4(end)-U(end))/abs(U(end));
    K2AB4 = AB4(f,ti,UK2,dt,T); K2B = abs(K2AB4(end)-U(end))/abs(U(end));
    K3AB4 = AB4(f,ti,UK3,dt,T); K3B = abs(K3AB4(end)-U(end))/abs(U(end));
    K4AB4 = AB4(f,ti,UK4,dt,T); K4B = abs(K4AB4(end)-U(end))/abs(U(end));
    blistTAB4 = [blistTAB4;TB];
    blistEAB4 = [blistEAB4;EB];
    blistK2AB4 = [blistK2AB4;K2B];
    blistK3AB4 = [blistK3AB4;K3B];
    blistK4AB4 = [blistK4AB4;K4B];

    %不同初值对AB4的收敛阶影响
    TG4 = GEAR4(f,ti,U,dt,T); TB = abs(TG4(end)-U(end))/abs(U(end)); %用精确初值AB4方法的末项相对误差
    EG4 = GEAR4(f,ti,UE,dt,T); EB = abs(EG4(end)-U(end))/abs(U(end));
    K2G4 = GEAR4(f,ti,UK2,dt,T); K2B = abs(K2G4(end)-U(end))/abs(U(end));
    K3G4 = GEAR4(f,ti,UK3,dt,T); K3B = abs(K3G4(end)-U(end))/abs(U(end));
    K4G4 = GEAR4(f,ti,UK4,dt,T); K4B = abs(K4G4(end)-U(end))/abs(U(end));
    blistTG4 = [blistTG4;TB];
    blistEG4 = [blistEG4;EB];
    blistK2G4 = [blistK2G4;K2B];
    blistK3G4 = [blistK3G4;K3B];
    blistK4G4 = [blistK4G4;K4B];
end

%%
%AM4在不同初值下的收敛速度
semilogy(n,(blistTAM4),'blacko-'); hold on; semilogy(n,(blistEAM4),'yd-');hold on;semilogy(n,(blistK2AM4),'rx-');hold on;semilogy(n,(blistK3AM4),'gx-');semilogy(n,(blistK4AM4),'bx-');title('不同初值下, A-M四阶格式的相对误差的对数值与i的关系');xlabel('i:\Delta{t}=2^{-i}');legend('精确初值','显式Euler初值','Kutta2初值','Kutta3初值','Kutta4初值')

%%
%AB4在不同初值下的收敛速度
semilogy(n,(blistTAB4),'blacko-'); hold on; semilogy(n,(blistEAB4),'yd-');hold on;semilogy(n,(blistK2AB4),'rx-');hold on;semilogy(n,(blistK3AB4),'gx-');semilogy(n,(blistK4AB4),'bx-');title('不同初值下, A-B四阶格式的相对误差的对数值与i的关系');xlabel('i:\Delta{t}=2^{-i}');legend('精确初值','显式Euler初值','Kutta2初值','Kutta3初值','Kutta4初值')

%%
%GEAR4在不同初值下的收敛速度
semilogy(n,(blistTG4),'blacko-'); hold on; semilogy(n,(blistEG4),'yd-');hold on;semilogy(n,(blistK2G4),'rx-');hold on;semilogy(n,(blistK3G4),'gx-');semilogy(n,(blistK4G4),'bx-');title('不同初值下, GEAR四阶格式的相对误差的对数值与i的关系');xlabel('i:\Delta{t}=2^{-i}');legend('精确初值','显式Euler初值','Kutta2初值','Kutta3初值','Kutta4初值')

%%
%后面开始考虑不同迭代格式在相同初值下的表现
semilogy(n,(blistTAM4),'bo-');hold on;semilogy(n,blistTAB4,'rd-');semilogy(n,blistTG4,'gx-');title('在精确初值下, 四阶格式的相对误差的对数值与i的关系');xlabel('i:\Delta{t}=2^{-i}');legend('A-M4','A-B4','GEAR4')

%%
%Euler初值
semilogy(n,(blistEAM4),'bo-');hold on;semilogy(n,blistEAB4,'rd-');semilogy(n,blistEG4,'gx-');title('在显式Euler初值下, 四阶格式的相对误差的对数值与i的关系');xlabel('i:\Delta{t}=2^{-i}');legend('A-M4','A-B4','GEAR4')

%%
%Kutta2
semilogy(n,(blistK2AM4),'bo-');hold on;semilogy(n,blistK2AB4,'rd-');semilogy(n,blistK2G4,'gx-');title('在Kutta二阶初值下, 四阶格式的相对误差的对数值与i的关系');xlabel('i:\Delta{t}=2^{-i}');legend('A-M4','A-B4','GEAR4')

%%
%Kutta3
semilogy(n,(blistK3AM4),'bo-');hold on;semilogy(n,blistK3AB4,'rd-');semilogy(n,blistK3G4,'gx-');title('在Kutta三阶初值下, 四阶格式的相对误差的对数值与i的关系');xlabel('i:\Delta{t}=2^{-i}');legend('A-M4','A-B4','GEAR4')

%%
%Kutta4
semilogy(n,(blistK4AM4),'bo-');hold on;semilogy(n,blistK4AB4,'rd-');semilogy(n,blistK4G4,'gx-');title('在Kutta四阶初值下, 四阶格式的相对误差的对数值与i的关系');xlabel('i:\Delta{t}=2^{-i}');legend('A-M4','A-B4','GEAR4')
