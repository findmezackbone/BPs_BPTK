clear;clc

%%
clear;clc
%先求初值为0的情形
r = [1,10,100,1000]; %lambda的取值
ui = 0;
ti = 0;
T = 1;
dt = 0.02;
N = floor((T-ti)/dt);
Tlist = [ti];
for i = 1:N
    Tlist = [Tlist;ti+i*dt];
end

for i = 1:length(r)
    k = r(i);
    f = @(t,u) k*(-u+cos(t));
    u = @(t) (-k^2/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %lambda取不同值下的精确表达式
    Utrue = [ui];

    %精确解
    for j = 1:N
        Utrue = [Utrue;u(ti+j*dt)];
    end

    %显式Euler
    UE = EulerExplicit(f,ti,ui,dt,T);

    %隐式Euler
    UI = EulerImplicit(f,ti,ui,dt,T);

    subplot(2,2,i);plot(Tlist,Utrue,'o-');hold on;plot(Tlist,UE,'d-');hold on;plot(Tlist,UI,'x-');hold on;title(sprintf('当初值为%d, lambda=%d时, 精确解与显隐式Euler近似解',ui,k));xlabel('时间t'); legend('Utrue','EulerExplicit','EulerImplicit');
end


%%
clear;clc
%先求初值为1的情形
r = [1,10,100,1000]; %lambda的取值
ui = 1;
ti = 0;
T = 1;
dt = 0.2;
N = floor((T-ti)/dt);
Tlist = [ti];
for i = 1:N
    Tlist = [Tlist;ti+i*dt];
end

for i = 1:length(r)
    k = r(i);
    f = @(t,u) k*(-u+cos(t));
    u = @(t) (1/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %lambda取不同值下的精确表达式
    Utrue = [ui];

    %精确解
    for j = 1:N
        Utrue = [Utrue;u(ti+j*dt)];
    end

    %显式Euler
    UE = EulerExplicit(f,ti,ui,dt,T);

    %隐式Euler
    UI = EulerImplicit(f,ti,ui,dt,T);

    subplot(2,2,i);plot(Tlist,Utrue,'bo-');hold on;plot(Tlist,UE,'gd-');hold on;plot(Tlist,UI,'rx-');hold on;title(sprintf('当初值为%d, lambda=%d时, 精确解与显隐式Euler近似解',ui,k)); legend('Utrue','EulerExplicit','EulerImplicit');
end

%%
%考虑第三小问
clear;clc
%先求初值为0的情形
r = [1000]; %lambda的取值
ui = 0;
ti = 0;
T = 1;
dt = 2^(-6);
N = floor((T-ti)/dt);
Tlist = [ti];
for i = 1:N
    Tlist = [Tlist;ti+i*dt];
end

for i = 1:length(r)
    k = r(i);
    f = @(t,u) k*(-u+cos(t));
    u = @(t) (-k^2/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %lambda取不同值下的精确表达式
    Utrue = [ui];

    %精确解
    for j = 1:N
        Utrue = [Utrue;u(ti+j*dt)];
    end

    %AB4
    UAB4 = AB4(f,ti,Utrue,dt,T);

    %AM4
    UAM4 = AM4(f,ti,Utrue,dt,T);

    %Gear4
    UG4 = GEAR4(f,ti,Utrue,dt,T);

    %plot(Tlist,Utrue,'blue');hold on;plot(Tlist,UAM4,'gd-');hold on;plot(Tlist,UAB4,'rx-');hold on;
    plot(Tlist,Utrue,'blue');hold on;plot(Tlist,UG4,'gd-');title(sprintf('当初值为%d, lambda=%d时, 精确解与Gear四阶格式近似解',ui,k));xlabel('时间t'); legend('Utrue','Gear4')
    %legend('Utrue','AM4','AB4','GEAR4',Location= 'southeast');
    %plot(Tlist,Utrue,'blue');hold on;plot(Tlist,UAB4,'gd-');hold on;plot(Tlist,UAM4,'rx-');title('精确解、A-B与A-M四阶格式的图像');legend('Utrue','UAB4','UAM4');
end
%%
%考虑第三小问
clear;clc
%先求初值为0的情形
r = [1000]; %lambda的取值
ui = 1;
ti = 0;
T = 1;
dt = 2^(-6);
N = floor((T-ti)/dt);
Tlist = [ti];
for i = 1:N
    Tlist = [Tlist;ti+i*dt];
end

for i = 1:length(r)
    k = r(i);
    f = @(t,u) k*(-u+cos(t));
    %u = @(t) (-k^2/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %lambda取不同值下的精确表达式
    u = @(t) (1/(1+k^2))*exp(-k*t)+(k/(1+k^2))*(sin(t)+k*cos(t)); %初值为1的精确表达式

    Utrue = [ui];

    %精确解
    for j = 1:N
        Utrue = [Utrue;u(ti+j*dt)];
    end

    %AB4
    UAB4 = AB4(f,ti,Utrue,dt,T);

    %AM4
    UAM4 = AM4(f,ti,Utrue,dt,T);

    %Gear4
    UG4 = GEAR4(f,ti,Utrue,dt,T);

    %plot(Tlist,Utrue,'blue');hold on;plot(Tlist,UAM4,'gd-');hold on;plot(Tlist,UAB4,'rx-');hold on;
    plot(Tlist,Utrue,'blue');hold on;plot(Tlist,UG4,'gd-');title(sprintf('当初值为%d, lambda=%d时, 精确解与Gear四阶格式近似解',ui,k));xlabel('时间t'); legend('Utrue','Gear4')
    %legend('Utrue','AM4','AB4','GEAR4',Location= 'southeast');
    %plot(Tlist,Utrue,'blue');hold on;plot(Tlist,UAB4,'gd-');hold on;plot(Tlist,UAM4,'rx-');title('精确解、A-B与A-M四阶格式的图像');legend('Utrue','UAB4','UAM4');
end