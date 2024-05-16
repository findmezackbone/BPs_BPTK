clear;clc

ui = 2;
%ui = 0.5;
ti = 0; dt = 0.0001; T = 1;
f = @(t,u) u-u^2; %例2.2.2的函数f
utrue = @(t) 1/(     (1/ui - 1)*exp(-t) +   1     ); %精确解
Utrue = [];Utrue(1,1)=utrue(ti);
N = floor((T-ti)/dt);
for i = 1:N
    Utrue = [Utrue;utrue(ti+i*dt)];
end


Ukutta2 = Kutta2(f,ti,ui,dt,T); U2end = Ukutta2(end);
Ukutta3 = Kutta3(f,ti,ui,dt,T); U3end = Ukutta3(end);
Ukutta4 = Kutta4(f,ti,ui,dt,T); U4end = Ukutta4(end);

Uend = Utrue(end);
b2 = abs(U2end-Uend);b3=abs(U3end-Uend);b4 = abs(U4end-Uend); %后续用于观察收敛阶
plot(0:(N),abs(Utrue-Ukutta2),'bo-');hold on; plot(0:(N),abs(Utrue-Ukutta3),'gx-');hold on; plot(0:(N),abs(Utrue-Ukutta4),'r+-');title('当u0=2与dt=0.08时，二到四阶Kutta方法与精确解的绝对误差');legend('Kutta2-bias','Kutta3-bias','Kutta4-bias');

