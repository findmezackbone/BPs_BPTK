clear;clc;

ui=1.5;
ti=0;
T=1;
f1 = @(t) 1/(((1/ui)-1)*exp(-t)+1);
f = @(t,u) u-u^2;

dt = 0.08; %根据题目要求只需要改动dt
n = floor((T-ti)/dt);
tlist = [ti];
for i = 1:n
    tnew = ti + i*dt;
    tlist = [tlist;tnew];
end

utrue=[];
for k = 1:length(tlist)
    utrue = [utrue;f1(tlist(k,1))];
end

%q=1，显示Euler
uEE=EulerExplicit(f,ti,ui,dt,T);

%q=2,
g1 = @(t,u) u-u^2 + (dt/2)*(1-2*u)*(u-u^2);
uq2 = EulerExplicit(g1,ti,ui,dt,T);

%q=3,
g2 = @(t,u) u-u^2 + (dt/2)*(1-2*u)*(u-u^2) + ((dt^2)/6)*(-2*((u-u^2)^2) + ((1-2*u)^2)*(u-u^2));
uq3 = EulerExplicit(g2,ti,ui,dt,T);

subplot(1,2,1);plot(tlist,utrue);hold on;plot(tlist,uEE,'o-');hold on;plot(tlist,uq2,'x-');hold on;plot(tlist,uq3,'+-');title('近似解');xlabel('t');legend('精确解','q=1','q=2','q=3');
subplot(1,2,2);semilogy(tlist,abs(utrue-uEE),'o-');hold on;semilogy(tlist,abs(utrue-uq2),'x-');hold on;semilogy(tlist,abs(utrue-uq3),'+-');title('用semilogy绘制的绝对误差');xlabel('t');legend('q=1','q=2','q=3');
