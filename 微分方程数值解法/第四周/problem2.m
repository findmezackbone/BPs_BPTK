clear;clc
a=-2;ui=1;T=1;ti=0;
dt = 0.05; %根据题目要求只需要改动dt
n = floor((T-ti)/dt);
tlist = [ti];
for i = 1:n
    tnew = ti + i*dt;
    tlist = [tlist;tnew];
end

%常微分方程的精确解为e^(at)
f1 = @(t) exp(a*t);
f = @(t,u) a*u;

utrue=[];
for k = 1:length(tlist)
    utrue = [utrue;f1(tlist(k,1))];
end

%显式Euler
uEE = EulerExplicit(f,ti,ui,dt,T); %显式Euler逼近值

%隐式Euler
uEI = EulerImplicit(f,ti,ui,dt,T); %隐式Euler逼近值

%改进Euler
uEP = EulerImproved(f,ti,ui,dt,T); %改进Euler逼近值

%修正Euler
uEM = EulerModified(f,ti,ui,dt,T); %修正Euler逼近值

%画图
plot(tlist,abs(utrue-uEE),'o-');hold on;plot(tlist,abs(utrue-uEI),'x-');hold on;plot(tlist,abs(utrue-uEP),'+-');hold on;plot(tlist,abs(utrue-uEM),'*-');title('四种不同Euler格式下u的绝对误差');xlabel('时间t');legend('显式Euler','隐式Euler','改进Euler','修正Euler');