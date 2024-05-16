clear;clc

ui = 2;
ti = 0;
T = 1;
f = @(t,u) u-u^2; %例2.2.2的函数f
u = @(t) 1/(     (1/ui - 1)*exp(-t) +   1     );

blist2 = [];blist3=[];blist4=[];list2 =[];list3=[];list4=[];
n = 1:16;
for i = n
    dt = 1/(2^i);
    UK2 = Kutta2(f,ti,ui,dt,T);
    UK3 = Kutta3(f,ti,ui,dt,T);
    UK4 = Kutta4(f,ti,ui,dt,T);
    N = floor((T-ti)/dt);
    U = [ui];
    for j = 1:N
        U = [U;u(ti+j*dt)];
    end

    bias2 = abs(U-UK2)./abs(U);bias3 = abs(U-UK3)./abs(U);bias4 = abs(U-UK4)./abs(U);

    en2 = bias2(end);en3 = bias3(end);en4=bias4(end);
    
%     en2 = (max(abs(U-UK2)./abs(UK2)));
%     en3 = (max(abs(U-UK3)./abs(UK3)));
%     en4 = (max(abs(U-UK4)./abs(UK4)));
    blist2 = [blist2;en2];blist3 = [blist3;en3];blist4 = [blist4;en4];
    list2 = [list2;bias2(end)];    list4 = [list4;bias4(end)];
    list3 = [list3;bias3(end)];

end

yx1=[];yx2=[];yx3=[];
for k = n
    yx1=[yx1;-2*log(2)*k];yx2=[yx2;-3*log(2)*k];yx3=[yx3;-4*log(2)*k];
end

%plot(n,yx,'black');hold on;plot(n,y2x,'blue');hold on;plot(n,y3x,'yellow');hold on;
semilogy(n,blist2,'bo-');hold on; semilogy(n,blist3,'gd-');hold on; semilogy(n,blist4,'rx-');title('用semilogy绘制的各方法的相对误差关于步长的变化曲线');legend('2阶','3阶','4阶');xlabel('i: \Delta{t} = 步长2^{-i}')

%%
en2 = ((abs(U-UK2)./abs(UK2)));
en3 = ((abs(U-UK3)./abs(UK3)));
en4 = ((abs(U-UK4)./abs(UK4)));
DT = []; I=[];
yx1=[];yx2=[];yx3=[];

for k = n
    DT = [DT;2^(-k)]; I = [I;exp(k)];yx1=[yx1;exp(-2*k*log(2))];yx2=[yx2;exp(-3*k*log(2))];yx3=[yx3;exp(-4*k*log(2))];
end
semilogy(n(3:12),yx1(3:12),'blue');hold on; semilogy(n(3:12),yx2(3:12),'green');hold on; semilogy(n(3:12),yx3(3:12),'red');hold on;semilogy(n,list2,'bo-');hold on; semilogy(n,list3,'gd-');hold on; semilogy(n,list4,'rx-');title('相对误差');legend('斜率-2ln2','斜率-3ln2','斜率-4ln2','Kutta2','Kutta3','Kutta4');xlabel('i: \Delta{t} = 步长2^{-i}')

