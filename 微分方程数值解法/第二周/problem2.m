clear;clc;
%下比较不动点迭代法与牛顿法迭代
% syms x
% G = @(x) exp(-x); 
 F = @(x) x-exp(-x);
% N = @(x) x - (x-exp(-x))/(1+exp(-x));


x0 = 1;
rG = [x0]; rN= [x0];
n = 25;
for i = 2:n
%     xnew = G(rG(1,i-1)); 
%     ynew = N((rN(1,i-1)));
    xnew = exp(-rG(1,i-1));
    ynew = rN(1,i-1)-(rN(1,i-1)-exp(-rN(1,i-1)))/(1+exp(-rN(1,i-1)));
    rG = [rG,xnew]; 
    rN = [rN,ynew];
end

xx = 0.567143290409784

semilogy([1:n-1],abs(rN(1,2:n)-xx),'r-*');hold on;semilogy([1:n-1],abs(rG(1,2:n)-xx)7,'g-*'); ylabel('点列的值');xlabel('迭代步数');title('对比不动点迭代与牛顿迭代');legend('牛顿迭代','不动点迭代')

%考虑他们的收敛
xs = fzero(F,rN(n)); 
abiasG = abs(rG(25)-xs)
abiasN = abs(rN(3)-xs)