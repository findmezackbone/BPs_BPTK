clear;clc;
f = @(x) abs(x-1/3);
I1=[];I2 = [];I3=[];I4=[];
N = [2,5,7,10,15,20,25,40,50,70,100,160,200]
for i = 1:length(N)
    I1 = [I1,fhzd(f,N(i))];I2 = [I2,fhtx(f,N(i))];I3 = [I3,fhsimpson(f,N(i))];I4 = [I4,fh38simpson(f,N(i))];
end
Q = (5/18)*ones(1,length(N))


semilogy(N,abs(Q-I1),'bo-');hold on;
semilogy(N,abs(Q-I2),'gx-');hold on;
semilogy(N,abs(Q-I3),'r+-');hold on;
semilogy(N,abs(Q-I4),'c*-');
title('semilogy绘制的不同复化求积公式绝对误差与分段数关系');
xlabel('分段个数n');
legend('中点','梯形','Simpson','3/8Simpson','Location','southeast')