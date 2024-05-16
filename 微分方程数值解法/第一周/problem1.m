clear;clc
%(1-cosx)/x^2趋于0的图像
f = @(x) (1-cos(x))/(x^2);
F = [];
n = linspace(0,0.001); %欲在（0，0.001）上表示图像 
for i = n
    F = [F,f(i)];
end
plot(n,F,'b-d'); ylabel('(1-cosx)/x^2'); xlabel('x');