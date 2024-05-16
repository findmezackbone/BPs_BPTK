function I = fhzd(f,n)
I = 0;
h=1/n;
for i = 1:n
    I = I+h*f((i-1/2)*h);
end