function I = fhtx(f,n)
h = 1/n;
I = 0;
for i = 1:n
    I = I + (h/2)*(f((i-1)*h)+f(i*h));
end