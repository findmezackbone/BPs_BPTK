function I = fhsimpson(f,n)
I = 0;
h=1/n;
for i = 1:n
    I = I + (h/6)*(f((i-1)*h)+f(i*h)+4*f((i-1/2)*h));
end