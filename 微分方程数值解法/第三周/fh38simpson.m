function I = fh38simpson(f,n)
I = 0;
h=1/n;
for i = 1:n
    I = I + (h/8)*(f((i-1)*h)+f(i*h)+3*f((i-2/3)*h)+3*f((i-1/3)*h));
end