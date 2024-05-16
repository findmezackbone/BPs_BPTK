function Unum = DN2D(Du,g0,N,k,theta)
U=[]; U_cpy=[]
%要求解Lu=-Du
%g0是初始猜想
%N代表了网格内点数为(N-1)x(N-1)
%k为最高的迭代次数
hy = 1/N; %这是y轴的步长
hx = hy/2; %这是x轴的步长, 由于\Gamma是从x=1/2截断, 故若要用五点差分格式求解, 则要将x轴步长减半 
Ay = (1/(hy^2))*(2*diag(ones(1,N-1))-diag(ones(1,N-2),1)-diag(ones(1,N-2),-1));
Ax = (1/(hx^2))*(2*diag(ones(1,N-1))-diag(ones(1,N-2),1)-diag(ones(1,N-2),-1));
Ix = diag(ones(1,N-1)); Iy = diag(ones(1,N-1));
A = kron(Ay,Ix) + kron(Iy,Ax);
B = A;

count = 0; %计数
gn = g0; 

while(count<k)
    %DN方法的一个迭代循环中的第一步是在左半平面\Omega_1做离散五点差分格式求解数值解, 而左半平面上的\Gamma上的值会影响到f,
    %因此要修改f

    %第一步
    f1 = []; %构造f
    for j = 1:N-1
        for i = 1:N-1
            f1 = [f1;-Du(i*hx,j*hy)];
            if i == N-1
                f1(end) = f1(end) + (1/(hx^2))*gn(j);
            end
        end
    end
    u1 = A\f1; 
    
    %欲对u1进行重排
    for s = 1:N-1
        v = u1((s-1)*(N-1)+1:s*(N-1),1);
        U(s,:) = v';
    end
    U = [U,gn];

    
    %第二步
    %由Neuman边值条件, 要修改矩阵A与f
    f2=[];
    for i = 1:N-1
        B((i-1)*(N-1)+1,(i-1)*(N-1)+1) = B((i-1)*(N-1)+1,(i-1)*(N-1)+1) - (1/(2*hx^2));
    end
    for j = 1:N-1
        for i = N+1:2*N-1
            f2 = [f2;-Du(i*hx,j*hy)];
        end
    end
    for i = 1:N-1
        f2((i-1)*(N-1)+1,1) = f2((i-1)*(N-1)+1,1) + (1/(2*hx^2))*U(i,N-1);
    end

    u2 = B\f2;
    B = A; %重置B
    
    T = [];
    %对u2重排
    for s = 1:N-1
        v = u2((s-1)*(N-1)+1:s*(N-1),1);
        T(s,:) = v';
    end
    

    
    U = [U,T];
    UN = (U(:,N-1)+U(:,N+1))./2;
    U_cpy=U;
    U = [];



    gn = theta*UN+(1-theta)*gn;

    count = count+1;

            
end
U = U_cpy;