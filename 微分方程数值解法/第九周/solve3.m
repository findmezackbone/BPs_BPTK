function u = solve3(N,a,b,c,f)
%这是专门求解144.3的函数，反馈了N+1点的数组，即u

h=1/N;
A=(1/h^2)*(diag(2*ones(1,N))+diag(-1*ones(1,N-1),-1)+diag(-1*ones(1,N-1),1)); A(N,:)=zeros(1,N);
B=(1/(2*h))*(diag(ones(1,N-1),1)+diag(-1*ones(1,N-1),-1)); %低阶项中心差商 
B(N,:)=zeros(1,N);
I=diag(ones(1,N)); I(N,N)=0;
Z=zeros(N,N); Z(N,N)=(c+2/(h^2)-b+2/h); Z(N,N-1)=-2/(h^2);
u = inv(a*A+b*B+c*I+Z)*f;
u=[0;u]; %对两端补0
end