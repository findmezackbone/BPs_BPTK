function u = solve313(N,a,b,c,f)
%这是专门求解3.1.3的函数，反馈了N+1点的数组，即u

h=1/N;
A=(1/h^2)*(diag(2*ones(1,N-1))+diag(-1*ones(1,N-2),-1)+diag(-1*ones(1,N-2),1));
B=(1/(2*h))*(diag(ones(1,N-2),1)+diag(-1*ones(1,N-2),-1)); %低阶项中心差商
I=diag(ones(1,N-1));
u = inv(a*A+b*B+c*I)*f;
u=[0;u;0]; %对两端补0
end