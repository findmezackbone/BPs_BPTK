function u = AB2(f,ti,Utrue,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=Utrue(1); u(2)=Utrue(2);



for i = 2:N
    tn= ti+(i-1)*dt;tnj1 = ti+(i-2)*dt;
    ug = u(i)+dt*f(ti+i*dt,u(i));
    
    u(i+1)=u(i)+(dt/2)*(3*f(tn,u(i))-f(tnj1,u(i-1)));

end