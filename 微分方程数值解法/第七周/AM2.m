function u = AM2(f,ti,Utrue,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=Utrue(1); 
for i = 1:N
    tn= ti+(i-1)*dt;
    ug = u(i)+dt*f(ti+i*dt,u(i));
    u(i+1)=fsolve(@(v) u(i)+(dt/2)*(f(tn+i*dt,v)+f(tn,u(i)))-v,ug);
    %v=u(i)+(dt/24)*(9*v+19*f(tn,u(i))-5*f(tnj1,u(i-1))+f(tnj2,u(i-2)))
end