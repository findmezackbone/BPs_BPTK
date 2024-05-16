function u = GEAR3(f,ti,Utrue,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=Utrue(1); u(2)=Utrue(2);u(3)=Utrue(3);

for i = 3:N
    tn= ti+(i-1)*dt;tnj1 = ti+(i-2)*dt;tnj2= ti+(i-3)*dt;
    ug = u(i)+dt*f(ti+(i)*dt,u(i));
    u(i+1)= fsolve(@(v) (11/6)*v-3*u(i)+(3/2)*u(i-1)-(1/3)*u(i-2)-dt*f(ti+(i)*dt,v),ug);
end