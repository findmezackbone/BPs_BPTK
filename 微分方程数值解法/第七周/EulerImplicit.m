function u = EulerImplicit(f,ti,ui,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=ui;

for i = 1:N
    u0 = u(i);
    t1 = ti + i*dt;
    ug = u0+dt*f(t1-dt,u0); %initial guess
    u1 = fsolve(@(v) (v-u0)/dt - f(t1,v),ug);
    u(i+1) = u1;
end