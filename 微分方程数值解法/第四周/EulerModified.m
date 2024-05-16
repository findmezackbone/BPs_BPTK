function u = EulerModified(f,ti,ui,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=ui;

for i = 1:N
    t0 = ti+(i-1)*dt;
    thalf = t0+dt/2;
    u0 = u(i);
    v=f(t0,u0);
    u1 = u0 + dt*f(thalf,u0+(dt/2)*v);
    u(i+1) = u1;
end