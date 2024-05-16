function u = EulerExplicit(f,ti,ui,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=ui;

for i = 1:N
    t0=ti+dt*(i-1);
    u0=u(i);
    u(i+1) = u0+dt*f(t0,u0);
    
end