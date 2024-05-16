function u = EulerImproved(f,ti,ui,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=ui;

for i = 1:N
    u0=u(i);
    t0=ti+(i-1)*dt;t1=ti+i*dt;
    ug = u0+dt*f(t1,u0); %initial guess
    u1 = fsolve(@(v) (v-u0)/dt-(1/2)*(f(t0,u0)+f(t1,v)),ug);
    u(i+1) =u1;
end