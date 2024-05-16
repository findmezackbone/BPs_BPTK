function u = Kutta3(f,ti,ui,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=ui;

for i = 1:N
    t1 = ti+(i-1)*dt;t2 = t1+(dt/2);t3 = t1 + dt;
    k1 = f(t1,u(i));
    k2 = f(t2,u(i)+(dt/2)*k1);
    k3 = f(t3,u(i)-dt*k1+2*dt*k2);
    u(i+1) = u(i) + (dt/6)*(k1+4*k2+k3);
end




end