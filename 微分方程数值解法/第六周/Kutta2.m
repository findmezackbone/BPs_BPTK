function u = Kutta2(f,ti,ui,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=ui;

for i = 1:N
    t1 = ti + (i-1)*dt; t2 = t1 + dt;
    k1 = f(t1,u(i));
    k2 = f(t2,u(i)+dt*k1);
    u(i+1) = u(i) + (dt/2)*( k1+k2  );
end
end