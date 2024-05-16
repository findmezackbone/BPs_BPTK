function u = Kutta4(f,ti,ui,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=ui;

for i = 1:N
    t1 = ti+(i-1)*dt;t2 = t1+(dt/3);t3 = t1 + (2/3)*dt;t4 = t1+dt;
    k1 = f(t1,u(i));
    k2 = f(t2,u(i)+(dt/3)*k1);
    k3 = f(t3,u(i)-(1/3)*dt*k1+dt*k2);
    k4 = f(t4,u(i)+dt*k1-dt*k2+dt*k3);
    u(i+1) = u(i) + (dt/8)*(k1+3*k2+3*k3+k4);
end




end