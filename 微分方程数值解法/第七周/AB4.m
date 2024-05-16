function u = AB4(f,ti,Utrue,dt,T)
N = floor((T-ti)/dt);
u = zeros(N+1,1);
u(1)=Utrue(1); u(2)=Utrue(2);u(3)=Utrue(3);u(4)=Utrue(4);



for i = 4:N
    tn= ti+(i-1)*dt;tnj1 = ti+(i-2)*dt;tnj2= ti+(i-3)*dt;tnj3=ti+(i-4)*dt;
    ug = u(i)+dt*f(ti+i*dt,u(i));
    
    u(i+1)=u(i)+(dt/24)*(55*f(tn,u(i))-59*f(tnj1,u(i-1))+37*f(tnj2,u(i-2))-9*f(tnj3,u(i-3)));

end