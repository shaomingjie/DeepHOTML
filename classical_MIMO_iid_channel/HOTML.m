function x=HOTML(x_ini,HH,Hy,L,P_per,lambda)

N=size(x_ini,1)/2;
v=zeros(2*N,1);
in_iter=100;
x=x_ini;

alpha=0.5;

for i_mu=1:200

    y_x=x;
    t=1;

    for i=1:in_iter
        grad_f=2*(HH*y_x-Hy)-2*lambda*v;
    
        x_buff=y_x-(1/L)*grad_f;
        ply_x=min(P_per,max(-P_per,x_buff));
        
        x_pre=x;
        x=ply_x;
        t1=(1+sqrt(1+4*t^2))/2;
        y_x=x+ (t-1)/t1 *(x-x_pre);
        t=t1;
        v=x;
        if norm(x-x_pre,'fro')<1e-4
            break
        end
    end
    lambda_pre= lambda;
    lambda=lambda+alpha/(i_mu)*(2*N -norm(x)^2);
    if abs(lambda_pre-lambda)<1e-5
        break
    end
end


