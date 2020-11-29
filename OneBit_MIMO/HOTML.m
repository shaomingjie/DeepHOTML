function [x_recover]=HOTML(x_init,Homega,K)

x=x_init;
lambda=0.001;

y_x=x;
L=0.1;
eta=1.2;
v=zeros(length(x),1);
alpha=0.2;

for i_mu=1:600
    t=1;
    
    for i=1:300
        % calculate the gradient
        Hy_x = Homega.'*y_x;
        B=normcdf(Hy_x);
        B(find(B==0))=1e-200;
        A=exp(-Hy_x.^2/2);
        grad_f=A./B;
        grad_all= - 1/sqrt(2*pi)*(Homega*grad_f) -2* lambda*v;
        % backtracking line search for step size
        while 1
            buff_x=y_x -(1/L)* grad_all;
            ply_x=max(min(buff_x,1),-1);
            sum_pl=normcdf(Homega.'*ply_x);
            F_l=-sum(sum(log(sum_pl)))-2* lambda* (ply_x.' * v);
            sum_y =B;
            Q_l=-sum(sum(log(sum_y)))-2* lambda* (y_x.' * v)...
                +(ply_x-y_x).'* grad_all+(L/2)*norm(ply_x-y_x,2)^2;
            
            if F_l>Q_l
                L=L*eta;
            else
                break
            end
        end
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
    
    lambda=lambda+alpha/sqrt(i_mu)*(2*K -norm(x)^2);
    if abs(lambda_pre-lambda)<1e-5
        break
    end
end

x_recover=x(1:K)+1i*x(K+1:2*K);
end