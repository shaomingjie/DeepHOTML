function X_estsign = DeepHOTML(y, Homega, LayerTied,W1,b1,InW1,Inb1,alpha1,lambdav,st2)

K = size(Homega,1)/2;
 
Xup = min(max(y'*W1+b1,-1),1).'; 
Z = Xup;
V = zeros( 2*K,1); 

for j =1:1:LayerTied
    Xpre = Xup; 
    utav_i = (Homega'*Z).*(InW1(j,:)')+ Inb1(j,:)'; 
    utav = 1/sqrt(2*pi)*exp(-utav_i.^2/2)./normcdf(utav_i);
    gradient = Homega*utav;
    Xup = min(max(Z - alpha1(j)* gradient+ lambdav(j)*V,-1),1);
    Z = Xup + st2(j)* (Xpre-Xup); 
    V = Xup;
end

X_est = Xup;
X_estsign = sign(X_est);

end
