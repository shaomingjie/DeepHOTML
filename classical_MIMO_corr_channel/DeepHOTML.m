function X_estsign = DeepHOTML(HTH,HY,y,W1,b1,N,M,LayerTied,alpha1,alpha2,lambdav,step2)

Xup = min(max(y'*W1+b1,-1),1);
Z = Xup;

V = zeros(1,2*N);

for j =1:1:LayerTied
    Xpre = Xup;
    R = Z - alpha1(j) * Z* HTH  + alpha2(j) * HY' +   lambdav(j)*  V ;
    Xup = max(min(R,1),-1);
    Z = Xup + step2(j) * (Xpre-Xup);
    V = Xup;
end

X_est = Xup;
X_estsign = sign(X_est);

end
