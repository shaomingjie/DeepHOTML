function x= NoInterference(H,s,n)

y=H*diag(s)+n *ones(1, size(H,2));
x=zeros(length(s),1);
for i=1:size(H,2)
    x(i)=H(:,i)'*y(:,i)/(norm(H(:,i))^2);
end
end