function s_hat=symbol_decode(y,modulation_order,theta,type)

switch type
    case 'SER'
        s_hat=pskdemod(y,2^modulation_order,theta);
        
    case 'BER'
        s=pskdemod(y,2^modulation_order,theta);
        s_hat=dec2bin(s,modulation_order)-'0';
        s_hat=reshape(s_hat.', modulation_order*length(s),1);
        
end
end
