%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program performs testing for the DeepHOTML and HOTML in the following paper
% ``Binary MIMO Detection via Homotopy Optimization and Its Deep Adaptation'' by Mingjie Shao and Wing-Kin Ma
% The demo is for one-bit MIMO detection with i.i.d. Gaussian channels.
% The demo loads the parameters of successfully trained DeepHOTML network.
% If you have any questions, please contact mjshao@link.cuhk.edu.hk
% Nov. 27, 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc

Ntrials = 1e1; % no. of Monte Carlo trials
T=10;  % no. of symbols per channel use

% MIMO system setting:
% N is the number of single-antenna users, M is the number of receive antennas
% Both N and M are complex-valued dimension. 
% For each case, we load the parameters of DeepHOTML

LayerTied = 20; % no. of layers of DeepHOTML

setting='128by24';
switch setting
    
    case '18by4'
        M = 18;
        N = 4;
        % load the parameters from DeepHOTML
        %------ 20 layers -----------------------
        load ('./DeepHOTML_para/OnebitNetwork_18by4SNR5_22Layer2042.mat')
        
    case '64by16'
        M = 64;
        N = 16;
        % load the parameters from DeepHOTML
        %------ 20 layers -----------------------
        load ('./DeepHOTML_para/OnebitNetwork_64by16SNR5_22Layer2048.mar')
        % -------- 10 layers --------------
        %  load ('./DeepHOTML_para/OnebitNetwork_64by16SNR5_22Layer1062.mat')
        % -------- 5 layers --------------------
        %  load ('./DeepHOTML_para/OnebitNetwork_64by16SNR5_22Layer569.mat')
        
    case '128by24'
        M = 128;
        N = 24;
        % load the parameters from DeepHOTML
        %------ 20 layers -----------------------
        load ('./DeepHOTML_para/OnebitNetwork_128by24SNR5_22Layer2075.mat')
        % ----- 10 layers ----------------
        %  load ('./DeepHOTML_para/OnebitNetwork_128by24SNR5_22Layer1073.mat')
        % ----- 5 layers --------------------
        %  load ('./DeepHOTML_para/OnebitNetwork_128by24SNR5_22Layer585.mat')
        
    case '256by48'
        M = 256;
        N = 48;
        % load the parameters from DeepHOTML
        %------- 20 layers -------------------
        load ('./DeepHOTML_para/OnebitNetwork_256by48SNR5_22Layer2051.mat')
        %------- 10 layers ----------
        %  load ('./DeepHOTML_para/OnebitNetwork_256by48SNR5_22Layer1048.mat')
        % ------- 5 layers --------------
        %  load ('./DeepHOTML_para/OnebitNetwork_256by48SNR5_22Layer584.mat')
        
end


% QPSK constellation is used
scenario='QPSK';
switch scenario
    
    case 'QPSK'
        modulation_order=2;
end
theta = pi/(2^modulation_order);


% SNR range
SNRdB=0:5:20;
SNR=10.^(SNRdB/10);
sigma_C=2*N./SNR;

sigma_real = sigma_C/2;
sigma_OE=sqrt(sigma_real)+0.5; % over-estimated noise standard variance

BER_ZF=zeros(length(SNR),1);
BER_DeepHOTML=zeros(length(SNR),1);
BER_HOTML=zeros(length(SNR),1);

%% -----------------Mote Caral----------------------------------
wb=waitbar(0,'plez wait');

for ntrials = 1:Ntrials
    
    fprintf('\n')
    display(['ntrials:' int2str(ntrials)]);
    waitbar(ntrials/Ntrials,wb);
    
    % generate channels
    H=(randn(M,N)+1i*randn(M,N))/sqrt(2);
    H_pinv=pinv(H);
    % for each channel, we evaluate T times symbol transmission.
    for t=1:T
        % ---------- i.i.d. generating symbols -----------
        
        Databits=round(rand(modulation_order,N));
        symbol_index=bin2dec(char(Databits+48)');
        symbol_mat=sqrt(2)* pskmod(symbol_index,2^modulation_order,theta);
        v_symbol=symbol_decode(symbol_mat,modulation_order,theta,'BER');
        % generating noise
        n_ch=(randn(M,1)+1i*randn(M,1))/sqrt(2);
        
        for snr_index=1:length(SNR)
            
            %display(['SNRdB:' int2str(SNRdB(snr_index))]);
            
            n=sqrt(sigma_C(snr_index))*n_ch;
            
            y=H* symbol_mat+n;
            %---------------- Real-valued form---------------------
            H_tilde=[real(H), -imag(H);imag(H), real(H)];
            y_tilde=[sign(real(y));sign(imag(y))];
            G_matrix = (diag(y_tilde)*H_tilde).';
            Homega = G_matrix/sigma_OE(snr_index);
            x_init=sqrt(2*N)*G_matrix*ones(2*M,1)/norm(G_matrix*ones(2*M,1),2);
            %% --------- ZF detector ------------
            
            y_bit_com=sign(real(y))+1i*sign(imag(y));
            x_ZF=H_pinv*y_bit_com;
            %------------ count BER -----------------------
            Bit_ZF=symbol_decode(x_ZF,modulation_order,theta,'BER');
            BER_ZF (snr_index)=BER_ZF (snr_index)+length(find(Bit_ZF-v_symbol));
            %% ----- DeepHOTML --------------------
            
            tic
            x_DeepHOTML = DeepHOTML(y_tilde, Homega, LayerTied,Iniaff_W1,Iniaff_b1,InnerW1,Innerb1,beta,gamma,alpha)';
            T_DeepHOTML =toc;
            %         display(['DeepHOTML time:' num2str(T_DeepHOTML)]);
            x_DeepHOTMLcom=x_DeepHOTML(1:N)+1i*x_DeepHOTML(N+1:2*N);
            %------------ count BER -----------------------
            Bit_DeepHOTML=symbol_decode(x_DeepHOTMLcom,modulation_order,theta,'BER');
            BER_DeepHOTML(snr_index)=BER_DeepHOTML (snr_index)+length(find(Bit_DeepHOTML-v_symbol));
            
            
            %% ----------- HOTML -------------
            
            tic
            [x_HOTML]=HOTML(x_init,Homega, N);
            T_HOTML=toc;
            %    display([HOTML time:' num2str(T_HOTML)])
            %------------ count BER -----------------------
            Bit_HOTML=symbol_decode(x_HOTML,modulation_order,theta,'BER');
            BER_HOTML(snr_index)=BER_HOTML (snr_index)+length(find(Bit_HOTML-v_symbol));
                        
        end
    end
end
close(wb)

%% show BER results

BER_ZF=BER_ZF/(T*modulation_order*N*Ntrials);
BER_DeepHOTML=BER_DeepHOTML/(T*modulation_order*N*Ntrials);
BER_HOTML=BER_HOTML/(T*modulation_order*N*Ntrials);

%--------- plot BER curve ----------------

H1 = figure;

semilogy(SNRdB,BER_ZF,'-xg', 'Linewidth',1.5,'markers',7);hold on;
semilogy(SNRdB,BER_HOTML,'-vb', 'Linewidth',1.5,'markers',7);hold on;
semilogy(SNRdB,BER_DeepHOTML,'-ok', 'Linewidth',1.5,'markers',7);hold on;

xlabel('\fontsize{12}SNR (dB)')
ylabel('\fontsize{12}Bit Error Rate (BER)')
legend('\fontsize{12}ZF', '\fontsize{12}HOTML', '\fontsize{12}DeepHOTML, K=20')
axis([0,20,1e-5,1]),
hold on
grid on
