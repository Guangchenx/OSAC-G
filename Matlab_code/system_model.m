clear all
close all
clc

% Consider a rectangular area with DxD m^2
% N distributed APs serves K terminals, they all randomly located in the area
N=40; %Number of access points (APs) 
M=4; %Number of antennas per AP
K=40; %Number of users

f = 2.0; % Frequency in GHz
B=20; %Bandwith in Mhz
D=1; %in kilometer
tau_p = K; % tau_p uplink training duration (in samples)
tau_c = 200; % each coherence interval
etak = 10; % QoS requirement at user k in Mbps
[U,S,V]=svd(randn(tau_p,tau_p)); % U(tau x tau) is unitary matrix includes tau orthogonal sequences(both columns and rows) 


k_b = 1.381*10^(-23); %Boltzmann constant in j/k
B_1 =  B*10^6; %Bandwith in Hz
n_f = 10^(9/10) ; %Noise figure 9dB
noise_p =  B_1*k_b*290*n_f ; %Noise power in w
noise_p_db = 10*log10(noise_p); %Noise power in db 
Pp = 0.2*tau_p;%pilot power
p_bh = 0.6; % Traffic-dependent power coefficient in W/Gbps
p_fix = 0.3; % Fixed RF chain power in W
p_tx = 0.3; % Maximum transmit power in W


Hb = 10; % Base station height in m
Hm = 1.65; % Mobile height in m
d0=0.01; % 0.01km = 10m
d1=0.05; % 0.05km = 50m
aL = (1.1*log10(f)-0.7)*Hm-(1.56*log10(f)-0.8); 
L = 46.3+33.9*log10(f)-13.82*log10(Hb)-aL;  




%%

N1=1; %Number of iterations
Rate_greedy = zeros(N1,K);
R_1 = zeros(N1,K);
number_iter = zeros(N,1);

for i=1:N1

N_data = 100; 

for n_data=1:N_data

%Randomly locations of N APs:
AP=unifrnd(-D/2,D/2,N,2); % Mx2 numbers in [-D/2, D/2]

%Wrapped around (8 neighbor cells)
D1=zeros(N,2);
D1(:,1)=D1(:,1)+ D*ones(N,1);
AP1=AP+D1;

D2=zeros(N,2);
D2(:,2)=D2(:,2)+ D*ones(N,1);
AP2=AP+D2;

D3=zeros(N,2);
D3(:,1)=D3(:,1)- D*ones(N,1);
AP3=AP+D3;

D4=zeros(N,2);
D4(:,2)=D4(:,2)- D*ones(N,1);
AP4=AP+D4;

D5=zeros(N,2);
D5(:,1)=D5(:,1)+ D*ones(N,1);
D5(:,2)=D5(:,2)- D*ones(N,1);
AP5=AP+D5;

D6=zeros(N,2);
D6(:,1)=D6(:,1)- D*ones(N,1);
D6(:,2)=D6(:,2)+ D*ones(N,1);
AP6=AP+D6;

D7=zeros(N,2);
D7=D7+ D*ones(N,2);
AP7=AP+D7;

D8=zeros(N,2);
D8=D8- D*ones(N,2);
AP8=AP+D8;

% Randomly locations of K terminals:
Ter=unifrnd(-D/2,D/2,K,2);
sigma_shd=8; %in dB
z_nk = 1 ; 

%Create an MxK beta_mk
BETAA = zeros(N,K);
dist=zeros(N,K);
betadB = zeros(N,K);
for n=1:N
    for k=1:K
    dist(n,k) = min([norm(AP(n,:)-Ter(k,:)), norm(AP1(n,:)-Ter(k,:)),norm(AP2(n,:)-Ter(k,:)),norm(AP3(n,:)-Ter(k,:)),norm(AP4(n,:)-Ter(k,:)),norm(AP5(n,:)-Ter(k,:)),norm(AP6(n,:)-Ter(k,:)),norm(AP7(n,:)-Ter(k,:)),norm(AP8(n,:)-Ter(k,:)) ]); %distance between Terminal k and AP m
    % Terminal chooses the nearest AP.
    if dist(n,k)<d0
         betadB=-L - 35*log10(d1) + 20*log10(d1) - 20*log10(d0);
    elseif ((dist(n,k)>=d0) && (dist(n,k)<=d1))
         betadB= -L - 35*log10(d1) + 20*log10(d1) - 20*log10(dist(n,k));
    else
    betadB = -L - 35*log10(dist(n,k)) + z_nk * sigma_shd*randn(1,1); %large-scale in dB 
    end

    BETAA(n,k)=10^(betadB/10); 

    end
end


%% Pilot Asignment
% Create Phii(tau x K) pilot matrix. 
Phii=zeros(tau_p,K);
for k=1:K 
   Point=randi([1,tau_p]); 
   Phii(:,k)=U(:,Point);
end 


%% Create Gamma matrix 
Gammaa = zeros(N,K); 
mau=zeros(N,K);
for n=1:N
    for k=1:K
        mau(n,k)=norm((BETAA(n,:).^(1/2)).*(Phii(:,k)'*Phii))^2;
    end
end


for n=1:N
    for k=1:K
        Gammaa(n,k)=Pp*BETAA(n,k)^2/(Pp*mau(n,k) + noise_p);
    end
end



end

end

