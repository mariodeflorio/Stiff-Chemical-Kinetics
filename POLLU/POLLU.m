%%
clear; close all; clc;
format long
%--------------------------------------------------------------------------
%{
  Physics-Informed X-TFC applied to Stiff Chemical Kinetics
  Test Case 4 - POLLU Problem

  Authors:
  Mario De Florio, PhD
  Enrico Schiassi, PhD
%}
%%
%--------------------------------------------------------------------------
%% Input

rng('default') % set random seed
format longE
start = tic;

t_0 = 0; % initial time
t_f = 60; % final time

%t_query = [0.1,1,2,5,10,20,30,40];

n_x = 10;    % Discretization order for x (-1,1)
L = 10;    % number of neurons

x = linspace(0,1,n_x)';

t_step = 0.001;

t_tot = (t_0:t_step:t_f)';
n_t = length(t_tot);

type_basis = 2; % 1 = Orthogonal polynomials ; 2 = activation functions

type_act = 2; % type activation function

%{
1= Logistic;
2= TanH;
3= Sine;
4= Cosine;
5= Gaussian; the best so far w/ m=11
6= ArcTan;
7= Hyperbolic Sine;
8= SoftPlus
9= Bent Identity;
10= Inverse Hyperbolic Sine
11= Softsign 
%}

LB = -1; % Lower boundary for weight and bias samplings
UB = 1; % Upper boundary for weight and bias samplings

% iterative least-square parameters

IterMax = 100;
IterTol = 1e-9;

%% Chemical Parameters definition

% rate constants
k1 = 0.35;              k6 = 0.15*10^5;         k11 = 0.22*10^-1;       k16 = 0.35*10^-3;       k21 = 0.21*10^1;
k2 = 0.266*10^2;        k7 = 0.13*10^-3;        k12 = 0.12*10^5;        k17 = 0.175*10^-1;      k22 = 0.578*10^1;
k3 = 0.123*10^5;        k8 = 0.24*10^5;         k13 = 0.188*10^1;       k18 = 0.1*10^9;         k23 = 0.474*10^-1;
k4 = 0.86*10^-3;        k9 = 0.165*10^5;        k14 =  0.163*10^5;      k19 = 0.444*10^12;      k24 = 0.178*10^4;
k5 = 0.82*10^-3;        k10 = 0.9*10^4;         k15 = 0.48*10^7;        k20 = 0.124*10^4;       k25 = 0.312*10^1;

switch type_basis
    
    case 1 % orthogonal polynomials
        [h, hd] = CP(x, L + 1);
        
        % Restore matrices (to start with the 2nd order polynom)
        I = 2:L+1;
        h = h(:,I);  hd = hd(:,I);
        h0=h(1,:); hf=h(end,:);
   

    case 2 % Activation function definition
        
        weight = unifrnd(LB,UB,L,1);
        bias = unifrnd(LB,UB,L,1);
        
        h= zeros(n_x,L); hd= zeros(n_x,L); hdd= zeros(n_x,L);
        
        for i = 1 : n_x
            
            for j = 1 : (L)
                [h(i, j), hd(i, j), hdd(i,j)] = act(x(i),weight(j), bias(j),type_act);
                
            end
            
        end
        
        h0 = h(1,:); hf= h(end,:);
        hd0 = hd(1,:); hdf = hd(end,:);
        hdd0 = hdd(1,:); hddf = hdd(end,:);
        
end

%% A,xi,B construction

Z = zeros(n_x,L);

y1 = zeros(n_t,1);  y6 = zeros(n_t,1);  y11 = zeros(n_t,1); y16 = zeros(n_t,1);
y2 = zeros(n_t,1);  y7 = zeros(n_t,1);  y12 = zeros(n_t,1); y17 = zeros(n_t,1);
y3 = zeros(n_t,1);  y8 = zeros(n_t,1);  y13 = zeros(n_t,1); y18 = zeros(n_t,1);
y4 = zeros(n_t,1);  y9 = zeros(n_t,1);  y14 = zeros(n_t,1); y19 = zeros(n_t,1);
y5 = zeros(n_t,1);  y10 = zeros(n_t,1); y15 = zeros(n_t,1); y20 = zeros(n_t,1);

% Initial Values

y1_0 = 0;       y6_0 = 0;       y11_0 = 0 ;     y16_0 = 0 ;
y2_0 = 0.2;     y7_0 = 0.1 ;    y12_0 = 0 ;     y17_0 = 0.007 ;
y3_0 = 0;       y8_0 = 0.3 ;    y13_0 = 0 ;     y18_0 = 0 ;
y4_0 = 0.04;    y9_0 = 0.01 ;   y14_0 = 0 ;     y19_0 = 0 ;
y5_0 = 0;       y10_0 = 0 ;     y15_0 = 0 ;     y20_0 = 0 ;

% assign the initial values to the solution vectors
y1(1) = y1_0;   y6(1) = y6_0;       y11(1) = y11_0;     y16(1) = y16_0;
y2(1) = y2_0;   y7(1) = y7_0;       y12(1) = y12_0;     y17(1) = y17_0;
y3(1) = y3_0;   y8(1) = y8_0;       y13(1) = y13_0;     y18(1) = y18_0;
y4(1) = y4_0;   y9(1) = y9_0;       y14(1) = y14_0;     y19(1) = y19_0;
y5(1) = y5_0;   y10(1) = y10_0;     y15(1) = y15_0;     y20(1) = y20_0;

training_err_vec = zeros(n_t-1,1);

tstart = tic;

for i = 1:(n_t-1)
    
    
    % mapping coefficient
    c_i = (x(end) - x(1)) / (t_tot(i+1) - t_tot(i));
    
    t = t_tot(i) + (1/c_i)*(x-x(1));
    
    xi_1_i = zeros(L,1);    xi_6_i = zeros(L,1);    xi_11_i = zeros(L,1);   xi_16_i = zeros(L,1);
    xi_2_i = zeros(L,1);    xi_7_i = zeros(L,1);    xi_12_i = zeros(L,1);   xi_17_i = zeros(L,1);
    xi_3_i = zeros(L,1);    xi_8_i = zeros(L,1);    xi_13_i = zeros(L,1);   xi_18_i = zeros(L,1);
    xi_4_i = zeros(L,1);    xi_9_i = zeros(L,1);    xi_14_i = zeros(L,1);   xi_19_i = zeros(L,1);
    xi_5_i = zeros(L,1);    xi_10_i = zeros(L,1);   xi_15_i = zeros(L,1);   xi_20_i = zeros(L,1);
    
    xi_i = [xi_1_i;xi_2_i;xi_3_i;xi_4_i;xi_5_i;xi_6_i;xi_7_i;xi_8_i;xi_9_i;xi_10_i; ...
        xi_11_i;xi_12_i;xi_13_i;xi_14_i;xi_15_i;xi_16_i;xi_17_i;xi_18_i;xi_19_i;xi_20_i ];
       
    %% Build Constrained Expressions
    
    y1_i = (h-h0)*xi_1_i + y1_0;        y1_dot_i = c_i*hd*xi_1_i;
    y2_i = (h-h0)*xi_2_i + y2_0;        y2_dot_i = c_i*hd*xi_2_i;
    y3_i = (h-h0)*xi_3_i + y3_0;        y3_dot_i = c_i*hd*xi_3_i;
    y4_i = (h-h0)*xi_4_i + y4_0;        y4_dot_i = c_i*hd*xi_4_i;
    y5_i = (h-h0)*xi_5_i + y5_0;        y5_dot_i = c_i*hd*xi_5_i;
    y6_i = (h-h0)*xi_6_i + y6_0;        y6_dot_i = c_i*hd*xi_6_i;
    y7_i = (h-h0)*xi_7_i + y7_0;        y7_dot_i = c_i*hd*xi_7_i;
    y8_i = (h-h0)*xi_8_i + y8_0;        y8_dot_i = c_i*hd*xi_8_i;
    y9_i = (h-h0)*xi_9_i + y9_0;        y9_dot_i = c_i*hd*xi_9_i;
    y10_i = (h-h0)*xi_10_i + y10_0;        y10_dot_i = c_i*hd*xi_10_i;
    y11_i = (h-h0)*xi_11_i + y11_0;        y11_dot_i = c_i*hd*xi_11_i;
    y12_i = (h-h0)*xi_12_i + y12_0;        y12_dot_i = c_i*hd*xi_12_i;
    y13_i = (h-h0)*xi_13_i + y13_0;        y13_dot_i = c_i*hd*xi_13_i;
    y14_i = (h-h0)*xi_14_i + y14_0;        y14_dot_i = c_i*hd*xi_14_i;
    y15_i = (h-h0)*xi_15_i + y15_0;        y15_dot_i = c_i*hd*xi_15_i;
    y16_i = (h-h0)*xi_16_i + y16_0;        y16_dot_i = c_i*hd*xi_16_i;
    y17_i = (h-h0)*xi_17_i + y17_0;        y17_dot_i = c_i*hd*xi_17_i;
    y18_i = (h-h0)*xi_18_i + y18_0;        y18_dot_i = c_i*hd*xi_18_i;
    y19_i = (h-h0)*xi_19_i + y19_0;        y19_dot_i = c_i*hd*xi_19_i;
    y20_i = (h-h0)*xi_20_i + y20_0;        y20_dot_i = c_i*hd*xi_20_i;
        
    % reaction velocities
    
    r1 = k1*y1_i;           r6 = k6*y7_i.*y6_i;         r11 = k11*y13_i;            r16 = k16*y4_i;             r21 = k21*y19_i;
    r2 = k2*y2_i.*y4_i;     r7 = k7*y9_i;               r12 = k12*y10_i.*y2_i;      r17 = k17*y4_i;             r22 = k22*y19_i;
    r3 = k3*y5_i.*y2_i;     r8 = k8*y9_i.*y6_i;         r13 = k13*y14_i;            r18 = k18*y16_i;            r23 = k23*y1_i.*y4_i;
    r4 = k4*y7_i;           r9 = k9*y11_i.*y2_i;        r14 = k14*y1_i.*y6_i;       r19 = k19*y16_i;            r24 = k24*y19_i.*y1_i;
    r5 = k5*y7_i;           r10 = k10*y11_i.*y1_i;      r15 = k15*y3_i;             r20 = k20*y17_i.*y6_i;      r25 = k25*y20_i;
    
    %% Build the Losses
    
    L_1 = y1_dot_i +r1+r10+r14+r23+r24-r2-r3-r9-r11-r12-r22-r25 ;
    L_2 = y2_dot_i +r2+r3+r9+r12-r1-r21 ;
    L_3 = y3_dot_i +r15-r1-r17-r19-r22 ;
    L_4 = y4_dot_i +r2+r16+r17+r23-r15 ;
    L_5 = y5_dot_i +r3-2*r4-r6-r7-r13-r20 ;
    L_6 = y6_dot_i +r6+r8+r14+r20-r3-2*r18 ;
    L_7 = y7_dot_i +r4+r5+r6-r13 ;
    L_8 = y8_dot_i -r4-r5-r6-r7 ;
    L_9 = y9_dot_i +r7+r8 ;
    L_10 = y10_dot_i +r12-r7-r9 ;
    L_11 = y11_dot_i +r9+r10-r8-r11 ;
    L_12 = y12_dot_i -r9 ;
    L_13 = y13_dot_i +r11-r10 ;
    L_14 = y14_dot_i +r13-r12 ;
    L_15 = y15_dot_i -r14 ;
    L_16 = y16_dot_i +r18+r19-r16 ;
    L_17 = y17_dot_i +r20 ;
    L_18 = y18_dot_i -r20 ;
    L_19 = y19_dot_i +r21+r22+r24-r23-r25 ;
    L_20 = y20_dot_i +r25-r24 ;
    
    Loss = [L_1 ; L_2 ; L_3 ; L_4 ; L_5 ; L_6 ; L_7 ; L_8 ; L_9 ; L_10 ; ...
        L_11 ; L_12 ; L_13 ; L_14 ; L_15 ; L_16 ; L_17 ; L_18 ; L_19 ; L_20 ];
    
    % X-TFC ILS loop
    l2 = [2 1];
    iter = 0;
    
    while abs(l2(2)) > IterTol &&  iter < IterMax && abs(l2(1) - l2(2)) > IterTol
        
        l2(1)= l2(2);
        
        % compute derivatives
        
        % L1
        L_y1_xi_1 = c_i*hd + ( k1 + k10*y11_i + k14*y6_i + k23*y4_i + k24*y19_i   ).*(h-h0)   ;
        L_y1_xi_2 = ( -k2*y4_i -k3*y5_i -k9*y11_i - k12*y10_i ).*(h-h0) ;
        L_y1_xi_4 = ( k23*y1_i - k2*y2_i ).*(h-h0) ;
        L_y1_xi_5 =  -k3*y2_i.*(h-h0) ;
        L_y1_xi_6 = k14*y1_i.*(h-h0)  ;
        L_y1_xi_10 = -k12*y2_i.*(h-h0)  ;
        L_y1_xi_11 = ( k10*y1_i - k9*y2_i ).*(h-h0)  ;
        L_y1_xi_13 = -k11.*(h-h0)  ;
        L_y1_xi_19 = ( k24*y1_i - k22 ).*(h-h0)  ;
        L_y1_xi_20 = -k25.*(h-h0)  ;
                
        %L2
        L_y2_xi_1 = -k1.*(h-h0)   ;
        L_y2_xi_2 = c_i*hd + ( k2*y4_i + k3*y5_i + k9*y11_i + k12*y10_i  ).*(h-h0);
        L_y2_xi_4 = k2*y2_i.*(h-h0) ;
        L_y2_xi_5 = k3*y2_i.*(h-h0) ;
        L_y2_xi_10 = k12*y2_i.*(h-h0) ;
        L_y2_xi_11 = k9*y2_i.*(h-h0) ;
        L_y2_xi_19 = -k21.*(h-h0) ;
        
        %L3
        L_y3_xi_1 = -k1.*(h-h0) ;
        L_y3_xi_3 = c_i*hd + k15.*(h-h0)  ;
        L_y3_xi_4 = -k17.*(h-h0) ;
        L_y3_xi_16 = -k19.*(h-h0) ;
        L_y3_xi_19 = -k22.*(h-h0) ;
        
        %L4
        L_y4_xi_1 = k23.*(h-h0)  ;
        L_y4_xi_2 = k2*y4_i.*(h-h0) ;
        L_y4_xi_3 = -k15.*(h-h0)  ;
        L_y4_xi_4 = c_i*hd + ( k2*y2_i + k16 + k17 + k23*y1_i ).*(h-h0) ;
        
        %L5
        L_y5_xi_2 = k3*y5_i.*(h-h0) ;
        L_y5_xi_5 = c_i*hd + k3*y2_i.*(h-h0)  ;
        L_y5_xi_6 = ( -k6*y7_i - k20*y17_i).*(h-h0) ;
        L_y5_xi_7 = ( -2*k4 - k6*y6_i).*(h-h0) ;
        L_y5_xi_9 =  -k7.*(h-h0);
        L_y5_xi_14 = -k13.*(h-h0) ;
        L_y5_xi_17 = -k20*y6_i.*(h-h0) ;
        
        %L6
        L_y6_xi_1 = k14*y6_i.*(h-h0) ;
        L_y6_xi_2 = -k3*y5_i.*(h-h0) ;
        L_y6_xi_5 = -k3*y2_i.*(h-h0) ;
        L_y6_xi_6 = c_i*hd + ( k6*y7_i + k8*y9_i + k14*y1_i + k20*y17_i ).*(h-h0) ;
        L_y6_xi_7 = k6*y6_i.*(h-h0) ;
        L_y6_xi_9 = k8*y6_i.*(h-h0) ;
        L_y6_xi_16 = -2*k18.*(h-h0) ;
        L_y6_xi_17 = k20.*(h-h0) ;
        
        %L7
        L_y7_xi_6 = k6*y7_i.*(h-h0) ;
        L_y7_xi_7 = c_i*hd + ( k4 + k5 + k6*y6_i ).*(h-h0) ;
        L_y7_xi_14 = -k13.*(h-h0) ;
        
        %L8
        L_y8_xi_6 = -k6*y7_i.*(h-h0) ;
        L_y8_xi_7 = ( - k4 - k5 - k6*y6_i ).*(h-h0) ;
        L_y8_xi_8 = c_i*hd ;
        L_y8_xi_9 = -k7.*(h-h0) ;
        
        %L9
        L_y9_xi_6 = k8*y9_i.*(h-h0) ;
        L_y9_xi_9 = c_i*hd + ( k8*y6_i + k7 ).*(h-h0)  ;
        
        %L10
        L_y10_xi_2 = ( -k9*y11_i + k12*y10_i ).*(h-h0) ;
        L_y10_xi_9 = -k7.*(h-h0) ;
        L_y10_xi_10 = c_i*hd + k12*y2_i.*(h-h0) ;
        L_y10_xi_11 = -k9*y2_i.*(h-h0) ;
        
        %L11
        L_y11_xi_1 =  k10*y11_i.*(h-h0)  ;
        L_y11_xi_2 = k9*y11_i.*(h-h0) ;
        L_y11_xi_6 = -k8*y9_i.*(h-h0)  ;
        L_y11_xi_9 = -k8*y6_i.*(h-h0)  ;
        L_y11_xi_11 = c_i*hd + ( k9*y2_i  + k10*y1_i  ).*(h-h0) ;
        L_y11_xi_13 = -k11.*(h-h0) ;
        
        %L12
        L_y12_xi_2 = -k9*y11_i.*(h-h0) ;
        L_y12_xi_11 = -k9*y2_i.*(h-h0) ;
        L_y12_xi_12 = c_i*hd ;
        
        %L13
        L_y13_xi_1 =  -k10*y11_i.*(h-h0)  ;
        L_y13_xi_2 = -k10*y1_i.*(h-h0) ;
        L_y13_xi_13 = c_i*hd + k11.*(h-h0) ;
        
        %L14
        L_y14_xi_2 = -k12*y10_i.*(h-h0) ;
        L_y14_xi_10 = -k12*y2_i.*(h-h0) ;
        L_y14_xi_14 = c_i*hd + k13.*(h-h0)  ;
        
        %L15
        L_y15_xi_1 =  -k14*y6_i.*(h-h0)  ;
        L_y15_xi_6 = -k14*y1_i.*(h-h0)  ;
        L_y15_xi_15 = c_i*hd ;
         
        %L16
        L_y16_xi_4 = -k16.*(h-h0) ;
        L_y16_xi_16 = c_i*hd + (k18 + k19).*(h-h0) ;
        
        %L17
        L_y17_xi_6 = k20*y17_i.*(h-h0) ;
        L_y17_xi_17 = c_i*hd + k20*y6_i.*(h-h0) ;       
        
        %L18
        L_y18_xi_6 = -k20*y17_i.*(h-h0) ;
        L_y18_xi_17 = -k20*y6_i.*(h-h0)  ;
        L_y18_xi_18 = c_i*hd ;       
        
        %L19
        L_y19_xi_1 = (k24*y19_i - k23*y4_i ).*(h-h0)   ;
        L_y19_xi_4 = -k23*y1_i.*(h-h0) ;
        L_y19_xi_19 = c_i*hd + ( k21 + k22 + k24*y1_i ).*(h-h0);
        L_y19_xi_20 = -k25.*(h-h0) ;
               
        %L20
        L_y20_xi_1 =  -k24*y19_i.*(h-h0)  ;
        L_y20_xi_19 = -k24*y1_i.*(h-h0) ;
        L_y20_xi_20 = c_i*hd + k25.*(h-h0);
           
        % Jacobian matrix
        JJ = [ L_y1_xi_1   , L_y1_xi_2   ,      Z      ,  L_y1_xi_4   ,  L_y1_xi_5  , L_y1_xi_6   ,      Z       ,     Z       ,     Z       , L_y1_xi_10  , L_y1_xi_11  ,      Z      , L_y1_xi_13  ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      , L_y1_xi_19  , L_y1_xi_20  ;
               L_y2_xi_1   , L_y2_xi_2   ,      Z      ,  L_y2_xi_4   ,  L_y2_xi_5  ,     Z       ,      Z       ,     Z       ,     Z       , L_y2_xi_10  , L_y2_xi_11  ,      Z      ,     Z       ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      , L_y2_xi_19  ,      Z      ;
               L_y3_xi_1   ,     Z       , L_y3_xi_3   ,  L_y3_xi_4   ,      Z      ,     Z       ,      Z       ,     Z       ,     Z       ,     Z       ,     Z       ,      Z      ,     Z       ,      Z      ,      Z      , L_y3_xi_16  ,      Z      ,      Z      , L_y3_xi_19  ,      Z      ;
               L_y4_xi_1   , L_y4_xi_2   , L_y4_xi_3   ,  L_y4_xi_4   ,      Z      ,     Z       ,      Z       ,     Z       ,     Z       ,     Z       ,     Z       ,      Z      ,     Z       ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ;
                   Z       , L_y5_xi_2   ,      Z      ,      Z       ,  L_y5_xi_5  , L_y5_xi_6   ,  L_y5_xi_7   ,     Z       , L_y5_xi_9   ,     Z       ,      Z      ,      Z      ,     Z       , L_y5_xi_14  ,      Z      ,      Z      , L_y5_xi_17  ,      Z      ,      Z      ,      Z      ;
               L_y6_xi_1   , L_y6_xi_2   ,      Z      ,      Z       ,  L_y6_xi_5  , L_y6_xi_6   ,  L_y6_xi_7   ,     Z       , L_y6_xi_9   ,     Z       ,     Z       ,      Z      ,     Z       ,      Z      ,      Z      , L_y6_xi_16  , L_y6_xi_17  ,      Z      ,      Z      ,      Z      ;
                   Z       ,     Z       ,      Z      ,      Z       ,      Z      , L_y7_xi_6   ,  L_y7_xi_7   ,     Z       ,     Z       ,     Z       ,     Z       ,      Z      ,     Z       , L_y7_xi_14  ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ;
                   Z       ,     Z       ,      Z      ,      Z       ,      Z      , L_y8_xi_6   ,  L_y8_xi_7   , L_y8_xi_8   , L_y8_xi_9   ,     Z       ,     Z       ,      Z      ,     Z       ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ;
                   Z       ,     Z       ,      Z      ,      Z       ,      Z      , L_y9_xi_6   ,       Z      ,      Z      , L_y9_xi_9   ,     Z       ,     Z       ,      Z      ,     Z       ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ; 
                   Z       , L_y10_xi_2  ,      Z      ,      Z       ,      Z      ,      Z      ,       Z      ,      Z      , L_y10_xi_9  , L_y10_xi_10 , L_y10_xi_11 ,      Z      ,     Z       ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ;
               L_y11_xi_1  , L_y11_xi_2  ,      Z      ,      Z       ,      Z      , L_y11_xi_6  ,       Z      ,      Z      , L_y11_xi_9  ,     Z       , L_y11_xi_11 ,      Z      , L_y11_xi_13 ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ;
                   Z       , L_y12_xi_2  ,      Z      ,      Z       ,      Z      ,      Z      ,       Z      ,      Z      ,      Z      ,     Z       , L_y12_xi_11 , L_y12_xi_12 ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ;
               L_y13_xi_1  , L_y13_xi_2  ,      Z      ,      Z       ,      Z      ,      Z      ,       Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      , L_y13_xi_13 ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ;
                    Z      , L_y14_xi_2  ,      Z      ,      Z       ,      Z      ,      Z      ,       Z      ,      Z      ,      Z      , L_y14_xi_10 ,      Z      ,      Z      ,      Z      , L_y14_xi_14 ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ;
               L_y15_xi_1  ,      Z      ,      Z      ,      Z       ,      Z      , L_y15_xi_6  ,       Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      , L_y15_xi_15 ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ;
                    Z      ,      Z      ,      Z      , L_y16_xi_4   ,      Z      ,      Z      ,       Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      , L_y16_xi_16 ,      Z      ,      Z      ,      Z      ,      Z      ;
                    Z      ,      Z      ,      Z      ,      Z       ,      Z      , L_y17_xi_6  ,       Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      , L_y17_xi_17 ,      Z      ,      Z      ,      Z      ;
                    Z      ,      Z      ,      Z      ,      Z       ,      Z      , L_y18_xi_6  ,       Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      , L_y18_xi_17 , L_y18_xi_18 ,      Z      ,      Z      ;
               L_y19_xi_1  ,      Z      ,      Z      , L_y19_xi_4   ,      Z      ,      Z      ,       Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      , L_y19_xi_19 , L_y19_xi_20 ;
               L_y20_xi_1  ,      Z      ,      Z      ,      Z       ,      Z      ,      Z      ,       Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      ,      Z      , L_y20_xi_19 , L_y20_xi_20
            ];
              
        % xi variation
        dxi = lsqminnorm(JJ,Loss);
        
        % update xi
        xi_i = xi_i - dxi;
        
        xi_1_i = xi_i((0*L)+1:1*L); xi_6_i = xi_i((5*L)+1:6*L);   xi_11_i = xi_i((10*L)+1:11*L);  xi_16_i = xi_i((15*L)+1:16*L);  
        xi_2_i = xi_i((1*L)+1:2*L); xi_7_i = xi_i((6*L)+1:7*L);   xi_12_i = xi_i((11*L)+1:12*L);  xi_17_i = xi_i((16*L)+1:17*L);  
        xi_3_i = xi_i((2*L)+1:3*L); xi_8_i = xi_i((7*L)+1:8*L);   xi_13_i = xi_i((12*L)+1:13*L);  xi_18_i = xi_i((17*L)+1:18*L);  
        xi_4_i = xi_i((3*L)+1:4*L); xi_9_i = xi_i((8*L)+1:9*L);   xi_14_i = xi_i((13*L)+1:14*L);  xi_19_i = xi_i((18*L)+1:19*L);  
        xi_5_i = xi_i((4*L)+1:5*L); xi_10_i = xi_i((9*L)+1:10*L); xi_15_i = xi_i((14*L)+1:15*L);  xi_20_i = xi_i((19*L)+1:20*L);         
        
        %% Re-Build Constrained Expressions
        
        y1_i = (h-h0)*xi_1_i + y1_0;        y1_dot_i = c_i*hd*xi_1_i;
        y2_i = (h-h0)*xi_2_i + y2_0;        y2_dot_i = c_i*hd*xi_2_i;
        y3_i = (h-h0)*xi_3_i + y3_0;        y3_dot_i = c_i*hd*xi_3_i;
        y4_i = (h-h0)*xi_4_i + y4_0;        y4_dot_i = c_i*hd*xi_4_i;
        y5_i = (h-h0)*xi_5_i + y5_0;        y5_dot_i = c_i*hd*xi_5_i;
        y6_i = (h-h0)*xi_6_i + y6_0;        y6_dot_i = c_i*hd*xi_6_i;
        y7_i = (h-h0)*xi_7_i + y7_0;        y7_dot_i = c_i*hd*xi_7_i;
        y8_i = (h-h0)*xi_8_i + y8_0;        y8_dot_i = c_i*hd*xi_8_i;
        y9_i = (h-h0)*xi_9_i + y9_0;        y9_dot_i = c_i*hd*xi_9_i;
        y10_i = (h-h0)*xi_10_i + y10_0;        y10_dot_i = c_i*hd*xi_10_i;
        y11_i = (h-h0)*xi_11_i + y11_0;        y11_dot_i = c_i*hd*xi_11_i;
        y12_i = (h-h0)*xi_12_i + y12_0;        y12_dot_i = c_i*hd*xi_12_i;
        y13_i = (h-h0)*xi_13_i + y13_0;        y13_dot_i = c_i*hd*xi_13_i;
        y14_i = (h-h0)*xi_14_i + y14_0;        y14_dot_i = c_i*hd*xi_14_i;
        y15_i = (h-h0)*xi_15_i + y15_0;        y15_dot_i = c_i*hd*xi_15_i;
        y16_i = (h-h0)*xi_16_i + y16_0;        y16_dot_i = c_i*hd*xi_16_i;
        y17_i = (h-h0)*xi_17_i + y17_0;        y17_dot_i = c_i*hd*xi_17_i;
        y18_i = (h-h0)*xi_18_i + y18_0;        y18_dot_i = c_i*hd*xi_18_i;
        y19_i = (h-h0)*xi_19_i + y19_0;        y19_dot_i = c_i*hd*xi_19_i;
        y20_i = (h-h0)*xi_20_i + y20_0;        y20_dot_i = c_i*hd*xi_20_i;
                
        % reaction velocities
        
        r1 = k1*y1_i;           r6 = k6*y7_i.*y6_i;         r11 = k11*y13_i;            r16 = k16*y4_i;             r21 = k21*y19_i;
        r2 = k2*y2_i.*y4_i;     r7 = k7*y9_i;               r12 = k12*y10_i.*y2_i;      r17 = k17*y4_i;             r22 = k22*y19_i;
        r3 = k3*y5_i.*y2_i;     r8 = k8*y9_i.*y6_i;         r13 = k13*y14_i;            r18 = k18*y16_i;            r23 = k23*y1_i.*y4_i;
        r4 = k4*y7_i;           r9 = k9*y11_i.*y2_i;        r14 = k14*y1_i.*y6_i;       r19 = k19*y16_i;            r24 = k24*y19_i.*y1_i;
        r5 = k5*y7_i;           r10 = k10*y11_i.*y1_i;      r15 = k15*y3_i;             r20 = k20*y17_i.*y6_i;      r25 = k25*y20_i;
                
        %% Re-Build the Losses
        
        L_1 = y1_dot_i +r1+r10+r14+r23+r24-r2-r3-r9-r11-r12-r22-r25 ;
        L_2 = y2_dot_i +r2+r3+r9+r12-r1-r21 ;
        L_3 = y3_dot_i +r15-r1-r17-r19-r22 ;
        L_4 = y4_dot_i +r2+r16+r17+r23-r15 ;
        L_5 = y5_dot_i +r3-2*r4-r6-r7-r13-r20 ;
        L_6 = y6_dot_i +r6+r8+r14+r20-r3-2*r18 ;
        L_7 = y7_dot_i +r4+r5+r6-r13 ;
        L_8 = y8_dot_i -r4-r5-r6-r7 ;
        L_9 = y9_dot_i +r7+r8 ;
        L_10 = y10_dot_i +r12-r7-r9 ;
        L_11 = y11_dot_i +r9+r10-r8-r11 ;
        L_12 = y12_dot_i -r9 ;
        L_13 = y13_dot_i +r11-r10 ;
        L_14 = y14_dot_i +r13-r12 ;
        L_15 = y15_dot_i -r14 ;
        L_16 = y16_dot_i +r18+r19-r16 ;
        L_17 = y17_dot_i +r20 ;
        L_18 = y18_dot_i -r20 ;
        L_19 = y19_dot_i +r21+r22+r24-r23-r25 ;
        L_20 = y20_dot_i +r25-r24 ;
               
        Loss = [L_1 ; L_2 ; L_3 ; L_4 ; L_5 ; L_6 ; L_7 ; L_8 ; L_9 ; L_10 ; ...
            L_11 ; L_12 ; L_13 ; L_14 ; L_15 ; L_16 ; L_17 ; L_18 ; L_19 ; L_20 ];
        
        
        l2(2) = norm(Loss);
        
        iter = iter+1;
        
    end
    
    training_err = sqrt(mean(abs(L_1.^2))) + sqrt(mean(abs(L_2.^2))) +  sqrt(mean(abs(L_3.^2))) ...
        +  sqrt(mean(abs(L_4.^2)))  +  sqrt(mean(abs(L_5.^2)))  +  sqrt(mean(abs(L_6.^2)))  +  sqrt(mean(abs(L_7.^2)))   ...
        +  sqrt(mean(abs(L_8.^2)))  +  sqrt(mean(abs(L_9.^2)))  +  sqrt(mean(abs(L_10.^2)))  +  sqrt(mean(abs(L_11.^2)))   ...
        +  sqrt(mean(abs(L_12.^2)))  +  sqrt(mean(abs(L_13.^2)))  +  sqrt(mean(abs(L_14.^2)))  +  sqrt(mean(abs(L_15.^2)))  ...
        +  sqrt(mean(abs(L_16.^2)))  +  sqrt(mean(abs(L_17.^2)))  +  sqrt(mean(abs(L_18.^2)))  +  sqrt(mean(abs(L_19.^2)))  +  sqrt(mean(abs(L_20.^2)))  ;
    
    % Update of constraints
    
    y1_0 = y1_i(end);   y6_0 = y6_i(end);   y11_0 = y11_i(end);  y16_0 = y16_i(end); 
    y2_0 = y2_i(end);   y7_0 = y7_i(end);   y12_0 = y12_i(end);  y17_0 = y17_i(end); 
    y3_0 = y3_i(end);   y8_0 = y8_i(end);   y13_0 = y13_i(end);  y18_0 = y18_i(end); 
    y4_0 = y4_i(end);   y9_0 = y9_i(end);   y14_0 = y14_i(end);  y19_0 = y19_i(end); 
    y5_0 = y5_i(end);   y10_0 = y10_i(end); y15_0 = y15_i(end);  y20_0 = y20_i(end);     
    
    y1(i+1) = y1_0;     y6(i+1) = y6_0;     y11(i+1) = y11_0;   y16(i+1) = y16_0;
    y2(i+1) = y2_0;     y7(i+1) = y7_0;     y12(i+1) = y12_0;   y17(i+1) = y17_0;
    y3(i+1) = y3_0;     y8(i+1) = y8_0;     y13(i+1) = y13_0;   y18(i+1) = y18_0;
    y4(i+1) = y4_0;     y9(i+1) = y9_0;     y14(i+1) = y14_0;   y19(i+1) = y19_0;
    y5(i+1) = y5_0;     y10(i+1) = y10_0;   y15(i+1) = y15_0;   y20(i+1) = y20_0;
        
    training_err_vec(i) = training_err;
         
end

xtfc_elapsedtime = toc(tstart) 

%% =======================================
% MATLAB ode15s solver

y0_ode15s = [0; 0.2; 0; 0.04; 0; 0; 0.1; 0.3; 0.01; 0; 0; 0; 0; 0; 0; 0; 0.007; 0; 0; 0];
options = odeset('RelTol',IterTol);
% test the automatic detection of a DAE.

tStart = tic;
[t_15s,y_15s] = ode15s(@pollu_ode15s_function, t_tot',y0_ode15s,options);
ode15s_elapsedtime = toc(tStart) 

%% errors

err_ode15s_1 = abs(y_15s(:,1) - y1) ;
err_ode15s_2 = abs(y_15s(:,2) - y2) ;
err_ode15s_3 = abs(y_15s(:,3) - y3) ;
err_ode15s_4 = abs(y_15s(:,4) - y4) ;
err_ode15s_5 = abs(y_15s(:,5) - y5) ;
err_ode15s_6 = abs(y_15s(:,6) - y6) ;
err_ode15s_7 = abs(y_15s(:,7) - y7) ;
err_ode15s_8 = abs(y_15s(:,8) - y8) ;
err_ode15s_9 = abs(y_15s(:,9) - y9) ;
err_ode15s_10 = abs(y_15s(:,10) - y10) ;
err_ode15s_11 = abs(y_15s(:,11) - y11) ;
err_ode15s_12 = abs(y_15s(:,12) - y12) ;
err_ode15s_13 = abs(y_15s(:,13) - y13) ;
err_ode15s_14 = abs(y_15s(:,14) - y14) ;
err_ode15s_15 = abs(y_15s(:,15) - y15) ;
err_ode15s_16 = abs(y_15s(:,16) - y16) ;
err_ode15s_17 = abs(y_15s(:,17) - y17) ;
err_ode15s_18 = abs(y_15s(:,18) - y18) ;
err_ode15s_19 = abs(y_15s(:,19) - y19) ;
err_ode15s_20 = abs(y_15s(:,20) - y20) ;

fprintf(' The average training error for X-TFC is: %g \n', mean(training_err_vec) )
% fprintf(' The average training error for ode15s is: %g \n', training_err_ode15s )



figure(1)
subplot(4,5,1)
hold on
grid on
plot(t_15s,y_15s(:,1), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y1,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y1')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,2)
hold on
grid on
plot(t_15s,y_15s(:,2), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y2,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y2')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,3)
hold on
grid on
plot(t_15s,y_15s(:,3), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y3,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y3')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,4)
hold on
grid on
plot(t_15s,y_15s(:,4), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y4,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y4')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,5)
hold on
grid on
plot(t_15s,y_15s(:,5), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y5,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y5')
hYLabel = get(gca,'YLabel');
legend('X-TFC','ode15s')
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,6)
hold on
grid on
plot(t_15s,y_15s(:,6), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y6,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y6')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,7)
hold on
grid on
plot(t_15s,y_15s(:,7), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y7,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y7')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,8)
hold on
grid on
plot(t_15s,y_15s(:,8), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y8,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y8')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,9)
hold on
grid on
plot(t_15s,y_15s(:,9), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y9,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y9')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,10)
hold on
grid on
plot(t_15s,y_15s(:,10), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y10,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y10')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,11)
hold on
grid on
plot(t_15s,y_15s(:,11), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y11,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y11')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,12)
hold on
grid on
plot(t_15s,y_15s(:,12), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y12,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y12')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,13)
hold on
grid on
plot(t_15s,y_15s(:,13), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y13,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y13')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,14)
hold on
grid on
plot(t_15s,y_15s(:,14), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y14,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y14')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,15)
hold on
grid on
plot(t_15s,y_15s(:,15), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y15,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y15')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,16)
hold on
grid on
plot(t_15s,y_15s(:,16), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y16,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y16')
xlabel('time (min)')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,17)
hold on
grid on
plot(t_15s,y_15s(:,17), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y17,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y17')
xlabel('time (min)')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,18)
hold on
grid on
plot(t_15s,y_15s(:,18), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y18,'--','LineWidth',3, 'Color', [17 50 50]/100)
ylabel('y18')
xlabel('time (min)')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,19)
hold on
grid on
plot(t_15s,y_15s(:,19), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y19,'--','LineWidth',3, 'Color', [17 50 50]/100)
xlabel('time (min)')
ylabel('y19')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,20)
hold on
grid on
plot(t_15s,y_15s(:,20), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y20,'--','LineWidth',3, 'Color', [17 50 50]/100)
xlabel('time (min)')
ylabel('y20')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on




figure(2)

subplot(4,5,1)
hold on
grid on 
plot(t_15s,err_ode15s_1,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y1')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
set(gca, 'YScale','log')
box on

subplot(4,5,2)

hold on
grid on 
plot(t_15s,err_ode15s_2,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y2')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,3)

hold on
grid on 
plot(t_15s,err_ode15s_3,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y3')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,4)

hold on
grid on 
plot(t_15s,err_ode15s_4,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y4')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,5)

hold on
grid on 
plot(t_15s,err_ode15s_5,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y5')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,6)

hold on
grid on 
plot(t_15s,err_ode15s_6,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y6')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,7)

hold on
grid on 
plot(t_15s,err_ode15s_7,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y7')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,8)

hold on
grid on 
plot(t_15s,err_ode15s_8,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y8')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,9)

hold on
grid on 
plot(t_15s,err_ode15s_9,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y9')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,10)

hold on
grid on 
plot(t_15s,err_ode15s_10,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y10')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,11)

hold on
grid on 
plot(t_15s,err_ode15s_11,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y11')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,12)

hold on
grid on 
plot(t_15s,err_ode15s_12,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y12')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on


subplot(4,5,13)

hold on
grid on 
plot(t_15s,err_ode15s_13,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y13')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,14)

hold on
grid on 
plot(t_15s,err_ode15s_14,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y14')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,15)

hold on
grid on 
plot(t_15s,err_ode15s_15,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('y15')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,16)

hold on
grid on 
plot(t_15s,err_ode15s_16,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
xlabel('time (min)')
ylabel('y16')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,17)

hold on
grid on 
plot(t_15s,err_ode15s_17,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
xlabel('time (min)')
ylabel('y17')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,18)

hold on
grid on 
plot(t_15s,err_ode15s_18,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
xlabel('time (min)')
ylabel('y18')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,19)

hold on
grid on 
plot(t_15s,err_ode15s_19,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
xlabel('time (min)')
ylabel('y19')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on

subplot(4,5,20)

hold on
grid on 
plot(t_15s,err_ode15s_20,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
xlabel('time (min)')
ylabel('y20')
set(gca, 'YScale','log')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
box on
