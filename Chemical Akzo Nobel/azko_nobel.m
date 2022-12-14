%%
clear; close all; clc;
format long
%--------------------------------------------------------------------------
%{
  Physics-Informed X-TFC applied to Stiff Chemical Kinetics
  Test Case 2 - Chemical Akzo Nobel Problem

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
t_f = 180; % final time

%t_query = [0.1,1,2,5,10,20,30,40]; 

n_x = 50;    % Discretization order for x (-1,1)
L = 20;    % number of neurons

x = linspace(0,1,n_x)';

t_step = 0.01;

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
IterTol = 1e-12;

%% Chemical Parameters definition

% rate constants
k1 = 18.7;
k2 = 0.58;
k3 = 0.09;
k4 = 0.42;
K = 34.4;
klA = 3.3;
pCO2 = 0.9;
H = 737;



switch type_basis
    
    case 1
        [h, hd] = CP(x, L + 1);
        
        % Restore matrices (to start with the 2nd order polynom)
        I = 2:L+1;
        h = h(:,I);  hd = hd(:,I);
        h0=h(1,:); hf=h(end,:);
        
%% Activation function definition

    case 2
% 
%         weight = unifrnd(-(L-10)/10 - 4,(L-10)/10 + 4,L,1);
%         centers = unifrnd(x(1),x(end),L,1);
%         bias = -weight.*centers;
        
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

y1 = zeros(n_t,1);
y2 = zeros(n_t,1);
y3 = zeros(n_t,1);
y4 = zeros(n_t,1);
y5 = zeros(n_t,1);
y6 = zeros(n_t,1);

% Initial Values

y1_0 = 0.437; 
y2_0 = 0.00123; 
y3_0 = 0;
y4_0 = 0;
y5_0 = 0;
y6_0 = 0.367;


y1(1) = y1_0;
y2(1) = y2_0;
y3(1) = y3_0;
y4(1) = y4_0;
y5(1) = y5_0;
y6(1) = y6_0;

training_err_vec = zeros(n_t-1,1);

tStart = tic;

for i = 1:(n_t-1)
     
    c_i = (x(end) - x(1)) / (t_tot(i+1) - t_tot(i));
    
    t = t_tot(i) + (1/c_i)*(x-x(1));

    xi_1_i = zeros(L,1);
    xi_2_i = zeros(L,1);
    xi_3_i = zeros(L,1);
    xi_4_i = zeros(L,1);
    xi_5_i = zeros(L,1);
    xi_6_i = zeros(L,1);

    xi_i = [xi_1_i;xi_2_i;xi_3_i;xi_4_i;xi_5_i;xi_6_i];
       
    %% Build Constrained Expressions
    
    y1_i = (h-h0)*xi_1_i + y1_0;        y1_dot_i = c_i*hd*xi_1_i;     
    y2_i = (h-h0)*xi_2_i + y2_0;        y2_dot_i = c_i*hd*xi_2_i;   
    y3_i = (h-h0)*xi_3_i + y3_0;        y3_dot_i = c_i*hd*xi_3_i;
    y4_i = (h-h0)*xi_4_i + y4_0;        y4_dot_i = c_i*hd*xi_4_i;
    y5_i = (h-h0)*xi_5_i + y5_0;        y5_dot_i = c_i*hd*xi_5_i;
    y6_i = (h-h0)*xi_6_i + y6_0;        y6_dot_i = c_i*hd*xi_6_i;
    
    % reaction velocities
    
    r1 = k1*(y1_i.^4).*y2_i.^(0.5) ;
    r2 = k2*y3_i.*y4_i ;
    r3 = (k2/K)*y1_i.*y5_i ;
    r4 = k3*y1_i.*y4_i.^2 ;
    r5 = k4*(y6_i.^2).*y2_i.^(0.5) ;
    
    %  inflow of oxygen per volume unit 
    
    F_in = klA.*((pCO2/H) - y2_i);
       
    %% Build the Losses  
    
    L_1 = y1_dot_i + 2*r1 - r2 + r3 + r4 ;
    L_2 = y2_dot_i + 0.5*r1 + r4 + 0.5*r5 - F_in ;
    L_3 = y3_dot_i - r1 + r2 - r3 ;
    L_4 = y4_dot_i + r2 - r3 + 2*r4 ;
    L_5 = y5_dot_i - r2 + r3 - r5 ;
    L_6 = y6_dot_i + r5 ;
    
    Loss = [L_1 ; L_2 ; L_3 ; L_4 ; L_5 ; L_6];
    
    % X-TFC ILS loop
    l2 = [2 1];
    iter = 0;
    
    while abs(l2(2)) > IterTol &&  iter < IterMax && abs(l2(1) - l2(2)) > IterTol
        
        l2(1)= l2(2);
        
        % compute derivatives
        
        % L1
        L_y1_xi_1 = c_i*hd + (8*k1*y1_i.^3.*y2_i.^0.5 + (k2/K)*y5_i + k3*y4_i.^2).*(h-h0) ;
        L_y1_xi_2 = 0.5*k1*y1_i.^4.*y2_i.^-0.5.*(h-h0) ;
        L_y1_xi_3 = -k2*y4_i.*(h-h0) ;
        L_y1_xi_4 = (-k2*y3_i + 2*k3*y1_i.*y4_i).*(h-h0);
        L_y1_xi_5 = (k2/K)*y1_i.*(h-h0) ;
        
        %L2
        L_y2_xi_1 = (2*k1*y1_i.^3.*y2_i.^0.5 + k3*y4_i.^2).*(h-h0) ;
        L_y2_xi_2 = c_i*hd + (0.25*k1*y1_i.^4.*y2_i.^-0.5 + 0.25*k4*y6_i.^2.*y2_i.^-0.5 + klA).*(h-h0) ;      
        L_y2_xi_4 = 2*k3*y1_i.*y4_i.*(h-h0) ;
        L_y2_xi_6 = k4*y6_i.*y2_i.^0.5.*(h-h0) ;
        
        %L3
        L_y3_xi_1 = (-4*k1*y1_i.^3.*y2_i.^0.5 - (k2/K)*y5_i).*(h-h0) ;
        L_y3_xi_2 = -0.5*k1*y1_i.^4.*y2_i.^-0.5.*(h-h0) ; 
        L_y3_xi_3 = c_i*hd + k2*y4_i.*(h-h0);        
        L_y3_xi_4 = k2*y3_i.*(h-h0);
        L_y3_xi_5 = - (k2/K)*y1_i.*(h-h0) ;
        
        %L4
        L_y4_xi_1 = ( -(k2/K)*y5_i + 2*k3*y4_i.^2).*(h-h0) ;
        L_y4_xi_3 = k2*y4_i.*(h-h0) ;
        L_y4_xi_4 = c_i*hd + (k2*y3_i + 4*k3*y1_i.*y4_i).*(h-h0) ;
        L_y4_xi_5 = (k2/K)*y1_i.*(h-h0) ;
        
        %L5
        L_y5_xi_1 = (k2/K)*y5_i.*(h-h0) ;
        L_y5_xi_2 = -0.5*k4*y6_i.^2.*y2_i.^-0.5.*(h-h0) ;
        L_y5_xi_3 = -k2*y4_i.*(h-h0) ;
        L_y5_xi_4 = -k2*y3_i.*(h-h0) ;
        L_y5_xi_5 = c_i*hd + (k2/K)*y1_i.*(h-h0);
        L_y5_xi_6 = -2*k4*y6_i.*y2_i.^0.5.*(h-h0) ;
        
        %L6
        L_y6_xi_2 = 0.5*k4*y6_i.^2.*y2_i.^-0.5.*(h-h0) ;
        L_y6_xi_6 = c_i*hd + 2*y6_i.*y2_i.^0.5.*(h-h0) ;
               
        % Jacobian matrix     
        JJ = [ L_y1_xi_1 , L_y1_xi_2 , L_y1_xi_3 , L_y1_xi_4 , L_y1_xi_5 ,     Z     ; 
               L_y2_xi_1 , L_y2_xi_2 ,     Z     , L_y2_xi_4 ,     Z     , L_y2_xi_6 ;
               L_y3_xi_1 , L_y3_xi_2 , L_y3_xi_3 , L_y3_xi_4 , L_y3_xi_5 ,     Z     ;
               L_y4_xi_1 ,     Z     , L_y4_xi_3 , L_y4_xi_4 , L_y4_xi_5 ,     Z     ;
               L_y5_xi_1 , L_y5_xi_2 , L_y5_xi_3 , L_y5_xi_4 , L_y5_xi_5 , L_y5_xi_6 ;
                   Z     , L_y6_xi_2 ,     Z     ,     Z     ,     Z     , L_y6_xi_6 
               ];
                 
        % xi variation
        dxi = lsqminnorm(JJ,Loss);
        
        % update xi
        xi_i = xi_i - dxi;
        
        xi_1_i = xi_i(1:L);
        xi_2_i = xi_i(L+1:2*L);
        xi_3_i = xi_i((2*L)+1:3*L);
        xi_4_i = xi_i((3*L)+1:4*L);
        xi_5_i = xi_i((4*L)+1:5*L);
        xi_6_i = xi_i((5*L)+1:6*L);
        
        %% Re-Build Constrained Expressions
        
        y1_i = (h-h0)*xi_1_i + y1_0;        y1_dot_i = c_i*hd*xi_1_i;
        y2_i = (h-h0)*xi_2_i + y2_0;        y2_dot_i = c_i*hd*xi_2_i;
        y3_i = (h-h0)*xi_3_i + y3_0;        y3_dot_i = c_i*hd*xi_3_i;
        y4_i = (h-h0)*xi_4_i + y4_0;        y4_dot_i = c_i*hd*xi_4_i;
        y5_i = (h-h0)*xi_5_i + y5_0;        y5_dot_i = c_i*hd*xi_5_i;
        y6_i = (h-h0)*xi_6_i + y6_0;        y6_dot_i = c_i*hd*xi_6_i;
        
        % reaction velocities
        
        r1 = k1*y1_i.^4.*y2_i.^0.5 ;
        r2 = k2*y3_i.*y4_i ;
        r3 = (k2/K)*y1_i.*y5_i ;
        r4 = k3*y1_i.*y4_i.^2 ;
        r5 = k4*y6_i.^2.*y2_i.^0.5 ;
        
        %  inflow of oxygen per volume unit
        
        F_in = klA.*((pCO2/H) - y2_i);
        
        %% Re-Build the Losses
        
        L_1 = y1_dot_i + 2*r1 - r2 + r3 + r4 ;
        L_2 = y2_dot_i + 0.5*r1 + r4 + 0.5*r5 - F_in ;
        L_3 = y3_dot_i - r1 + r2 - r3 ;
        L_4 = y4_dot_i + r2 - r3 + 2*r4 ;
        L_5 = y5_dot_i - r2 + r3 - r5 ;
        L_6 = y6_dot_i + r5 ;
        
        Loss = [L_1 ; L_2 ; L_3 ; L_4 ; L_5 ; L_6];
        
        l2(2) = norm(Loss);
        
        iter = iter+1;
      
    end
    
    training_err = sqrt(mean(abs(L_1.^2))) + sqrt(mean(abs(L_2.^2))) +  sqrt(mean(abs(L_3.^2))) ...
        +  sqrt(mean(abs(L_4.^2)))  +  sqrt(mean(abs(L_5.^2)))  +  sqrt(mean(abs(L_6.^2)))   ;
    
    % Update of constraints
    
    y1_0 = y1_i(end);
    y2_0 = y2_i(end);
    y3_0 = y3_i(end);
    y4_0 = y4_i(end);
    y5_0 = y5_i(end);
    y6_0 = y6_i(end);
        
	y1(i+1) = y1_0;
    y2(i+1) = y2_0;
    y3(i+1) = y3_0;
    y4(i+1) = y4_0;
    y5(i+1) = y5_0;
    y6(i+1) = y6_0;
    
    training_err_vec(i) = training_err;
                   
end


xtfc_elapsedtime = toc(tStart) ;

fprintf('The elapsed time for x-tfc is: %g \n', xtfc_elapsedtime );

%% =======================================
% MATLAB ode15s solver

y0_ode15s = [0.437; 0.00123; 0; 0; 0; 0.367];
options = odeset('RelTol',IterTol);
% test the automatic detection of a DAE.

tStart = tic;
[t_15s,y_15s] = ode15s(@akzonobel_ode15s_function, t_tot',y0_ode15s,options);
ode15s_elapsedtime = toc(tStart) ;

fprintf('The elapsed time for ode15s is: %g \n', ode15s_elapsedtime );

dy1dt = gradient(y_15s(:,1));
dy2dt = gradient(y_15s(:,2));
dy3dt = gradient(y_15s(:,3));
dy4dt = gradient(y_15s(:,4));
dy5dt = gradient(y_15s(:,5));
dy6dt = gradient(y_15s(:,6));

% loss1 = dy1dt + 0.04*y_15s(:,1) - 10^4*y_15s(:,2).*y_15s(:,3);
% loss2 = dy2dt - 0.04*y_15s(:,1) + 10^4*y_15s(:,2).*y_15s(:,3) + (3*10^7)*y_15s(:,2).^2;
% loss3 = dy3dt - (3*10^7)*y_15s(:,2).^2;
% 
% training_err_ode15s = sqrt(mean(abs(loss1.^2))) + sqrt(mean(abs(loss2.^2))) +  sqrt(mean(abs(loss3.^2)))   ;


%% errors



err_ode15s_1 = abs(y_15s(:,1) - y1) ;
err_ode15s_2 = abs(y_15s(:,2) - y2) ;
err_ode15s_3 = abs(y_15s(:,3) - y3) ;
err_ode15s_4 = abs(y_15s(:,4) - y4) ;
err_ode15s_5 = abs(y_15s(:,5) - y5) ;
err_ode15s_6 = abs(y_15s(:,6) - y6) ;


fprintf('The average training error for X-TFC is: %g \n', mean(training_err_vec) )


subplot(4,3,1)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,y_15s(:,1), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y1,'--','LineWidth',3 , 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('concentration')
box on
title('y1', 'FontWeight', 'Normal')

subplot(4,3,2)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,y_15s(:,2), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y2,'--','LineWidth',3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
box on
title('y2', 'FontWeight', 'Normal')

subplot(4,3,3)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,y_15s(:,3), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y3,'--','LineWidth',3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
legend('ode15s','X-TFC')
box on
title('y3', 'FontWeight', 'Normal')


subplot(4,3,4)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,err_ode15s_1,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('abs(error)')
set(gca, 'YScale','log')
box on

subplot(4,3,5)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,err_ode15s_2,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
set(gca, 'YScale','log')
box on

subplot(4,3,6)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,err_ode15s_3,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
set(gca, 'YScale','log')
box on




subplot(4,3,7)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,y_15s(:,4), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y4,'--','LineWidth',3 , 'Color', [17 50 50]/100)
xlim([t_0 t_f])
ylabel('concentration')
box on
title('y4', 'FontWeight', 'Normal')

subplot(4,3,8)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,y_15s(:,5), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y5,'--','LineWidth',3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
box on
title('y5', 'FontWeight', 'Normal')

subplot(4,3,9)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,y_15s(:,6), 'LineWidth',3, 'Color', [80 32.5 9]/100)
plot(t_tot,y6,'--','LineWidth',3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
box on
title('y6', 'FontWeight', 'Normal')


subplot(4,3,10)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,err_ode15s_4,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
xlabel('time (s)')
ylabel('abs(error)')
set(gca, 'YScale','log')
box on

subplot(4,3,11)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,err_ode15s_5,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
xlabel('time (s)')
set(gca, 'YScale','log')
box on

subplot(4,3,12)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,err_ode15s_6,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
xlim([t_0 t_f])
xlabel('time (s)')
set(gca, 'YScale','log')
box on



figure(2)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_tot,y1,t_tot,y2,t_tot,y3,t_tot,y4,t_tot,y5,t_tot,y6, 'LineWidth',1.5)
xlim([t_0 t_f])
xlabel('time(s)')
% set(gca, 'XScale','log')
%xlim([10^(-4) t_tot(end)])
legend('y1','y2','y3','y4','y5','y6')
title('All concentrations', 'FontWeight', 'Normal')


