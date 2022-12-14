%%
clear; close all; clc;
format long
%--------------------------------------------------------------------------
%{
  Physics-Informed X-TFC appleid to Stiff Chemical Kinetics

  Test Case 1 - The ROBER problem

  Authors:
  Ing. Mario De Florio - PhD Student, The University of Arizona
  Ing. Enrico Schiassi - PhD Student, The University of Arizona
%}
%%
%--------------------------------------------------------------------------
%% Input

rng('default') % set random seed
format longE


start = tic;

t_0 = 10e-5; % initial time
t_f = 10e5; % final time

n_x = 10;    % Discretization order for x (-1,1)
L = 10;    % number of neurons

x = linspace(0,1,n_x)';

t_step = 20000;


t_tot = (t_0:t_step:t_f)';
n_t = length(t_tot);
t_tot = logspace(-5,5,n_t)';

type_basis = 2; % 1 = Orthogonal polynomials ; 2 = activation functions 
type_act = 2; % activation functions

LB = -1; % Lower boundary for weight and bias samplings
UB = 1; % Upper boundary for weight and bias samplings

% iterative least-square parameters

IterMax = 100;
IterTol = 1e-12;

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

%% Chemical Parameters definition

% rate constants
k1 = 0.04;
k2 = 3*(10^7);
k3 = 10^4;


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

% Initial Values

y1_0 = 1; 
y2_0 = 0; 
y3_0 = 0;

y1(1) = y1_0;
y2(1) = y2_0;
y3(1) = y3_0;

training_err_vec = zeros(n_t-1,1);

tStart = tic;

for i = 1:(n_t-1)
    
    c_i = (x(end) - x(1)) / (t_tot(i+1) - t_tot(i));  
    t = t_tot(i) + (1/c_i)*(x-x(1));

    xi_1_i = ones(L,1);
    xi_2_i = zeros(L,1);
    xi_3_i = zeros(L,1);

    xi_i = [xi_1_i;xi_2_i;xi_3_i];

        
    %% Build Constrained Expressions
    
    y1_i = (h-h0)*xi_1_i + y1_0;        y1_dot_i = c_i*hd*xi_1_i;      
    y2_i = (h-h0)*xi_2_i + y2_0;        y2_dot_i = c_i*hd*xi_2_i;   
    y3_i = (h-h0)*xi_3_i + y3_0;        y3_dot_i = c_i*hd*xi_3_i;    

    
    %% Build the Losses  
    
    L_1 = y1_dot_i + k1*y1_i - k3*y2_i.*y3_i;
    L_2 = y2_dot_i - k1*y1_i + k2*(y2_i.^2) + k3*y2_i.*y3_i;
    L_3 = y3_dot_i - k2*(y2_i.^2);
    
    Loss = [L_1 ; L_2 ; L_3 ];
    
    % X-TFC ILS loop
    l2 = [2 1];
    iter = 0;
    
    while abs(l2(2)) > IterTol &&  iter < IterMax && abs(l2(1) - l2(2)) > IterTol
        
        l2(1)= l2(2);
        
        % compute derivatives
        
        L_y1_xi_1 = c_i*hd + k1*(h-h0);
        L_y1_xi_2 = -k3*y3_i.*(h-h0);
        L_y1_xi_3 = -k3*y2_i.*(h-h0);
        
        L_y2_xi_1 = -k1*(h-h0);
        L_y2_xi_2 = c_i*hd + 2*k2*y2_i.*(h-h0) + k3*y3_i.*(h-h0);
        L_y2_xi_3 = k3*y2_i.*(h-h0);        
        
        L_y3_xi_2 = -2*k2*y2_i.*(h-h0); 
        L_y3_xi_3 = c_i*hd;        
             
        JJ = [ L_y1_xi_1 , L_y1_xi_2 , L_y1_xi_3 ; 
               L_y2_xi_1 , L_y2_xi_2 , L_y2_xi_3 ;
                   Z     , L_y3_xi_2 , L_y3_xi_3 ];  
     
        % xi variation
        dxi = lsqminnorm(JJ,Loss);
        
        % update xi
        xi_i = xi_i - dxi;
        
        xi_1_i = xi_i(1:L);
        xi_2_i = xi_i(L+1:2*L);
        xi_3_i = xi_i((2*L)+1:3*L);
        
        %% Re-Build Constrained Expressions
        
        y1_i = (h-h0)*xi_1_i + y1_0;        y1_dot_i = c_i*hd*xi_1_i;
        y2_i = (h-h0)*xi_2_i + y2_0;        y2_dot_i = c_i*hd*xi_2_i; 
        y3_i = (h-h0)*xi_3_i + y3_0;        y3_dot_i = c_i*hd*xi_3_i; 
           
        %% Re-Build the Losses
        
        L_1 = y1_dot_i + k1*y1_i - k3*y2_i.*y3_i;
        L_2 = y2_dot_i - k1*y1_i + k2*y2_i.^2 + k3*y2_i.*y3_i;
        L_3 = y3_dot_i - k2*y2_i.^2;
        
        Loss = [L_1 ; L_2 ; L_3]; 
            
        l2(2) = norm(Loss);
        
        iter = iter+1;
          
    end
    
    training_err = sqrt(mean(abs(L_1.^2))) + sqrt(mean(abs(L_2.^2))) +  sqrt(mean(abs(L_3.^2)))   ;
    
    % Update of constraints
    
    y1_0 = y1_i(end);
    y2_0 = y2_i(end);
    y3_0 = y3_i(end);
        
	y1(i+1) = y1_0;
    y2(i+1) = y2_0;
    y3(i+1) = y3_0;
    
    training_err_vec(i) = training_err;
              
end

xtfc_elapsedtime = toc(tStart) ;

fprintf('The elapsed time for x-tfc is: %g \n', xtfc_elapsedtime );

%% =======================================
% MATLAB ode15s solver

y0_ode15s = [1; 0; 0];
options = odeset('RelTol',IterTol);
% test the automatic detection of a DAE.

tStart = tic;
[t_15s,y_15s] = ode15s(@rober_ode15s_function, t_tot',y0_ode15s,options);
ode15s_elapsedtime = toc(tStart) ;

fprintf('The elapsed time for ode15s is: %g \n', ode15s_elapsedtime );

dy1dt = gradient(y_15s(:,1));
dy2dt = gradient(y_15s(:,2));
dy3dt = gradient(y_15s(:,3));

loss1 = dy1dt + 0.04*y_15s(:,1) - 10^4*y_15s(:,2).*y_15s(:,3);
loss2 = dy2dt - 0.04*y_15s(:,1) + 10^4*y_15s(:,2).*y_15s(:,3) + (3*10^7)*y_15s(:,2).^2;
loss3 = dy3dt - (3*10^7)*y_15s(:,2).^2;

training_err_ode15s = sqrt(mean(abs(loss1.^2))) + sqrt(mean(abs(loss2.^2))) +  sqrt(mean(abs(loss3.^2)))   ;

%%


%% =======================================
% MATLAB ode15i solver


y0_ode15i = [1; 0; 1e-3];
yp0 = [0; 0; 0];

tStart = tic;
[y0,yp0] = decic(@rober_ode15i_function,0,y0_ode15i,[1 1 0],yp0,[],options); 
[t_15i,y_15i] = ode15i(@rober_ode15i_function,t_tot',y0_ode15i,yp0,options);
ode15i_elapsedtime = toc(tStart) ;

fprintf('The elapsed time for ode15i is: %g \n', ode15i_elapsedtime );

dy1dt = gradient(y_15i(:,1));
dy2dt = gradient(y_15i(:,2));
dy3dt = gradient(y_15i(:,3));

loss1 = dy1dt + 0.04*y_15i(:,1) - 10^4*y_15i(:,2).*y_15i(:,3);
loss2 = dy2dt - 0.04*y_15i(:,1) + 10^4*y_15i(:,2).*y_15i(:,3) + (3*10^7)*y_15i(:,2).^2;
loss3 = dy3dt - (3*10^7)*y_15i(:,2).^2;

training_err_ode15i = sqrt(mean(abs(loss1.^2))) + sqrt(mean(abs(loss2.^2))) +  sqrt(mean(abs(loss3.^2)))   ;

%% errors


err_ode15s_1 = abs(y_15s(:,1) - y1) ;
err_ode15s_2 = abs(y_15s(:,2) - y2) ;
err_ode15s_3 = abs(y_15s(:,3) - y3) ;

err_ode15i_1 =  abs(y_15i(:,1) - y1)  ;
err_ode15i_2 =  abs(y_15i(:,2) - y2) ;
err_ode15i_3 =  abs(y_15i(:,3) - y3) ;

fprintf('\n')
fprintf('The average training error for X-TFC is: %g \n', mean(training_err_vec) )
fprintf('The average training error for ode15s is: %g \n', training_err_ode15s )
fprintf('The average training error for ode15s is: %g \n', training_err_ode15i )



%% PLOTS

subplot(3,2,1)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,y_15s(:,1), 'LineWidth',3, 'Color', [80 32.5 9]/100)
% plot(t_15i,y_15i(:,1),'g--','LineWidth',1.2)
plot(t_tot,y1,'--','LineWidth',3 , 'Color', [17 50 50]/100)
ylabel('y_1')
set(gca, 'XScale','log')
box on
title('concentration', 'FontWeight', 'Normal')


subplot(3,2,2)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,err_ode15s_1,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
% plot(t_15i,err_ode15i_1,'r*')
ylabel('abs(error)')
set(gca, 'XScale','log')
set(gca, 'YScale','log')
box on
title('abs(error)', 'FontWeight', 'Normal')







subplot(3,2,3)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,y_15s(:,2), 'LineWidth',3, 'Color', [80 32.5 9]/100)
%plot(t_15i,y_15i(:,2),'rx',  'MarkerSize',10,  'LineWidth',1.2)
plot(t_tot,y2,'--','LineWidth',3, 'Color', [17 50 50]/100)
set(gca, 'XScale','log')
ylabel('y_2')
box on


subplot(3,2,4)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,err_ode15s_2,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
% plot(t_15i,err_ode15i_2,'r*')
set(gca, 'XScale','log')
set(gca, 'YScale','log')
box on



subplot(3,2,5)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,y_15s(:,3), 'LineWidth',3, 'Color', [80 32.5 9]/100)
%plot(t_15i,y_15i(:,3),'rx',  'MarkerSize',10,  'LineWidth',1.2)
plot(t_tot,y3,'--','LineWidth',3, 'Color', [17 50 50]/100)
xlabel('time (s)')
legend('ode15s','X-TFC')
set(gca, 'XScale','log')
ylabel('y_3')
box on





subplot(3,2,6)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_15s,err_ode15s_3,'*','LineWidth',1.3, 'Color', [17 50 50]/100)
% plot(t_15i,err_ode15i_3,'r*','LineWidth',1)
xlabel('time (s)')
set(gca, 'XScale','log')
set(gca, 'YScale','log')
box on

figure(4)
set(gca,'Fontsize',12)
hold on
grid on
plot(t_tot,y1, 'LineWidth',1.5)
xlabel('time(s)')
set(gca, 'XScale','log')
xlim([10^(-4) t_tot(end)])
title('y1', 'FontWeight', 'Normal')

figure(5)
set(gca,'Fontsize',12)
hold on
grid on
plot(t_tot,y2, 'LineWidth',1.5)
xlabel('time(s)')
set(gca, 'XScale','log')
xlim([10^(-4) t_tot(end)])
title('y2', 'FontWeight', 'Normal')

figure(6)
set(gca,'Fontsize',12)
hold on
grid on
plot(t_tot,y3, 'LineWidth',1.5)
xlabel('time(s)')
set(gca, 'XScale','log')
xlim([10^(-4) t_tot(end)])
title('y3', 'FontWeight', 'Normal')




