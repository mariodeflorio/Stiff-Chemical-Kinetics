%%
clear; close all; clc;
format long
%--------------------------------------------------------------------------
%{ 
  Physics-Informed X-TFC applied to Stiff Chemical Kinetics
  Test Case 3 - Belousov-Zhabotinsky Reaction

  Authors:
  Mario De Florio, PhD
  Enrico Schiassi, PhD
%}
%%
%--------------------------------------------------------------------------
%% Input

rng('default') % set random seed

start = tic;

t_0 = 0; % initial time
t_f = 40; % final time

%t_query = [0.1,1,2,5,10,20,30,40]; 

n_x = 20;    % Discretization order for x (-1,1)
L = 20;    % number of neurons

x = linspace(0,1,n_x)';

t_step = 0.01;
t_step_ode15s = 0.0001;
t_step_ode15i = 0.0001;

t_tot = (t_0:t_step:t_f)';
n_t = length(t_tot);

type_basis = 2; % 1 = Orthogonal polynomials ; 2 = activation functions 

type_act = 2; % activation functions

LB = -1; % Lower boundary for weight and bias samplings
UB = 1; % Upper boundary for weight and bias samplings

% iterative least-square parameters

IterMax = 100;
IterTol = 1e-9;
IterTol_ode15s = 1e-9;
IterTol_ode15i = 1e-9;


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
k1 = 4.72;
k2 = 3*10^9;
k3 = 1.5*10^4;
k4 = 4*10^7;
k5 = 1;

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
y7 = zeros(n_t,1);

% Initial Values

y1_0 = 0.066; 
y2_0 = 0; 
y3_0 = 0;
y4_0 = 0;
y5_0 = 0.066;
y6_0 = 0.002;
y7_0 = 0;


y1(1) = y1_0;
y2(1) = y2_0;
y3(1) = y3_0;
y4(1) = y4_0;
y5(1) = y5_0;
y6(1) = y6_0;
y7(1) = y7_0;


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
    xi_7_i = zeros(L,1);
    

    xi_i = [xi_1_i;xi_2_i;xi_3_i;xi_4_i;xi_5_i;xi_6_i;xi_7_i];

        
    %% Build Constrained Expressions
    
    y1_i = (h-h0)*xi_1_i + y1_0;        y1_dot_i = c_i*hd*xi_1_i;     
    y2_i = (h-h0)*xi_2_i + y2_0;        y2_dot_i = c_i*hd*xi_2_i;   
    y3_i = (h-h0)*xi_3_i + y3_0;        y3_dot_i = c_i*hd*xi_3_i;
    y4_i = (h-h0)*xi_4_i + y4_0;        y4_dot_i = c_i*hd*xi_4_i;
    y5_i = (h-h0)*xi_5_i + y5_0;        y5_dot_i = c_i*hd*xi_5_i;
    y6_i = (h-h0)*xi_6_i + y6_0;        y6_dot_i = c_i*hd*xi_6_i;
    y7_i = (h-h0)*xi_7_i + y7_0;        y7_dot_i = c_i*hd*xi_7_i;
    

    
    %% Build the Losses  
    
    L_1 = y1_dot_i + k1*y1_i.*y2_i ;
    L_2 = y2_dot_i + k1*y1_i.*y2_i + k2*y3_i.*y2_i - k5*y6_i;
    L_3 = y3_dot_i + k2*y3_i.*y2_i - k3*y3_i.*y5_i + 2*k4*y3_i.^2 - k1*y1_i.*y2_i;
    L_4 = y4_dot_i - k2*y3_i.*y2_i ;
    L_5 = y5_dot_i + k3*y5_i.*y3_i ;
    L_6 = y6_dot_i - k3*y5_i.*y3_i + k5*y6_i ;
    L_7 = y7_dot_i - k4*y3_i.^2 ;
    
    Loss = [L_1 ; L_2 ; L_3 ; L_4 ; L_5 ; L_6 ; L_7];
    
    % X-TFC ILS loop
    l2 = [2 1];
    iter = 0;
    
    while abs(l2(2)) > IterTol &&  iter < IterMax && abs(l2(1) - l2(2)) > IterTol
        
        l2(1)= l2(2);
        
        % compute derivatives
        
        % L1
        L_y1_xi_1 = c_i*hd + k1*y2_i.*(h-h0) ;
        L_y1_xi_2 = k1*y1_i.*(h-h0) ;
        
        %L2
        L_y2_xi_1 = k1*y2_i.*(h-h0)  ;
        L_y2_xi_2 = c_i*hd + k1*y1_i.*(h-h0) + k2*y3_i.*(h-h0)  ;
        L_y2_xi_3 = k2*y2_i.*(h-h0)  ;       
        L_y2_xi_6 = - k5*(h-h0) ;
        
        %L3
        L_y3_xi_1 = - k1*y2_i.*(h-h0) ;
        L_y3_xi_2 = k2*y3_i.*(h-h0) - k1*y1_i.*(h-h0) ; 
        L_y3_xi_3 = c_i*hd + k2*y2_i.*(h-h0) - k3*y5_i.*(h-h0) + 4*k4*y3_i.*(h-h0) ;        
        L_y3_xi_5 =  -k3*y3_i.*(h-h0) ;
        
        %L4
        L_y4_xi_2 = - k2*y3_i.*(h-h0) ;
        L_y4_xi_3 = - k2*y2_i.*(h-h0) ;
        L_y4_xi_4 = c_i*hd ;
        
        %L5
        L_y5_xi_3 = k3*y5_i.*(h-h0)  ;
        L_y5_xi_5 = c_i*hd + k3*y3_i.*(h-h0) ;
        
        %L6
        L_y6_xi_3 = - k3*y5_i.*(h-h0) ;
        L_y6_xi_5 = - k3*y3_i.*(h-h0) ;
        L_y6_xi_6 = c_i*hd + k5*(h-h0)  ;

        %L7
        L_y7_xi_3 = - 2*k4*y3_i.*(h-h0) ;
        L_y7_xi_7 = c_i*hd ;
        
        % Jacobian matrix     
        JJ = [ L_y1_xi_1 , L_y1_xi_2 ,     Z     ,     Z     ,     Z     ,     Z     ,     Z     ; 
               L_y2_xi_1 , L_y2_xi_2 , L_y2_xi_3 ,     Z     ,     Z     , L_y2_xi_6 ,     Z     ;
               L_y3_xi_1 , L_y3_xi_2 , L_y3_xi_3 ,     Z     , L_y3_xi_5 ,     Z     ,     Z     ;
                   Z     , L_y4_xi_2 , L_y4_xi_3 , L_y4_xi_4 ,     Z     ,     Z     ,     Z     ;
                   Z     ,     Z     , L_y5_xi_3 ,     Z     , L_y5_xi_5 ,     Z     ,     Z     ;
                   Z     ,     Z     , L_y6_xi_3 ,     Z     , L_y6_xi_5 , L_y6_xi_6 ,     Z     ;
                   Z     ,     Z     , L_y7_xi_3 ,     Z     ,     Z     ,     Z     , L_y7_xi_7 ];
            
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
        xi_7_i = xi_i((6*L)+1:7*L);
        
        %% Re-Build Constrained Expressions
        
        y1_i = (h-h0)*xi_1_i + y1_0;        y1_dot_i = c_i*hd*xi_1_i;
        y2_i = (h-h0)*xi_2_i + y2_0;        y2_dot_i = c_i*hd*xi_2_i;
        y3_i = (h-h0)*xi_3_i + y3_0;        y3_dot_i = c_i*hd*xi_3_i;
        y4_i = (h-h0)*xi_4_i + y4_0;        y4_dot_i = c_i*hd*xi_4_i;
        y5_i = (h-h0)*xi_5_i + y5_0;        y5_dot_i = c_i*hd*xi_5_i;
        y6_i = (h-h0)*xi_6_i + y6_0;        y6_dot_i = c_i*hd*xi_6_i;
        y7_i = (h-h0)*xi_7_i + y7_0;        y7_dot_i = c_i*hd*xi_7_i;
        
        %% Re-Build the Losses
        
        L_1 = y1_dot_i + k1*y1_i.*y2_i ;
        L_2 = y2_dot_i + k1*y1_i.*y2_i + k2*y3_i.*y2_i - k5*y6_i;
        L_3 = y3_dot_i + k2*y3_i.*y2_i - k3*y3_i.*y5_i + 2*k4*y3_i.^2 - k1*y1_i.*y2_i;
        L_4 = y4_dot_i - k2*y3_i.*y2_i ;
        L_5 = y5_dot_i + k3*y5_i.*y3_i ;
        L_6 = y6_dot_i - k3*y5_i.*y3_i + k5*y6_i ;
        L_7 = y7_dot_i - k4*y3_i.^2 ;
        
        Loss = [L_1 ; L_2 ; L_3 ; L_4 ; L_5 ; L_6 ; L_7];
        
        l2(2) = norm(Loss);
        
        iter = iter+1;
        
    end
    
    training_err = sqrt(mean(abs(L_1.^2))) + sqrt(mean(abs(L_2.^2))) +  sqrt(mean(abs(L_3.^2))) ...
        +  sqrt(mean(abs(L_4.^2)))  +  sqrt(mean(abs(L_5.^2)))  +  sqrt(mean(abs(L_6.^2)))   +  sqrt(mean(abs(L_7.^2)))  ;    
    % Update of constraints
    
    y1_0 = y1_i(end);
    y2_0 = y2_i(end);
    y3_0 = y3_i(end);
    y4_0 = y4_i(end);
    y5_0 = y5_i(end);
    y6_0 = y6_i(end);
    y7_0 = y7_i(end);
        
	y1(i+1) = y1_0;
    y2(i+1) = y2_0;
    y3(i+1) = y3_0;
    y4(i+1) = y4_0;
    y5(i+1) = y5_0;
    y6(i+1) = y6_0;
    y7(i+1) = y7_0;
    
    training_err_vec(i) = training_err;
              
end

xtfc_elapsedtime = toc(tStart) ;

fprintf('The elapsed time for x-tfc is: %g \n', xtfc_elapsedtime );

%% =======================================
% MATLAB ode15s solver

y0_ode15s = [0.066; 0; 0; 0; 0.066; 0.002; 0];
options_ode15s = odeset('RelTol',IterTol_ode15s);


tStart = tic;
[t_15s,y_15s] = ode15s(@belousov_zhabotinsky_ode15s_function, (t_0:t_step_ode15s:t_f)',y0_ode15s,options_ode15s);
ode15s_elapsedtime = toc(tStart) ;

fprintf('The elapsed time for ode15s is: %g \n', ode15s_elapsedtime );

%% =======================================
% MATLAB ode15i solver

y0_ode15i = [0.066; 0; 0; 0; 0.066; 0.002; 0];
yp0 = [0; 0; 0; 0; 0; 0; 0];
options_ode15i = odeset('RelTol',IterTol_ode15i);

tStart = tic;
[y0,yp0] = decic(@belousov_zhabotinsky_ode15i_function,0,y0_ode15i,[0 1 0 0 0 0 0],yp0,[],options_ode15i); 
[t_15i,y_15i] = ode15i(@belousov_zhabotinsky_ode15i_function,(t_0:t_step_ode15i:t_f)',y0_ode15i,yp0,options_ode15i);
ode15i_elapsedtime = toc(tStart) ;

fprintf('The elapsed time for ode15i is: %g \n', ode15i_elapsedtime );

%% plots

fprintf('\n')
fprintf('The average training error for X-TFC is: %g \n', mean(training_err_vec) )


subplot(3,3,1)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_tot,y1,'LineWidth',3 , 'Color', [17 50 50]/100)
plot(t_15i,y_15i(:,1),'--','LineWidth',3, 'Color', [90 61.5 25]/100)
plot(t_15s,y_15s(:,1),':','LineWidth',3, 'Color', [80 32.5 9]/100)
xlim([t_0 t_f])
ylabel('concentration')
box on
title('y1', 'FontWeight', 'Normal')

subplot(3,3,2)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_tot,y2,'LineWidth',3, 'Color', [17 50 50]/100)
plot(t_15i,y_15i(:,2),'--','LineWidth',3, 'Color', [90 61.5 25]/100)
plot(t_15s,y_15s(:,2),':','LineWidth',3, 'Color', [80 32.5 9]/100)
xlim([t_0 t_f])
box on
title('y2', 'FontWeight', 'Normal')

subplot(3,3,3)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_tot,y3,'LineWidth',3, 'Color', [17 50 50]/100)
plot(t_15i,y_15i(:,3),'--','LineWidth',3, 'Color', [90 61.5 25]/100)
plot(t_15s,y_15s(:,3),':','LineWidth',3, 'Color', [80 32.5 9]/100)
xlim([t_0 t_f])
legend('X-TFC','ode15i','ode15s')
box on
title('y3', 'FontWeight', 'Normal')




subplot(3,3,7)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_tot,y7,'LineWidth',3 , 'Color', [17 50 50]/100)
plot(t_15i,y_15i(:,7),'--','LineWidth',3, 'Color', [90 61.5 25]/100)
plot(t_15s,y_15s(:,7),':','LineWidth',3, 'Color', [80 32.5 9]/100)
xlim([t_0 t_f])
xlabel('time (s)')
ylabel('concentration')
box on
title('y7', 'FontWeight', 'Normal')







subplot(3,3,4)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_tot,y4,'LineWidth',3 , 'Color', [17 50 50]/100)
plot(t_15i,y_15i(:,4),'--','LineWidth',3, 'Color', [90 61.5 25]/100)
plot(t_15s,y_15s(:,4),':','LineWidth',3, 'Color', [80 32.5 9]/100)
xlim([t_0 t_f])
ylabel('concentration')
box on
title('y4', 'FontWeight', 'Normal')

subplot(3,3,5)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_tot,y5,'LineWidth',3, 'Color', [17 50 50]/100)
plot(t_15i,y_15i(:,5),'--','LineWidth',3, 'Color', [90 61.5 25]/100)
plot(t_15s,y_15s(:,5),':','LineWidth',3, 'Color', [80 32.5 9]/100)
xlim([t_0 t_f])
xlabel('time (s)')
box on
title('y5', 'FontWeight', 'Normal')

subplot(3,3,6)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t_tot,y6,'LineWidth',3, 'Color', [17 50 50]/100)
plot(t_15i,y_15i(:,6),'--','LineWidth',3, 'Color', [90 61.5 25]/100)
plot(t_15s,y_15s(:,6),':','LineWidth',3, 'Color', [80 32.5 9]/100)
xlim([t_0 t_f])
xlabel('time (s)')
box on
title('y6', 'FontWeight', 'Normal')



% subplot(4,4,12)
% set(gca,'Fontsize',12)
% hold on
% grid on 
% plot(t_15s,err_ode15s_7,'*','LineWidth',1.3, 'Color', [80 32.5 9]/100)
% plot(t_15i,err_ode15i_7,'*','LineWidth',1.3, 'Color', [90 61.5 25]/100)
% xlim([t_0 t_f])
% xlabel('time (s)')
% ylabel('abs(error)')
% set(gca, 'YScale','log')
% box on
% 
% 
% 
% 
% 
% subplot(4,4,13)
% set(gca,'Fontsize',12)
% hold on
% grid on 
% plot(t_15s,err_ode15s_1,'*','LineWidth',1.3, 'Color', [80 32.5 9]/100)
% plot(t_15i,err_ode15i_1,'*','LineWidth',1.3, 'Color', [90 61.5 25]/100)
% xlim([t_0 t_f])
% xlabel('time (s)')
% ylabel('abs(error)')
% set(gca, 'YScale','log')
% box on
% 
% subplot(4,4,14)
% set(gca,'Fontsize',12)
% hold on
% grid on 
% plot(t_15s,err_ode15s_5,'*','LineWidth',1.3, 'Color', [80 32.5 9]/100)
% plot(t_15i,err_ode15i_5,'*','LineWidth',1.3, 'Color', [90 61.5 25]/100)
% xlim([t_0 t_f])
% xlabel('time (s)')
% set(gca, 'YScale','log')
% box on
% 
% subplot(4,4,15)
% set(gca,'Fontsize',12)
% hold on
% grid on 
% plot(t_15s,err_ode15s_6,'*','LineWidth',1.3, 'Color', [80 32.5 9]/100)
% plot(t_15i,err_ode15i_6,'*','LineWidth',1.3, 'Color', [90 61.5 25]/100)
% xlim([t_0 t_f])
% xlabel('time (s)')
% set(gca, 'YScale','log')
% box on



figure(2)
set(gca,'Fontsize',12)
hold on
grid on 
title('X-TFC solutions')
plot(t_tot,y2, 'LineWidth', 2, 'Color', [17 50 50]/100)
plot(t_tot,y4, 'LineWidth', 2, 'Color', [80 32.5 9]/100)
plot(t_tot,y6, 'LineWidth', 2, 'Color', [90 61.5 25]/100)
plot(t_tot,y7, 'LineWidth', 2)
xlabel('time(s)')
legend('Y','P','Z','Q')
% set(gca, 'XScale','log')
%xlim([10^(-4) t_tot(end)])
% 
% 
% figure(3)
% set(gca,'Fontsize',12)
% hold on
% grid on 
% title('ode15s solutions')
% plot(t_15s,y_15s(:,2),t_15s,y_15s(:,4),t_15s,y_15s(:,6),t_15s,y_15s(:,7), 'LineWidth',1.5)
% xlabel('time(s)')
% legend('Y','P','Z','Q')
% % set(gca, 'XScale','log')
% %xlim([10^(-4) t_tot(end)])
% 
% 
% figure(4)
% set(gca,'Fontsize',12)
% hold on
% grid on 
% title('ode15i solutions')
% plot(t_15s,y_15i(:,2),'LineWidth',3 , 'Color', [17 50 50]/100)
% plot(t_15i,y_15i(:,2),t_15s,y_15i(:,4),t_15s,y_15i(:,6),t_15s,y_15i(:,7), 'LineWidth',1.5)
% xlabel('time(s)')
% legend('Y','P','Z','Q')
% set(gca, 'XScale','log')
%xlim([10^(-4) t_tot(end)])
