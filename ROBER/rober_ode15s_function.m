function dydt = rober_ode15s_function(t,y)
%rober_ode15s_function  Evaluate the ROBER problem
%
% Authors:
% Mario De Florio
% Enrico Schiassi

dydt = [-0.04*y(1) + 10^4*y(2)*y(3); 
         0.04*y(1) - 10^4*y(2)*y(3) - (3*10^7)*y(2)^2 ; 
        (3*10^7)*y(2)^2 ];
