function res = rober_ode15i_function(t,y,yp)
%rober_ode15i_function  Evaluate the ROBER problem
%
% Authors:
% Mario De Florio
% Enrico Schiassi

res = [yp(1) + 0.04*y(1) - 1e4*y(2)*y(3);
   yp(2) - 0.04*y(1) + 1e4*y(2)*y(3) + 3e7*y(2)^2;
   yp(3)  - 3*10^7*y(2)^2 ];

% y(1) + y(2) + y(3) - 1