function res = belousov_zhabotinsky_ode15i_function(t,y,yp)
%rober_ode15i_function  Evaluate the ROBER problem
%
% Authors:
% Mario De Florio
% Enrico Schiassi

% rate constants
k1 = 4.72;
k2 = 3*10^9;
k3 = 1.5*10^4;
k4 = 4*10^7;
k5 = 1;

res = [ yp(1) + k1*y(1)*y(2) ;
        yp(2) + k1*y(1)*y(2) + k2*y(3)*y(2) - k5*y(6) ;
        yp(3) + k2*y(3)*y(2) - k3*y(3)*y(5) + 2*k4*y(3)^2 - k1*y(1)*y(2) ;
        yp(4) - k2*y(3)*y(2) ;
        yp(5) + k3*y(3)*y(5) ;
        yp(6) - k3*y(3)*y(5) + k5*y(6) ;
        yp(7) - k4*y(3)^2
        ];
