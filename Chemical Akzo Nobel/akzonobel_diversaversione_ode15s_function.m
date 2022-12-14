function dydt = akzonobel_diversaversione_ode15s_function(t,y)
%akzonobel_ode15s_function  Evaluate the Chemical Akzo Nobel problem
%
% Authors:
% Mario De Florio
% Enrico Schiassi

k1 = 18.7;
k2 = 0.58;
k3 = 0.09;
k4 = 0.42;
K = 34.4;
klA = 3.3;
pCO2 = 0.9;
H = 737;

% reaction velocities
r1 = k1*(y(1)^4)*y(2)^(0.5) ;
r2 = k2*y(3)*y(4) ;
r3 = (k2/K)*y(1)*y(5) ;
r4 = k3*y(1)*y(4)^2 ;
r5 = k4*(y(6)^2)*y(2)^(0.5) ;

%  inflow of oxygen per volume unit
F_in = klA*((pCO2/H) - y(2));

dydt = [ -2*r1 + r2 - r3 - r4 ;
         -0.5*r1 - r4 - 0.5*r5 + F_in ;
         r1 - r2 + r3 ;
         - r2 + r3 - 2*r4 ;
         r2 - r3 + r5 ;
         - r5
        ];
