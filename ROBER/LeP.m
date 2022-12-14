function [F, FD, FDD, FDDD] = LeP(x, deg)

%% Computes the first deg Legendre polynomials (F), with 1-st (FD),
%%     2-nd (FDD), and 3-rd (FDDD) derivatives
%% Daniele Mortari  -  Texas A&M University  -  April 23, 2017

x = x(:);
N = length(x);
Zero = zeros(N, 1);
One = ones(N, 1);

% Initialization
if deg == 0
    F = One;
    if nargout > 1, FD   = Zero; end
    if nargout > 2, FDD  = Zero; end
    if nargout > 3, FDDD = Zero; end
    return
elseif deg == 1
    F = [One, x];
    if nargout > 1, FD   = [Zero, One]; end
    if nargout > 2, FDD  = zeros(N, 2); end
    if nargout > 3, FDDD = zeros(N, 2); end
else
    F = [One, x, zeros(N, deg-1)];
    if nargout > 1, FD   = [Zero, One, zeros(N, deg-1)]; end
    if nargout > 2, FDD  = zeros(N, deg+1); end
    if nargout > 3, FDDD = zeros(N, deg+1); end
    
    %% Legendre Polynomials and 1st, 2nd, and 3rd derivatives
    for k = 2:deg
        F(:, k+1) = ((2*k + 1)*x.*F(:, k) - k*F(:, k-1))/(k + 1);
        if nargout > 1
            FD(:, k+1) = ((2*k + 1)*(F(:,k) + x.*FD(:, k)) - k*FD(:, k-1))/(k + 1);
        end
        if nargout > 2
            FDD(:, k+1) = ((2*k + 1)*(2*FD(:,k) + x.*FDD(:, k)) - k*FDD(:, k-1))/(k + 1);
        end
        if nargout > 3
            FDDD(:, k+1) = ((2*k + 1)*(3*FDD(:,k) + x.*FDDD(:, k)) - k*FDDD(:, k-1))/(k + 1);
        end
    end
    
end

end