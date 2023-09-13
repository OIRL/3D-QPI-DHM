%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: reference_wave                                                        %
%                                                                              %
% The function computes the digital reference wave [Eq. (5)]                   %
%                                                                              %                                                                             
% Authors: Raul Castaneda and Ana Doblas                                       %
% Department of Electrical and Computer Engineering, The University of Memphis,% 
% Memphis, TN 38152, USA.                                                      %   
%                                                                              %
% Email: rcstdqnt@memphis.edu and adoblas@memphis                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ref_wave] = reference_wave(M,N,m,n,lambda,dxy,fx_max,fy_max)
    fx_0 = N/2;
    fy_0 = M/2;
    k = 2 * pi / lambda;
    theta_x = asin((fx_0 - fx_max) * lambda / (N * dxy));%Eq. 6
    theta_y = asin((fy_0 - fy_max) * lambda / (M * dxy));%Eq. 7
    ref_wave = exp(1i * k * (sin(theta_x) * m * dxy + sin(theta_y) * n * dxy));%digital reference wave
end