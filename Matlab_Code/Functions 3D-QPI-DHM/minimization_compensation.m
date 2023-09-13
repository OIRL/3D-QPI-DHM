%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: cost_function                                                         %
%                                                                              %
% The function computing the cost function estimated by Eq. (8)                %  
%                                                                              %                                                                             
% Authors: Raul Castaneda and Ana Doblas                                       %
% Department of Electrical and Computer Engineering, The University of Memphis,% 
% Memphis, TN 38152, USA.                                                      %   
%                                                                              %
% Email: rcstdqnt@memphis.edu and adoblas@memphis.                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [J] = minimization_compensation(seed_maxPeak,lambda,dxy,M,N,m,n,holo_filt)
    J = 0;
    fx_max = seed_maxPeak(1,1);
    fy_max = seed_maxPeak(1,2);
    [ref_wave] = reference_wave(M,N,m,n,lambda,dxy,fx_max,fy_max);
    holo_rec = holo_filt .* ref_wave;%Eq. (4)
    phase = angle(holo_rec);
    phase = phase + pi; 
    ib = imbinarize(phase, 0.5);
    J = M*N - sum(ib(:));
end