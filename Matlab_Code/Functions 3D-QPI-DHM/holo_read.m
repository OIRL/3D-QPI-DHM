%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: holo_read                                                             %
%                                                                              %
% The function is implemented to read a hologram, the hologram is cut to get   %
% square dimension, the square has the dimension of the higher side of the     %
% digital camera used to record the hologram, and the cut is making from the   %
% center of the hologram.                                                      %
%                                                                              %                                                                        
% Authors: Raul Castaneda and Ana Doblas                                       %
% Department of Electrical and Computer Engineering, The University of Memphis,% 
% Memphis, TN 38152, USA.                                                      %   
%                                                                              %
% Email: adoblas@memphis.edu                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [holo,M,N,m,n] = holoRead(filename)
    holo = double(imread(filename));
    holo = holo(:,:,1);
    [M,N] = size(holo);
    [m,n] = meshgrid(-N/2:N/2-1,-M/2:M/2-1);
end