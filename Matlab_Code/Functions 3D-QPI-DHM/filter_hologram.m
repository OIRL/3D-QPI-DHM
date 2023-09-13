%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: filter_holgram                                                        %
%                                                                              %
% The function is used to filter the +1 order of the hologram                  %
%                                                                              %                                                                           
% Authors: Raul Castaneda,    Ana Doblas                                       %
% Department of Electrical and Computer Engineering, The University of Memphis,% 
% Memphis, TN 38152, USA.                                                      %   
%                                                                              %
% Email: adoblas@memphis.edu                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [holo_filter,fx_max,fy_max] = filter_hologram(holo,M,N,visual,factor)
    fft_holo = fftshift(fft2(fftshift(holo))); 
    %figure,imagesc(log(abs(fft_holo).^2)),colormap(gray),title('FT Hologram'),daspect([1 1 1]) 

    % select order of diffraction for find the best compensation
    % max value first cuadrant
    mask = ones(M,N);
    mask(M/2-100:M/2+100,N/2-100:N/2+100)=0;
    fft_holo_I = fft_holo .* mask;
    mask = ones(M,N);
    mask(1:10,1:10)=0;
    fft_holo_I = fft_holo_I .* mask;
    %figure,imagesc(log(abs( fft_holo_I).^2)),colormap(gray),title('FT Hologram'),daspect([1 1 1]) 

    maxValue_1 = max(max(abs(fft_holo_I)));
    [fy_max_1 fx_max_1] = find(abs(fft_holo_I) == maxValue_1);

    mask(fy_max_1 - 50:fy_max_1 + 50,fx_max_1 - 50:fx_max_1 + 50)=0;
    fft_holo_I = fft_holo_I .* mask;
    %figure,imagesc(log(abs(fft_holo_I).^2)),colormap(gray),title('FT Hologram'),daspect([1 1 1])
    
     maxValue_1 = max(max(abs(fft_holo_I)));
    [fy_max_1 fx_max_1] = find(abs(fft_holo_I) == maxValue_1);
    fx_max = fx_max_1(1); fy_max = fy_max_1(1);
    % mask to filter and filter
    distance = sqrt((fx_max - M/2)^2+(fy_max - N/2)^2);
    resc = M/factor;
    filter = ones(M,N);
    for r=1:M
        for p=1:N
            if sqrt((r-fy_max)^2+(p-fx_max)^2)>resc
                filter(r,p)=0;
            end
        end
    end
    % figure,imagesc(filter),colormap(gray),title('Filter')

    fft_filter_holo = fft_holo .* filter;
    %figure,imagesc(log(abs(fft_filter_holo).^2)),colormap(gray),title('FT Filtered Hologram'),daspect([1 1 1]) 

    %filtered_spect = log(abs(fft_filtered_holo).^2);
    [num,idx] = max(fft_filter_holo(:));
    %New hologram filtered
    holo_filter = fftshift(ifft2(fftshift(fft_filter_holo)));
    if visual == 'Yes'
         figure,imagesc(log(abs(fft_filter_holo).^2)),colormap(gray),title('FT Filter Hologram'),daspect([1 1 1]) 
         figure,imagesc((abs(holo_filter).^2)),colormap(gray),title('Filter Hologram'),daspect([1 1 1]) 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mask for DC termn
%     ftHolo = fftshift(fft2(fftshift(holo)));
%     figure,imagesc(log(abs(ftHolo).^2)),colormap(gray),title('FT Hologram'),daspect([1 1 1])
%     mask = zeros(M,N);
%     mask(1:M/2-50,1:N)=1;
%     fft_holo_I = ftHolo .* mask;
%     [num,idx] = max(abs(fft_holo_I(:)));
%     [fy_max,fx_max] = ind2sub([M,N],idx);
%     data = log(abs(fft_holo_I).^2);
%     %figure,imagesc(mask),colormap(gray),title('mask I'),daspect([1 1 1]) 
%     figure,imagesc(data),colormap(gray),title('FT Hologram'),daspect([1 1 1])
% 
%     % find the best diameter filter for the spatial filter
%     levels = max(data(:));
%     BW = imbinarize(data,levels- (levels*(0.45)));
%     %figure,imagesc(BW),colormap(gray),title('FFT Hologram binarized'),daspect([1 1 1])
% 
%     stats = regionprops('table',BW,'Centroid',...
%         'MajorAxisLength','MinorAxisLength');
% 
%     centers = stats.Centroid;
%     diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
%     radii = diameters/2;
% 
%     hold on
%     viscircles(centers,radii);
%     hold off
% 
%     [num,idx] = max(stats.MajorAxisLength);
%     [num2,idx] = max(stats.MinorAxisLength);
%     radiip = (num + num2)/2;
%     radiii = radiip/4;%   star target radiii = radiii/3; USAF target radiii = radiii/5;
%     filter = ones(M,N);
%     for r=1:N
%         for p=1:M
%             if sqrt((r-fy_max)^2+(p-fx_max)^2)>radiii*2
%                 filter(r,p)=0;
%             end
%         end
%     end
%     %figure,imagesc(filter),colormap(gray),title('Best mask'),daspect([1 1 1]) 
%     fft_filtered_holo = ftHolo .* filter;
%     % New hologram filtered
%     holo_filter = fftshift(ifft2(fftshift(fft_filtered_holo)));
% 
%     
%     if visual == 'yes'
%         figure,imagesc(log(abs(fft_filtered_holo).^2)),colormap(gray),title('FT Filter Hologram'),daspect([1 1 1]) 
%     end