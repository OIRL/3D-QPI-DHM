%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: 3D DHM-QPI                                                            %
%                                                                              %
% Code implementation of the synergetic phase compensation and autofocus       %
% procedure in CPU using a heuristics approach.                                % 
%                                                                              %
% Authors: Raul Castaneda*, Carlos Trujilo* and Ana Doblas**                   %
%                                                                              % 
%  *School of Applied Sciences and Engineering                                 %
%   EAFIT University                                                           %
%   Medellín, Colombia                                                         %
%                                                                              %
% **Department of Electrical and Computer Engineering                          %
%   UMass Dartmouth University                                                 %
%   Boston, USA                                                                %
%                                                                              %
% Email: rcstdqnt@memphis.edu // catrujilla@eafit.educo // adoblas@memphis.    %
% version 1.0 (2023)                                                           %
%                                                                              %
% ------------------------------Specifications---------------------------------% 
% Input:                                                                       %
%     holo = Recorded Off-axis Hologram operating in telecentric regimen       %
%     lambda = wavelength                                                      % 
%     pixel size = dxy                                                         %
%                                                                              %
% Output: Reconstructed in-focus phase and amplituded images                   %
%                                                                              %
% Functions: The algortihm implements the following functions                  %
%   - holo_read                                                                %
%   - filter_hologram                                                          %
%   - reference_wave                                                           % 
%   - cost_function                                                            %
%   - phase_reconstruction                                                     %
%   - CF_integratedII                                                          %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear memory
clc              % clear command window
close all        % close all windows
clear all        % clear of memory all variable

%% Lines to add folders for reading images and/or functions
% Please make sure to add the folder where you saved the functions necessary to run the code.

% my computer
addpath('C:\Users\racastaneq\Documents\MEGA\MEGAsync\RACQ\Universities\05 EAFIT\Research projects\2023\Autofocus\Algorithms\Final version\GitHub versions\Matlab_codes\Functions\')


% Lines to load the hologram and parameters for the reconstruction
name = 'holo-RBC-20p22-3-2.png';

[holo,M,N,m,n] = holo_read(name);
figure,imagesc(holo),colormap(gray),title('Hologram'),daspect([1 1 1])

lambda = 0.000532;
dxy = 0.0024;


% spatial frequency filter
%Choose 'Yes' to visualized the spatial filter in the Fourier Domain
[holo_filter,fx_max,fy_max] = filter_hologram(holo,M,N,'Yes',12);

% minimization --- to the best compensation
seed_maxPeak = [fx_max - 1,fy_max - 1];
J = minimization_compensation(seed_maxPeak,lambda,dxy,M,N,m,n,holo_filter);
options = optimset('Display','iter', 'MaxIter', 100,'TolX',1e-3);
[MaxPeaks, J] = ...
	fminunc(@(t)(minimization_compensation(t,lambda,dxy,M,N,m,n,holo_filter)), seed_maxPeak, options);

fx_max_temp = MaxPeaks(1,1);
fy_max_temp = MaxPeaks(1,2);

fprintf('Cost at peaks found by fminunc: %f\n', J);
fprintf('fx: %f\n', fx_max_temp);
fprintf('fy: %f\n', fy_max_temp);

[ref_wave] = reference_wave(M,N,m,n,lambda,dxy,fx_max_temp,fy_max_temp);
field_compensate = ref_wave.*holo_filter;
phase_1 = angle(field_compensate);
figure,imagesc(phase_1),colormap(gray),title('Image plane phase reconstruction'),daspect([1 1 1])

%% Run this block if you are working with holograms with defocus samples at the same distances.

% select a ROI to center the algoritm in a specific region
[pointsX,pointsY] = ginput(2);
    p = round(pointsY(1));
    q = round(pointsY(2));
    r = round(pointsX(1));
    s = round(pointsX(2));

mask = zeros(M,N);
mask(p:q,r:s) = 1;
%figure,imagesc(mask),colormap(gray),title('mask'),daspect([1 1 1])

distance = 0;

% minimization --- synergetic
seed = [fx_max_temp, fy_max_temp, distance];
CFI = CF_integratedII(seed,field_compensate,lambda,dxy,mask,p,q,r,s);
%options = optimset('Display','iter', 'MaxIter', 300,'TolX',1e-1);
options = optimset('MaxIter', 300,'TolX',1e-1);
[Val, CFI] = fminunc(@(t)(CF_integratedII(t,field_compensate,lambda,dxy,mask,p,q,r,s)), seed, options);
 
fx_max_best = Val(1,1);
fy_max_best = Val(1,2);
distance_best = Val(1,3);
fprintf('Cost at peaks found by fminunc: %f\n', CFI);
fprintf('fx: %f\n', fx_max_best);
fprintf('fy: %f\n', fy_max_best);
fprintf('z: %f\n', distance_best);

% best - reconstruction
[ref_wave] = reference_wave(M,N,m,n,lambda,dxy,fx_max_best,fy_max_best);
field_compensate = ref_wave.*holo_filter;
outputII = ang_spectrum(field_compensate,distance_best,lambda,dxy,dxy);
phase = angle(outputII);
amplitude = abs(outputII);

figure,imagesc(amplitude),colormap(gray),colorbar, title(['Amplitude reconstruction ' ,num2str(distance_best)]),daspect([1 1 1])
figure,imagesc(phase),colormap(gray),colorbar, title(['Phase reconstruction ' ,num2str(distance_best)]),daspect([1 1 1])

phase_Save = uint8(255 * mat2gray(phase));
amplitude_Save = uint8(255 * mat2gray(amplitude));

% Uncomment if you desire to save the outcomes. 
% imwrite(phase_Save,['phase_RBC_SCF','.png'])
%imwrite(amplitude_Save,['amplitude_RBC_SCF','.png'])

%% Run this block if you are working with a hologram that contains samples at different propagation distances.


% segmentation -- free feel to implement your segmentation algorithm 
phase = phase_1 *-1;
phase_bin = imbinarize(phase, 0.3);
figure,imagesc(phase_bin),colormap(gray),title('segmentation'),daspect([1 1 1])

% Found centroids and AxisLength
regions = regionprops('table',phase_bin,'Centroid','MajorAxisLength','MinorAxisLength')
centers = regions.Centroid;
diameters = mean([regions.MajorAxisLength regions.MinorAxisLength],2);
% figure,imagesc(phase_bin),colormap(gray),title('phase bin '),daspect([1 1 1])
% hold on
% viscircles(centers, diameters/2);
% hold off
ind = find(diameters < 20);
diameters(ind) = [];
centers(ind,:) = [];
figure,imagesc(phase_bin),colormap(gray),title('phase bin '),daspect([1 1 1])
hold on
viscircles(centers, diameters/2);
hold off

% Implement 3D QPI-DHM for each segmenteded sample 

contCell = 1;
phase_stack = [];
amplitude_stack = [];
distance = 0;
factor = 8;
array_distance = [];
array_fx = [];
array_fy = [];

for contCell = 1:1:length(diameters-1);
    figure,imagesc(phase_bin),colormap(gray),title('phase bin '),daspect([1 1 1])
    mask = zeros(M,N);
      
    radious = (diameters(contCell)/2)*factor;
    circle = drawcircle('Center',[centers(contCell,1) centers(contCell,2)],'Radius',radious);
    mask_temp = createMask(circle);
    mask = or(mask, mask_temp);
    p = centers(contCell,2) - 90;
    q = p + 180
    r = centers(contCell,1) - 90;
    s = r + 180
    mask2 = zeros(M,N);
    mask2(p:q,r:s) = 1;
        
    % minimization --- synergetic
    seed = [fx_max_temp, fy_max_temp, distance];
    CFI = CF_integratedII(seed,field_compensate,lambda,dxy,mask,p,q,r,s);
    options = optimset('Display','iter', 'MaxIter', 300,'TolX',1e-1);
    [Val, CFI] = fminunc(@(t)(CF_integratedII(t,field_compensate,lambda,dxy,mask,p,q,r,s)), seed, options);
    fx_max_best = Val(1,1);
    fy_max_best = Val(1,2);
    distance_best = Val(1,3);
    fprintf('Cost at peaks found by fminunc: %f\n', CFI);
    fprintf('fx: %f\n', fx_max_best);
    fprintf('fy: %f\n', fy_max_best);
    fprintf('z: %f\n', distance_best);
    
    array_distance(contCell) = distance_best;
    array_fx(contCell) = fx_max_best;
    array_fy(contCell) = fy_max_best;

    
    [ref_wave] = reference_wave(M,N,m,n,lambda,dxy,fx_max_best,fy_max_best);
    field_compensate = ref_wave.*holo_filter;
    outputII = ang_spectrum(field_compensate,distance_best,lambda,dxy,dxy);
    phase = angle(outputII);
    amplitude = abs(outputII);

    amplitude = amplitude.*mask;
    phase = phase.*mask;
   
    amplitude_display = uint8(255 * mat2gray(amplitude));
    amplitude_stack = cat(3, amplitude_stack, amplitude_display);

    phase_display = uint8(255 * mat2gray(phase));
    phase_stack = cat(3, phase_stack, phase_display);
   
end    

% Stack visualization
[~,~,l] = size(phase_stack);
implay(phase_stack,l);