function [CFI] = CF_integratedII(seed,complex_field,lambda,dxy,mask,p,q,r,s)
 % laoding the seeds
    CFI = 0;
    fx_max = seed(1,1);
    fy_max = seed(1,2);
    distance = seed(1,3);
    
    [M,N] = size(complex_field);
    [m,n] = meshgrid(-N/2:N/2-1,-M/2:M/2-1);

    dfx = 1 / (dxy * M);
    dfy = 1 / (dxy * N);

    %propagation
    field_spec = fftshift(fft2(fftshift(complex_field)));    
    phase = exp(1i * distance * 2 * pi * sqrt((1 / lambda)^2 - ((m*dfx).^2 + (n*dfy).^2)));
    % phase = padarray(phase,[floor(N/2) floor(M/2)]);
    complex_field = ifftshift(ifft2(ifftshift(field_spec.*phase)));
     
    % fine compensation
    [ref_wave] = reference_wave(M,N,m,n,lambda,dxy,fx_max,fy_max);
    complex_field2 = complex_field.*ref_wave;

    amplitude = abs(complex_field2).*mask;
    %amplitude = amplitude(p:q,r:s);
    %phase = angle(complex_field2);
    phase = angle(complex_field2).*mask;
    %figure,imagesc(amplitude),colormap(gray),colorbar, title('amplitude'),daspect([1 1 1])

    %cost-function propagation()
    NV = amplitude - mean2(amplitude);
    NV = NV.^2;
    NV = sum(NV(:))/mean2(amplitude);
    %NV = -1*NV;
    
    %cost-function compensation()
    phase = phase + pi; 
    %ib = imbinarize(phase,0.5);
    %J = M*N - sum(ib(:));
    J = std2(phase);
    
    CFI = J + NV;

end
