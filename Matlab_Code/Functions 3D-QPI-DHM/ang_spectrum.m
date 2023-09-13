function[out]=ang_spectrum(field,z,lambda,dx,dy)
%ANG_SPECTRUM Function to diffract a complex field using Angular Spectrum
%method
%   out = ANG_SPECTRUM(field,z,lambda,dx,dy)
%
%       field       complex field
%       z           propagation distance
%       lambda      wavelength
%       dx/dy       sampling pitches

[N,M] = size(field);
[m,n] = meshgrid(1-M/2:M/2,1-N/2:N/2);

dfx = 1 / (dx * M);
dfy = 1 / (dy * N);

% field = padarray(field,[floor(N/2) floor(M/2)]);
field_spec = fftshift(fft2(fftshift(field)));
% field_spec = padarray(field_spec,[floor(N/2) floor(M/2)]);

phase = exp(1i * z * 2 * pi * sqrt((1 / lambda)^2 - ((m*dfx).^2 + (n*dfy).^2)));
%phase = np.exp2(1j * z * pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))

% phase = padarray(phase,[floor(N/2) floor(M/2)]);

out = ifftshift(ifft2(ifftshift(field_spec.*phase)));
% out = out(floor(N/2)+1:floor(N/2)+N,floor(M/2)+1:floor(M/2)+M);

return