import numpy as np
from math import pi

from scipy.fftpack import fftshift, ifftshift, fft2, ifft2

def angularSpectrum(field, z, wavelength, dxy):
    '''
    # Function to diffract a complex field using the angular spectrum approximation
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx,dy - sampling pitches
    '''
    N, M = field.shape
    m, n = np.meshgrid(np.arange(1 - M / 2, M / 2 + 1), np.arange(1 - N / 2, N / 2 + 1))

    dfx = 1 / (dxy * M)
    dfy = 1 / (dxy * N)

    field_spec = fftshift(fft2(fftshift(field)))
    phase = np.exp(1j * z * 2 * np.pi * np.sqrt((1 / wavelength)**2 - ((m * dfx)**2 + (n * dfy)**2)))
    out = ifftshift(ifft2(ifftshift(field_spec * phase)))
    return out
