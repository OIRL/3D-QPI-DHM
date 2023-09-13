from matplotlib import pyplot as plt
import numpy as np
import math
import os
import cv2
import angularSpectrum

from scipy.fftpack import fftshift, ifftshift, fft2, ifft2

#pycuda imports #If you do not have a GPU-enabled system, nor pycuda properly installed, comment the following imports
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import struct
import imageio
import skcuda.fft as cu_fft

#time counting imports
from timeit import default_timer as timer

#pyCUDA kernels
mod = SourceModule("""
  #include <cuComplex.h>
  #include <math_functions.h>

  #define BLOCK_SIZE 128

__global__ void Compensacion_SalidaFase(float *__restrict__ odata_real, float *__restrict__ odata_imag, float *__restrict__ odata_temp, float *__restrict__ odata_temp2, float theta_x, float theta_y, float k, float dx, float dy, int width, int height)
{
	//Descriptores de cada hilo    
    int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	
	//real and imaginary parts of the tilted reference
	float ref_real = __cosf( k * (__sinf(theta_x) * (fila-(width/2)) * dx + __sinf(theta_y) * (col-(height/2)) * dy));

	float ref_imag = __sinf( k * (__sinf(theta_x) * (fila-(width/2)) * dx + __sinf(theta_y) * (col-(height/2)) * dy));

    //Pointwise multiplication of complex "arrays""
	odata_temp[col*width + fila] = (odata_real[col*width + fila] * ref_real) - (odata_imag[col*width + fila] * ref_imag);
	odata_temp2[col*width + fila] = (odata_imag[col*width + fila] * ref_real) + (odata_real[col*width + fila] * ref_imag);

    //odata_real[col*width + fila] = odata_temp[col*width + fila];
    //odata_imag[col*width + fila] = odata_temp2[col*width + fila];
    
    //odata_temp[col*width + fila] = atan2( odata_temp2[col*width + fila], odata_temp[col*width + fila] );
    
}

  __global__ void getStats(float *__restrict__ pArray, float *__restrict__ pMaxResults, float *__restrict__ pMinResults)
  {
	// Declare arrays to be in shared memory.
	// 256 elements * (4 bytes / element) * 2 = 2KB.
	__shared__ float min[256];
	__shared__ float max[256];

	// Calculate which element this thread reads from memory
	int arrayIndex = 256 * 128 * blockIdx.y + 256 * blockIdx.x + threadIdx.x;
	min[threadIdx.x] = max[threadIdx.x] = pArray[arrayIndex];
	__syncthreads();


	int nTotalThreads = blockDim.x;	// Total number of active threads

	while (nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < halfPoint)
		{
			// Get the shared value stored by another thread
			float temp = min[threadIdx.x + halfPoint];
			if (temp < min[threadIdx.x]) min[threadIdx.x] = temp;
			temp = max[threadIdx.x + halfPoint];
			if (temp > max[threadIdx.x]) max[threadIdx.x] = temp;
		}


		nTotalThreads = (nTotalThreads >> 1);	// divide by two.
	}

	// At this point in time, thread zero has the min, max, and average
	// It's time for thread zero to write it's final results.
	// Note that the address structure of pResults is different, because
	// there is only one value for every thread block.

	if (threadIdx.x == 0)
	{
		pMaxResults[128 * blockIdx.y + blockIdx.x] = max[0];
		pMinResults[128 * blockIdx.y + blockIdx.x] = min[0];

	}
  }
  
  __global__ void escalamiento(float *__restrict__ temp, float *__restrict__ temp2, int width, int height, float maximo, float minimo)
  {

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	temp2[fila*width + col] = (temp[fila*width + col]) - minimo;
	temp2[fila*width + col] = (temp2[fila*width + col]) / (maximo - minimo);
	temp2[fila*width + col] = (temp2[fila*width + col]) * 255;
	//Ac? tenemos todas los pixeles escalados a 8 bits (255 niveles de gris)

  }
  
  __global__ void Umbralizacion( float *__restrict__ temp, float umbral, int width, int height) 
  {
	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	//Calculo de la intensidad
	if (temp[fila*width + col] > umbral) {
		temp[fila*width + col] = 1;
	}
	else {
		temp[fila*width + col] = 0;
	}

  }

  
  __global__ void Sumatoria(float *__restrict__ pArray, float *__restrict__ pDesviacion) 
  {
	// Declare arrays to be in shared memory.
	// 128 elements * (4 bytes / element) * 2 = 2KB.
	__shared__ float avg[256];

	// Calculate which element this thread reads from memory
	int arrayIndex = 256 * 128 * blockIdx.y + 256 * blockIdx.x + threadIdx.x;
	avg[threadIdx.x] = pArray[arrayIndex];
	__syncthreads();


	int nTotalThreads = blockDim.x;	// Total number of active threads

	while (nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < halfPoint)
		{
			// when calculating the average, sum and divide
			avg[threadIdx.x] += avg[threadIdx.x + halfPoint];
			//avg[threadIdx.x] /= 2;
		}

		nTotalThreads = (nTotalThreads >> 1);	// divide by two.
	}

	if (threadIdx.x == 0)
	{
		pDesviacion[128 * blockIdx.y + blockIdx.x] = avg[0];

	}

  }
  
  __global__ void fft_shift_complex(cuComplex* __restrict__ arregloC, float *__restrict__ d_temp13x, int width, int height)
  {

	int m2 = width / 2;
	int n2 = height / 2;

	//Descriptores de cada hilo
    int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	int fila2 = blockIdx.x*blockDim.x + threadIdx.x + m2;
	int col2 = blockIdx.y*blockDim.y + threadIdx.y + n2;
    
    d_temp13x[col*width + fila] = arregloC[col*width + fila].x;  //Guardo el primer cuadrante
	arregloC[col*width + fila].x = arregloC[col2*width + fila2].x;  //en el primer cuadrante estoy poniendo lo que hay en el tercero
	arregloC[col2*width + fila2].x = d_temp13x[col*width + fila];//En el tercer cuadrante estoy poniendo lo que habia en el primero

	d_temp13x[col*width + fila] = arregloC[col*width + fila].y;  //Lo mismo anterior pero para los imaginarios
	arregloC[col*width + fila].y = arregloC[col2*width + fila2].y;
	arregloC[col2*width + fila2].y = d_temp13x[col*width + fila];

	d_temp13x[col*width + fila] = arregloC[col*width + fila2].x;//Guardo Cuadrante dos
	arregloC[col*width + fila2].x = arregloC[col2*width + fila].x;  //En el segundo guardo lo que hay en el cuarto
	arregloC[col2*width + fila].x = d_temp13x[col*width + fila];//En el cuarto guardo lo que estaba en el segundo

	d_temp13x[col*width + fila] = arregloC[col*width + fila2].y; //Lo mismo que en el anterior
	arregloC[col*width + fila2].y = arregloC[col2*width + fila].y;
	arregloC[col2*width + fila].y = d_temp13x[col*width + fila];
  }
  
  __global__ void CambioTipoVariableUnaMatrix(float *__restrict__ real, float *__restrict__ imag, cuComplex* __restrict__ arregloC, int width, int height)
  {

    //Descriptores de cada hilo
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int fila = blockIdx.y*blockDim.y + threadIdx.y;

    arregloC[fila*width + col].x = real[fila*width + col];
    arregloC[fila*width + col].y = imag[fila*width + col];
  } 
  
  __global__ void Multiplicacion(cuComplex* __restrict__ arregloC, float *__restrict__ temp, float z, float lambda, float dfx, float dfy, int width, int height)
{

	//Descriptores de cada hilo
	int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y ;

	float M = col - (height / 2);
	float N = fila - (width / 2);
	
	float term1 = (1 / lambda)*(1 / lambda);
	float term2 = (dfx * M)*(dfx * M);
	float term3 = (dfy * N)*(dfy * N);

	//phase in angular spectrum  kernel
	float phase_real = __cosf(1 * (3.141592) * z * sqrt( term1 - term2 - term3 ) );
	float phase_imag = __sinf(1 * (3.141592) * z * sqrt( term1 - term2 - term3 ) );

	//Pointwise multiplication of complex "arrays""
	temp[col*width + fila] = (arregloC[col*width + fila].x * phase_real) - (arregloC[col*width + fila].y * phase_imag);
	arregloC[col*width + fila].y = (arregloC[col*width + fila].y * phase_real) + (arregloC[col*width + fila].x * phase_imag);
	arregloC[col*width + fila].x = temp[col*width + fila] ;	

    //temp[col*width + fila] = (arregloC[col*width + fila].x * arregloC[col*width + fila].x)+(arregloC[col*width + fila].y * arregloC[col*width + fila].y);
    
    //temp[col*width + fila] = log(temp[col*width + fila]);


}

  __global__ void fft_inverse_correction(cuComplex *__restrict__ arregloC, int width, int height) 
  {
	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

    arregloC[fila*width + col].x = arregloC[fila*width + col].x;
    arregloC[fila*width + col].y = (-1)*arregloC[fila*width + col].y;

  }
  
  
    __global__ void module(cuComplex* __restrict__ arregloC, float *__restrict__ temp, int width, int height)
  {

	//Descriptores de cada hilo
    int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
    
    temp[col*width + fila] = (arregloC[col*width + fila].x * arregloC[col*width + fila].x) + (arregloC[col*width + fila].y * arregloC[col*width + fila].y); 

    //temp[col*width + fila] = log(temp[col*width + fila]);

  }

  """)


# Function to display an image
def imageShow(inp, title):
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image
    plt.imshow(inp, cmap='gray'), plt.title(title)  # image in gray scale
    plt.show()  # show image
    return


# Function to compute the intensity representation of a given complex field
def intensity(inp, log):
    # Inputs:
    # inp - The input complex field
    # log - boolean variable to determine if a log representation is applied
    out = np.abs(inp)
    out = out * out
    if log == True:
        out = 20 * np.log(out)
        out[out == np.inf] = 0
        out[out == -np.inf] = 0

    return out


# Function to compute the phase representation of a given complex field
def phase(inp):
    # Inputs:
    # inp - The input complex field
    out = np.angle(inp)
    return out

def spatialFiltering(holo, M, N, factor):
    print('Spatial filtering started....')
    fft_holo = fftshift(fft2(fftshift(holo)))
    #plt.imshow(np.log(np.abs(fft_holo)**2), cmap='gray')
    #plt.title('FT Hologram')
    #plt.show()

    mask = np.ones((M, N))
    mask[M // 2 - 100:M // 2 + 100, N // 2 - 100:N // 2 + 100] = 0
    fft_holo_I = fft_holo * mask
    mask = np.ones((M, N))
    mask[:10, :10] = 0
    fft_holo_I = fft_holo_I * mask

    maxValue_1 = np.max(np.abs(fft_holo_I))
    fy_max_1, fx_max_1 = np.where(np.abs(fft_holo_I) == maxValue_1)

    mask[fy_max_1[0] - 50:fy_max_1[0] + 50, fx_max_1[0] - 50:fx_max_1[0] + 50] = 0
    fft_holo_I = fft_holo_I * mask

    maxValue_1 = np.max(np.abs(fft_holo_I))
    fy_max_1, fx_max_1 = np.where(np.abs(fft_holo_I) == maxValue_1)
    fx_max, fy_max = fx_max_1[0], fy_max_1[0]
    
    distance = np.sqrt((fx_max - M / 2)**2 + (fy_max - N / 2)**2)
    resc = M / factor
    filter = np.ones((M, N))
    
    for r in range(M):
        for p in range(N):
            if np.sqrt((r - fy_max)**2 + (p - fx_max)**2) > resc:
                filter[r, p] = 0
    
    fft_filter_holo = fft_holo * filter
    num = np.max(fft_filter_holo)
    idx = np.argmax(fft_filter_holo)
    
    holo_filter = fftshift(ifft2(fftshift(fft_filter_holo)))
    print('Spatial filtering finished....')    
    return holo_filter, fx_max, fy_max


def digitalReferenceWave(field, fx_max, fy_max, wavelength, dxy):

    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')
    
    fx_0 = N / 2
    fy_0 = M / 2
    k = 2 * np.pi / wavelength
    theta_x = np.arcsin((fx_0 - fx_max) * wavelength / (N * dxy))  # Eq. 6
    theta_y = np.arcsin((fy_0 - fy_max) * wavelength / (M * dxy))  # Eq. 7
    ref_wave = np.exp(1j * k * (np.sin(theta_x) * X * dxy + np.sin(theta_y) * Y * dxy))  # digital reference wave
    return ref_wave
    
# Function to create a circular mask
def circular(M, N, radius, centers, display):
    cent_x = centers[0, 0]
    cent_y = centers[0, 1]
    circle = np.zeros((radius * 2, radius * 2), dtype=int)
    for p in range(0, radius * 2):
        for q in range(0, radius * 2):
            if np.sqrt((p - radius) ** 2 + (q - radius) ** 2) < radius:
                circle[p, q] = 1
    (minC, maxC) = circle.shape
    mask = np.zeros((M, N), dtype=int)
    minX = int(cent_x - minC / 2)
    maxX = int(cent_x + minC / 2)
    minY = int(cent_y - minC / 2)
    maxY = int(cent_y+ maxC / 2)
    mask[minY:maxY, minX:maxX] = circle[:, :]

    if (display == True):
        imageShow(mask, 'mask')

    return mask    


#Function to compute the synergectic metric using the CPU numpy library
def func_minimize(complex_field, distance, fx, fy, wavelength, dxy, mask):
    
    ref_wave = digitalReferenceWave(complex_field, fx, fy, wavelength, dxy)
    
    complex_field = complex_field * ref_wave
    
    complex_field2 = angularSpectrum.angularSpectrum(complex_field, distance, wavelength, dxy)
    
    amplitude = np.abs(complex_field2) * mask
    minVal = np.amin(amplitude)
    maxVal = np.amax(amplitude)
    amplitude_sca = (amplitude - minVal) / (maxVal - minVal)
    
    #Propagation cost function
    NV = amplitude_sca - np.mean(amplitude_sca)
    NV = NV ** 2
    NV = np.sum(NV.flatten()) / np.mean(amplitude_sca)
    NV = NV / np.sum(mask.flatten())

    phase = np.angle(complex_field2)
                
    #Compensation cost function
    phase = phase + np.pi
    # Thresholding process
    minVal = np.amin(phase)
    maxVal = np.amax(phase)
    phase_sca = (phase - minVal) / (maxVal - minVal)
    binary_phase = (phase_sca > 0.2)
    summ = np.sum(np.sum(binary_phase))
    M, N = binary_phase.shape
    J = M * N - summ
                
    #Synergetic metric
    val = 0.0037*J - NV
    
    return val
    
# Function to compute the compensated autofocused reconstruction with CPU
def autofocus_compensation_CPU(complex_field, distance, fx, fy, wavelength, dxy, mask):
    
    s = 1 # Specify the range size for fx and fy
    steps = 0.5 # Specify the step size for fx and fy

    ds = 20 # Specify the range size for distance
    dsteps = 10 # Specify the step size for distance

    min_val = float('inf')  # Initialize with a large value
    optimal_fx = None
    optimal_fy = None
    optimal_distance = None

    for fx_temp in np.arange(fx - s, fx + s + steps, steps):
        for fy_temp in np.arange(fy - s, fy + s + steps, steps):
            for distance_temp in np.arange(distance - ds, distance + ds + dsteps, dsteps):
                val = func_minimize(complex_field, distance_temp, fx_temp, fy_temp, wavelength, dxy, mask)
                if val < min_val:
                    min_val = val
                    optimal_fx = fx_temp
                    optimal_fy = fy_temp
                    optimal_distance = distance_temp
                
    print("val=", min_val)
    return optimal_fx, optimal_fy, optimal_distance


#Function to compute the Angular Spectrum of a complex wavefiled with GPU boosting
def angular_spectrum_GPU(devicereal, deviceimag, bufferFFT1, devicetemp, M, N, distance, wavelength, dfx, dfy):
    
    #Variables to define the size of each block, thus, the number of blocks in each dimension of the grid. 
    #The latter based on the number of threads and the size oft he array (image)
    block_size_x = 16
    block_size_y = 16
    block_dim = (block_size_x, block_size_y,1)
    
    #let's put the data in a cuComplex-type array
    grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
    CambioTipoVariableUnaMatrix = mod.get_function("CambioTipoVariableUnaMatrix")
    CambioTipoVariableUnaMatrix(devicereal, deviceimag, bufferFFT1, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
    pycuda.driver.Context.synchronize()
    
    grid_dim = (M // (2*block_dim[0]), N // (2*block_dim[1]),1)
    fft_shift2 = mod.get_function("fft_shift_complex")
    fft_shift2(bufferFFT1, devicetemp, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
    pycuda.driver.Context.synchronize()
    
    # Forward FFT
    plan_forward = cu_fft.Plan(bufferFFT1.shape, np.complex64, np.complex64)
    cu_fft.fft(bufferFFT1, bufferFFT1, plan_forward)
    pycuda.driver.Context.synchronize()

    grid_dim = (M // (2*block_dim[0]), N // (2*block_dim[1]),1)
    fft_shift2 = mod.get_function("fft_shift_complex")
    fft_shift2(bufferFFT1, devicetemp, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
    pycuda.driver.Context.synchronize()
     
    # Complex matrix multiplication (the propagation kernel)
    grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
    Multiplicacion = mod.get_function("Multiplicacion")
    Multiplicacion(bufferFFT1, devicetemp, np.float32(distance), np.float32(wavelength), np.float32(dfx), np.float32(dfy), np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
    pycuda.driver.Context.synchronize()  
    
    #Let's come back to spatial domain
    #IFFT
    #Let's correct the output of this foward fft (we need the inverse fft)
    grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
    fft_inverse_correction = mod.get_function("fft_inverse_correction")
    fft_inverse_correction(bufferFFT1, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
    pycuda.driver.Context.synchronize()
    
    grid_dim = (M // (2*block_dim[0]), N // (2*block_dim[1]),1)
    fft_shift3 = mod.get_function("fft_shift_complex")
    fft_shift3(bufferFFT1, devicetemp, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
    pycuda.driver.Context.synchronize()

    cu_fft.fft(bufferFFT1, bufferFFT1, plan_forward)
    pycuda.driver.Context.synchronize()

    grid_dim = (M // (2*block_dim[0]), N // (2*block_dim[1]),1)
    fft_shift3 = mod.get_function("fft_shift_complex")
    fft_shift3(bufferFFT1, devicetemp, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
    pycuda.driver.Context.synchronize()
    
    #Let's correct the output of this foward fft (we need the inverse fft)
    grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
    fft_inverse_correction = mod.get_function("fft_inverse_correction")
    fft_inverse_correction(bufferFFT1, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
    pycuda.driver.Context.synchronize()
    
    return bufferFFT1


# Function to compute the compensated autofocused reconstruction with GPU boosting
def autofocus_compensation_GPU(complex_field, distance, fy, fx, wavelength, dxy, mask):
    
    s = 1  # Specify the range size for fx and fy
    steps = 0.5 # Specify the step size for fx and fy

    ds = 20  # Specify the range size for distance
    dsteps = 10  # Specify the step size for distance

    min_val = float('inf')  # Initialize with a large value
    optimal_fx = None
    optimal_fy = None
    optimal_distance = None
    
    M, N = complex_field.shape
    
    fx_0 = M/2
    fy_0 = N/2
    
    
    #Variables to define the size of each block, thus, the number of blocks in each dimension of the grid. 
    #The latter based on the number of threads and the size oft he array (image)
    block_size_x = 16
    block_size_y = 16
    block_dim = (block_size_x, block_size_y,1)
    
    numElements = N * M

    #Reduction process variables (search of the maximum and minimum)
    THREADS_PER_BLOCK = int(256)
    BLOCKS_PER_GRID_ROW = int(128)
    
    blockGridWidth = int(BLOCKS_PER_GRID_ROW)
    blockGridHeight = int((numElements / THREADS_PER_BLOCK) / blockGridWidth)

    #h_resultMax[] = new float[numElements / THREADS_PER_BLOCK * Sizeof.FLOAT];
    h_resultMax = np.zeros(int(numElements / THREADS_PER_BLOCK * struct.calcsize('f')), dtype=np.float32)
    #h_resultMax = np.zeros(int(numElements / THREADS_PER_BLOCK * 4), dtype=np.float32)
    h_resultMin = np.zeros(int(numElements / THREADS_PER_BLOCK * struct.calcsize('f')), dtype=np.float32)
    #h_resultMin = np.zeros(int(numElements / THREADS_PER_BLOCK * 4), dtype=np.float32)
    h_resultDes = np.zeros(int(numElements / THREADS_PER_BLOCK * struct.calcsize('f')), dtype=np.float32)

    #Allocate and fill the host input data
    temp = np.zeros(numElements, dtype=np.float32)
    
    #Allocate device devicetemp memory
    devicetemp = cuda.mem_alloc(temp.nbytes)
    cuda.memcpy_htod(devicetemp, temp)
    
    #Allocate device devicetemp2 memory
    devicetemp2 = cuda.mem_alloc(temp.nbytes)
    cuda.memcpy_htod(devicetemp2, temp)
    
        #Allocate device devicetemp2 memory
    devicetemp3 = cuda.mem_alloc(temp.nbytes)
    cuda.memcpy_htod(devicetemp3, temp)

    #Allocate and fill the host input data
    real = np.zeros(numElements, dtype=np.float32)

    # Allocate the device input data, and copy the
    # host input data to the device
    deviceout = gpuarray.to_gpu(real)# From numpy array to GPUarray
    
    bufferFFT1 = gpuarray.zeros((N,M), np.complex64)
    cpu_bufferFFT1 = np.zeros((N,M), dtype=np.complex64)
    
    #Allocate device memory for the reductions
    d_resultMax = cuda.mem_alloc(h_resultMax.nbytes)
    cuda.memcpy_htod(d_resultMax, h_resultMax)
    d_resultMin = cuda.mem_alloc(h_resultMin.nbytes)
    cuda.memcpy_htod(d_resultMin, h_resultMin)
    d_resultDes = cuda.mem_alloc(h_resultDes.nbytes)
    cuda.memcpy_htod(d_resultDes, h_resultDes)
    
    # Initialise complex_field GPUarray 
        
    real_part = np.transpose(np.real(complex_field))
    imaginary_part = np.transpose(np.imag(complex_field))

    # Allocate the device input data, and copy the
    devicereal = cuda.mem_alloc(real_part.astype(np.float32).nbytes)
    cuda.memcpy_htod(devicereal, real_part.astype(np.float32))

    #Allocate device imaginary_part memory
    deviceimag = cuda.mem_alloc(imaginary_part.astype(np.float32).nbytes)
    cuda.memcpy_htod(deviceimag, imaginary_part.astype(np.float32))

    for fx_temp in np.arange(fx - s, fx + s + steps, steps):
        for fy_temp in np.arange(fy - s, fy + s + steps, steps):
            for distance_temp in np.arange(distance - ds, distance + ds + dsteps, dsteps):
    
                theta_x = math.asin((fx_0 - fx_temp) * wavelength / (M * dxy))
                theta_y = math.asin((fy_0 - fy_temp) * wavelength / (N * dxy))
                pi = 3.141592
                k = 2 * pi / wavelength
        
                # Phase Compensation with specific theta_x and theta_y
                grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
                Compensacion_SalidaFase = mod.get_function("Compensacion_SalidaFase")
                Compensacion_SalidaFase(devicereal, deviceimag, devicetemp, devicetemp2, np.float32(theta_x), np.float32(theta_y), np.float32(k), np.float32(dxy), np.float32(dxy), np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
                pycuda.driver.Context.synchronize()
          
                #Propagation with angular spectrum
                dfx = 1 / (dxy * N)
                dfy = 1 / (dxy * M)            
                bufferFFT1 = angular_spectrum_GPU(devicetemp, devicetemp2, bufferFFT1, devicetemp3, M, N, distance_temp, wavelength, dfx, dfy)
                
                cpu_bufferFFT1 = bufferFFT1.get()
                
                cpu_bufferFFT1 = np.transpose(cpu_bufferFFT1)
                
                amplitude = np.abs(cpu_bufferFFT1) * mask
                minVal = np.amin(amplitude)
                maxVal = np.amax(amplitude)
                amplitude_sca = (amplitude - minVal) / (maxVal - minVal)
    
                #Propagation cost function
                NV = amplitude_sca - np.mean(amplitude_sca)
                NV = NV ** 2
                NV = np.sum(NV.flatten()) / np.mean(amplitude_sca)
                NV = NV / np.sum(mask.flatten())
    
                phase = np.angle(cpu_bufferFFT1)
                
                #Compensation cost function
                phase = phase + np.pi
                # Thresholding process
                minVal = np.amin(phase)
                maxVal = np.amax(phase)
                phase_sca = (phase - minVal) / (maxVal - minVal)
                binary_phase = (phase_sca > 0.2)
                summ = np.sum(np.sum(binary_phase))
                M, N = binary_phase.shape
                J = M * N - summ
                
                #Synergetic metric
                val = 0.001*J - NV               
            
                if val < min_val:
                    min_val = val
                    optimal_fx = fx_temp
                    optimal_fy = fy_temp
                    optimal_distance = -distance_temp/3
                
    print("val=", min_val)
    return optimal_fy, optimal_fx, optimal_distance