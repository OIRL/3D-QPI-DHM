'''

Code implementation of the synergetic phase compensation and autofocus procedure in both CPU/GPU. The GPU version uses pyCUDA.
This code applies a 3D grid search (fx, fy and z) to accurately compute well compensated and focused DHM holograms.
Original code: Raul Castañeda's phase compensation + autofocusing of image-plane off-axis DHM holograms.

'functions.py' includes the auxiliary functions and methods to run this code.

Authors: Raúl Castañeda, Ana Doblas and Carlos Trujillo
Original date: June 17th, 2023.
Last version: August 11, 2023.

'''

from PIL import Image, ImageOps
import numpy as np
import functions
import skimage
import cv2
import imutils
import angularSpectrum

#time counting imports
from timeit import default_timer as timer

# Lines to read and show the hologram
hologram = Image.open('holo-RBC-20p22-5-2.png')
functions.imageShow(hologram, 'hologram')


# Line to set the reconstruction parameters
#wavelength = 0.000633
#dxy = 0.00375

wavelength = 0.000532
dxy = 0.0024


# Lines to perform the spatial filter +1 term
field = np.array(hologram)
M, N = field.shape
holoFilter, fx_max, fy_max = functions.spatialFiltering(field, M, N, 12)
#print("fx_max=", fx_max, " and fy_max=", fy_max)

# Lines to compute the phase reconstruction from a set of propagation distance
ref_wave = functions.digitalReferenceWave(holoFilter, fx_max, fy_max, wavelength, dxy)
compensation = holoFilter * ref_wave
#functions.imageShow(np.abs(compensation), 'Filtered hologram')
#functions.imageShow(np.angle(ref_wave), 'ref_wave')

'''
# Uncomment if you want to build the phase reconstructions for the whole inspection volume (it takes time...)
#path = 'F:/My Drive/CurrentResearch/DHM/DHM - PhaseComp+Autofocus/phase_Images_stack/'
array_distance = np.arange(-200, 200, 20)
print('Propagation to different distances started....')
cont = 1
for distance in array_distance:
    output = angularSpectrum.angularSpectrum(compensation, distance, wavelength, dxy)
    intensity = functions.intensity(output, False)
    functions.imageShow(intensity, 'Intensity at ' + str(distance))
    #intensity_scale = ((intensity - intensity.min()) * (1 / (intensity.max() - intensity.min()) * 255)).astype('uint8')
    #filename = path + str(cont) + '_intensity_' + str(distance) + 'mm' + '.jpg'
    #cv2.imwrite(filename, intensity_scale)
    cont = cont + 1
print('Propagation to different distances finished....')
'''

########################################################################################################################
########################################################################################################################
########################################################################################################################

## This part of the code depends on the type of specimens that are visualized. Feel free to use your own segmentation 
## method. 
## No matter the selected segmentation method, the information for each individual specimen's mask must be provided 
## to the algorithm in the following stage.

# Lines for segmentation
phase = functions.phase(compensation)
phase_to_binary = ((phase - phase.min()) * (1 / (phase.max() - phase.min()) * 255)).astype('uint8')
threshold = skimage.filters.threshold_triangle(phase_to_binary)
binary = (np.invert(phase_to_binary > threshold - 10) * 1) * 255
binary = binary.astype(np.uint8)
#functions.imageShow(binary, 'Triangle binary image')

# find contours in the thresholded image
print("Segmentation started....")
cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("[INFO] {} contours found".format(len(cnts)))
''' Uncomment if you want to see the detected specimens
for (i, c) in enumerate(cnts):
    # draw the contour
    ((x, y), _) = cv2.minEnclosingCircle(c)
    cv2.putText(phase_to_binary, "#{}".format(i + 1), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                2)
    cv2.drawContours(phase_to_binary, [c], -1, (0, 255, 0), 2)
cv2.imshow("Image", phase_to_binary)
cv2.waitKey(0)
'''

# Lines to select the sperm cells, the number are taken from the phase image - 1 position
#array_Sperm = (cnts[40 - 1], cnts[34 - 1], cnts[29 - 1])
array_Sperm = (cnts[2 - 1], cnts[3 - 1]) #Single object holos

print("Segmentation finished....")

########################################################################################################################
########################################################################################################################
########################################################################################################################


#Synergetic implementation
print('Starting synergetic search')
for s in (np.arange(0, len(array_Sperm)).tolist()):
    
    start = timer()	#Start to count time
    
    temp = np.asarray(array_Sperm[s])
    centers = temp[1]
    #mask = functions.circular(M, N, 90, centers, False)
    mask = np.ones((M, N)) #For single object holograms
    
    distance = 0 #We are assuming the user has no clue where the specimens are within the inspection volume
    
    #If you do not have a GPU-enabled system, run the following command (it will take time).
    fx_best, fy_best, distance_best = functions.autofocus_compensation_CPU(holoFilter, distance, fx_max, fy_max, wavelength, dxy, mask)
    
    #If you have a GPU-enabled device, use the following function for speed-up reconstruction (see pycuda installation instructions) 
    #fx_best, fy_best, distance_best = functions.autofocus_compensation_GPU(holoFilter, distance, fx_max, fy_max, wavelength, dxy, mask)
    
    ref_wave = functions.digitalReferenceWave(holoFilter, fx_best, fy_best, wavelength, dxy)
    best_reconstruction = holoFilter * ref_wave
    output = angularSpectrum.angularSpectrum(best_reconstruction, distance_best, wavelength, dxy)
    phase_best = functions.phase(output)
    int_best = functions.intensity(output, False)
    
    print("Processing time:", timer()-start) #Time for execution
    
    print("fx_best=", fx_best, " fy_best=", fy_best, " and distance_best=", distance_best)
    
    path = 'F:\My Drive\CurrentResearch\DHM\DHM - PhaseComp+Autofocus/'
    
    phase_scale = ((phase_best - phase_best.min()) * (1 / (phase_best.max() - phase_best.min()) * 255)).astype('uint8')
    filename = path + 'comp_phase_' + str(s + 1) + '_' + str(distance_best) + 'mm' + '.jpg'
    cv2.imwrite(filename, phase_scale)
        
    functions.imageShow(phase_best, 'rec ' + str(s + 1) + 'phase ' + str(distance_best) + 'mm')
    functions.imageShow(phase_best*mask, 'rec ' + str(s + 1) + ' phase*mask ' + str(distance_best) + 'mm')
    functions.imageShow(int_best*mask, 'rec ' + str(s + 1) + ' intensity ' + str(distance_best) + 'mm')

print('Synergetic search finished')