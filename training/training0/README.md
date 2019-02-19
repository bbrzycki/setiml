## Training 0

This training set will explore classifying narrow-band signals between 4 categories; noise, constant intensity signal, simple rfi (with 0 drift), and Gaussian-shaped intensity scintillation. We generate 32x1024 pixel spectrograms containing a single signal (or none in the case of noise), and assume a sample resolution of 1.4 Hz, 1.4 s.

Files are saved into an appropriate location in scratch space on BL machines in the colocation facility. For each data array, we save both a .png and .npy version and note certain signal parameters such as starting index, drift rate, line width, signal intensity (level), and freq spread (only really applicable for the simple rfi implementation, which varies the signal in the frequency direction in a Gaussian manner according to the value of the spread).
