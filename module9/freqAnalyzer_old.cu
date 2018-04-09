/////////////////////////
// freqAnalyzer_old.cu // 
// Andrew Krepps       //
// Module 9 Assignment //
// 4/9/2018            //
/////////////////////////

#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cufft.h>

///////////////////////////////////////////////////////////////////////////////
/// \brief find the magnitude of each frequency bin from FFT results
/// 
/// \param [in] in the complex frequency-domain signal
/// \param [out] out the magnitude of each frequency bin
/// \param [in] n the number of samples
///////////////////////////////////////////////////////////////////////////////
__global__ void calcMagnitudes(const cufftComplex* in, float* out, const unsigned int n)
{
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (idx < n) {
		out[idx] = cuCabsf(in[idx]);
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief run an FFT on a real signal and find the magnitude of each frequency
/// bin
/// 
/// \param [in] timeSignal the real time-domain signal
/// \param [out] freqSignalMagnitudes the magnitude of each frequency bin
/// \param [in] n the number of samples
/// \param [in] blockSize the number of threads per block
///////////////////////////////////////////////////////////////////////////////
void runFFT(const float* timeSignal, float* freqSignalMagnitudes, const unsigned int n, const unsigned int blockSize)
{
	// create FFT plan
	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_R2C, 1);
	
	// allocate device memory
	cufftReal* d_timeSignal;
	cufftComplex* d_freqSignal;
	float* d_freqSignalMagnitudes;
	
	const unsigned int realBytes = n*sizeof(cufftReal);
	const unsigned int complexBytes = n*sizeof(cufftComplex);
	cudaMalloc((void**)&d_timeSignal, realBytes);
	cudaMalloc((void**)&d_freqSignal, complexBytes);
	cudaMalloc((void**)&d_freqSignalMagnitudes, realBytes);
	
	// copy input data to device
	cudaMemcpy(d_timeSignal, timeSignal, realBytes, cudaMemcpyHostToDevice);
	
	// perform FFT
	cufftExecR2C(plan, d_timeSignal, d_freqSignal);
	cudaDeviceSynchronize();

	// find magnitudes of frequency signal
	const unsigned int numBlocks = n/blockSize;
	calcMagnitudes<<<numBlocks, blockSize>>>(d_freqSignal, d_freqSignalMagnitudes, n);
	
	// copy output data to host
	cudaMemcpy(freqSignalMagnitudes, d_freqSignalMagnitudes, realBytes, cudaMemcpyDeviceToHost);
	
	// free device memory
	cudaFree(d_timeSignal);
	cudaFree(d_freqSignal);
	cudaFree(d_freqSignalMagnitudes);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief initialize input data on the host
/// 
/// \param [out] timeSignal the time-domain input signal
/// \param [in] frequency the signal frequency (in Hz)
/// \param [in] samplingRate the signal sampling rate (in Hz)
/// \param [in] n the number of samples
///////////////////////////////////////////////////////////////////////////////
void initializeInputData(float* timeSignal, float frequency, float samplingRate, const unsigned int n)
{
	printf("Generating %f Hz signal\n", frequency);
	for (unsigned int i = 0; i < n; ++i) {
		timeSignal[i] = sin(2.0f*M_PI*frequency*i/samplingRate);
	}
}

///////////////////////////////////////////////////////////////////////////////
/// \brief extract strongest frequency bin from FFT results
/// 
/// \param [in] freqSignalMagnitudes the magnitude of each frequency bin
/// \param [in] samplingRate the signal sampling rate (in Hz)
/// \param [in] n the number of samples
/// 
/// \returns the frequency (in Hz) of the bin with the largest magnitude
///////////////////////////////////////////////////////////////////////////////
float extractFrequency(const float* freqSignalMagnitudes, float samplingRate, const unsigned int n)
{
	// find frequency bin with highest magnitude
	unsigned int maxIdx = 0;
	float maxVal = 0.0f;
	
	// since input was a real signal, we only need to consider the first half of the output
	for (unsigned int i = 0; i < n/2; ++i) {
		if (freqSignalMagnitudes[i] > maxVal) {
			maxVal = freqSignalMagnitudes[i];
			maxIdx = i;
		}
	}
	
	// calculate frequency of selected bin
	return maxIdx*samplingRate/n;
}

int main(int argc, char** argv)
{
	// configure run
	unsigned int dataSize = 512;
	unsigned int blockSize = 256;
	float frequency = 128.0f;
	float samplingRate = 512.0f;
	if (argc > 1) {
		dataSize = atoi(argv[1]);
	}
	if (argc > 2) {
		blockSize = atoi(argv[2]);
	}
	if (argc > 3) {
		frequency = atof(argv[3]);
	}
	if (argc > 4) {
		samplingRate = atof(argv[4]);
	}
	
	// allocate and initialize host memory
	const unsigned int numBytes = dataSize*sizeof(float);
	float* timeSignal = (float*)malloc(numBytes);
	float* freqSignalMagnitudes = (float*)malloc(numBytes);
	initializeInputData(timeSignal, frequency, samplingRate, dataSize);
	
	// dummy execution to avoid startup cost	
	runFFT(timeSignal, freqSignalMagnitudes, dataSize, blockSize);
	
	// measure total exectution time (including frequency extraction from FFT results)
	auto start = std::chrono::high_resolution_clock::now();
	
	// run FFT and extract dominant frequency
	runFFT(timeSignal, freqSignalMagnitudes, dataSize, blockSize);
	float foundFrequency = extractFrequency(freqSignalMagnitudes, samplingRate, dataSize);
	
	// stop clock and calculate time in ms
	auto stop = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> duration(stop - start);
	float ms = duration.count()*1000.0f;
	
	// show results
	printf("Strongest frequency bin was %f Hz (execution time = %.3f ms)\n", foundFrequency, ms);
	
	// free host memory
	free(timeSignal);
	free(freqSignalMagnitudes);
}
