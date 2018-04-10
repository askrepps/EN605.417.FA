////////////////////////////
// freqAnalyzer_thrust.cu // 
// Andrew Krepps          //
// Module 9 Assignment    //
// 4/9/2018               //
////////////////////////////

#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cufft.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

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
/// \brief run an FFT on a real signal and extract the dominant frequency
/// 
/// \param [in] timeSignal the real time-domain signal
/// \param [in] samplingRate the sampling rate (in Hz)
/// \param [in] blockSize the number of threads per block
///////////////////////////////////////////////////////////////////////////////
float analyzeFrequency(const thrust::host_vector<float>& timeSignal, const float samplingRate, const unsigned int blockSize)
{
	// create FFT plan
	cufftHandle plan;
	cufftPlan1d(&plan, timeSignal.size(), CUFFT_R2C, 1);
	
	// allocate device memory and copy input data
	thrust::device_vector<cufftReal> d_timeSignal = timeSignal;
	
	// allocate device memory for output data
	thrust::device_vector<cufftComplex> d_freqSignal(timeSignal.size());
	thrust::device_vector<float> d_freqSignalMagnitudes(timeSignal.size());
	
	// perform FFT
	cufftExecR2C(plan, thrust::raw_pointer_cast(&d_timeSignal[0]), thrust::raw_pointer_cast(&d_freqSignal[0]));
	cudaDeviceSynchronize();
	
	// find magnitudes of frequency signal
	const unsigned int numBlocks = timeSignal.size()/blockSize;
	calcMagnitudes<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(&d_freqSignal[0]), thrust::raw_pointer_cast(&d_freqSignalMagnitudes[0]), timeSignal.size());
	cudaDeviceSynchronize();
	
	// find index of frequency bin of highest magnitude and calculate frequency
	unsigned int maxIdx = thrust::max_element(d_freqSignalMagnitudes.begin(), d_freqSignalMagnitudes.end()) - d_freqSignalMagnitudes.begin();
	float freq = maxIdx*samplingRate/d_freqSignalMagnitudes.size();
	
	return freq;
	
	// device memory is freed automatically by device_vector destructors
}

///////////////////////////////////////////////////////////////////////////////
/// \brief initialize input data on the host
/// 
/// \param [out] timeSignal the time-domain input signal
/// \param [in] frequency the signal frequency (in Hz)
/// \param [in] samplingRate the signal sampling rate (in Hz)
///////////////////////////////////////////////////////////////////////////////
void initializeInputData(thrust::host_vector<float>& timeSignal, float frequency, float samplingRate)
{
	printf("Generating %f Hz signal\n", frequency);
	for (unsigned int i = 0; i < timeSignal.size(); ++i) {
		timeSignal[i] = sin(2.0f*M_PI*frequency*i/samplingRate);
	}
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
	thrust::host_vector<float> timeSignal(dataSize);
	initializeInputData(timeSignal, frequency, samplingRate);
	
	// dummy execution to avoid startup cost	
	analyzeFrequency(timeSignal, samplingRate, blockSize);
	
	// measure total exectution time (including frequency extraction from FFT results)
	auto start = std::chrono::high_resolution_clock::now();
	
	// run FFT and extract dominant frequency
	float foundFrequency = analyzeFrequency(timeSignal, samplingRate, blockSize);
	
	// stop clock and calculate time in ms
	auto stop = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> duration(stop - start);
	float ms = duration.count()*1000.0f;
	
	// show results
	printf("Strongest frequency bin was %f Hz (execution time = %.3f ms)\n", foundFrequency, ms);
	
	// host memory is automatically freed by host_vector destructors
}
