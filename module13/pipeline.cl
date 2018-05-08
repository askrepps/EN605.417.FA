//////////////////////////
// pipline.cl           //
// Andrew Krepps        //
// Module 13 Assignment //
// 7 May 2018           //
//////////////////////////

// This is a simple example demonstrating different stages of a pipeline of simple operations

__kernel void addFive(
	const __global float* inputBuffer,
	__global float* outputBuffer,
	size_t n)
{
	size_t id = get_global_id(0);
	
	if (id < n)
	{
		outputBuffer[id] = inputBuffer[id] + 5.0f;
	}
}

__kernel void multiplyByTwo(
	const __global float* inputBuffer,
	__global float* outputBuffer,
	size_t n)
{
	size_t id = get_global_id(0);
	
	if (id < n)
	{
		outputBuffer[id] = inputBuffer[id] * 2.0f;
	}
}

__kernel void subtractSeven(
	const __global float* inputBuffer,
	__global float* outputBuffer,
	size_t n)
{
	size_t id = get_global_id(0);
	
	if (id < n)
	{
		outputBuffer[id] = inputBuffer[id] - 7.0f;
	}
}