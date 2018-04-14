//////////////////////////
// HelloWorld.cl        //
// Andrew Krepps        //
// Module 10 Assignment //
// 16 April 2018        //
//////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// \brief Perform element-wise addition on two arrays
/// 
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [out] result the output array
///////////////////////////////////////////////////////////////////////////////
__kernel void add(
	__global const float* a,
	__global const float* b,
	__global float* result)
{
	int gid = get_global_id(0);
	
	result[gid] = a[gid] + b[gid];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Perform element-wise subtraction on two arrays
/// 
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [out] result the output array
///////////////////////////////////////////////////////////////////////////////
__kernel void sub(
	__global const float* a,
	__global const float* b,
	__global float* result)
{
	int gid = get_global_id(0);
	
	result[gid] = a[gid] - b[gid];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Perform element-wise multiplication on two arrays
/// 
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [out] result the output array
///////////////////////////////////////////////////////////////////////////////
__kernel void mult(
	__global const float* a,
	__global const float* b,
	__global float* result)
{
	int gid = get_global_id(0);
	
	result[gid] = a[gid] * b[gid];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Perform element-wise division on two arrays
/// 
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [out] result the output array
///////////////////////////////////////////////////////////////////////////////
__kernel void div(
	__global const float* a,
	__global const float* b,
	__global float* result)
{
	int gid = get_global_id(0);
	
	result[gid] = a[gid] / b[gid];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Perform element-wise exponentiation on two arrays
/// 
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [out] result the output array
///////////////////////////////////////////////////////////////////////////////
__kernel void my_pow(
	__global const float* a,
	__global const float* b,
	__global float* result)
{
	int gid = get_global_id(0);
	
	result[gid] = pow(a[gid], b[gid]);
}