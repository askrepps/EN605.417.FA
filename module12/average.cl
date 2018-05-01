//////////////////////////
// average.cl           //
// Andrew Krepps        //
// Module 12 Assignment //
// 30 April 2018        //
//////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// \brief Average the values in an input buffer and store the result in an
/// output buffer
/// 
/// \param [in] input the input buffer
/// \param [in] inputSize the number of elements in the input buffer
/// \param [out] output the output buffer
/// \param [in] outputIndex the output element location
///////////////////////////////////////////////////////////////////////////////
__kernel void average(
    __global const float* input,
    const unsigned int inputSize,
    __global float* output,
    const unsigned int outputIndex)
{
    float sum = 0.0f;
    for (unsigned int i = 0; i < inputSize; ++i) {
        sum += input[i];
    }
    
    output[outputIndex] = sum/inputSize;
}
