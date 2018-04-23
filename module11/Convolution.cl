//////////////////////////
// Convolution.cl       //
// Andrew Krepps        //
// Module 11 Assignment //
// 23 April 2018        //
//////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// \brief Perform 2D signal convolution for a given output signal value
/// 
/// Note: This assumes that all data is in row-major order, the filter mask is
/// square, and the input signal is the same size or larger than the output
/// signal (depending on the mask size) such that all input locations read are
/// valid and no zero-padding is required.
/// 
/// \param [in] input the input signal
/// \param [in] mask the convolution mask
/// \param [out] output the output signal
/// \param [in] inputWidth the width of the input signal
/// \param [in] maskWidth the width of the filter mask
///////////////////////////////////////////////////////////////////////////////
__kernel void convolve(
    const __global  float * const input,
    __constant float * const mask,
    __global  float * const output,
    const int inputWidth,
    const int maskWidth)
{
    // get output coordinates (work items and all data are in row-major order)
    const int outRow = get_global_id(0);
    const int outCol = get_global_id(1);
    
    // generate output sum using each input element's contribution
    float sum = 0.0f;
    for (int maskRow = 0; maskRow < maskWidth; maskRow++)
    {
        // pre-calculate start of the subset of the input row for this row of the mask
        const int inRowStart = (maskRow + outRow)*inputWidth + outCol;
        
        for (int maskCol = 0; maskCol < maskWidth; maskCol++)
        {
            sum += mask[maskRow*maskWidth + maskCol] * input[inRowStart + maskCol];
        }
    }
    
    // set output signal value
    output[outRow*get_global_size(0) + outCol] = sum;
}