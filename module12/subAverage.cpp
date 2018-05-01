//////////////////////////
// subAverage.cpp       //
// Andrew Krepps        //
// Module 12 Assignment //
// 30 April 2018        //
//////////////////////////

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define NUM_BUFFER_ELEMENTS 16
#define NUM_SUBBUFFER_ELEMENTS 4

#define NUM_SUBBUFFERS 13

// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Init OpenCL GPU device and create context
/// 
/// \param [out] deviceID the device ID
/// \param [out] context the context
///////////////////////////////////////////////////////////////////////////////
void initDeviceAndContext(cl_device_id* deviceID, cl_context* context)
{
    cl_uint numPlatforms;
    cl_uint numDevices;
    
    // First, select an OpenCL platform to run on.
    cl_int errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
    
    cl_platform_id* platformIDs = (cl_platform_id *)alloca(sizeof(cl_platform_id)*numPlatforms);
    
    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");
    
    // Iterate through the list of platforms until we find one that supports
    // a CPU device, otherwise fail with an error.
    cl_uint i;
    cl_device_id* deviceIDs = NULL;
    for (i = 0; i < numPlatforms; ++i)
    {
        errNum = clGetDeviceIDs(
            platformIDs[i],
            CL_DEVICE_TYPE_GPU,
            0,
            NULL,
            &numDevices);
        if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
        {
            checkErr(errNum, "clGetDeviceIDs");
        }
        else if (numDevices > 0) 
        {
            deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
            errNum = clGetDeviceIDs(
                platformIDs[i],
                CL_DEVICE_TYPE_GPU,
                numDevices,
                &deviceIDs[0],
                NULL);
            checkErr(errNum, "clGetDeviceIDs");
            
            *deviceID = deviceIDs[0];
            break;
       }
    }
    
    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    
    *context = clCreateContext(
        contextProperties,
        numDevices,
        deviceIDs,
        NULL,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateContext");
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Init OpenCL program from kernel source code
/// 
/// \param [in] context the context
/// \param [in] deviceID the device ID
/// \param [out] program the built program
///////////////////////////////////////////////////////////////////////////////
void initProgram(cl_context context, cl_device_id deviceID, cl_program* program)
{
    std::ifstream srcFile("average.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading average.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char* src = srcProg.c_str();
    size_t length = srcProg.length();

    // Create program from source
    cl_int errNum;
    *program = clCreateProgramWithSource(
        context,
        1,
        &src,
        &length,
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(
        *program,
        1,
        &deviceID,
        NULL,
        NULL,
        NULL);
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            *program,
            deviceID,
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog),
            buildLog,
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Init host input data, buffers, and sub-buffers
/// 
/// \param [in] context the context
/// \param [out] inputData the input host data
/// \param [out] inputBuffer the input buffer
/// \param [out] outputBuffer the output buffer
/// \param [out] subBuffers the input sub-buffers
///////////////////////////////////////////////////////////////////////////////
void initBuffers(cl_context context, float* inputData, cl_mem* inputBuffer, cl_mem* outputBuffer, std::vector<cl_mem>& subBuffers)
{
    // initialize input host data
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; ++i)
    {
        inputData[i] = (float)i;
    }

    // create a single buffer to cover all the input data
    cl_int errNum;
    *inputBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * NUM_BUFFER_ELEMENTS,
        inputData,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // create a buffer to hold the output data
    *outputBuffer = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(float) * NUM_SUBBUFFERS,  // 1 output element per sub-buffer
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");
    
    // create sub-buffers with different views of the input data
    for (unsigned int i = 0; i < NUM_SUBBUFFERS; i++)
    {
        cl_buffer_region region = 
            {
                i*sizeof(float),
                NUM_SUBBUFFER_ELEMENTS*sizeof(float)
            };
        cl_mem buffer = clCreateSubBuffer(
            *inputBuffer,
            CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        subBuffers.push_back(buffer);
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Init command queues and set up kernels
/// 
/// \param [in] context the context
/// \param [in] deviceID the device ID
/// \param [in] program the program
/// \param [in] subBuffers the input sub-buffers
/// \param [in] outputBuffer the ouptut buffer
/// \param [out] queues the command queues
/// \param [out] kernels the kernels
///////////////////////////////////////////////////////////////////////////////
void initQueuesAndKernels(
    cl_context context,
    cl_device_id deviceID,
    cl_program program,
    const std::vector<cl_mem>& subBuffers,
    cl_mem outputBuffer,
    std::vector<cl_command_queue>& queues,
    std::vector<cl_kernel>& kernels)
{
    cl_int errNum;
    
    // create command queues
    const unsigned int inputSize = NUM_SUBBUFFER_ELEMENTS;
    for (unsigned int i = 0; i < NUM_SUBBUFFERS; ++i)
    {
        cl_command_queue queue = 
            clCreateCommandQueue(
                context,
                deviceID,
                0,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);

        cl_kernel kernel = clCreateKernel(
            program,
            "average",
            &errNum);
        checkErr(errNum, "clCreateKernel(average)");

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&subBuffers[i]);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &inputSize);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputBuffer);
        errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &i);
        checkErr(errNum, "clSetKernelArg(average)");

        kernels.push_back(kernel);
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Run all kernels to completion
/// 
/// \param [in] queues the command queues
/// \param [in] kernels the kernels
///////////////////////////////////////////////////////////////////////////////
void runAllKernels(const std::vector<cl_command_queue>& queues, const std::vector<cl_kernel>& kernels)
{
    // call kernel for each sub-buffer
    for (unsigned int i = 0; i < NUM_SUBBUFFERS; ++i)
    {
        size_t gWI = 1;

        cl_int errNum = clEnqueueNDRangeKernel(
            queues[i],
            kernels[i],
            1,
            NULL,
            (const size_t*)&gWI,
            (const size_t*)NULL,
            0,
            0,
            NULL);
        checkErr(errNum, "clEnqueueNDRangeKernel");
    }
    
    // wait for each queue to finish
    for (unsigned int i = 0; i < NUM_SUBBUFFERS; ++i)
    {
        clFinish(queues[i]);
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Copy output data back to the host
/// 
/// \param [in] queue the command queue
/// \param [in] outputBuffer the output buffer
/// \param [out] outputData the host output data
///////////////////////////////////////////////////////////////////////////////
void getOutputData(cl_command_queue queue, cl_mem outputBuffer, float* outputData)
{
    // copy output data to host
    clEnqueueReadBuffer(
        queue,
        outputBuffer,
        CL_TRUE,
        0,
        sizeof(float)*NUM_SUBBUFFERS,
        (void*)outputData,
        0,
        NULL,
        NULL);
}

int main(int argc, char** argv)
{
    cl_device_id deviceID;
    cl_context context;
    cl_program program;
    cl_mem inputBuffer;
    cl_mem outputBuffer;
    std::vector<cl_mem> subBuffers;
    std::vector<cl_command_queue> queues;
    std::vector<cl_kernel> kernels;
    
    float* inputData = new float[NUM_BUFFER_ELEMENTS];
    float* outputData = new float[NUM_SUBBUFFERS];

    // initialize OpenCL platform/device/context and prepare memory buffers/sub-buffers and kernels for execution
    initDeviceAndContext(&deviceID, &context);
    initProgram(context, deviceID, &program);
    initBuffers(context, inputData, &inputBuffer, &outputBuffer, subBuffers);
    initQueuesAndKernels(context, deviceID, program, subBuffers, outputBuffer, queues, kernels);
    
    // time execution of all kernels
    auto start = std::chrono::high_resolution_clock::now();
    runAllKernels(queues, kernels);
    auto stop = std::chrono::high_resolution_clock::now();    
    
    float ms = std::chrono::duration<float>(stop - start).count()*1000.0f;
    std::cout << "Execution time: " << ms << " ms" << std::endl;
    
    // display output data
    getOutputData(queues[0], outputBuffer, outputData);
    
    std::cout << "Averaged output:" << std::endl;
    for (unsigned int i = 0; i < NUM_SUBBUFFERS; ++i)
    {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;
    
    delete [] inputData;
    delete [] outputData;
    
    return EXIT_SUCCESS;
}
