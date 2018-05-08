//////////////////////////
// pipline.cpp          //
// Andrew Krepps        //
// Module 13 Assignment //
// 7 May 2018           //
//////////////////////////

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define NUM_STAGES 3
#define NUM_EVENTS 4

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
/// \param [in] sourcePath the path to the kernel source file
/// \param [in] context the context
/// \param [in] deviceID the device ID
/// \param [out] program the built program
///////////////////////////////////////////////////////////////////////////////
void initProgram(const std::string& sourcePath, cl_context context, cl_device_id deviceID, cl_program* program)
{
    std::ifstream srcFile(sourcePath);
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, (std::string("reading ") + sourcePath).c_str());
    
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
/// \brief Init device buffers
/// 
/// \param [in] context the context
/// \param [in] inputSize the input host data size
/// \param [out] buffers the device buffers
///////////////////////////////////////////////////////////////////////////////
void initBuffers(cl_context context, size_t inputSize, std::vector<cl_mem>& buffers)
{
    // create input buffer
    cl_int errNum;
    cl_mem buffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizeof(float) * inputSize,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");
    buffers.push_back(buffer);
    
    // create output buffer for each stage in the pipeline
    for (unsigned int i = 0; i < NUM_STAGES; ++i)
    {
        buffer = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(float) * inputSize,
            NULL,
            &errNum);
        checkErr(errNum, "clCreateBuffer");
        buffers.push_back(buffer);
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Init kernels
/// 
/// \param [in] program the program
/// \param [in] buffers the device buffers
/// \param [in] inputSize the input host data size
/// \param [out] kernels the kernels
///////////////////////////////////////////////////////////////////////////////
void initKernels(cl_program program, const std::vector<cl_mem>& buffers, size_t inputSize, std::vector<cl_kernel>& kernels)
{
    const char* kernelNames[NUM_STAGES] = {"addFive", "multiplyByTwo", "subtractSeven"};
    
    cl_int errNum;
    for (unsigned int i = 0; i < NUM_STAGES; ++i)
    {
        cl_kernel kernel = clCreateKernel(
            program,
            kernelNames[i],
            &errNum);
        checkErr(errNum, "clCreateKernel");
        
        // set up kernel arguments (there are NUM_STAGES+1 buffers,
        // and the output of one is the input of the next)
        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
        errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buffers[i+1]);
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &inputSize);
        checkErr(errNum, "clSetKernelArg");
        
        kernels.push_back(kernel);
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Init command queues
/// 
/// \param [in] context the context
/// \param [in] deviceID the device ID
/// \param [out] queues the command queues
///////////////////////////////////////////////////////////////////////////////
void initQueues(cl_context context, cl_device_id deviceID, std::vector<cl_command_queue>& queues)
{
    cl_int errNum;
    for (unsigned int i = 0; i < NUM_STAGES; ++i)
    {
        cl_command_queue queue = 
            clCreateCommandQueue(
                context,
                deviceID,
                0,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");
        queues.push_back(queue);
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Queue up commands such that as soon as inputEvent is triggered
/// the entire pipline will be executed with each command queue waiting for
/// the previous stage to finish, with a final output event being triggered
/// when the output data is ready to copy to the host
/// 
/// \param [in] inputEvent the input event that will trigger execution
/// \param [in] inputData the pointer to host input data (not yet initialized)
/// \param [in] inputSize the host input data size
/// \param [in] queues the command queues (1 per pipeline stage)
/// \param [in] inputBuffer the input device buffer
/// \param [in] kernels the computation kernels (1 per pipeline stage)
/// \param [out] the output event that will be triggered by the final stage
///////////////////////////////////////////////////////////////////////////////
void queueAllPipelineCommands(
    const cl_event& inputEvent,
    const float* inputData,
    size_t inputSize,
    const std::vector<cl_command_queue>& queues,
    const cl_mem& inputBuffer,
    const std::vector<cl_kernel>& kernels,
    cl_event& outputEvent)
{
    cl_int errNum;
    cl_event events[NUM_EVENTS];
    const size_t globalWorkSize = inputSize;
    const size_t localWorkSize = 1;
    
    // queue command to read input data (that will wait until input data is available)
    errNum = clEnqueueWriteBuffer(
        queues[0],
        inputBuffer,
        CL_FALSE,
        0,
        sizeof(float)*inputSize,
        inputData,
        1,
        &inputEvent,  // don't copy data until input event is triggered
        &events[0]);
    checkErr(errNum, "clEnqueueWriteBuffer");
    
    // create cascading commands on each queue that wait for the previous stage to finish
    for (unsigned int i = 1; i <= NUM_STAGES; ++i)
    {
        errNum = clEnqueueNDRangeKernel(
            queues[i-1],
            kernels[i-1],
            1,
            NULL,
            &globalWorkSize,
            &localWorkSize,
            1,
            &events[i-1],
            &events[i]);
        checkErr(errNum, "clEnqueueNDRangeKernel");
    }
    
    // save last event to see when output is ready
    outputEvent = events[NUM_EVENTS - 1];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Copy output data back to the host
/// 
/// \param [in] queue the command queue
/// \param [in] buffer the output device buffer
/// \param [out] outputData the host output data
/// \param [in] outputSize the host output data size
///////////////////////////////////////////////////////////////////////////////
void getOutputData(cl_command_queue queue, cl_mem buffer, float* outputData, size_t outputSize)
{
    // copy output data to host
    clEnqueueReadBuffer(
        queue,
        buffer,
        CL_TRUE,
        0,
        sizeof(float)*outputSize,
        (void*)outputData,
        0,
        NULL,
        NULL);
}

int main(int argc, char** argv)
{
    // configure run
    const size_t inputSize = 5;
    float inputData[5];
    float outputData[5];
    
    // initialize components of OpenCL pipeline
    cl_int errNum;
    cl_device_id deviceID;
    cl_context context;
    cl_program program;
    std::vector<cl_mem> buffers;
    std::vector<cl_command_queue> queues;
    std::vector<cl_kernel> kernels;
    cl_event inputEvent;
    cl_event outputEvent;
    
    initDeviceAndContext(&deviceID, &context);
    initProgram("pipeline.cl", context, deviceID, &program);
    initBuffers(context, inputSize, buffers);
    initKernels(program, buffers, inputSize, kernels);
    initQueues(context, deviceID, queues);
    
    // set up command pipeline (execution will wait for input event to be triggered)
    inputEvent = clCreateUserEvent(context, &errNum);
    checkErr(errNum, "clCreateUserEvent");
    queueAllPipelineCommands(inputEvent, inputData, 5, queues, buffers[0], kernels, outputEvent);
    
    // retrieve input
    inputData[0] = 1.0f;
    inputData[1] = 2.0f;
    inputData[2] = 3.0f;
    inputData[3] = 4.0f;
    inputData[4] = 5.0f;
    
    // time execution (input event will trigger pipeline execution)
    auto start = std::chrono::high_resolution_clock::now();
    errNum = clSetUserEventStatus(inputEvent, CL_COMPLETE);
    checkErr(errNum, "clSetUserEventStatus");
    
    // output event means that pipline is finished and output is ready
    errNum = clWaitForEvents(1, &outputEvent);
    auto stop = std::chrono::high_resolution_clock::now();
    
    float ms = std::chrono::duration<float>(stop - start).count()*1000.0f;
    std::cout << "Execution time: " << ms << " ms" << std::endl;
    
    // display output data
    getOutputData(queues[0], buffers[NUM_STAGES], outputData, 5);
    std::cout << "Output:" << std::endl;
    for (unsigned int i = 0; i < 5; ++i)
    {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;
    
    //delete [] inputData;
    //delete [] outputData;
    
    return EXIT_SUCCESS;
}