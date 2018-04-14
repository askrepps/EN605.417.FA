//////////////////////////
// HelloWorld.cpp       //
// Andrew Krepps        //
// Module 10 Assignment //
// 16 April 2018        //
//////////////////////////

#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Constants
//
//const int ARRAY_SIZE = 1000;

#define NUM_KERNELS 5
#define NUM_ARRAYS 3

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;
    
    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }
    
    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }
    
    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;
    
    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }
    
    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }
    
    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }
    
    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }
    
    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;
    
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }
    
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }
    
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }
    
    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[NUM_ARRAYS],
                      float *a, float *b, size_t arraySize)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * arraySize, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * arraySize, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * arraySize, NULL, NULL);
    
    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }
    
    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernels[NUM_KERNELS],
             cl_mem memObjects[NUM_ARRAYS], float* a, float* b, float* result)
{
    for (int i = 0; i < NUM_ARRAYS; i++)
    {
        if (memObjects[i] != 0)
        {
            clReleaseMemObject(memObjects[i]);
        }
    }
    
    if (commandQueue != 0)
    {
        clReleaseCommandQueue(commandQueue);
    }
    
    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        if (kernels[i] != 0)
        {
            clReleaseKernel(kernels[i]);
        }
    }
    
    if (program != 0)
    {
        clReleaseProgram(program);
    }
    
    if (context != 0)
    {
        clReleaseContext(context);
    }
    
    free(a);
    free(b);
    free(result);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Check if two floating-point values are almost equal
/// 
/// \param [in] a the first value
/// \param [in] b the second value
/// \param [in] eps the relative error threshold (optional, default = 1e-6)
/// 
/// \returns true if the values are almost equal, false otherwise
///////////////////////////////////////////////////////////////////////////////
bool almostEq(float a, float b, float eps = 1e-6)
{
    // implementation taken from Boost test documentation
    return std::fabs(a - b)/std::fabs(a) <= eps
        && std::fabs(a - b)/std::fabs(b) <= eps;
}

// typedef validation function pointer so that it can be passed around more easily
typedef void (*ValidationFunc)(const float*, const float*, const float*, size_t);

///////////////////////////////////////////////////////////////////////////////
/// \brief Validate the results of the add operation and display an error
/// message for evey incorrect value
/// 
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [in] result the output array
/// \param [in] arraySize the size of the arrays
///////////////////////////////////////////////////////////////////////////////
void validateAdd(const float* a, const float* b, const float* result, size_t arraySize)
{
    for (size_t i = 0; i < arraySize; ++i)
    {
        if (!almostEq(a[i] + b[i], result[i]))
        {
            std::cerr << "Error: " << a[i] << " + " << b[i] << " != " << result[i] << std::endl;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Validate the results of the sub operation and display an error
/// message for evey incorrect value
/// 
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [in] result the output array
/// \param [in] arraySize the size of the arrays
///////////////////////////////////////////////////////////////////////////////
void validateSub(const float* a, const float* b, const float* result, size_t arraySize)
{
    for (size_t i = 0; i < arraySize; ++i)
    {
        if (!almostEq(a[i] - b[i], result[i]))
        {
            std::cerr << "Error: " << a[i] << " - " << b[i] << " != " << result[i] << std::endl;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Validate the results of the mult operation and display an error
/// message for evey incorrect value
/// 
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [in] result the output array
/// \param [in] arraySize the size of the arrays
///////////////////////////////////////////////////////////////////////////////
void validateMult(const float* a, const float* b, const float* result, size_t arraySize)
{
    for (size_t i = 0; i < arraySize; ++i)
    {
        if (!almostEq(a[i] * b[i], result[i]))
        {
            std::cerr << "Error: " << a[i] << " * " << b[i] << " != " << result[i] << std::endl;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Validate the results of the div operation and display an error
/// message for evey incorrect value
/// 
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [in] result the output array
/// \param [in] arraySize the size of the arrays
///////////////////////////////////////////////////////////////////////////////
void validateDiv(const float* a, const float* b, const float* result, size_t arraySize)
{
    for (size_t i = 0; i < arraySize; ++i)
    {
        if (!almostEq(a[i] / b[i], result[i]))
        {
            std::cerr << "Error: " << a[i] << " / " << b[i] << " != " << result[i] << std::endl;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Validate the results of the pow operation and display an error
/// message for evey incorrect value
/// 
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [in] result the output array
/// \param [in] arraySize the size of the arrays
///////////////////////////////////////////////////////////////////////////////
void validatePow(const float* a, const float* b, const float* result, size_t arraySize)
{
    for (size_t i = 0; i < arraySize; ++i)
    {
        if (!almostEq(std::pow(a[i], b[i]), result[i]))
        {
            std::cerr << "Error: " << a[i] << " ^ " << b[i] << " != " << result[i] << std::endl;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Run a kernel to completion
/// 
/// \param [in] commandQueue the command queue
/// \param [in] kernel the kernel
/// \param [in] arraySize the size of the arrays
///////////////////////////////////////////////////////////////////////////////
void runKernel(cl_command_queue commandQueue, cl_kernel kernel, size_t arraySize)
{
    // set up work size
    size_t globalWorkSize[1] = { arraySize };
    size_t localWorkSize[1] = { 1 };
    
    // launch kernel
    cl_int errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                           globalWorkSize, localWorkSize,
                                           0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queueing kernel for execution." << std::endl;
    }
    
    // wait for kernel to finish
    clFinish(commandQueue);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Run a kernel, capture the execution time, and validate the reuslts
/// 
/// \param [in] commandQueue the command queue
/// \param [in] kernel the kernel
/// \param [in] memObjects the device memory objects
/// \param [in] a the first input array
/// \param [in] b the second input array
/// \param [in/out] result the output array
/// \param [in] arraySize the size of the arrays
/// \param [in] validateFunc a pointer to the output validation function
/// 
/// \returns the execution time (not including data transfer) in ms
///////////////////////////////////////////////////////////////////////////////
float runTest(
    cl_command_queue commandQueue,
    cl_kernel kernel,
    cl_mem memObjects[NUM_ARRAYS],
    const float* a,
    const float* b,
    float* result,
    size_t arraySize,
    ValidationFunc validateFunc)
{
    // time kernel exection
    auto start = std::chrono::high_resolution_clock::now();
    runKernel(commandQueue, kernel, arraySize);
    auto stop = std::chrono::high_resolution_clock::now();
    
    // copy data to host
    cl_int errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                        0, arraySize * sizeof(float), result,
                                        0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
    }
    
    // validate results
    validateFunc(a, b, result, arraySize);
    
    // calculate execution time in ms
    std::chrono::duration<float> duration(stop - start);
    return duration.count()*1000.0f;
}

int main(int argc, char** argv)
{
    // configure run
    size_t arraySize = 1024;
    if (argc > 1)
    {
        arraySize = atoi(argv[1]);
    }
    
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernels[NUM_KERNELS] = { 0, 0, 0, 0, 0 };
    cl_mem memObjects[NUM_ARRAYS] = { 0, 0, 0 };
    cl_int errNum;
    
    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }
    
    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    size_t numBytes = arraySize*sizeof(float);
    float* result = (float*)malloc(numBytes);
    float* a = (float*)malloc(numBytes);
    float* b = (float*)malloc(numBytes);
    for (int i = 0; i < arraySize; i++)
    {
        a[i] = (float)((i + 1)*1e-9);
        b[i] = (float)((i + 1)*2e-9);
    }
    
    if (!CreateMemObjects(context, memObjects, a, b, arraySize))
    {
        Cleanup(context, commandQueue, program, kernels, memObjects, a, b, result);
        return 1;
    }
    
    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, program, kernels, memObjects, a, b, result);
        return 1;
    }
    
    // Create OpenCL program from HelloWorld.cl kernel source
    program = CreateProgram(context, device, "HelloWorld.cl");
    if (program == NULL)
    {
        Cleanup(context, commandQueue, program, kernels, memObjects, a, b, result);
        return 1;
    }
    
    // Create OpenCL kernels and test each one
    const char* kernelNames[NUM_KERNELS] = { "add", "sub", "mult", "div", "my_pow" };
    ValidationFunc validateFuncs[NUM_KERNELS] { validateAdd, validateSub, validateMult, validateDiv, validatePow };
    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        kernels[i] = clCreateKernel(program, kernelNames[i], NULL);
        if (kernels[i] == NULL)
        {
            std::cerr << "Failed to create " << kernelNames[i] << " kernel" << std::endl;
            Cleanup(context, commandQueue, program, kernels, memObjects, a, b, result);
            return 1;
        }
        
        errNum = clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &memObjects[0]);
        errNum |= clSetKernelArg(kernels[i], 1, sizeof(cl_mem), &memObjects[1]);
        errNum |= clSetKernelArg(kernels[i], 2, sizeof(cl_mem), &memObjects[2]);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Error setting " << kernelNames[i] << " kernel arguments." << std::endl;
            Cleanup(context, commandQueue, program, kernels, memObjects, a, b, result);
            return 1;
        }
        
        // one dummy execution to avoid startup cost
        if (i == 0)
        {
            runTest(commandQueue, kernels[i], memObjects, a, b, result, arraySize, validateFuncs[i]);
        }
        
        // run kernel test and display timing results
        float ms = runTest(commandQueue, kernels[i], memObjects, a, b, result, arraySize, validateFuncs[i]);
        std::cout << "Execution time for " << kernelNames[i] << ": " << ms << " ms" << std::endl;
    }
    
    Cleanup(context, commandQueue, program, kernels, memObjects, a, b, result);
    
    return 0;
}
