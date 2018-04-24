//////////////////////////
// Convolution.cpp      //
// Andrew Krepps        //
// Module 11 Assignment //
// 23 April 2018        //
//////////////////////////

#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Constants

// input signal (defined in row-major order)
const unsigned int INPUT_SIGNAL_HEIGHT = 49;
const unsigned int INPUT_SIGNAL_WIDTH  = 49;
cl_float inputSignal[INPUT_SIGNAL_HEIGHT][INPUT_SIGNAL_WIDTH];

// output signal (defined in row-major order)
const unsigned int OUTPUT_SIGNAL_HEIGHT = 43;
const unsigned int OUTPUT_SIGNAL_WIDTH  = 43;
cl_float outputSignal[OUTPUT_SIGNAL_HEIGHT][OUTPUT_SIGNAL_WIDTH];

// convolution mask (defined in row-major order)
const unsigned int MASK_HEIGHT = 7;
const unsigned int MASK_WIDTH  = 7;
cl_float mask[MASK_HEIGHT][MASK_WIDTH] =
{
    {0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f},
    {0.25f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.25f},
    {0.25f, 0.50f, 0.75f, 0.75f, 0.75f, 0.50f, 0.25f},
    {0.25f, 0.50f, 0.75f, 1.00f, 0.75f, 0.50f, 0.25f},
    {0.25f, 0.50f, 0.75f, 0.75f, 0.75f, 0.50f, 0.25f},
    {0.25f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.25f},
    {0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f}
};

///
// Function to check and handle OpenCL errors
inline void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

bool readInputSignal(const std::string& fileName)
{
    // open file
    std::ifstream fileStream(fileName);
    if (!fileStream.is_open()) {
        std::cerr << "Could not open input signal CSV file: '" << fileName << "'" << std::endl;
        return false;
    }
    
    // read each data row
    std::string line;
    unsigned int rowNumber = 0;
    while (rowNumber < INPUT_SIGNAL_HEIGHT && std::getline(fileStream, line)) {
        // parse each data row into tokens
        std::istringstream lineStream(line);
        std::string token;
        unsigned int colNumber = 0;
        while (colNumber < INPUT_SIGNAL_WIDTH && std::getline(lineStream, token, ',')) {
            inputSignal[rowNumber][colNumber] = std::stof(token);
            ++colNumber;
        }
    
        ++rowNumber;
    }
    
    // close file
    fileStream.close();
    
    return true;
}

void runKernel(cl_command_queue queue, cl_kernel kernel)
{
    const size_t globalWorkSize[2] = { OUTPUT_SIGNAL_HEIGHT, OUTPUT_SIGNAL_WIDTH };
    const size_t localWorkSize[2]  = { 1, 1 };
    
    // wueue the kernel up for execution across the array
    cl_int errNum = clEnqueueNDRangeKernel(
        queue, 
        kernel, 
        2, 
        NULL,
        globalWorkSize, 
        localWorkSize,
        0, 
        NULL, 
        NULL);
    checkErr(errNum, "clEnqueueNDRangeKernel");
    
    // wait for the kernel to finish
    clFinish(queue);
}

void runTest()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context = NULL;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem inputSignalBuffer;
    cl_mem outputSignalBuffer;
    cl_mem maskBuffer;
    
    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
    
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);
    
    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");
    
    // Iterate through the list of platforms until we find one that supports
    // a CPU device, otherwise fail with an error.
    deviceIDs = NULL;
    cl_uint i;
    for (i = 0; i < numPlatforms; i++)
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
    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");
    
    std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");
    
    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));
    
    const char * src = srcProg.c_str();
    size_t length = srcProg.length();
    
    // Create program from source
    program = clCreateProgramWithSource(
        context, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");
    
    // Build program
    errNum = clBuildProgram(
        program,
        numDevices,
        deviceIDs,
        NULL,
        NULL,
        NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
            buildLog, 
            NULL);
        
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        checkErr(errNum, "clBuildProgram");
    }
    
    // Create kernel object
    kernel = clCreateKernel(
        program,
        "convolve",
        &errNum);
    checkErr(errNum, "clCreateKernel");
    
    // Now allocate buffers
    inputSignalBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float) * INPUT_SIGNAL_HEIGHT * INPUT_SIGNAL_WIDTH,
        static_cast<void *>(inputSignal),
        &errNum);
    checkErr(errNum, "clCreateBuffer(inputSignal)");
    
    maskBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float) * MASK_HEIGHT * MASK_WIDTH,
        static_cast<void *>(mask),
        &errNum);
    checkErr(errNum, "clCreateBuffer(mask)");
    
    outputSignalBuffer = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_float) * OUTPUT_SIGNAL_HEIGHT * OUTPUT_SIGNAL_WIDTH,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer(outputSignal)");
    
    // Pick the first device and create command queue.
    queue = clCreateCommandQueue(
        context,
        deviceIDs[0],
        0,
        &errNum);
    checkErr(errNum, "clCreateCommandQueue");
    
    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &INPUT_SIGNAL_WIDTH);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &MASK_WIDTH);
    checkErr(errNum, "clSetKernelArg");
    
    // dummy execution to avoid startup performance hit
    runKernel(queue, kernel);
    
    // time kernel execution
    auto start = std::chrono::high_resolution_clock::now();
    runKernel(queue, kernel);
    auto stop = std::chrono::high_resolution_clock::now();
    
    float ms = std::chrono::duration<float>(stop - start).count()*1000.0f;
    std::cerr << "Execution time: " << ms << " ms" << std::endl;
    
    // copy results to host memory
    errNum = clEnqueueReadBuffer(
        queue, 
        outputSignalBuffer, 
        CL_TRUE,
        0, 
        sizeof(cl_float) * OUTPUT_SIGNAL_HEIGHT * OUTPUT_SIGNAL_WIDTH, 
        outputSignal,
        0, 
        NULL, 
        NULL);
    checkErr(errNum, "clEnqueueReadBuffer");
}

///
//  main() for Convoloution example
//
int main(int argc, char** argv)
{
    // configure run
    if (argc < 1) {
        std::cerr << "Input signal CSV must be provided!" << std::endl;
        return EXIT_FAILURE;
    }
    
    readInputSignal(argv[1]);
    runTest();
    
    return EXIT_SUCCESS;
}