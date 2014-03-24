#ifndef UTIL_HEADER
#define UTIL_HEADER

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <string.h>
#include <stdlib.h> 
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "error.h"
#include "types.h"
#include "tic_toc.h"


#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

//#define INFO

/*
Create an OpenCL Context 
using first found device.
Return context.
*/
cl_context CreateContext()
{
    cl_int status;
    cl_uint numPlatforms, numDevices;
    cl_platform_id *PlatformIds;
    cl_context context = NULL;
#ifdef INFO
    size_t size;
#endif
    // Select te first OpenCL platform to run on                                                                                                                                              
    HANDLE_OPENCL_ERROR(clGetPlatformIDs(0, NULL, &numPlatforms));
    if (numPlatforms <= 0)
      {
	std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
      }
    PlatformIds = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);


    // Choose OpenCL platform to run on                                                                        
    HANDLE_OPENCL_ERROR(clGetPlatformIDs(numPlatforms, PlatformIds, NULL));

    int found_device = -1;
    for( unsigned int i = 0; i < numPlatforms; ++i)
      {

#ifdef INFO
        HANDLE_OPENCL_ERROR(clGetPlatformInfo(PlatformIds[i], CL_PLATFORM_NAME, 0, NULL, &size));
        char * name = (char *)malloc(sizeof(char) * size);
        HANDLE_OPENCL_ERROR(clGetPlatformInfo(PlatformIds[i], CL_PLATFORM_NAME, size, name, NULL));

	printf("Platform: \nName: %s\n", name);	
	
	HANDLE_OPENCL_ERROR(clGetPlatformInfo(PlatformIds[i], CL_PLATFORM_VERSION, 0, NULL, &size));
	char * version = (char *)malloc(sizeof(char) * size);
	HANDLE_OPENCL_ERROR(clGetPlatformInfo(PlatformIds[i], CL_PLATFORM_VERSION, size, version, NULL));	
	printf("Version: %s\n", version);

#endif

	clGetDeviceIDs(PlatformIds[i], DEVICE_TYPE, 0, NULL, &numDevices);
	if( numDevices > 0 )
	  {
#ifdef INFO
	    printf("Device found\n\n");
#endif
	    found_device = i;
	    break;
	  }
#ifdef INFO
	else printf("No device found\n\n");
#endif
      }

    if( found_device == -1 )
      {
	printf("No device found on any platform\n");
	return NULL;
      }

    // Create an OpenCL context on the platform
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)PlatformIds[found_device],
        0
    };
    
    //Create an OpenCL context from a device type that identifies the specific device to use.
    context = clCreateContextFromType(contextProperties, DEVICE_TYPE,
                                      NULL, NULL, &status);
    HANDLE_OPENCL_ERROR(status);

    return context;
}


/*
Create a command queue for selected device.
Set profiling_info parameter to 1 if specific information 
about device and kernel is needed. 
 */
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device, int profiling_info)
{
    cl_int status;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

#ifdef INFO
    cl_uint freq, units;
    cl_ulong mem, mem_l;
    size_t size;
    size_t max_items[3];
#endif

    // First get the size of the devices (related to this context) buffer
    HANDLE_OPENCL_ERROR(clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize));
    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    HANDLE_OPENCL_ERROR(clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL));
 
    if(profiling_info == 1)
      commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
    else
      commandQueue = clCreateCommandQueue(context, devices[0], 0, &status);
    HANDLE_OPENCL_ERROR(status);


#ifdef INFO

    HANDLE_OPENCL_ERROR(clGetDeviceInfo(devices[0], CL_DEVICE_VENDOR, 0, NULL, &size));
    char * vendor = (char *)malloc(sizeof(char) * size);
    HANDLE_OPENCL_ERROR(clGetDeviceInfo(devices[0], CL_DEVICE_VENDOR, size, vendor, NULL));

    HANDLE_OPENCL_ERROR(clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &size));
    char * name = (char *)malloc(sizeof(char) * size);
    HANDLE_OPENCL_ERROR(clGetDeviceInfo(devices[0], CL_DEVICE_NAME, size, name, NULL));

    HANDLE_OPENCL_ERROR(clGetDeviceInfo(devices[0], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &freq, NULL));
    HANDLE_OPENCL_ERROR(clGetDeviceInfo(devices[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem, NULL)); 
    HANDLE_OPENCL_ERROR(clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_l, NULL));     
    HANDLE_OPENCL_ERROR(clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &units, NULL));
    HANDLE_OPENCL_ERROR(clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, max_items, NULL));

    /***************************/
    printf("Device:\n");
    printf("Name: %s\n", name);
    printf("Vendor: %s\n", vendor);
    printf("Max. clock frequency: %u\n", freq);
    printf("Global memory size: %lu\n", mem);
    printf("Local memory size: %lu\n", mem_l);
    printf("Max. compute units: %u\n", units);
    printf("Max. work item sizes: (%lu, %lu, %lu)\n\n", max_items[0], max_items[1], max_items[2]);

    /***************************/
    
#endif

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

/*
 Create an OpenCL program from the kernel source file
*/
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName1, const char* fileName2)
{
    cl_int status;
    cl_program program;

    std::ifstream kernelFile(fileName1, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName1 << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr[2];
    srcStdStr[0] = oss.str();

    std::ifstream kernelFile2(fileName2, std::ios::in);
    if (!kernelFile2.is_open())
      {
	std::cerr << "Failed to open file for reading: " << fileName2 << std::endl;
        return NULL;
      }

  
    std::ostringstream oss2;
    oss2 << kernelFile2.rdbuf();
    srcStdStr[1] = oss2.str();


    const char *srcStr[2];
    srcStr[0] = srcStdStr[0].c_str();
    srcStr[1] = srcStdStr[1].c_str();
    program = clCreateProgramWithSource(context, 2,
                                        (const char**)srcStr,
                                        NULL, &status);
    HANDLE_OPENCL_ERROR(status);

    status = clBuildProgram(program, 0, NULL, "-w", NULL, NULL);
    /*
    if(status == CL_OUT_OF_HOST_MEMORY)
      printf("CL_OUT_OF_HOST_MEMORY");
    if(status == CL_BUILD_PROGRAM_FAILURE)
      printf("CL_BUILD_PROGRAM_FAILURE");
    if(status == CL_INVALID_DEVICE)
      printf("CL_INVALID_DEVICE");
    if(status == CL_INVALID_BINARY)
      printf("CL_INVALID_BINARY");
    if(status == CL_INVALID_OPERATION)
      printf("CL_INVALID_OPERATION");
    */

    if (status != CL_SUCCESS)
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


/*
  Cleanup any created OpenCL resources
*/
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel)
{
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}


#endif
