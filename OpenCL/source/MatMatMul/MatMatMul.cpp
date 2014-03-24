/*
OpenCL
Dense matrix matrix multiplication: C = A*B
*/


#include "../../common/util.h"
#include "../../matrix/MatrixParams.h"


#define PROFILE 1
#define PRINT_INFO 1

int WORK_GROUP_SIZE  = 16; // square block with WORK_GROUP_SIZE side


int main(int argc, char** argv)
{
    cl_context       context      = 0;
    cl_command_queue commandQueue = 0;
    cl_program       program      = 0;
    cl_device_id     device       = 0;
    cl_kernel        kernel       = 0;
    cl_int           status;
   
    int profiling_info = 0;
    cl_event myEvent, myEvent2;

    if( argc < 3 || argc > 4 )
      {
	printf("Usage: %s matrix1_file matrix2_file [ref_matrix_file]\n", argv[0]);
	return EXIT_FAILURE;
      }
    
    int verify = (argc == 4 ? 1 : 0);
    int dim = atoi(argv[1]);
   
    char Afilename[50];
    char Bfilename[50];
    char reffilename[50];
    
    strcpy(Afilename, argv[1]);
    strcpy(Bfilename, argv[2]);
    if(verify)
      strcpy(reffilename, argv[3]);
    
    
#ifdef PROFILE
    cl_ulong startTime, endTime, startTime2, endTime2;
    cl_ulong kernelExecTimeNs, readFromGpuTime;
    profiling_info = 1;
#endif

    char filename[]   = "../../kernels/MatMatMul_kernel.cl";
    char filename2[] = "../../common/types_kernel.h";

    /*  READING DATA FROM FILE  */

    struct MatrixParams paramsA, paramsB, paramsC;
    real *A;
    real *B;
    real *C;

    
    std::ifstream Afile;
    Afile.open (Afilename, std::ios::in);
    if (!Afile.is_open())
      {
	printf("Error: cannot open file\n");
	return EXIT_FAILURE;
      }
    
    Afile >> paramsA.NRows;
    Afile >> paramsA.NCols;
    
    HANDLE_ALLOC_ERROR(A = (real*)malloc(paramsA.NRows*paramsA.NCols*sizeof(real)));
  
    for( int i = 0; i < paramsA.NRows; i++)
      for( int j = 0; j < paramsA.NCols; j++)
	Afile >> A[i*paramsA.NCols+j];
    
    Afile.close();

    
    std::ifstream Bfile;
    Bfile.open (Bfilename, std::ios::in);
    if (!Bfile.is_open())
      {
	printf("Error: cannot open file\n");
	return EXIT_FAILURE;
      }
    
    Bfile >> paramsB.NRows;
    Bfile >> paramsB.NCols;
    
    assert(paramsA.NCols == paramsB.NRows);

    HANDLE_ALLOC_ERROR(B = (real*)malloc(paramsB.NRows*paramsB.NCols*sizeof(real)));
  
    for( int i = 0; i < paramsB.NRows; i++)
      for( int j = 0; j < paramsB.NCols; j++)
	Bfile >> B[i*paramsB.NCols+j];

    Bfile.close();

    paramsC.NRows = paramsA.NRows;
    paramsC.NCols = paramsB.NCols;
    HANDLE_ALLOC_ERROR(C = (real*)malloc(paramsA.NRows*paramsB.NCols*sizeof(real)));


    TIME start = tic();
    TIME init = tic();

    // Create an OpenCL context
    context = CreateContext();
    if(context == NULL)
      {
	std::cerr << "Failed to create OpenCL context." << std::endl;
	Cleanup(context, commandQueue, program, kernel);
	return EXIT_FAILURE;
      }

    // Create a command queue
    commandQueue = CreateCommandQueue(context, &device, profiling_info);
    if(commandQueue == NULL)
      {
	std::cerr << "Failed to create OpenCL command queue." << std::endl;
	Cleanup(context, commandQueue, program, kernel);
	return EXIT_FAILURE;
      }

    program = CreateProgram(context, device, filename, filename2);
    if (program == NULL)
      {
	Cleanup(context, commandQueue, program, kernel);
	return 1;
      }
    
    // Create OpenCL kernel
    kernel = clCreateKernel(program, "MatMatMul", NULL);
    if(kernel == NULL)
      {
	std::cerr << "Failed to create kernel." << std::endl;
	Cleanup(context, commandQueue, program, kernel);
	return EXIT_FAILURE;
      }
    
    printf("Initialization time: %lf\n", toc(init));

    /*     QUERYING DEVICE INFO     */
    
    size_t kernelWorkGroupSize; // maximum work-group size that can be used to execute a kernel
    size_t sizeOfWarp;          // the preferred multiple of workgroup size for launch
    cl_ulong localMemSize;      // the amount of local memory in bytes being used by a kernel
    cl_ulong privateMemSize;    // the minimum amount of private memory, in bytes, used by each workitem in the kernel. 

    HANDLE_OPENCL_ERROR(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelWorkGroupSize, NULL));
    HANDLE_OPENCL_ERROR(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &sizeOfWarp, NULL));

#ifdef PRINT_INFO    
    HANDLE_OPENCL_ERROR(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, NULL));
    HANDLE_OPENCL_ERROR(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &privateMemSize, NULL));
#endif

#ifdef PRINT_INFO 
    printf("------------  Some info: --------------\n");
    printf("kernelWorkGroupSize = %lu \n", kernelWorkGroupSize);
    printf("sizeOfWarp          = %lu \n", sizeOfWarp);
    printf("localMemSize        = %lu \n", localMemSize);
    printf("privateMemSize      = %lu \n", privateMemSize);
    printf("------------------------ --------------\n");
#endif

    if( WORK_GROUP_SIZE > kernelWorkGroupSize )
      {
	printf("Error: wrong work group size\n");
	return EXIT_FAILURE;
      }
    
    size_t localWorkSize[2]  = {WORK_GROUP_SIZE, WORK_GROUP_SIZE};
    size_t globalWorkSize[2] = {((paramsB.NCols-1)/WORK_GROUP_SIZE+1)*WORK_GROUP_SIZE, ((paramsA.NRows-1)/WORK_GROUP_SIZE+1)*WORK_GROUP_SIZE};
 

    TIME t = tic();

    cl_mem DEV_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				  sizeof(real)*paramsA.NRows*paramsA.NCols, A, &status);
    HANDLE_OPENCL_ERROR(status);
    
    cl_mem DEV_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				  sizeof(real)*paramsB.NRows*paramsB.NCols, B, &status);
    HANDLE_OPENCL_ERROR(status);
 
    cl_mem DEV_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				  sizeof(real)*paramsA.NRows*paramsB.NCols, NULL, &status);
    HANDLE_OPENCL_ERROR(status);

    int params_dim = 3;
    int params[] = {paramsA.NRows, paramsA.NCols, paramsB.NCols};
    cl_mem DEV_params = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(int)*params_dim, params, &status);
    HANDLE_OPENCL_ERROR(status);


    int n = 0;
    status |= clSetKernelArg(kernel, n++, sizeof(cl_mem), (void*)&DEV_A);
    status |= clSetKernelArg(kernel, n++, sizeof(cl_mem), (void*)&DEV_B);
    status |= clSetKernelArg(kernel, n++, sizeof(cl_mem), (void*)&DEV_C);
    status |= clSetKernelArg(kernel, n++, sizeof(cl_mem), (void*)&DEV_params);
    status |= clSetKernelArg(kernel, n++, WORK_GROUP_SIZE*WORK_GROUP_SIZE * sizeof(real),  NULL);
    status |= clSetKernelArg(kernel, n++, WORK_GROUP_SIZE*WORK_GROUP_SIZE * sizeof(real),  NULL);
    HANDLE_OPENCL_ERROR(status);



    printf("Transfer from CPU to GPU time: %lf\n", toc(t));



    // Queue the kernel 
    HANDLE_OPENCL_ERROR(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
					       globalWorkSize, localWorkSize,
					       0, NULL, &myEvent));


    clFinish(commandQueue);


    // Read the output buffer back to the Host
    HANDLE_OPENCL_ERROR(clEnqueueReadBuffer(commandQueue, DEV_C, CL_TRUE,
					    0, paramsA.NRows*paramsB.NCols*sizeof(real), C,
    					    0, NULL, &myEvent2));
    
 
    clFinish(commandQueue); // wait for all events to finish

    double elapsed_time = toc(start);



    /*  CHECK RESULTS   */
    if( verify )
      {
	std::ifstream reffile;
	reffile.open (reffilename, std::ios::in);
	if (!reffile.is_open())
	  {
	    printf("Error: cannot open file\n");
	    return EXIT_FAILURE;
	  }
	
	real *Ref;
	MatrixParams paramsRef;
	reffile >> paramsRef.NRows;
	reffile >> paramsRef.NCols;
	
	assert(paramsRef.NRows == paramsC.NRows);
	assert(paramsRef.NCols == paramsC.NCols);

	HANDLE_ALLOC_ERROR(Ref = (real*)malloc(paramsRef.NRows*paramsRef.NCols*sizeof(real)));

	for( int i = 0; i < paramsRef.NRows; i++)
	  for( int j = 0; j < paramsRef.NCols; j++)
	    reffile >> Ref[i*paramsRef.NCols+j];

	reffile.close();

	for( int i = 0; i < paramsRef.NRows*paramsRef.NCols; i++)
	  {
	    //std::cout << i <<": " <<  Ref[i] << "=="  << C[i] << ",   ";
	    assert(abs(C[i] - Ref[i]) < TOL);
	  }

	std::cout << "Verified..." << std::endl;
      }


#ifdef PROFILE
    clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_START, 
			    sizeof(cl_ulong), &startTime, NULL);
    clGetEventProfilingInfo(myEvent, CL_PROFILING_COMMAND_END,
			    sizeof(cl_ulong), &endTime, NULL);
    clGetEventProfilingInfo(myEvent2, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &startTime2, NULL);
    clGetEventProfilingInfo(myEvent2, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &endTime2, NULL);

    kernelExecTimeNs = endTime-startTime;
    readFromGpuTime = endTime2-startTime2;

    printf("Transfer from GPU to CPU: %lf\n", (double)readFromGpuTime/1000000000.0);
    printf("Kernel execution time: %lf\n", (double)kernelExecTimeNs/1000000000.0);
#endif
    printf("Total execution time: %lf\n", elapsed_time);

    Cleanup(context, commandQueue, program, kernel);
    free(A);
    free(B);
    free(C);
    clReleaseMemObject(DEV_A); 
    clReleaseMemObject(DEV_B); 
    clReleaseMemObject(DEV_C); 
    clReleaseMemObject(DEV_params); 

    return EXIT_SUCCESS;
}

