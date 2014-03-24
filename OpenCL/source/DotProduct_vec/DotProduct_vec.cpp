#include "../../common/util.h"

#define PROFILE 1
//#define PRINT_INFO 1

#define WORK_GROUP_SIZE 256
#define MAX_NUM_WORK_GROUPS 100

int main(int argc, char** argv)
{
  cl_context       context      = 0;
  cl_command_queue commandQueue = 0;
  cl_program       program      = 0;
  cl_device_id     device       = 0;
  cl_kernel        kernel       = 0;
  cl_int           status;

  char filename[]   = "../../kernels/DotProduct_vec_kernel.cl";
  char filename2[] = "../../common/types_kernel.h";

  int profiling_info = 0;
  cl_event myEvent, myEvent2;
  if( argc != 3 )
    {
      printf("Usage: %s vector_file1 vector_file2 \n", argv[0]);
      return EXIT_FAILURE;
    }

  char xfilename[50];
  char yfilename[50];
  
  strcpy(xfilename, argv[1]);
  strcpy(yfilename, argv[2]);

#ifdef PROFILE
  cl_ulong startTime, endTime, startTime2, endTime2;
  cl_ulong kernelExecTimeNs, readFromGpuTime;
  profiling_info = 1;
#endif
  
 /*  READING DATA FROM FILE  */
  
  real *x;
  real *y;
  int N, M, N4;

  std::ifstream xfile;
  xfile.open (xfilename, std::ios::in);
  if (!xfile.is_open())
    {
      printf("Error: cannot open file\n");
      return EXIT_FAILURE;
    }
  
    xfile >> N;

    // it must be N%4 == 0
    N4 = ((N & (4-1)) == 0) ? N : N+(4-(N&3));
    
    HANDLE_ALLOC_ERROR(x = (real*)malloc(N4*sizeof(real)));
  
    for( int i = 0; i < N; i++)
 	xfile >> x[i];
    for(int i = N; i < N4; ++i)
      x[i] = 0;
    
    xfile.close();
    
    std::ifstream yfile;
    yfile.open (yfilename, std::ios::in);
    if (!yfile.is_open())
      {
	printf("Error: cannot open file\n");
	return EXIT_FAILURE;
      }
    
    yfile >> M;
    assert(N==M);
    
    HANDLE_ALLOC_ERROR(y = (real*)malloc(N4*sizeof(real)));
    
    for( int i = 0; i < N; i++)
      yfile >> y[i];
    for(int i = N; i < N4; i++)
      y[i] = 0;
    
    yfile.close();

    int Ndev4 = N4/4;


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
  kernel = clCreateKernel(program, "DotProduct", NULL);
  if(kernel == NULL)
    {
      std::cerr << "Failed to create kernel." << std::endl;
      Cleanup(context, commandQueue, program, kernel);
      return EXIT_FAILURE;
    }
  
  printf("%lf\n", toc(init));
     
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

    int WARP_SIZE = sizeOfWarp;

    size_t localWorkSize[1] = {WORK_GROUP_SIZE};
    int numWorkGroups = (Ndev4-1)/WORK_GROUP_SIZE+1;
    numWorkGroups = (numWorkGroups <= MAX_NUM_WORK_GROUPS) ? numWorkGroups : MAX_NUM_WORK_GROUPS;
    size_t globalWorkSize[1] = {numWorkGroups*WORK_GROUP_SIZE};

    //printf("Number of work groups: %d\n", numWorkGroups);

    TIME t = tic();

    cl_mem DEV_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				  sizeof(real)*N4, x, &status);
    HANDLE_OPENCL_ERROR(status);
    
    cl_mem DEV_y = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				  sizeof(real)*N4, y, &status);
    HANDLE_OPENCL_ERROR(status);

    cl_mem DEV_results = clCreateBuffer(context, CL_MEM_READ_WRITE,
				  sizeof(real)*numWorkGroups, NULL, &status);
    HANDLE_OPENCL_ERROR(status);
    
    int params_dim = 2;
    int params[] = {Ndev4, WARP_SIZE};
    
    cl_mem DEV_params = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				  sizeof(int)*params_dim, params, &status);
    HANDLE_OPENCL_ERROR(status);

    int n = 0;
    status  = clSetKernelArg(kernel, n++, sizeof(cl_mem), (void*)&DEV_x);
    status |= clSetKernelArg(kernel, n++, sizeof(cl_mem), (void*)&DEV_y);
    status |= clSetKernelArg(kernel, n++, sizeof(cl_mem), (void*)&DEV_results);
    status |= clSetKernelArg(kernel, n++, sizeof(real)*WORK_GROUP_SIZE, NULL);
    status |= clSetKernelArg(kernel, n++, sizeof(int)*params_dim, (void*)&DEV_params);
    HANDLE_OPENCL_ERROR(status);
    printf("%lf\n", toc(t));
    
    // Queue the kernel 
    HANDLE_OPENCL_ERROR(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
					       globalWorkSize, localWorkSize,
					       0, NULL, &myEvent));

    real *results;
    HANDLE_ALLOC_ERROR(results = (real*)malloc(numWorkGroups*sizeof(real)));

    // Read the output buffer back to the Host
    HANDLE_OPENCL_ERROR(clEnqueueReadBuffer(commandQueue, DEV_results, CL_TRUE,
					    0, numWorkGroups*sizeof(real), results,
					    0, NULL, &myEvent2));
    clFinish(commandQueue); // wait for all events to finish
   

    double elapsed_time = toc(start);

    real sum = 0;
    for (int i = 0; i < numWorkGroups; i++)
      sum += results[i];

    TIME start_seq = tic();
    real ref = 0;
    for(int i = 0; i < N; i++)
      ref += x[i] * y[i];
    double elapsed_time_seq = toc(start_seq);


    assert(ref < 10000000);
    //std::cout << ref <<"" << sum << std::endl;

    //assert(abs(ref - sum) < TOL);
    //printf("Verified...\n");


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
    printf(/*"Kernel execution time: %lf\n"*/"%lf\n", (double)readFromGpuTime/1000000000.0);
    printf(/*"Kernel execution time: %lf\n"*/"%lf\n", (double)kernelExecTimeNs/1000000000.0);
#endif
    printf(/*"Total execution time: %lf\n"*/"%lf\n", elapsed_time);
    printf(/*"Total execution time (seq): */"%lf\n", elapsed_time_seq);

    Cleanup(context, commandQueue, program, kernel);
    free(x);
    free(y);
    clReleaseMemObject(DEV_x);
    clReleaseMemObject(DEV_y);

    return EXIT_SUCCESS;
}

