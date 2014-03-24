#include "../../common/types_kernel.h"

__kernel void DotProduct( __global real4 * x, __global real4 * y, __global real * results, __local real *local_res, __global int *params )
{
  int itemIDglobal = get_global_id(0);
  int itemIDlocal  = get_local_id(0);
  int globalSize   = get_global_size(0);
  int localSize    = get_local_size(0);
  int groupID      = get_group_id(0);

  int i = itemIDglobal;
  int reductionSize;

  int N = params[0];
  int WARP_SIZE = params[1];

  local_res[itemIDlocal] = 0;

  while( i < N )
    { 
      local_res[itemIDlocal] += dot(x[i], y[i]);
      i += globalSize;
    }

  // parallel reduction inside a block
  reductionSize = localSize >> 1;

  barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
  for( reductionSize; reductionSize > WARP_SIZE; reductionSize >>= 1 )
    if (itemIDlocal < reductionSize)
      {
	local_res[itemIDlocal] += local_res[itemIDlocal + reductionSize];
	barrier(CLK_LOCAL_MEM_FENCE);
      }

  // continue reduction inside a warp without syncronization
#pragma unroll
  for( reductionSize; reductionSize >= 1; reductionSize >>= 1 )
    if (itemIDlocal < reductionSize)
      local_res[itemIDlocal] += local_res[itemIDlocal + reductionSize];


  // write result to global memory
  if(itemIDlocal == 0)
    results[groupID] = local_res[0];

}
