#include  "../../common/types_kernel.h"


__kernel void MatVecMul( __global real *Values, __global int *ColInd, __global int *RowPtr, 
			 __global real *x, __global real *res, __global int *params, 
			 __local real *tempSum /*every work item in work group will save its result here*/)
{

  int ITEMS_PER_ROW = params[0];
  int NRows = params[1];

  int itemIDglobal = get_global_id(0);                        // ID in grid
  int itemIDlocal  = get_local_id(0);                         // ID in work group
  int itemIDwarp   = itemIDglobal & (ITEMS_PER_ROW - 1);   // ID in warp
  int warpIDlocal  = itemIDlocal / ITEMS_PER_ROW;          // ID of warp in work group
  int warpIDglobal = itemIDglobal / ITEMS_PER_ROW;         // ID of warp in grid
  int globalSize   = get_global_size(0);                      // total number of work items in grid
  int localSize    = get_local_size(0);                       // total number of work items in work group
  int numWarps     = globalSize / ITEMS_PER_ROW; 

  int row_start, row_end;
  int row, col;
  real local_sum;
  int reductionSize;

  // each warp get one row
  for( row = warpIDglobal; row < NRows; row += numWarps )
    {
      row_start = RowPtr[row];
      row_end   = RowPtr[row+1];
      
      local_sum = 0;
      for( col = row_start + itemIDwarp; col < row_end; col += ITEMS_PER_ROW )
	local_sum += Values[col] * x[ColInd[col]];

      // save result to the local memory
      tempSum[itemIDlocal] = local_sum;

      // warp's are implicitly syncronized
 
      // parallel reduction inside a warp
      reductionSize = ITEMS_PER_ROW >> 1;
      __local real* p = tempSum + warpIDlocal*ITEMS_PER_ROW + itemIDwarp;

#pragma unroll
      for( reductionSize; reductionSize > 0; reductionSize >>= 1 )
	if (itemIDwarp < reductionSize)
	  p[0] += p[reductionSize];
      
      // Write the result of the reduction to global memory
      if (itemIDwarp == 0)
         res[row] = p[0];
    }

}


